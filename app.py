from collections import deque
from threading import Event, Lock, Thread
import time

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
import tensorflow as tf

from mouth_pipeline import StabilizedMouthExtractor


SEQ_LEN = 75
CROP_SIZE = (140, 46)
CROP_MODE = "grid_like"
MODEL_PATH = "final_model_gru.h5"
PREDICT_EVERY_FRAMES = 8
CONFIDENCE_THRESHOLD = 0.40
STABLE_REPEATS = 1
MOTION_THRESHOLD = 0.003
SILENCE_SECONDS = 1.5
SHOW_RAW_DEBUG = True
VOCAB = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
MODEL_LOCK = Lock()


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    char_to_num = tf.keras.layers.StringLookup(vocabulary=VOCAB, oov_token="")
    num_to_char = tf.keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(),
        oov_token="",
        invert=True,
    )
    return model, num_to_char


MODEL, NUM_TO_CHAR = load_assets()


def decode_prediction(prediction: np.ndarray) -> tuple[str, float]:
    decoded, _ = tf.keras.backend.ctc_decode(
        prediction,
        input_length=np.array([prediction.shape[1]]),
        greedy=True,
    )

    token_ids = decoded[0][0].numpy()
    token_ids = token_ids[token_ids >= 0]
    text = tf.strings.reduce_join(NUM_TO_CHAR(token_ids)).numpy().decode("utf-8")

    frame_conf = float(np.max(prediction[0], axis=-1).mean())
    return text.strip(), frame_conf


def is_plausible_text(text: str) -> bool:
    compact = text.replace(" ", "")
    if not compact:
        return False
    if len(compact) > 3 and len(set(compact)) <= 1:
        return False
    return True


class LipReadingProcessor(VideoProcessorBase):
    def __init__(self):
        self.extractor = StabilizedMouthExtractor(output_size=CROP_SIZE, realtime=True, crop_mode=CROP_MODE)
        self.frames = deque(maxlen=SEQ_LEN)
        self.lock = Lock()
        self.predict_event = Event()
        self.pending_sequence = None
        self.latest_text = ""
        self.raw_text = ""
        self.latest_confidence = 0.0
        self.stable_text = ""
        self.candidate_text = ""
        self.candidate_repeats = 0
        self.previous_crop = None
        self.motion_score = 0.0
        self.last_motion_time = 0.0
        self.speech_active = False
        self.frame_count = 0
        self.last_time = time.perf_counter()
        self.fps = 0.0
        self.worker = Thread(target=self._prediction_worker, daemon=True)
        self.worker.start()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        crop, mouth_quad = self.extractor.extract(image)

        if crop is not None:
            self.frames.append(crop)
            self._update_motion(crop)
            if mouth_quad is not None:
                cv2.polylines(
                    image,
                    [mouth_quad.astype(np.int32)],
                    isClosed=True,
                    color=(0, 220, 120),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

        self.frame_count += 1
        self.speech_active = (time.perf_counter() - self.last_motion_time) <= SILENCE_SECONDS

        if not self.speech_active:
            with self.lock:
                self.latest_text = ""
                self.raw_text = ""
                self.stable_text = ""
                self.candidate_text = ""
                self.candidate_repeats = 0
                self.latest_confidence = 0.0

        if (
            self.speech_active
            and len(self.frames) == SEQ_LEN
            and self.frame_count % PREDICT_EVERY_FRAMES == 0
        ):
            with self.lock:
                if self.pending_sequence is None:
                    self.pending_sequence = np.asarray(self.frames, dtype=np.float32)
                    self.predict_event.set()

        now = time.perf_counter()
        elapsed = now - self.last_time
        if elapsed > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / elapsed) if self.fps else 1.0 / elapsed
        self.last_time = now

        self._draw_overlay(image)
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def _update_motion(self, crop: np.ndarray) -> None:
        if self.previous_crop is None:
            self.previous_crop = crop
            return

        motion = float(np.mean(np.abs(crop - self.previous_crop)))
        self.motion_score = 0.85 * self.motion_score + 0.15 * motion
        self.previous_crop = crop

        if self.motion_score >= MOTION_THRESHOLD:
            self.last_motion_time = time.perf_counter()

    def _prediction_worker(self) -> None:
        while True:
            self.predict_event.wait()
            with self.lock:
                sequence = self.pending_sequence
                self.pending_sequence = None
                self.predict_event.clear()

            if sequence is None:
                continue

            x = np.expand_dims(sequence, axis=(0, -1))
            with MODEL_LOCK:
                pred = MODEL.predict(x, verbose=0)

            text, confidence = decode_prediction(pred)
            with self.lock:
                self.raw_text = text
                self.latest_confidence = confidence
                if text and confidence >= CONFIDENCE_THRESHOLD and is_plausible_text(text):
                    self.latest_text = text
                    if text == self.candidate_text:
                        self.candidate_repeats += 1
                    else:
                        self.candidate_text = text
                        self.candidate_repeats = 1

                    if self.candidate_repeats >= STABLE_REPEATS:
                        self.stable_text = text
                else:
                    self.latest_text = ""
                    self.raw_text = text
                    self.candidate_text = ""
                    self.candidate_repeats = 0

    def _draw_overlay(self, image: np.ndarray) -> None:
        h, w = image.shape[:2]
        with self.lock:
            subtitle = self.stable_text or self.latest_text or "Listening..."
            mode = "speaking" if self.speech_active else "silent"
            status = (
                f"{mode}  {len(self.frames)}/{SEQ_LEN} frames  {self.fps:0.1f} FPS  "
                f"conf {self.latest_confidence:0.2f}  motion {self.motion_score:0.3f}"
            )
            if SHOW_RAW_DEBUG and self.raw_text:
                status = f"{status}  raw:{self.raw_text[:24]}"

        cv2.rectangle(image, (0, h - 96), (w, h), (0, 0, 0), -1)
        cv2.putText(
            image,
            subtitle[:80],
            (24, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            status,
            (24, h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (160, 230, 180),
            1,
            cv2.LINE_AA,
        )


st.set_page_config(page_title="Live Lip Reading", layout="centered")
st.title("Live Lip Reading")

st.caption("The webcam stream uses MediaPipe Face Mesh mouth stabilization, a 75-frame rolling buffer, and your saved TensorFlow CTC model.")

webrtc_streamer(
    key="lip-reader",
    video_processor_factory=LipReadingProcessor,
    media_stream_constraints={
        "video": {"width": 480, "height": 360, "frameRate": {"ideal": 24, "max": 30}},
        "audio": False,
    },
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    async_processing=True,
)
