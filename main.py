from collections import deque
from threading import Event, Lock, Thread
import time

import cv2
import numpy as np
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


def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    char_to_num = tf.keras.layers.StringLookup(vocabulary=VOCAB, oov_token="")
    num_to_char = tf.keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(),
        oov_token="",
        invert=True,
    )
    return model, num_to_char


def decode_prediction(prediction, num_to_char):
    decoded, _ = tf.keras.backend.ctc_decode(
        prediction,
        input_length=np.array([prediction.shape[1]]),
        greedy=True,
    )
    token_ids = decoded[0][0].numpy()
    token_ids = token_ids[token_ids >= 0]
    text = tf.strings.reduce_join(num_to_char(token_ids)).numpy().decode("utf-8")
    confidence = float(np.max(prediction[0], axis=-1).mean())
    return text.strip(), confidence


def is_plausible_text(text):
    compact = text.replace(" ", "")
    if not compact:
        return False
    if len(compact) > 3 and len(set(compact)) <= 1:
        return False
    return True


def draw_overlay(frame, text, raw_text, confidence, frame_count, fps, motion_score, speech_active, crop):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 95), (w, h), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text or "Listening...",
        (24, h - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    mode = "speaking" if speech_active else "silent"
    status = f"{mode}  {frame_count}/{SEQ_LEN} frames  {fps:0.1f} FPS  conf {confidence:0.2f}  motion {motion_score:0.3f}"
    if SHOW_RAW_DEBUG and raw_text:
        status = f"{status}  raw:{raw_text[:24]}"
    cv2.putText(
        frame,
        status,
        (24, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (160, 230, 180),
        1,
        cv2.LINE_AA,
    )
    if crop is not None:
        preview = (crop * 255).clip(0, 255).astype(np.uint8)
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
        preview = cv2.resize(preview, (280, 92), interpolation=cv2.INTER_NEAREST)
        frame[12:104, 12:292] = preview
        cv2.rectangle(frame, (12, 12), (292, 104), (0, 220, 120), 1)


def main():
    print("Loading model...")
    model, num_to_char = load_assets()
    extractor = StabilizedMouthExtractor(output_size=CROP_SIZE, realtime=True, crop_mode=CROP_MODE)
    buffer = deque(maxlen=SEQ_LEN)
    state = {
        "latest_text": "",
        "raw_text": "",
        "stable_text": "",
        "candidate_text": "",
        "candidate_repeats": 0,
        "latest_confidence": 0.0,
        "pending_sequence": None,
    }
    lock = Lock()
    predict_event = Event()
    previous_crop = None
    motion_score = 0.0
    last_motion_time = 0.0
    speech_active = False
    frame_index = 0
    last_time = time.perf_counter()
    fps = 0.0
    last_crop = None

    def prediction_worker():
        while True:
            predict_event.wait()
            with lock:
                sequence = state["pending_sequence"]
                state["pending_sequence"] = None
                predict_event.clear()

            if sequence is None:
                continue

            x = np.expand_dims(sequence, axis=(0, -1))
            prediction = model.predict(x, verbose=0)
            text, confidence = decode_prediction(prediction, num_to_char)

            with lock:
                state["raw_text"] = text
                state["latest_confidence"] = confidence
                if text and confidence >= CONFIDENCE_THRESHOLD and is_plausible_text(text):
                    state["latest_text"] = text
                    if text == state["candidate_text"]:
                        state["candidate_repeats"] += 1
                    else:
                        state["candidate_text"] = text
                        state["candidate_repeats"] = 1

                    if state["candidate_repeats"] >= STABLE_REPEATS:
                        state["stable_text"] = text
                else:
                    state["latest_text"] = ""
                    state["candidate_text"] = ""
                    state["candidate_repeats"] = 0

    Thread(target=prediction_worker, daemon=True).start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 24)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Webcam started. Press q to quit, c to clear text.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        crop, mouth_quad = extractor.extract(frame)
        if crop is not None:
            last_crop = crop
            buffer.append(crop)
            if previous_crop is not None:
                motion = float(np.mean(np.abs(crop - previous_crop)))
                motion_score = 0.85 * motion_score + 0.15 * motion
                if motion_score >= MOTION_THRESHOLD:
                    last_motion_time = time.perf_counter()
            previous_crop = crop

            if mouth_quad is not None:
                cv2.polylines(
                    frame,
                    [mouth_quad.astype(np.int32)],
                    isClosed=True,
                    color=(0, 220, 120),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

        frame_index += 1
        speech_active = (time.perf_counter() - last_motion_time) <= SILENCE_SECONDS

        if not speech_active:
            with lock:
                state["latest_text"] = ""
                state["raw_text"] = ""
                state["stable_text"] = ""
                state["candidate_text"] = ""
                state["candidate_repeats"] = 0
                state["latest_confidence"] = 0.0

        if speech_active and len(buffer) == SEQ_LEN and frame_index % PREDICT_EVERY_FRAMES == 0:
            with lock:
                if state["pending_sequence"] is None:
                    state["pending_sequence"] = np.asarray(buffer, dtype=np.float32)
                    predict_event.set()

        now = time.perf_counter()
        elapsed = now - last_time
        if elapsed > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps else 1.0 / elapsed
        last_time = now

        with lock:
            text = state["stable_text"] or state["latest_text"]
            raw_text = state["raw_text"]
            confidence = state["latest_confidence"]
        draw_overlay(frame, text, raw_text, confidence, len(buffer), fps, motion_score, speech_active, last_crop)
        cv2.imshow("Lip Reading Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            with lock:
                state["latest_text"] = ""
                state["raw_text"] = ""
                state["stable_text"] = ""
                state["candidate_text"] = ""
                state["candidate_repeats"] = 0
                state["latest_confidence"] = 0.0

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
