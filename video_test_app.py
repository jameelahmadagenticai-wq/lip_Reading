import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from mouth_pipeline import StabilizedMouthExtractor


SEQ_LEN = 75
CROP_SIZE = (140, 46)
VOCAB = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]


@st.cache_resource
def load_model_and_decoder(model_path: str):
    model = tf.keras.models.load_model(model_path, compile=False)
    char_to_num = tf.keras.layers.StringLookup(vocabulary=VOCAB, oov_token="")
    num_to_char = tf.keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(),
        oov_token="",
        invert=True,
    )
    return model, num_to_char


def decode_prediction(prediction: np.ndarray, num_to_char) -> tuple[str, float]:
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


def alignment_to_text(alignment: str) -> str:
    words = []
    for line in alignment.splitlines():
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        word = parts[2].lower()
        if word not in {"sil", "sp"}:
            words.append(word)
    return " ".join(words)


def edit_distance(a, b) -> int:
    dp = list(range(len(b) + 1))
    for i, item_a in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, item_b in enumerate(b, 1):
            old = dp[j]
            cost = 0 if item_a == item_b else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = old
    return dp[-1]


def error_rates(expected: str, predicted: str) -> tuple[float, float]:
    expected_words = expected.split()
    predicted_words = predicted.split()
    wer = edit_distance(expected_words, predicted_words) / max(len(expected_words), 1)

    expected_chars = list(expected.replace(" ", ""))
    predicted_chars = list(predicted.replace(" ", ""))
    cer = edit_distance(expected_chars, predicted_chars) / max(len(expected_chars), 1)
    return wer, cer


def fit_sequence(frames: np.ndarray) -> np.ndarray:
    if len(frames) == 0:
        raise RuntimeError("No usable mouth frames were extracted.")
    if len(frames) == SEQ_LEN:
        return frames.astype(np.float32)
    if len(frames) > SEQ_LEN:
        idx = np.linspace(0, len(frames) - 1, SEQ_LEN).round().astype(np.int32)
        return frames[idx].astype(np.float32)

    pad = np.repeat(frames[-1][None, ...], SEQ_LEN - len(frames), axis=0)
    return np.concatenate([frames, pad], axis=0).astype(np.float32)


def extract_fixed_grid_crop(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop = gray[190:236, 80:220]
        if crop.shape != (46, 140):
            crop = cv2.resize(crop, CROP_SIZE)
        frames.append(crop.astype(np.float32) / 255.0)

    cap.release()
    return fit_sequence(np.asarray(frames))


def extract_mediapipe_crop(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    extractor = StabilizedMouthExtractor(output_size=CROP_SIZE, realtime=False)
    frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        crop, _ = extractor.extract(frame)
        if crop is not None:
            frames.append(crop)

    cap.release()
    extractor.close()
    return fit_sequence(np.asarray(frames))


def predict_video(video_path: Path, model_path: str, crop_mode: str):
    model, num_to_char = load_model_and_decoder(model_path)

    if crop_mode == "Fixed GRID crop":
        sequence = extract_fixed_grid_crop(video_path)
    else:
        sequence = extract_mediapipe_crop(video_path)

    x = np.expand_dims(sequence, axis=(0, -1))
    prediction = model.predict(x, verbose=0)
    text, confidence = decode_prediction(prediction, num_to_char)
    return text, confidence, sequence


st.set_page_config(page_title="Video Lip Reading Test", layout="centered")
st.title("Video Lip Reading Test")

model_path = st.selectbox(
    "Model",
    ["final_model_gru.h5", "final_model.h5"],
    index=0,
)

crop_mode = st.radio(
    "Crop mode",
    ["Fixed GRID crop", "MediaPipe mouth crop"],
    index=0,
    horizontal=True,
)

uploaded = st.file_uploader("Upload a video", type=["mpg", "mp4", "avi", "mov"])

alignment_text = st.text_area(
    "Optional alignment / expected text",
    placeholder="Paste .align lines here, or type expected words like: set white with v nine again",
)

if uploaded is not None:
    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        video_path = Path(tmp.name)

    st.video(str(video_path))

    if st.button("Predict", type="primary"):
        with st.spinner("Extracting mouth frames and running model..."):
            try:
                text, confidence, sequence = predict_video(video_path, model_path, crop_mode)
            except Exception as exc:
                st.error(str(exc))
            else:
                st.metric("Confidence", f"{confidence:.2f}")
                st.subheader("Prediction")
                st.write(text or "(blank)")

                if alignment_text.strip():
                    expected = alignment_to_text(alignment_text)
                    if not expected:
                        expected = alignment_text.strip().lower()
                    wer, cer = error_rates(expected, text.lower())

                    st.subheader("Expected")
                    st.write(expected)
                    col1, col2 = st.columns(2)
                    col1.metric("WER", f"{wer:.2%}")
                    col2.metric("CER", f"{cer:.2%}")

                preview = sequence[np.linspace(0, len(sequence) - 1, 8).round().astype(np.int32)]
                cols = st.columns(4)
                for i, frame in enumerate(preview):
                    cols[i % 4].image(frame, clamp=True, caption=f"Frame {i + 1}")
