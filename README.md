# 🧠 Silent Speech Visual Decoder (Lip Reading System)

## 📌 Overview

This project is an end-to-end **lip reading system** that converts visual mouth movements from video/webcam into predicted words — without using any audio. It focuses on building a **real-time, lightweight pipeline** using computer vision and deep learning.

---

## 🎯 Problem Statement

* Audio-based systems fail in noisy or silent environments
* Need a system that can understand speech **purely from visual input**
* Handle real-world challenges like lighting, angles, and face movement

---

## 🚀 Solution

* Extract mouth region using **MediaPipe Face Mesh**
* Convert video into **frame sequences (.npy format)**
* Train **CNN + LSTM model** to learn lip motion patterns
* Apply smoothing and confidence filtering for stable predictions

---

## 📂 Project Structure

```
.
├── app.py                     # Main application (entry point)
├── main.py                    # Core inference logic (model + webcam)
├── extract_mouth_sequence.py  # Converts video → mouth sequence (.npy)
├── mouth_pipeline.py          # Mouth extraction pipeline using MediaPipe
├── video_test_app.py          # Testing pipeline for video input
├── face_landmarker.task       # MediaPipe model for face landmarks
├── requirements.txt           # Dependencies
```

---

## 🔄 Workflow (End-to-End)

```
[ Video / Webcam Input ]
            ↓
[ Face Detection (MediaPipe) ]
            ↓
[ Mouth Region Extraction ]
            ↓
[ Resize + Normalize Frames ]
            ↓
[ Convert to Sequence (.npy) ]
            ↓
[ CNN (Spatial Features) ]
            ↓
[ TimeDistributed + Flatten ]
            ↓
[ LSTM (Temporal Learning) ]
            ↓
[ Dense Layer (Prediction) ]
            ↓
[ Confidence + Smoothing ]
            ↓
[ Final Word Output ]
```

---

## 🧠 Key Learnings

* Dataset quality (angles, lighting, consistency) is critical
* MediaPipe errors can corrupt entire dataset if not handled carefully
* Pretrained face models are not suitable for lip-only tasks
* Frame size must remain consistent in training and inference
* Generators are essential to avoid RAM crashes in Colab
* Conv3D works directly on sequence data but is heavier than CNN+LSTM
* Small datasets limit generalization → need larger data for scaling
* CTC / Transformer models are better for sentence-level prediction

---

## ⚙️ Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv myvenv
myvenv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Extract Dataset

```bash
python extract_mouth_sequence.py
```

### Run Webcam Prediction

```bash
python main.py
```

### Test with Video

```bash
python video_test_app.py
```

---

## ⚠️ Limitations

* Works only on **fixed vocabulary (trained words)**
* Sensitive to lighting and face angle
* Requires consistent preprocessing
* Limited dataset reduces accuracy

---

## 🔮 Future Improvements

* Move to **CTC-based sequence model** for sentence prediction
* Use **Transformer / Mamba models** for large-scale datasets
* Improve dataset diversity (angles, lighting, speakers)
* Optimize for real-time performance (quantization, streaming)

---

## 🏁 Conclusion

This project demonstrates a complete pipeline from **raw video → real-time lip prediction**, including data processing, model training, optimization, and inference. It lays the foundation for advanced systems like **silent speech recognition and accessibility tools**.

---
