import argparse
from pathlib import Path

import cv2
import numpy as np

from mouth_pipeline import StabilizedMouthExtractor


SEQ_LEN = 75
CROP_SIZE = (140, 46)


def normalize_video_to_npy(
    video_path: Path,
    output_path: Path,
    sequence_length: int = SEQ_LEN,
    target_fps: float | None = None,
) -> np.ndarray:
    extractor = StabilizedMouthExtractor(output_size=CROP_SIZE, realtime=False)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_stride = max(int(round(source_fps / target_fps)), 1) if target_fps else 1

    frames: list[np.ndarray] = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if index % frame_stride == 0:
            crop, _ = extractor.extract(frame)
            if crop is not None:
                frames.append(crop)
        index += 1

    cap.release()
    extractor.close()

    if not frames:
        raise RuntimeError("No face/mouth landmarks were detected in the video.")

    sequence = _fit_sequence(np.asarray(frames, dtype=np.float32), sequence_length)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, sequence)
    return sequence


def _fit_sequence(frames: np.ndarray, sequence_length: int) -> np.ndarray:
    if len(frames) == sequence_length:
        return frames
    if len(frames) > sequence_length:
        indices = np.linspace(0, len(frames) - 1, sequence_length).round().astype(np.int32)
        return frames[indices]

    padding = np.repeat(frames[-1][None, ...], sequence_length - len(frames), axis=0)
    return np.concatenate([frames, padding], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a stabilized mouth crop sequence as a .npy file.")
    parser.add_argument("video", type=Path, help="Input video path.")
    parser.add_argument("-o", "--output", type=Path, default=Path("mouth_sequence.npy"))
    parser.add_argument("--sequence-length", type=int, default=SEQ_LEN)
    parser.add_argument("--target-fps", type=float, default=None)
    args = parser.parse_args()

    sequence = normalize_video_to_npy(
        video_path=args.video,
        output_path=args.output,
        sequence_length=args.sequence_length,
        target_fps=args.target_fps,
    )
    print(f"Saved {args.output} with shape {sequence.shape} and dtype {sequence.dtype}")


if __name__ == "__main__":
    main()
