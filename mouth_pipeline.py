from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


OUTER_LIP = np.array(
    [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
    dtype=np.int32,
)
INNER_LIP = np.array(
    [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
    dtype=np.int32,
)
LIP_INDICES = np.unique(np.concatenate([OUTER_LIP, INNER_LIP]))


@dataclass(frozen=True)
class MouthExtraction:
    crop: np.ndarray | None
    quad: np.ndarray | None


class StabilizedMouthExtractor:
    """Landmark-normalized mouth extractor for GRID-style lip-reading models."""

    def __init__(
        self,
        output_size: tuple[int, int] = (140, 46),
        realtime: bool = True,
        crop_mode: str = "warped_mouth",
        model_asset_path: str = "face_landmarker.task",
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ):
        self.output_size = output_size
        self.crop_mode = crop_mode
        self.realtime = realtime
        self.timestamp_ms = 0
        self.detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=model_asset_path),
                running_mode=vision.RunningMode.VIDEO if realtime else vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
        )

    def extract(self, frame_bgr: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        result = self.extract_result(frame_bgr)
        return result.crop, result.quad

    def extract_result(self, frame_bgr: np.ndarray) -> MouthExtraction:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.realtime:
            self.timestamp_ms += 33
            results = self.detector.detect_for_video(mp_image, self.timestamp_ms)
        else:
            results = self.detector.detect(mp_image)

        if not results.face_landmarks:
            return MouthExtraction(None, None)

        landmarks = results.face_landmarks[0]
        points = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks], dtype=np.float32)
        if self.crop_mode == "grid_like":
            quad = self._grid_like_quad(points)
        else:
            quad = self._mouth_quad(points)

        crop_w, crop_h = self.output_size
        dst = np.array(
            [[0, 0], [crop_w - 1, 0], [crop_w - 1, crop_h - 1], [0, crop_h - 1]],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        crop = cv2.warpPerspective(
            frame_bgr,
            transform,
            self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return MouthExtraction(crop, quad)

    def close(self) -> None:
        self.detector.close()

    def _mouth_quad(self, points: np.ndarray) -> np.ndarray:
        left = points[61]
        right = points[291]
        upper = (points[13] + points[0] + points[267] + points[37]) / 4.0
        lower = (points[14] + points[17] + points[314] + points[84]) / 4.0
        center = (left + right + upper + lower) / 4.0

        x_axis_3d = right - left
        y_axis_3d = lower - upper
        x_axis = x_axis_3d[:2]
        y_axis = y_axis_3d[:2]

        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1.0:
            return self._bounding_quad(points)
        x_axis = x_axis / x_norm

        y_axis = y_axis - x_axis * float(np.dot(y_axis, x_axis))
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1.0:
            y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float32)
        else:
            y_axis = y_axis / y_norm

        lip_points = points[LIP_INDICES]
        lip_centered = lip_points[:, :2] - center[:2]
        half_w = max(abs(lip_centered @ x_axis).max() * 1.55, x_norm * 0.72)
        half_h = max(abs(lip_centered @ y_axis).max() * 2.15, half_w * 0.26)

        yaw_compensation = 1.0 + min(abs(float(x_axis_3d[2])) / max(np.linalg.norm(x_axis_3d), 1.0), 0.35)
        half_w *= yaw_compensation

        c = center[:2]
        return np.array(
            [
                c - half_w * x_axis - half_h * y_axis,
                c + half_w * x_axis - half_h * y_axis,
                c + half_w * x_axis + half_h * y_axis,
                c - half_w * x_axis + half_h * y_axis,
            ],
            dtype=np.float32,
        )

    def _grid_like_quad(self, points: np.ndarray) -> np.ndarray:
        """Lower-face crop shaped like the original GRID frame[190:236, 80:220].

        The trained model saw more than lips: lower nose, mouth, cheeks, and a bit
        of chin. This crop deliberately keeps that context instead of tightly
        ortho-rectifying only the lip contour.
        """
        left = points[61]
        right = points[291]
        upper_lip = (points[13] + points[0]) / 2.0
        lower_lip = (points[14] + points[17]) / 2.0
        nose_base = points[2] if len(points) > 2 else upper_lip

        x_axis_3d = right - left
        x_axis = x_axis_3d[:2]
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1.0:
            return self._bounding_quad(points)
        x_axis = x_axis / x_norm
        y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float32)

        mouth_center = ((left + right + upper_lip + lower_lip) / 4.0)[:2]
        nose_to_mouth = mouth_center - nose_base[:2]
        if np.linalg.norm(nose_to_mouth) > 1.0 and float(np.dot(nose_to_mouth, y_axis)) < 0:
            y_axis = -y_axis

        aspect = self.output_size[0] / self.output_size[1]
        half_w = x_norm * 1.28
        half_h = half_w / aspect

        # Move slightly downward from the lip center so the crop includes chin
        # while still retaining lower-nose context, similar to GRID crops.
        center = mouth_center + y_axis * (half_h * 0.08)

        yaw_compensation = 1.0 + min(abs(float(x_axis_3d[2])) / max(np.linalg.norm(x_axis_3d), 1.0), 0.25)
        half_w *= yaw_compensation
        half_h *= yaw_compensation

        return np.array(
            [
                center - half_w * x_axis - half_h * y_axis,
                center + half_w * x_axis - half_h * y_axis,
                center + half_w * x_axis + half_h * y_axis,
                center - half_w * x_axis + half_h * y_axis,
            ],
            dtype=np.float32,
        )

    def _bounding_quad(self, points: np.ndarray) -> np.ndarray:
        lip_points = points[LIP_INDICES, :2]
        x_min, y_min = lip_points.min(axis=0)
        x_max, y_max = lip_points.max(axis=0)
        pad_x = (x_max - x_min) * 0.45
        pad_y = (y_max - y_min) * 0.9
        return np.array(
            [
                [x_min - pad_x, y_min - pad_y],
                [x_max + pad_x, y_min - pad_y],
                [x_max + pad_x, y_max + pad_y],
                [x_min - pad_x, y_max + pad_y],
            ],
            dtype=np.float32,
        )
