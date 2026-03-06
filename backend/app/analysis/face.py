from __future__ import annotations

import base64

import cv2
import mediapipe as mp
import numpy as np

from .common import clamp01
from .mediapipe_tasks import get_face_model_path


TARGET_FPS = 12
MOUTH_ALPHA = 0.65
EYE_BETA = 0.35


LEFT_MOUTH = 61
RIGHT_MOUTH = 291
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
LEFT_FACE_ANCHOR = 234
RIGHT_FACE_ANCHOR = 454


def analyze_face_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sample_step = max(1, int(round(fps / TARGET_FPS)))

    mouth_vals: list[float] = []
    eye_vals: list[float] = []
    total_frames = 0
    valid_frames = 0
    best_frame_rgb = None
    best_points = None
    best_metrics = None
    best_score = -1.0

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=get_face_model_path()),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
    )

    with FaceLandmarker.create_from_options(options) as face_mesh:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total_frames += 1
            idx += 1
            if idx % sample_step != 0:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((idx / max(fps, 1e-6)) * 1000)
            results = face_mesh.detect_for_video(mp_image, timestamp_ms)

            if not results.face_landmarks:
                continue

            landmarks = results.face_landmarks[0]
            face_width = abs(landmarks[LEFT_FACE_ANCHOR].x - landmarks[RIGHT_FACE_ANCHOR].x)
            if face_width < 1e-6:
                continue

            mouth_diff = abs(landmarks[LEFT_MOUTH].y - landmarks[RIGHT_MOUTH].y)
            eye_open_left = abs(landmarks[LEFT_EYE_TOP].y - landmarks[LEFT_EYE_BOTTOM].y)
            eye_open_right = abs(landmarks[RIGHT_EYE_TOP].y - landmarks[RIGHT_EYE_BOTTOM].y)
            eye_diff = abs(eye_open_left - eye_open_right)

            mouth_vals.append(mouth_diff / face_width)
            eye_vals.append(eye_diff / face_width)
            valid_frames += 1

            side_deviation = abs(
                ((landmarks[LEFT_MOUTH].x + landmarks[RIGHT_MOUTH].x) / 2)
                - ((landmarks[LEFT_FACE_ANCHOR].x + landmarks[RIGHT_FACE_ANCHOR].x) / 2)
            ) / face_width
            frame_score = MOUTH_ALPHA * (mouth_diff / face_width) + EYE_BETA * (eye_diff / face_width)
            if frame_score > best_score:
                best_score = float(frame_score)
                best_frame_rgb = rgb.copy()
                best_points = {
                    "left_mouth": {"x": landmarks[LEFT_MOUTH].x, "y": landmarks[LEFT_MOUTH].y},
                    "right_mouth": {"x": landmarks[RIGHT_MOUTH].x, "y": landmarks[RIGHT_MOUTH].y},
                    "left_eye_top": {"x": landmarks[LEFT_EYE_TOP].x, "y": landmarks[LEFT_EYE_TOP].y},
                    "left_eye_bottom": {"x": landmarks[LEFT_EYE_BOTTOM].x, "y": landmarks[LEFT_EYE_BOTTOM].y},
                    "right_eye_top": {"x": landmarks[RIGHT_EYE_TOP].x, "y": landmarks[RIGHT_EYE_TOP].y},
                    "right_eye_bottom": {"x": landmarks[RIGHT_EYE_BOTTOM].x, "y": landmarks[RIGHT_EYE_BOTTOM].y},
                    "left_anchor": {"x": landmarks[LEFT_FACE_ANCHOR].x, "y": landmarks[LEFT_FACE_ANCHOR].y},
                    "right_anchor": {"x": landmarks[RIGHT_FACE_ANCHOR].x, "y": landmarks[RIGHT_FACE_ANCHOR].y},
                }
                best_metrics = {
                    "mouth_diff_norm": float(mouth_diff / face_width),
                    "eye_diff_norm": float(eye_diff / face_width),
                    "side_deviation_norm": float(side_deviation),
                }

    cap.release()

    mouth_mean = float(np.mean(mouth_vals)) if mouth_vals else 0.0
    eye_mean = float(np.mean(eye_vals)) if eye_vals else 0.0

    raw_score = MOUTH_ALPHA * mouth_mean + EYE_BETA * eye_mean
    score = clamp01(raw_score)

    overlay = None
    if best_frame_rgb is not None and best_points is not None and best_metrics is not None:
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(best_frame_rgb, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 82],
        )
        if ok:
            overlay = {
                "image_base64": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"),
                "image_width": int(best_frame_rgb.shape[1]),
                "image_height": int(best_frame_rgb.shape[0]),
                "keypoints": best_points,
                "metrics": best_metrics,
            }

    return {
        "score": score,
        "details": {
            "mouth_mean": mouth_mean,
            "eye_mean": eye_mean,
            "valid_frames": valid_frames,
            "total_frames": total_frames,
            "quality_confidence": round(valid_frames / max(total_frames, 1), 3),
        },
        "overlay": overlay,
    }
