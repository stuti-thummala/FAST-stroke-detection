from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from .common import clamp01, norm
from .mediapipe_tasks import get_pose_model_path


TARGET_FPS = 12
W_DRIFT = 0.6
W_FINAL = 0.4


LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


def _torso_length(landmarks) -> float:
    shoulder_center_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2
    hip_center_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2
    length = abs(hip_center_y - shoulder_center_y)
    return max(length, 1e-4)


def analyze_arm_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sample_step = max(1, int(round(fps / TARGET_FPS)))

    times: list[float] = []
    left_series: list[float] = []
    right_series: list[float] = []

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=get_pose_model_path()),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
    )

    with PoseLandmarker.create_from_options(options) as pose:
        frame_idx = 0
        sampled_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % sample_step != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_idx / max(fps, 1e-6)) * 1000)
            res = pose.detect_for_video(mp_image, timestamp_ms)
            if not res.pose_landmarks:
                continue

            lm = res.pose_landmarks[0]
            torso = _torso_length(lm)
            shoulder_ref = (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2

            left_y = (lm[LEFT_WRIST].y - shoulder_ref) / torso
            right_y = (lm[RIGHT_WRIST].y - shoulder_ref) / torso

            sampled_idx += 1
            times.append(sampled_idx / TARGET_FPS)
            left_series.append(float(left_y))
            right_series.append(float(right_y))

    cap.release()

    if len(times) < 2:
        slope_l = 0.0
        slope_r = 0.0
        intercept_l = 0.0
        intercept_r = 0.0
        drift_metric = 0.0
        final_height_diff = 0.0
        score = 0.0
        asymmetry_regions: list[dict] = []
    else:
        t = np.array(times)
        y_l = np.array(left_series)
        y_r = np.array(right_series)

        slope_l, intercept_l = [float(v) for v in np.polyfit(t, y_l, 1)]
        slope_r, intercept_r = [float(v) for v in np.polyfit(t, y_r, 1)]
        drift_metric = abs(slope_l - slope_r)
        final_height_diff = abs(y_l[-1] - y_r[-1])

        score = clamp01(W_DRIFT * norm(drift_metric, 0.08) + W_FINAL * norm(final_height_diff, 0.6))

        asymmetry_regions = []
        diff = np.abs(y_l - y_r)
        threshold = 0.2
        start_idx = None
        for i, value in enumerate(diff):
            if value >= threshold and start_idx is None:
                start_idx = i
            elif value < threshold and start_idx is not None:
                segment = diff[start_idx:i]
                asymmetry_regions.append(
                    {
                        "start": float(t[start_idx]),
                        "end": float(t[i - 1]),
                        "max_diff": float(np.max(segment)),
                    }
                )
                start_idx = None
        if start_idx is not None:
            segment = diff[start_idx:]
            asymmetry_regions.append(
                {
                    "start": float(t[start_idx]),
                    "end": float(t[-1]),
                    "max_diff": float(np.max(segment)),
                }
            )

    return {
        "score": score,
        "details": {
            "slope_left": slope_l,
            "slope_right": slope_r,
            "intercept_left": intercept_l,
            "intercept_right": intercept_r,
            "drift_metric": drift_metric,
            "final_height_diff": final_height_diff,
            "quality_confidence": round(len(times) / max(len(times) + 2, 1), 3),
        },
        "timeseries": {
            "time": times,
            "left_wrist_y": left_series,
            "right_wrist_y": right_series,
            "start_markers": {
                "left": {"t": times[0] if times else 0.0, "y": left_series[0] if left_series else 0.0},
                "right": {"t": times[0] if times else 0.0, "y": right_series[0] if right_series else 0.0},
            },
            "end_markers": {
                "left": {"t": times[-1] if times else 0.0, "y": left_series[-1] if left_series else 0.0},
                "right": {"t": times[-1] if times else 0.0, "y": right_series[-1] if right_series else 0.0},
            },
            "slope_overlay": {
                "left": {
                    "start": {"t": times[0] if times else 0.0, "y": (slope_l * times[0] + intercept_l) if times else 0.0},
                    "end": {"t": times[-1] if times else 0.0, "y": (slope_l * times[-1] + intercept_l) if times else 0.0},
                },
                "right": {
                    "start": {"t": times[0] if times else 0.0, "y": (slope_r * times[0] + intercept_r) if times else 0.0},
                    "end": {"t": times[-1] if times else 0.0, "y": (slope_r * times[-1] + intercept_r) if times else 0.0},
                },
            },
            "asymmetry_regions": asymmetry_regions,
        },
    }
