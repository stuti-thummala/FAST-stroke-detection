from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def _ensure_model(url: str, name: str) -> str:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / name
    if not model_path.exists():
        urlretrieve(url, str(model_path))
    return str(model_path)


def get_face_model_path() -> str:
    return _ensure_model(FACE_MODEL_URL, "face_landmarker.task")


def get_pose_model_path() -> str:
    return _ensure_model(POSE_MODEL_URL, "pose_landmarker_lite.task")
