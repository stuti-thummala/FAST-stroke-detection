from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import librosa
import numpy as np

from .common import clamp01, norm


HOP = 256
FRAME = 1024


def _load_audio_16k_mono(path: str):
    try:
        return librosa.load(path, sr=16000, mono=True)
    except Exception:
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError(
                "Speech conversion requires ffmpeg for webm audio. Install ffmpeg and retry."
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name
        try:
            result = subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-i",
                    path,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    wav_path,
                ],
                capture_output=True,
                check=False,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError("Audio conversion failed; ensure ffmpeg is installed.")
            return librosa.load(wav_path, sr=16000, mono=True)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Audio conversion timed out while invoking ffmpeg.") from exc
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)


def _pause_durations(pause_mask: np.ndarray, frame_seconds: float) -> list[float]:
    durations: list[float] = []
    run = 0
    for flag in pause_mask:
        if flag:
            run += 1
        elif run > 0:
            durations.append(run * frame_seconds)
            run = 0
    if run > 0:
        durations.append(run * frame_seconds)
    return durations


def _mask_to_segments(mask: np.ndarray, frame_seconds: float, min_frames: int = 1) -> list[dict]:
    segments: list[dict] = []
    start = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= min_frames:
                segments.append(
                    {
                        "start": round(start * frame_seconds, 3),
                        "end": round((idx - 1) * frame_seconds, 3),
                    }
                )
            start = None
    if start is not None and len(mask) - start >= min_frames:
        segments.append(
            {
                "start": round(start * frame_seconds, 3),
                "end": round((len(mask) - 1) * frame_seconds, 3),
            }
        )
    return segments


def _downsample(values: np.ndarray, count: int) -> list[float]:
    if len(values) <= count:
        return [float(v) for v in values]
    idx = np.linspace(0, len(values) - 1, count).astype(int)
    return [float(values[i]) for i in idx]


def analyze_speech_audio(audio_path: str) -> dict:
    y, sr = _load_audio_16k_mono(audio_path)
    if len(y) == 0:
        return {
            "score": 0.0,
            "details": {
                "pause_var": 0.0,
                "articulation_dev": 0.0,
                "mfcc_var": 0.0,
                "duration": 0.0,
            },
        }

    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=FRAME, hop_length=HOP)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP)
    mfcc_delta = librosa.feature.delta(mfcc)

    duration = len(y) / sr
    frame_seconds = HOP / sr

    rms_threshold = max(0.01, float(np.percentile(rms, 30)))
    pause_mask = rms < rms_threshold
    pauses = _pause_durations(pause_mask, frame_seconds)
    pause_var = float(np.var(pauses)) if len(pauses) > 1 else 0.0

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    peaks = librosa.util.peak_pick(
        onset_env,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=5,
        delta=0.5,
        wait=max(1, int(np.ceil(float(np.mean(onset_env))))),
    )
    articulation_rate = len(peaks) / max(duration, 1e-6)
    articulation_dev = abs(articulation_rate - 2.2) / 2.2

    mfcc_var = float(np.var(mfcc) + 0.5 * np.var(mfcc_delta) + 0.1 * np.var(zcr))
    rms_delta = np.abs(np.diff(rms, prepend=rms[0]))
    unstable_mask = rms_delta > float(np.percentile(rms_delta, 85))

    score = clamp01(
        0.4 * norm(pause_var, 0.06)
        + 0.3 * norm(articulation_dev, 1.0)
        + 0.3 * norm(mfcc_var, 220.0)
    )

    return {
        "score": score,
        "details": {
            "pause_var": pause_var,
            "articulation_dev": articulation_dev,
            "articulation_rate": articulation_rate,
            "mfcc_var": mfcc_var,
            "duration": duration,
            "quality_confidence": round(min(1.0, duration / 5.0), 3),
        },
        "timeline": {
            "time": [round(i * frame_seconds, 3) for i in range(len(rms))],
            "rms": [float(v) for v in rms],
            "pause_segments": _mask_to_segments(pause_mask, frame_seconds, min_frames=max(2, int(0.35 / frame_seconds))),
            "unstable_segments": _mask_to_segments(unstable_mask, frame_seconds, min_frames=2),
            "waveform": _downsample(y, 400),
        },
    }
