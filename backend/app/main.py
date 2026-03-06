from __future__ import annotations

import os
import tempfile
from io import BytesIO
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .analysis.arm import analyze_arm_video
from .analysis.face import analyze_face_video
from .analysis.risk import build_baseline_comparison, build_explainability, build_report
from .analysis.speech import analyze_speech_audio
from .reporting import generate_pdf
from .storage import append_session, latest_baseline, latest_session, list_sessions

import qrcode


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="FAST Stroke Screening API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/analyze")
async def analyze_fast(
    face_video: UploadFile = File(...),
    arm_video: UploadFile = File(...),
    speech_audio: UploadFile = File(...),
    is_baseline: bool = Form(False),
    patient_id: str | None = Form(None),
    screener_name: str | None = Form(None),
    language: str = Form("English"),
    retake_reason: str | None = Form(None),
    face_capture_id: str | None = Form(None),
    arm_capture_id: str | None = Form(None),
    speech_capture_id: str | None = Form(None),
):
    tmp_paths: list[str] = []
    try:
        for upload in (face_video, arm_video, speech_audio):
            suffix = Path(upload.filename or "").suffix or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                data = await upload.read()
                tmp.write(data)
                tmp_paths.append(tmp.name)

        try:
            face_result = analyze_face_video(tmp_paths[0])
            arm_result = analyze_arm_video(tmp_paths[1])
            speech_result = analyze_speech_audio(tmp_paths[2])
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        confidence_score = (
            face_result["details"].get("quality_confidence", 0.0)
            + arm_result["details"].get("quality_confidence", 0.0)
            + speech_result["details"].get("quality_confidence", 0.0)
        ) / 3.0

        report = build_report(
            facial_score=face_result["score"],
            arm_score=arm_result["score"],
            speech_score=speech_result["score"],
            rule_inputs={},
            confidence_score=confidence_score,
        )

        explainability = build_explainability(face_result, arm_result, speech_result)

        baseline = None
        prior = latest_baseline(patient_id) or latest_session(patient_id)
        if prior:
            prior_report = prior.get("report", {})
            baseline = {
                "facial": float(prior_report.get("facial_asymmetry_score", 0.0)),
                "arm": float(prior_report.get("arm_drift_score", 0.0)),
                "speech": float(prior_report.get("speech_instability_score", 0.0)),
            }

        baseline_comparison = build_baseline_comparison(
            current={
                "facial": face_result["score"],
                "arm": arm_result["score"],
                "speech": speech_result["score"],
            },
            baseline=baseline,
        )

        session_payload = {
            "patient_id": patient_id,
            "screener_name": screener_name,
            "language": language,
            "retake_reason": retake_reason,
            "is_baseline": is_baseline,
            "media_capture_ids": {
                "face": face_capture_id,
                "arm": arm_capture_id,
                "speech": speech_capture_id,
            },
            "report": report,
            "baseline_comparison": baseline_comparison,
            "explainability": explainability,
        }
        append_session(session_payload)

        if is_baseline:
            return {
                "status": "baseline_saved",
                "patient_id": patient_id,
                "report": report,
            }

        return {
            "facial": face_result,
            "arm": arm_result,
            "speech": speech_result,
            "report": report,
            "explainability": explainability,
            "baseline_comparison": baseline_comparison,
            "confidence_breakdown": {
                "face":   round(face_result["details"].get("quality_confidence", 0.0), 3),
                "arm":    round(arm_result["details"].get("quality_confidence", 0.0), 3),
                "speech": round(speech_result["details"].get("quality_confidence", 0.0), 3),
            },
            "recent_sessions": list_sessions(patient_id=patient_id, limit=5) if patient_id else [],
        }
    finally:
        for path in tmp_paths:
            if os.path.exists(path):
                os.remove(path)


@app.get("/api/sessions")
def get_sessions(patient_id: str | None = Query(default=None), limit: int = Query(default=10, ge=1, le=50)):
    return {"sessions": list_sessions(patient_id=patient_id, limit=limit)}


@app.get("/api/qr/start-session")
def get_start_session_qr(target: str = Query(..., min_length=1, max_length=2048)):
    qr = qrcode.QRCode(version=None, box_size=8, border=2)
    qr.add_data(target)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@app.post("/api/report/pdf")
def export_report_pdf(payload: dict = Body(...)):
    pdf_bytes = generate_pdf(payload)
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=fast_screening_report.pdf"},
    )
