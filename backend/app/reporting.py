from __future__ import annotations

import base64
import io
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def _draw_wrapped_text(pdf: canvas.Canvas, text: str, x: int, y: int, max_width: int, line_height: int = 14) -> int:
    words = text.split()
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if pdf.stringWidth(candidate, "Helvetica", 10) <= max_width:
            line = candidate
        else:
            pdf.drawString(x, y, line)
            y -= line_height
            line = word
    if line:
        pdf.drawString(x, y, line)
        y -= line_height
    return y


def _data_url_to_image_reader(data_url: str | None):
    if not data_url or "," not in data_url:
        return None
    _, encoded = data_url.split(",", 1)
    return ImageReader(io.BytesIO(base64.b64decode(encoded)))


def generate_pdf(report_payload: dict) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(40, y, "FAST Stroke Screening Report")
    y -= 26

    pdf.setFont("Helvetica", 10)
    pdf.drawString(40, y, f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    y -= 16
    pdf.drawString(40, y, f"Patient ID: {report_payload.get('patient_id') or 'N/A'}")
    y -= 16
    pdf.drawString(40, y, f"Screener: {report_payload.get('screener_name') or 'N/A'}")
    y -= 16
    pdf.drawString(40, y, f"Language: {report_payload.get('language') or 'N/A'}")
    y -= 24

    report = report_payload.get("report", {})
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Scores")
    y -= 18
    pdf.setFont("Helvetica", 10)
    score_lines = [
        f"Facial Asymmetry Score: {report.get('facial_asymmetry_score', 0):.3f}",
        f"Arm Drift Score: {report.get('arm_drift_score', 0):.3f}",
        f"Speech Instability Score: {report.get('speech_instability_score', 0):.3f}",
        f"FAST Risk Index: {report.get('fast_risk_index', 0):.3f}",
        f"Risk Category: {report.get('category', 'N/A')}",
        f"Quality/Confidence: {report.get('quality_confidence', 0):.3f} ({report.get('confidence_band', 'N/A')})",
    ]
    for line in score_lines:
        pdf.drawString(50, y, line)
        y -= 14

    y -= 10
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Recommendation")
    y -= 18
    pdf.setFont("Helvetica", 10)
    y = _draw_wrapped_text(pdf, report.get("recommendation", "N/A"), 50, y, 500)
    y -= 8

    baseline = report_payload.get("baseline_comparison")
    if baseline:
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(40, y, "Baseline Comparison")
        y -= 18
        pdf.setFont("Helvetica", 10)
        y = _draw_wrapped_text(pdf, baseline.get("summary", ""), 50, y, 500)
        for item in baseline.get("components", {}).values():
            y = _draw_wrapped_text(
                pdf,
                f"{item['label']}: baseline {item['baseline']:.3f}, current {item['current']:.3f}, delta {item['delta']:.3f} — {item['status']}",
                50,
                y,
                500,
            )
        y -= 8

    explainability = report_payload.get("explainability", {})
    if explainability:
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(40, y, "Explainability")
        y -= 18
        pdf.setFont("Helvetica", 10)
        for key in ("facial", "arm", "speech"):
            if explainability.get(key):
                y = _draw_wrapped_text(pdf, explainability[key], 50, y, 500)
        y -= 8

    graphs = report_payload.get("graphs", {})
    for title, key in (
        ("Arm Drift Graph", "arm_graph"),
        ("Facial Overlay", "face_overlay"),
        ("Speech Timeline", "speech_waveform"),
    ):
        image = _data_url_to_image_reader(graphs.get(key))
        if image is None:
            continue
        if y < 220:
            pdf.showPage()
            y = height - 40
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(40, y, title)
        y -= 12
        pdf.drawImage(image, 40, y - 180, width=520, height=180, preserveAspectRatio=True, mask="auto")
        y -= 196

    pdf.setFont("Helvetica", 9)
    y = max(y, 70)
    y = _draw_wrapped_text(pdf, report.get("disclaimer", ""), 40, y, 520, line_height=12)

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()
