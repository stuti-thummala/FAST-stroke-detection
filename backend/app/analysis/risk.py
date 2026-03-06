from __future__ import annotations

from .common import clamp01


DISCLAIMER = (
    "This tool is for screening support only and is not a diagnosis. "
    "If stroke is suspected, contact emergency services immediately."
)


def _confidence_band(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.45:
        return "Moderate"
    return "Low"


def _category(risk: float) -> str:
    if risk < 0.30:
        return "Low"
    if risk < 0.60:
        return "Moderate"
    return "High"


def _recommendation(category: str, rule_inputs: dict[str, bool]) -> str:
    sudden_onset = not rule_inputs.get("onset_gradual", False)
    if category == "High":
        return "Urgent emergency escalation recommended now (activate stroke/emergency pathway)."
    if category == "Moderate":
        if sudden_onset:
            return "Repeat capture only if quality was poor, but escalate promptly if onset is sudden or symptoms are new."
        return "Prompt clinical evaluation recommended as soon as possible."
    return "No major abnormality detected in this screening, but clinical judgment is still required if symptoms persist or worsen."


def _rule_based_impression(rule_inputs: dict[str, bool]) -> str:
    droop_only = rule_inputs.get("facial_droop_without_arm", False)
    speech_intact = rule_inputs.get("speech_intact", False)
    onset_gradual = rule_inputs.get("onset_gradual", False)

    if droop_only and speech_intact and onset_gradual:
        return "Likely mimic pattern"
    return "Likely stroke pattern"


def build_report(
    facial_score: float,
    arm_score: float,
    speech_score: float,
    rule_inputs: dict[str, bool],
    confidence_score: float = 0.5,
) -> dict:
    fast_risk = clamp01(0.4 * facial_score + 0.4 * arm_score + 0.2 * speech_score)
    category = _category(fast_risk)
    recommendation = _recommendation(category, rule_inputs)
    impression = _rule_based_impression(rule_inputs)

    return {
        "facial_asymmetry_score": facial_score,
        "arm_drift_score": arm_score,
        "speech_instability_score": speech_score,
        "fast_risk_index": fast_risk,
        "category": category,
        "quality_confidence": round(confidence_score, 3),
        "confidence_band": _confidence_band(confidence_score),
        "rule_based_impression": impression,
        "recommendation": recommendation,
        "disclaimer": DISCLAIMER,
    }


def build_explainability(face_result: dict, arm_result: dict, speech_result: dict) -> dict:
    mouth_mean = face_result["details"].get("mouth_mean", 0.0)
    eye_mean = face_result["details"].get("eye_mean", 0.0)
    face_driver = "mouth asymmetry" if mouth_mean >= eye_mean else "eye opening asymmetry"

    drift_metric = arm_result["details"].get("drift_metric", 0.0)
    final_diff = arm_result["details"].get("final_height_diff", 0.0)
    arm_driver = "trajectory drift slope" if drift_metric >= final_diff else "final arm height difference"

    pause_var = speech_result["details"].get("pause_var", 0.0)
    articulation_dev = speech_result["details"].get("articulation_dev", 0.0)
    mfcc_var = speech_result["details"].get("mfcc_var", 0.0)
    speech_driver = max(
        {
            "pause variability": pause_var,
            "articulation pace": articulation_dev,
            "spectral instability": mfcc_var,
        }.items(),
        key=lambda item: item[1],
    )[0]

    return {
        "facial": f"Facial score driven mainly by {face_driver}.",
        "arm": f"Arm score driven mainly by {arm_driver}.",
        "speech": f"Speech score driven mainly by {speech_driver}.",
    }


def build_baseline_comparison(current: dict[str, float], baseline: dict[str, float] | None) -> dict | None:
    if not baseline:
        return None

    comparisons = {}
    flag = False
    for key, label in (
        ("facial", "Facial asymmetry"),
        ("arm", "Arm drift"),
        ("speech", "Speech instability"),
    ):
        current_value = float(current.get(key, 0.0))
        baseline_value = float(baseline.get(key, 0.0))
        delta = round(current_value - baseline_value, 3)
        worsened = delta > 0.08
        flag = flag or worsened
        comparisons[key] = {
            "label": label,
            "baseline": round(baseline_value, 3),
            "current": round(current_value, 3),
            "delta": delta,
            "status": "New deviation from baseline detected" if worsened else "Within expected baseline range",
        }

    return {
        "summary": "New deviation from baseline detected" if flag else "Current screening is near baseline.",
        "components": comparisons,
    }
