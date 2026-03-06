from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from twilio.rest import Client

# Load .env from the project root (three levels up from this file)
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
EMERGENCY_CONTACT_NUMBER = os.getenv("EMERGENCY_CONTACT_NUMBER", "+14049327005")


def dispatch_code_stroke(patient_id: str | None, location: str = "Triage Bay 1") -> bool:
    """
    Triggers an automated emergency SMS and phone call via Twilio
    when a high-risk stroke scenario is detected by the AI.

    Returns True if both the SMS and call were dispatched successfully.
    """
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        logging.warning("Twilio credentials not found. Skipping Code Stroke dispatch.")
        return False

    patient_label = patient_id or "Unknown"

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        # 1. SMS alert
        sms_body = (
            f"\U0001f6a8 CODE STROKE ALERT \U0001f6a8\n"
            f"Patient {patient_label} at {location} has been flagged by FAST AI "
            f"with HIGH asymmetry/drift scores.\n"
            f"Immediate neurological evaluation recommended."
        )
        message = client.messages.create(
            body=sms_body,
            from_=TWILIO_PHONE_NUMBER,
            to=EMERGENCY_CONTACT_NUMBER,
        )
        logging.info("Code Stroke SMS dispatched: %s", message.sid)

        # 2. Automated voice call via TwiML text-to-speech
        twiml = (
            "<Response>"
            "<Say voice='Polly.Matthew' language='en-US'>"
            "Code Stroke Alert. Code Stroke Alert. "
            f"Patient {patient_label} at {location} requires immediate evaluation. "
            "High risk facial asymmetry detected. "
            "Please respond immediately."
            "</Say>"
            "</Response>"
        )
        call = client.calls.create(
            twiml=twiml,
            from_=TWILIO_PHONE_NUMBER,
            to=EMERGENCY_CONTACT_NUMBER,
        )
        logging.info("Code Stroke call dispatched: %s | SMS: %s", call.sid, message.sid)
        return True

    except Exception as exc:
        logging.error("Failed to dispatch Code Stroke alert: %s", exc)
        raise
