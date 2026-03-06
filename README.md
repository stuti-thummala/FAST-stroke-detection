# FAST Stroke Detection (Automated FAST Screening)

Browser-based guided FAST exam with a local Python backend that quantifies:

- Facial Asymmetry Score (MediaPipe FaceMesh)
- Arm Drift Score (MediaPipe Pose)
- Speech Instability Score (librosa audio features)

It returns a triage report with a composite FAST risk index, category, recommendation, and disclaimer.

## Medical intent

This project digitizes the FAST protocol for **screening support**:

- **F**ace drooping
- **A**rm weakness
- **S**peech difficulty
- **T**ime to call emergency services

It is not intended to replace diagnosis. It can support settings where clinicians are not continuously available.

## Features

- Guided localhost workflow to record:
	- 5s smile video
	- 5s both-arms-raised video
	- 5s speech audio for a fixed sentence
- Local backend analysis of uploaded media
- Composite FAST risk:
	- `FAST_Risk = 0.4*Facial + 0.4*Arm + 0.2*Speech`
	- Low `< 0.30`, Moderate `0.30–0.60`, High `>= 0.60`
- Rule-based impression for likely stroke vs likely mimic pattern
- Result dashboard with score table, arm drift chart, recommendation, disclaimer

## Project structure

```text
backend/
	app/
		main.py
		analysis/
			face.py
			arm.py
			speech.py
			risk.py
			common.py
		static/
			index.html
			main.js
			styles.css
requirements.txt
```

## Prerequisites

- Python 3.10+ (3.11 recommended)
- `ffmpeg` installed and available on PATH (used for some audio conversions)
- Internet access on first run to download MediaPipe `.task` model files (cached under `backend/models/`)

On macOS (Homebrew):

```bash
brew install ffmpeg
```

## Setup

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run locally

```bash
uvicorn backend.app.main:app --reload
```

Open:

- `http://127.0.0.1:8000/`

Then perform all three recordings and click **Analyze FAST**.

## Output schema

API endpoint: `POST /api/analyze`

Returns:

- `facial.score`, details (`mouth_mean`, `eye_mean`, frame counts)
- `arm.score`, details (slopes, drift/final diff), `timeseries` for graphing
- `speech.score`, details (`pause_var`, `articulation_dev`, `mfcc_var`, duration)
- `report` containing:
	- 3 scores
	- `fast_risk_index`
	- `category`
	- `rule_based_impression`
	- `recommendation`
	- `disclaimer`

## Clinical disclaimer

This software is for triage/screening support only and not a medical diagnosis.
If stroke is suspected, emergency care should be activated immediately.