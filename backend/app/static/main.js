const facePreview = document.getElementById("facePreview");
const armPreview = document.getElementById("armPreview");
const faceRecordBtn = document.getElementById("faceRecordBtn");
const armRecordBtn = document.getElementById("armRecordBtn");
const speechRecordBtn = document.getElementById("speechRecordBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const exportPdfBtn = document.getElementById("exportPdfBtn");
const printBtn = document.getElementById("printBtn");
const startQrImage = document.getElementById("startQrImage");
const startQrLink = document.getElementById("startQrLink");

const faceStatus = document.getElementById("faceStatus");
const armStatus = document.getElementById("armStatus");
const speechStatus = document.getElementById("speechStatus");
const analyzeStatus = document.getElementById("analyzeStatus");

const patientIdInput = document.getElementById("patientId");
const screenerNameInput = document.getElementById("screenerName");
const languageSelect = document.getElementById("languageSelect");
const retakeReasonInput = document.getElementById("retakeReason");

// Baseline recording elements
const toggleBaselineBtn  = document.getElementById("toggleBaselineBtn");
const baselineDrawer     = document.getElementById("baselineDrawer");
const bFaceBtn           = document.getElementById("bFaceBtn");
const bArmBtn            = document.getElementById("bArmBtn");
const bSpeechBtn         = document.getElementById("bSpeechBtn");
const saveBaselineBtn    = document.getElementById("saveBaselineBtn");
const bFaceStatus        = document.getElementById("bFaceStatus");
const bArmStatus         = document.getElementById("bArmStatus");
const bSpeechStatus      = document.getElementById("bSpeechStatus");
const saveBaselineStatus = document.getElementById("saveBaselineStatus");
const baselineSavedBanner = document.getElementById("baselineSavedBanner");
const baselineRecordStatus = document.getElementById("baselineRecordStatus");

const faceCoach = document.getElementById("faceCoach");
const armCoach = document.getElementById("armCoach");
const speechCoach = document.getElementById("speechCoach");
const speechPrompt = document.getElementById("speechPrompt");
const speechPlayback = document.getElementById("speechPlayback");
const speechWaveform = document.getElementById("speechWaveform");
const speechPauseInfo = document.getElementById("speechPauseInfo");
const speechPaceInfo = document.getElementById("speechPaceInfo");

const reportCard = document.getElementById("reportCard");
const facialScoreEl = document.getElementById("facialScore");
const armScoreEl = document.getElementById("armScore");
const speechScoreEl = document.getElementById("speechScore");
const riskScoreEl = document.getElementById("riskScore");
const riskCategoryEl = document.getElementById("riskCategory");
const ruleImpressionEl = document.getElementById("ruleImpression");
const confidenceInfoEl = document.getElementById("confidenceInfo");
const reportTimestampEl = document.getElementById("reportTimestamp");
const recommendationEl = document.getElementById("recommendation");
const recommendationPanel = document.getElementById("recommendationPanel");
const disclaimerEl = document.getElementById("disclaimer");
const armChart = document.getElementById("armChart");
const faceOverlayCanvas = document.getElementById("faceOverlayCanvas");
const speechTimeline = document.getElementById("speechTimeline");
const baselinePanel = document.getElementById("baselinePanel");
const explainabilityPanel = document.getElementById("explainabilityPanel");
const sessionHistory = document.getElementById("sessionHistory");

// Confidence breakdown panel
const confidenceBreakdown = document.getElementById("confidenceBreakdown");
const confOverallEl     = document.getElementById("confOverall");
const confFaceFill      = document.getElementById("confFaceFill");
const confArmFill       = document.getElementById("confArmFill");
const confSpeechFill    = document.getElementById("confSpeechFill");
const confFaceVal       = document.getElementById("confFaceVal");
const confArmVal        = document.getElementById("confArmVal");
const confSpeechVal     = document.getElementById("confSpeechVal");
// Gauge
const gaugeFillEl   = document.getElementById("gaugeFill");
const gaugeNeedleEl = document.getElementById("gaugeNeedle");
const gaugeValueEl  = document.getElementById("gaugeValue");
const gaugeCatEl    = document.getElementById("gaugeCat");

const LANGUAGE_PROMPTS = {
  English: "Please read: “Today is a bright sunny day and I feel well.”",
  Spanish: "Lea: “Hoy hace un día soleado y me siento bien.”",
  Hindi: "कहें: “आज का दिन उजला है और मैं ठीक महसूस कर रहा हूँ।”",
  Telugu: "చెప్పండి: “ఈ రోజు వెలుగైన రోజు, నేను బాగున్నాను.”",
  "Neutral / Non-verbal cadence": "Speak any short familiar phrase naturally, or count from one to ten at a steady pace.",
};

let faceBlob = null;
let armBlob = null;
let speechBlob = null;
let bFaceBlob = null;
let bArmBlob = null;
let bSpeechBlob = null;
let cameraStream = null;
let currentReportPayload = null;
let speechObjectUrl = null;
let currentCaptureIds = { face: null, arm: null, speech: null };

function generateCaptureId(prefix) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function fmt(value) {
  return Number(value || 0).toFixed(3);
}

function initStartQr() {
  if (!startQrImage || !startQrLink) return;
  const sessionUrl = `${window.location.origin}/?start=1`;
  const qrApi = `/api/qr/start-session?target=${encodeURIComponent(sessionUrl)}`;
  startQrImage.src = qrApi;
  startQrLink.href = sessionUrl;
  startQrLink.textContent = sessionUrl;
}

function setPrompt() {
  speechPrompt.textContent = LANGUAGE_PROMPTS[languageSelect.value] || LANGUAGE_PROMPTS.English;
}

async function ensureCamera() {
  if (cameraStream) return cameraStream;
  cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  facePreview.srcObject = cameraStream;
  armPreview.srcObject = cameraStream;
  return cameraStream;
}

async function recordStream(stream, seconds, mimeType) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    const recorder = new MediaRecorder(stream, { mimeType });
    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) chunks.push(event.data);
    };
    recorder.onerror = (event) => reject(event.error || new Error("Recording failed"));
    recorder.onstop = () => resolve(new Blob(chunks, { type: mimeType }));
    recorder.start();
    setTimeout(() => recorder.stop(), seconds * 1000);
  });
}

function drawWaveform(canvas, waveform, pauseSegments = [], unstableSegments = []) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (!waveform || !waveform.length) {
    ctx.fillStyle = "#64748b";
    ctx.fillText("No waveform available.", 20, 30);
    return;
  }

  const pad = 20;
  const mid = canvas.height / 2;
  const w = canvas.width - 2 * pad;
  const h = canvas.height - 2 * pad;
  const maxAbs = Math.max(...waveform.map((v) => Math.abs(v)), 1e-6);

  pauseSegments.forEach((segment) => {
    const x1 = pad + (segment.start / Math.max(segment.total || segment.end || 1, 1)) * w;
    const x2 = pad + (segment.end / Math.max(segment.total || segment.end || 1, 1)) * w;
    ctx.fillStyle = "rgba(245, 158, 11, 0.15)";
    ctx.fillRect(x1, pad, Math.max(4, x2 - x1), h);
  });

  unstableSegments.forEach((segment) => {
    const x1 = pad + (segment.start / Math.max(segment.total || segment.end || 1, 1)) * w;
    const x2 = pad + (segment.end / Math.max(segment.total || segment.end || 1, 1)) * w;
    ctx.fillStyle = "rgba(239, 68, 68, 0.12)";
    ctx.fillRect(x1, pad, Math.max(4, x2 - x1), h);
  });

  ctx.strokeStyle = "#1f6feb";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  waveform.forEach((value, index) => {
    const x = pad + (index / Math.max(waveform.length - 1, 1)) * w;
    const y = mid - (value / maxAbs) * (h / 2);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.strokeStyle = "#d0d7de";
  ctx.beginPath();
  ctx.moveTo(pad, mid);
  ctx.lineTo(pad + w, mid);
  ctx.stroke();
}

async function analyzeSpeechPreview(blob) {
  try {
    const arrayBuffer = await blob.arrayBuffer();
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    const channel = audioBuffer.getChannelData(0);
    const bucketCount = 300;
    const bucketSize = Math.max(1, Math.floor(channel.length / bucketCount));
    const waveform = [];
    const rms = [];

    for (let i = 0; i < bucketCount; i += 1) {
      const start = i * bucketSize;
      const end = Math.min(channel.length, start + bucketSize);
      if (start >= channel.length) break;
      let sum = 0;
      let peak = 0;
      for (let j = start; j < end; j += 1) {
        const sample = channel[j];
        sum += sample * sample;
        peak = Math.max(peak, Math.abs(sample));
      }
      waveform.push(peak);
      rms.push(Math.sqrt(sum / Math.max(end - start, 1)));
    }

    const threshold = Math.max(0.015, [...rms].sort((a, b) => a - b)[Math.floor(rms.length * 0.3)] || 0.015);
    const frameSec = audioBuffer.duration / Math.max(rms.length, 1);
    const pauseSegments = [];
    let pauseStart = null;
    rms.forEach((value, index) => {
      if (value < threshold && pauseStart == null) pauseStart = index;
      if ((value >= threshold || index === rms.length - 1) && pauseStart != null) {
        const endIndex = value >= threshold ? index - 1 : index;
        if (endIndex - pauseStart >= 2) {
          pauseSegments.push({
            start: pauseStart * frameSec,
            end: endIndex * frameSec,
            total: audioBuffer.duration,
          });
        }
        pauseStart = null;
      }
    });

    const peaks = rms.filter((value, index) => {
      if (index === 0 || index === rms.length - 1) return false;
      return value > rms[index - 1] && value > rms[index + 1] && value > threshold * 1.4;
    }).length;
    const articulationRate = peaks / Math.max(audioBuffer.duration, 1e-6);

    drawWaveform(speechWaveform, waveform, pauseSegments, []);
    speechPauseInfo.textContent = pauseSegments.length ? `${pauseSegments.length} detected` : "No major pauses";
    speechPaceInfo.textContent = `${articulationRate.toFixed(2)} peaks/sec`;
    audioContext.close();

    return { waveform, pauseSegments, articulationRate, duration: audioBuffer.duration };
  } catch (error) {
    speechPauseInfo.textContent = "Preview unavailable";
    speechPaceInfo.textContent = "Preview unavailable";
    return null;
  }
}

function startSpeechMeter(stream) {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const audioContext = new AudioCtx();
  const source = audioContext.createMediaStreamSource(stream);
  const analyser = audioContext.createAnalyser();
  analyser.fftSize = 1024;
  source.connect(analyser);
  const data = new Uint8Array(analyser.fftSize);
  let active = true;
  let rafId = null;

  function tick() {
    analyser.getByteTimeDomainData(data);
    let total = 0;
    for (let i = 0; i < data.length; i += 1) {
      const centered = (data[i] - 128) / 128;
      total += centered * centered;
    }
    const rms = Math.sqrt(total / data.length);
    if (rms < 0.035) speechCoach.textContent = "Coaching: Please speak louder.";
    else if (rms > 0.25) speechCoach.textContent = "Coaching: Good signal level detected.";
    else speechCoach.textContent = "Coaching: Speak clearly and maintain a steady pace.";
    if (active) rafId = requestAnimationFrame(tick);
  }

  tick();

  return async () => {
    active = false;
    if (rafId) cancelAnimationFrame(rafId);
    source.disconnect();
    analyser.disconnect();
    await audioContext.close();
  };
}

async function recordFace() {
  faceRecordBtn.disabled = true;
  faceStatus.textContent = "Recording...";
  faceCoach.textContent = "Coaching: Hold steady, keep your face centered, and smile naturally.";
  try {
    const stream = await ensureCamera();
    faceBlob = await recordStream(stream, 5, "video/webm");
    currentCaptureIds.face = generateCaptureId("face");
    faceStatus.textContent = `Recorded (${Math.round(faceBlob.size / 1024)} KB)`;
    faceCoach.textContent = "Face recording complete.";
  } catch (err) {
    faceStatus.textContent = `Error: ${err.message}`;
  } finally {
    faceRecordBtn.disabled = false;
  }
}

async function recordArm() {
  armRecordBtn.disabled = true;
  armStatus.textContent = "Recording...";
  armCoach.textContent = "Coaching: Keep both wrists visible and hold them level.";
  try {
    const stream = await ensureCamera();
    armBlob = await recordStream(stream, 5, "video/webm");
    currentCaptureIds.arm = generateCaptureId("arm");
    armStatus.textContent = `Recorded (${Math.round(armBlob.size / 1024)} KB)`;
    armCoach.textContent = "Arm recording complete.";
  } catch (err) {
    armStatus.textContent = `Error: ${err.message}`;
  } finally {
    armRecordBtn.disabled = false;
  }
}

async function recordSpeech() {
  speechRecordBtn.disabled = true;
  speechStatus.textContent = "Recording...";
  speechCoach.textContent = "Coaching: Listening...";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    const stopMeter = startSpeechMeter(stream);
    speechBlob = await recordStream(stream, 5, "audio/webm");
    await stopMeter();
    stream.getTracks().forEach((track) => track.stop());
    currentCaptureIds.speech = generateCaptureId("speech");

    if (speechObjectUrl) URL.revokeObjectURL(speechObjectUrl);
    speechObjectUrl = URL.createObjectURL(speechBlob);
    speechPlayback.src = speechObjectUrl;
    await analyzeSpeechPreview(speechBlob);

    speechStatus.textContent = `Recorded (${Math.round(speechBlob.size / 1024)} KB)`;
    speechCoach.textContent = "Speech recording complete.";
  } catch (err) {
    speechStatus.textContent = `Error: ${err.message}`;
  } finally {
    speechRecordBtn.disabled = false;
  }
}

function severityColor(value) {
  if (value >= 0.12) return "#dc2626";
  if (value >= 0.05) return "#d97706";
  return "#16a34a";
}

function drawFaceOverlay(facial) {
  const ctx = faceOverlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, faceOverlayCanvas.width, faceOverlayCanvas.height);
  if (!facial?.overlay?.image_base64) {
    ctx.fillStyle = "#64748b";
    ctx.fillText("No facial overlay available.", 20, 30);
    return;
  }

  const image = new Image();
  image.onload = () => {
    ctx.clearRect(0, 0, faceOverlayCanvas.width, faceOverlayCanvas.height);
    ctx.drawImage(image, 0, 0, faceOverlayCanvas.width, faceOverlayCanvas.height);

    const { keypoints, metrics } = facial.overlay;
    const scaleX = faceOverlayCanvas.width;
    const scaleY = faceOverlayCanvas.height;

    function point(name) {
      return {
        x: keypoints[name].x * scaleX,
        y: keypoints[name].y * scaleY,
      };
    }

    const lm = point("left_mouth");
    const rm = point("right_mouth");
    const letp = point("left_eye_top");
    const leb = point("left_eye_bottom");
    const retp = point("right_eye_top");
    const reb = point("right_eye_bottom");

    const mouthColor = severityColor(metrics.mouth_diff_norm);
    const eyeColor = severityColor(metrics.eye_diff_norm);
    const sideColor = severityColor(metrics.side_deviation_norm);

    ctx.lineWidth = 4;
    ctx.strokeStyle = mouthColor;
    ctx.beginPath();
    ctx.moveTo(lm.x, lm.y);
    ctx.lineTo(rm.x, rm.y);
    ctx.stroke();

    [[letp, leb, eyeColor], [retp, reb, eyeColor]].forEach(([a, b, color]) => {
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    });

    [
      [lm, mouthColor],
      [rm, mouthColor],
      [letp, eyeColor],
      [leb, eyeColor],
      [retp, eyeColor],
      [reb, eyeColor],
    ].forEach(([pt, color]) => {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 7, 0, Math.PI * 2);
      ctx.fill();
    });

    const midX = (lm.x + rm.x) / 2;
    ctx.strokeStyle = sideColor;
    ctx.setLineDash([8, 6]);
    ctx.beginPath();
    ctx.moveTo(midX, 20);
    ctx.lineTo(midX, faceOverlayCanvas.height - 20);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "rgba(15,23,42,0.78)";
    ctx.fillRect(12, 12, 310, 82);
    ctx.fillStyle = "#fff";
    ctx.font = "14px sans-serif";
    ctx.fillText(`Mouth asymmetry: ${fmt(metrics.mouth_diff_norm)}`, 24, 36);
    ctx.fillText(`Eye asymmetry: ${fmt(metrics.eye_diff_norm)}`, 24, 58);
    ctx.fillText(`Side deviation: ${fmt(metrics.side_deviation_norm)}`, 24, 80);
  };
  image.src = facial.overlay.image_base64;
}

function drawArmChart(series = {}) {
  const ctx = armChart.getContext("2d");
  ctx.clearRect(0, 0, armChart.width, armChart.height);

  const times = series.time || [];
  const left = series.left_wrist_y || [];
  const right = series.right_wrist_y || [];
  if (!times.length) {
    ctx.fillStyle = "#4b5563";
    ctx.fillText("No arm time-series data available.", 20, 30);
    return;
  }

  const minT = times[0];
  const maxT = times[times.length - 1] || 1;
  const allY = [...left, ...right];
  const minY = Math.min(...allY);
  const maxY = Math.max(...allY);
  const pad = 38;
  const w = armChart.width - 2 * pad;
  const h = armChart.height - 2 * pad;

  function x(t) {
    return pad + ((t - minT) / Math.max(maxT - minT, 1e-6)) * w;
  }
  function y(v) {
    return pad + (1 - (v - minY) / Math.max(maxY - minY, 1e-6)) * h;
  }

  ctx.strokeStyle = "#cbd5e1";
  ctx.beginPath();
  ctx.moveTo(pad, pad + h);
  ctx.lineTo(pad + w, pad + h);
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, pad + h);
  ctx.stroke();

  (series.asymmetry_regions || []).forEach((region) => {
    const x1 = x(region.start);
    const x2 = x(region.end);
    ctx.fillStyle = "rgba(220, 38, 38, 0.12)";
    ctx.fillRect(x1, pad, Math.max(5, x2 - x1), h);
  });

  ctx.strokeStyle = "#1f6feb";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  left.forEach((v, i) => (i === 0 ? ctx.moveTo(x(times[i]), y(v)) : ctx.lineTo(x(times[i]), y(v))));
  ctx.stroke();

  ctx.strokeStyle = "#d1242f";
  ctx.beginPath();
  right.forEach((v, i) => (i === 0 ? ctx.moveTo(x(times[i]), y(v)) : ctx.lineTo(x(times[i]), y(v))));
  ctx.stroke();

  ctx.setLineDash([7, 5]);
  ctx.lineWidth = 1.8;
  if (series.slope_overlay?.left) {
    ctx.strokeStyle = "#60a5fa";
    ctx.beginPath();
    ctx.moveTo(x(series.slope_overlay.left.start.t), y(series.slope_overlay.left.start.y));
    ctx.lineTo(x(series.slope_overlay.left.end.t), y(series.slope_overlay.left.end.y));
    ctx.stroke();
  }
  if (series.slope_overlay?.right) {
    ctx.strokeStyle = "#f87171";
    ctx.beginPath();
    ctx.moveTo(x(series.slope_overlay.right.start.t), y(series.slope_overlay.right.start.y));
    ctx.lineTo(x(series.slope_overlay.right.end.t), y(series.slope_overlay.right.end.y));
    ctx.stroke();
  }
  ctx.setLineDash([]);

  function marker(mark, color, label) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x(mark.t), y(mark.y), 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText(label, x(mark.t) + 6, y(mark.y) - 6);
  }

  marker(series.start_markers.left, "#1f6feb", "L start");
  marker(series.start_markers.right, "#d1242f", "R start");
  marker(series.end_markers.left, "#1f6feb", "L end");
  marker(series.end_markers.right, "#d1242f", "R end");

  ctx.fillStyle = "#1f6feb";
  ctx.fillText("Left wrist trajectory", pad + 4, 20);
  ctx.fillStyle = "#d1242f";
  ctx.fillText("Right wrist trajectory", pad + 170, 20);
  ctx.fillStyle = "#64748b";
  ctx.fillText("Red shaded area = highlighted asymmetry region", pad + 345, 20);
}

function drawSpeechTimeline(timeline = {}) {
  const ctx = speechTimeline.getContext("2d");
  ctx.clearRect(0, 0, speechTimeline.width, speechTimeline.height);
  const time = timeline.time || [];
  const rms = timeline.rms || [];
  if (!time.length || !rms.length) {
    ctx.fillStyle = "#64748b";
    ctx.fillText("No speech timeline available.", 20, 30);
    return;
  }

  const pad = 30;
  const w = speechTimeline.width - 2 * pad;
  const h = speechTimeline.height - 2 * pad;
  const maxT = time[time.length - 1] || 1;
  const minR = 0;
  const maxR = Math.max(...rms, 1e-6);

  function x(t) {
    return pad + (t / maxT) * w;
  }
  function y(v) {
    return pad + (1 - (v - minR) / Math.max(maxR - minR, 1e-6)) * h;
  }

  (timeline.pause_segments || []).forEach((segment) => {
    ctx.fillStyle = "rgba(245, 158, 11, 0.18)";
    ctx.fillRect(x(segment.start), pad, Math.max(4, x(segment.end) - x(segment.start)), h);
  });
  (timeline.unstable_segments || []).forEach((segment) => {
    ctx.fillStyle = "rgba(239, 68, 68, 0.12)";
    ctx.fillRect(x(segment.start), pad, Math.max(4, x(segment.end) - x(segment.start)), h);
  });

  ctx.strokeStyle = "#1f6feb";
  ctx.lineWidth = 2;
  ctx.beginPath();
  rms.forEach((v, i) => (i === 0 ? ctx.moveTo(x(time[i]), y(v)) : ctx.lineTo(x(time[i]), y(v))));
  ctx.stroke();

  ctx.strokeStyle = "#cbd5e1";
  ctx.beginPath();
  ctx.moveTo(pad, pad + h);
  ctx.lineTo(pad + w, pad + h);
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, pad + h);
  ctx.stroke();

  ctx.fillStyle = "#475569";
  ctx.fillText("Orange = long pauses, Red = unstable amplitude regions", pad, 18);
}

function renderBaselineComparison(baselineComparison) {
  if (!baselineComparison) {
    baselinePanel.textContent = "No baseline comparison available for this screening.";
    return;
  }
  const items = Object.values(baselineComparison.components || {})
    .map(
      (item) => `
        <div class="baseline-item">
          <strong>${item.label}</strong><br />
          Baseline: ${fmt(item.baseline)} | Current: ${fmt(item.current)} | Delta: ${fmt(item.delta)}<br />
          ${item.status}
        </div>`
    )
    .join("");
  baselinePanel.innerHTML = `<strong>${baselineComparison.summary}</strong>${items}`;
}

function renderExplainability(explainability = {}) {
  explainabilityPanel.innerHTML = ["facial", "arm", "speech"]
    .filter((key) => explainability[key])
    .map((key) => `<div class="explain-item"><strong>${key.toUpperCase()}</strong><br />${explainability[key]}</div>`)
    .join("") || "Explainability unavailable.";
}

function renderSessionHistory(sessions = []) {
  if (!sessions.length) {
    sessionHistory.textContent = "No previous sessions loaded.";
    return;
  }

  sessionHistory.innerHTML = sessions
    .map((session) => {
      const report = session.report || {};
      return `
        <div class="history-item">
          <strong>${session.patient_id || "Unknown patient"}</strong> — ${session.logged_at || "Unknown time"}<br />
          Risk ${fmt(report.fast_risk_index)} (${report.category || "N/A"}) | Confidence ${fmt(report.quality_confidence)}<br />
          Retake reason: ${session.retake_reason || "None"}
        </div>`;
    })
    .join("");
}

async function fetchHistory(patientId) {
  if (!patientId) {
    renderSessionHistory([]);
    return;
  }
  try {
    const response = await fetch(`/api/sessions?patient_id=${encodeURIComponent(patientId)}&limit=5`);
    if (!response.ok) return;
    const payload = await response.json();
    renderSessionHistory(payload.sessions || []);
  } catch {
    // ignore passive history errors
  }
}

function animateRiskGauge(riskIndex, category) {
  if (!gaugeFillEl || !gaugeNeedleEl) return;
  // Delay slightly so CSS transition fires after hidden is removed
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      gaugeFillEl.style.strokeDashoffset = String(1 - riskIndex);
      const deg = (riskIndex * 180) - 90;
      gaugeNeedleEl.style.transform = `rotate(${deg}deg)`;
    });
  });
  gaugeValueEl.textContent = `${Math.round(riskIndex * 100)}%`;
  const cat = (category || "low").toLowerCase();
  gaugeCatEl.textContent = category || "—";
  gaugeCatEl.className = `gauge-cat ${cat}`;
}

function confBand(v) {
  if (v >= 0.75) return "high";
  if (v >= 0.45) return "moderate";
  return "low";
}

function renderConfidenceBreakdown(breakdown, overall, band) {
  if (!breakdown || !confidenceBreakdown) return;
  const pct = (v) => `${Math.round(v * 100)}%`;

  [[confFaceFill,   confFaceVal,   breakdown.face],
   [confArmFill,    confArmVal,    breakdown.arm],
   [confSpeechFill, confSpeechVal, breakdown.speech]
  ].forEach(([fill, val, v]) => {
    if (!fill || !val) return;
    const level = confBand(v);
    fill.style.width = pct(v);
    fill.className = `conf-bar-fill ${level}`;
    val.textContent = pct(v);
  });

  if (confOverallEl) {
    confOverallEl.textContent = `Overall ${pct(overall)} · ${band || "—"}`;
  }
  confidenceBreakdown.hidden = false;
}

function updateRecommendationPanel(report) {
  recommendationEl.textContent = report.recommendation || "";
  recommendationPanel.className = "recommendation-panel";
  recommendationPanel.classList.add((report.category || "low").toLowerCase());
}

function populateReport(payload) {
  currentReportPayload = payload;
  const report = payload.report;

  facialScoreEl.textContent = fmt(report.facial_asymmetry_score);
  armScoreEl.textContent = fmt(report.arm_drift_score);
  speechScoreEl.textContent = fmt(report.speech_instability_score);
  riskScoreEl.textContent = fmt(report.fast_risk_index);
  riskCategoryEl.textContent = report.category;
  ruleImpressionEl.textContent = report.rule_based_impression;
  confidenceInfoEl.textContent = `${fmt(report.quality_confidence)} (${report.confidence_band})`;
  reportTimestampEl.textContent = new Date().toLocaleString();
  disclaimerEl.textContent = report.disclaimer;

  updateRecommendationPanel(report);
  drawArmChart(payload.arm.timeseries);
  drawFaceOverlay(payload.facial);
  drawSpeechTimeline(payload.speech.timeline);
  renderBaselineComparison(payload.baseline_comparison);
  renderExplainability(payload.explainability);
  renderSessionHistory(payload.recent_sessions || []);
  animateRiskGauge(report.fast_risk_index ?? 0, report.category);
  renderConfidenceBreakdown(
    payload.confidence_breakdown,
    report.quality_confidence ?? 0,
    report.confidence_band
  );

  reportCard.hidden = false;
  exportPdfBtn.disabled = false;
  printBtn.disabled = false;
}

function appendOptionalNumber(form, key, input) {
  const value = input.value.trim();
  if (value !== "") form.append(key, value);
}

async function analyze() {
  if (!faceBlob || !armBlob || !speechBlob) {
    analyzeStatus.textContent = "Please complete all three recordings first.";
    return;
  }

  analyzeBtn.disabled = true;
  analyzeStatus.textContent = "Analyzing...";

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 90000);
    const form = new FormData();
    form.append("face_video", faceBlob, "face.webm");
    form.append("arm_video", armBlob, "arm.webm");
    form.append("speech_audio", speechBlob, "speech.webm");
    form.append("patient_id", patientIdInput.value.trim());
    form.append("screener_name", screenerNameInput.value.trim());
    form.append("language", languageSelect.value);
    form.append("retake_reason", retakeReasonInput.value.trim());
    form.append("face_capture_id", currentCaptureIds.face || generateCaptureId("face"));
    form.append("arm_capture_id", currentCaptureIds.arm || generateCaptureId("arm"));
    form.append("speech_capture_id", currentCaptureIds.speech || generateCaptureId("speech"));

    const response = await fetch("/api/analyze", {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      let detail = "Unknown error";
      try {
        const errPayload = await response.json();
        detail = errPayload.detail || detail;
      } catch {
        detail = await response.text();
      }
      throw new Error(`Backend error (${response.status}): ${detail}`);
    }

    const payload = await response.json();
    populateReport(payload);
    analyzeStatus.textContent = "Analysis complete.";
  } catch (err) {
    if (err.name === "AbortError") analyzeStatus.textContent = "Error: analysis timed out after 90 seconds.";
    else analyzeStatus.textContent = `Error: ${err.message}`;
  } finally {
    analyzeBtn.disabled = false;
  }
}

async function exportPdf() {
  if (!currentReportPayload) return;
  exportPdfBtn.disabled = true;
  try {
    const response = await fetch("/api/report/pdf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        patient_id: patientIdInput.value.trim(),
        screener_name: screenerNameInput.value.trim(),
        language: languageSelect.value,
        report: currentReportPayload.report,
        baseline_comparison: currentReportPayload.baseline_comparison,
        explainability: currentReportPayload.explainability,
        graphs: {
          arm_graph: armChart.toDataURL("image/png"),
          face_overlay: faceOverlayCanvas.toDataURL("image/png"),
          speech_waveform: speechTimeline.toDataURL("image/png"),
        },
      }),
    });
    if (!response.ok) throw new Error("Unable to generate PDF");
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "fast_screening_report.pdf";
    link.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    analyzeStatus.textContent = `Error: ${err.message}`;
  } finally {
    exportPdfBtn.disabled = false;
  }
}

faceRecordBtn.addEventListener("click", recordFace);
armRecordBtn.addEventListener("click", recordArm);
speechRecordBtn.addEventListener("click", recordSpeech);
analyzeBtn.addEventListener("click", analyze);
exportPdfBtn.addEventListener("click", exportPdf);
printBtn.addEventListener("click", () => window.print());
languageSelect.addEventListener("change", setPrompt);
patientIdInput.addEventListener("change", () => fetchHistory(patientIdInput.value.trim()));

// ─── Baseline recording ───────────────────────────────────────────────────────

function checkBaselineSaveReady() {
  saveBaselineBtn.disabled = !(bFaceBlob && bArmBlob && bSpeechBlob);
}

async function recordBaselineClip(btn, statusEl, blobSetter) {
  btn.disabled = true;
  statusEl.textContent = "Recording...";
  try {
    const stream = await ensureCamera();
    const blob = await recordStream(stream, 5, "video/webm");
    blobSetter(blob);
    statusEl.textContent = `✓ ${Math.round(blob.size / 1024)} KB`;
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  } finally {
    btn.disabled = false;
    checkBaselineSaveReady();
  }
}

async function recordBaselineSpeechClip() {
  bSpeechBtn.disabled = true;
  bSpeechStatus.textContent = "Recording...";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    const blob = await recordStream(stream, 5, "audio/webm");
    stream.getTracks().forEach((t) => t.stop());
    bSpeechBlob = blob;
    bSpeechStatus.textContent = `✓ ${Math.round(blob.size / 1024)} KB`;
  } catch (err) {
    bSpeechStatus.textContent = `Error: ${err.message}`;
  } finally {
    bSpeechBtn.disabled = false;
    checkBaselineSaveReady();
  }
}

async function saveBaseline() {
  if (!bFaceBlob || !bArmBlob || !bSpeechBlob) return;
  saveBaselineBtn.disabled = true;
  saveBaselineStatus.textContent = "Saving…";
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 90000);
    const form = new FormData();
    form.append("face_video",   bFaceBlob,   "face.webm");
    form.append("arm_video",    bArmBlob,    "arm.webm");
    form.append("speech_audio", bSpeechBlob, "speech.webm");
    form.append("patient_id",    patientIdInput.value.trim());
    form.append("screener_name", screenerNameInput.value.trim());
    form.append("language",      languageSelect.value);
    form.append("is_baseline",   "true");
    const response = await fetch("/api/analyze", { method: "POST", body: form, signal: controller.signal });
    clearTimeout(timeoutId);
    if (!response.ok) {
      const e = await response.json().catch(() => ({}));
      throw new Error(e.detail || response.statusText);
    }
    saveBaselineStatus.textContent = "";
    baselineSavedBanner.classList.add("visible");
    baselineSavedBanner.textContent =
      `✓ Baseline saved${patientIdInput.value.trim() ? " for " + patientIdInput.value.trim() : ""} — future screenings will compare against it.`;
    baselineRecordStatus.textContent = "✓ Baseline on file";
    // reset blobs
    bFaceBlob = bArmBlob = bSpeechBlob = null;
    [bFaceStatus, bArmStatus, bSpeechStatus].forEach((el) => (el.textContent = "Not recorded"));
    saveBaselineBtn.disabled = true;
  } catch (err) {
    if (err.name === "AbortError") saveBaselineStatus.textContent = "Timed out.";
    else saveBaselineStatus.textContent = `Error: ${err.message}`;
    saveBaselineBtn.disabled = false;
  }
}

toggleBaselineBtn.addEventListener("click", () => {
  const open = baselineDrawer.classList.toggle("open");
  toggleBaselineBtn.textContent = open ? "Hide Baseline ↑" : "Record Baseline →";
});
bFaceBtn.addEventListener("click", () =>
  recordBaselineClip(bFaceBtn, bFaceStatus, (b) => { bFaceBlob = b; })
);
bArmBtn.addEventListener("click", () =>
  recordBaselineClip(bArmBtn, bArmStatus, (b) => { bArmBlob = b; })
);
bSpeechBtn.addEventListener("click", recordBaselineSpeechClip);
saveBaselineBtn.addEventListener("click", saveBaseline);

setPrompt();
ensureCamera().catch((err) => {
  faceStatus.textContent = `Camera unavailable: ${err.message}`;
  armStatus.textContent = `Camera unavailable: ${err.message}`;
});
fetchHistory(patientIdInput.value.trim());
initStartQr();
