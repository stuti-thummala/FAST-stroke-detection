/**
 * landmark-overlay.js
 * Stylish real-time landmark overlay for the FAST Stroke Screening tool.
 *
 * Design intent: NOT the default ugly green MediaPipe lines.
 *   Face  → glowing cyan eye rings, pink lip curve, indigo brows, faint jaw silhouette
 *   Arm   → purple-to-cyan gradient skeleton, animated wrist pulse rings
 *   Both  → HUD corner brackets + sweep scan-line on init + status badge
 *
 * Loads MediaPipe Tasks Vision from CDN (requires internet on first load; then cached).
 * Fails gracefully — nothing breaks if CDN is unreachable.
 */

import { FaceLandmarker, PoseLandmarker, FilesetResolver }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/+esm";

// ── CDN ──────────────────────────────────────────────────────────────────────
const WASM =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm";
const FACE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const POSE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";

// ── Face contour groups (MediaPipe 478-pt mesh, official index sets) ──────────
// We draw only these curated contours as smooth curves — no raw triangulation mesh
const FC = {
  leftEye:   [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
  rightEye:  [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398],
  lips:      [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
  leftBrow:  [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
  rightBrow: [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
  jaw:       [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
              379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
              234, 127, 162, 21, 54, 103, 67, 109],
};

// Style per contour group — color, glow, width, alpha, whether to close the path
const FS = {
  leftEye:   { color: "#22d3ee", glow: "#22d3ee", w: 1.6, blur: 10, a: 0.92, close: true  },
  rightEye:  { color: "#22d3ee", glow: "#22d3ee", w: 1.6, blur: 10, a: 0.92, close: true  },
  lips:      { color: "#f472b6", glow: "#f472b6", w: 1.5, blur: 8,  a: 0.88, close: true  },
  leftBrow:  { color: "#a5b4fc", glow: "#818cf8", w: 1.2, blur: 6,  a: 0.80, close: false },
  rightBrow: { color: "#a5b4fc", glow: "#818cf8", w: 1.2, blur: 6,  a: 0.80, close: false },
  jaw:       { color: "rgba(34,211,238,0.22)", glow: null, w: 1.0, blur: 0, a: 0.55, close: false },
};

// Specific landmark indices to draw as bright junction dots
const FACE_DOTS = [33, 133, 263, 362, 0, 17, 1];   // eye corners, lip midpoints, nose tip

// ── Pose arm skeleton ─────────────────────────────────────────────────────────
const POSE_COLORS = {
  11: "#a855f7", 12: "#a855f7",   // shoulders — purple
  13: "#6366f1", 14: "#6366f1",   // elbows    — indigo
  15: "#06b6d4", 16: "#06b6d4",   // wrists    — cyan
  23: "#7c3aed", 24: "#7c3aed",   // hips      — deep purple
};

const POSE_LINKS = [
  [11, 13], [13, 15],   // left arm
  [12, 14], [14, 16],   // right arm
  [11, 12],             // shoulder bar
  [11, 23], [12, 24],   // torso side edges (subtle)
];

// ── State ─────────────────────────────────────────────────────────────────────
let faceDetector = null;
let poseDetector = null;
let modelsReady  = false;
let active       = false;
let rafId        = null;
let scanStart    = null;         // timestamp for the sweep animation
const SCAN_MS    = 1900;
const THROTTLE   = 60;          // ~16fps — balanced quality vs CPU
let lastTick     = 0;
let lastFaceTs   = -1;
let lastPoseTs   = -1;

// DOM refs (resolved in init)
let faceVideo, armVideo, faceCanvas, armCanvas, faceBadge, armBadge;

// ── Canvas helpers ─────────────────────────────────────────────────────────────
function syncCanvasSize(canvas, video) {
  const vw = video.videoWidth  || 640;
  const vh = video.videoHeight || 480;
  if (canvas.width !== vw || canvas.height !== vh) {
    canvas.width  = vw;
    canvas.height = vh;
  }
}

function pt(landmarks, idx, w, h) {
  const p = landmarks[idx];
  return p ? { x: p.x * w, y: p.y * h } : null;
}

// ── Drawing primitives  ────────────────────────────────────────────────────────
// Smooth quadratic-bezier curve through an array of {x,y} points
function smoothPath(ctx, pts, close) {
  if (pts.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length - 1; i++) {
    const mx = (pts[i].x + pts[i + 1].x) / 2;
    const my = (pts[i].y + pts[i + 1].y) / 2;
    ctx.quadraticCurveTo(pts[i].x, pts[i].y, mx, my);
  }
  if (close) {
    const last = pts[pts.length - 1];
    ctx.quadraticCurveTo(last.x, last.y, (last.x + pts[0].x) / 2, (last.y + pts[0].y) / 2);
    ctx.closePath();
  } else {
    ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
  }
}

function drawContour(ctx, lms, indices, s, w, h) {
  const pts = indices.map(i => pt(lms, i, w, h)).filter(Boolean);
  if (pts.length < 2) return;
  ctx.save();
  ctx.lineWidth    = s.w;
  ctx.strokeStyle  = s.color;
  ctx.globalAlpha  = s.a;
  if (s.glow) { ctx.shadowBlur = s.blur; ctx.shadowColor = s.glow; }
  smoothPath(ctx, pts, s.close);
  ctx.stroke();
  ctx.restore();
}

function glowDot(ctx, x, y, r, fill, glowColor, blur = 12) {
  ctx.save();
  ctx.shadowBlur  = blur;
  ctx.shadowColor = glowColor ?? fill;
  ctx.fillStyle   = fill;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// Animated pulsing ring around a point (wrists)
function pulseRing(ctx, x, y, baseR, color, now) {
  const phase = (Math.sin(now / 380) + 1) / 2;
  ctx.save();
  ctx.globalAlpha = 0.7 - phase * 0.5;
  ctx.strokeStyle = color;
  ctx.lineWidth   = 1.5;
  ctx.shadowBlur  = 8;
  ctx.shadowColor = color;
  ctx.beginPath();
  ctx.arc(x, y, baseR + 4 + phase * 7, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

// Gradient line between two points for skeleton connections
function gradLine(ctx, x1, y1, x2, y2, c1, c2, lw = 2.5) {
  const grad = ctx.createLinearGradient(x1, y1, x2, y2);
  grad.addColorStop(0, c1);
  grad.addColorStop(1, c2);
  ctx.save();
  ctx.lineWidth   = lw;
  ctx.strokeStyle = grad;
  ctx.shadowBlur  = 9;
  ctx.shadowColor = c1;
  ctx.globalAlpha = 0.82;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.restore();
}

// Subtle L-bracket HUD corners
function hudCorners(ctx, w, h) {
  const L = 22, lw = 1.5, color = "rgba(34,211,238,0.42)";
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth   = lw;
  ctx.shadowBlur  = 5;
  ctx.shadowColor = color;
  [
    [0, 0,  1, 0,  0, 1],
    [w, 0, -1, 0,  0, 1],
    [0, h,  1, 0,  0,-1],
    [w, h, -1, 0,  0,-1],
  ].forEach(([cx, cy, dx, dy, ex, ey]) => {
    ctx.beginPath();
    ctx.moveTo(cx + dx * L, cy + dy * L);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx + ex * L, cy + ey * L);
    ctx.stroke();
  });
  ctx.restore();
}

// Sweep scan line — returns true while still animating
function scanLine(ctx, w, h, now) {
  if (!scanStart) scanStart = now;
  const elapsed = now - scanStart;
  if (elapsed >= SCAN_MS) return false;
  const y = (elapsed / SCAN_MS) * h;
  const g = ctx.createLinearGradient(0, y - 28, 0, y + 28);
  g.addColorStop(0,   "rgba(34,211,238,0)");
  g.addColorStop(0.5, "rgba(34,211,238,0.26)");
  g.addColorStop(1,   "rgba(34,211,238,0)");
  ctx.save();
  ctx.fillStyle = g;
  ctx.fillRect(0, y - 28, w, 56);
  ctx.restore();
  return true;
}

// ── Badge ─────────────────────────────────────────────────────────────────────
function setBadge(badge, state) {
  if (!badge) return;
  const M = {
    loading:   ["◌", "Loading…",  "loading"],
    searching: ["◎", "Scanning…", "searching"],
    tracking:  ["◉", "Tracking",  "tracking"],
  };
  const [icon, label, cls] = M[state] ?? M.searching;
  badge.innerHTML   = `<span class="lb-dot">${icon}</span><span class="lb-text">${label}</span>`;
  badge.className   = `landmark-badge ${cls}`;
}

// ── Per-frame draw routines ───────────────────────────────────────────────────
function drawFace(ctx, result, w, h, now) {
  ctx.clearRect(0, 0, w, h);
  hudCorners(ctx, w, h);
  scanLine(ctx, w, h, now);

  const faces = result?.faceLandmarks;
  if (!faces?.length || !faces[0]) return false;

  const lms = faces[0];

  // Contour groups
  for (const [key, indices] of Object.entries(FC)) {
    drawContour(ctx, lms, indices, FS[key], w, h);
  }

  // Junction dots at eye corners, lip mid, nose
  FACE_DOTS.forEach(idx => {
    const p = pt(lms, idx, w, h);
    if (p) glowDot(ctx, p.x, p.y, 2.5, "#fff", "#22d3ee", 14);
  });

  // Larger glowing nose-tip dot
  const nose = pt(lms, 1, w, h);
  if (nose) glowDot(ctx, nose.x, nose.y, 3.2, "#22d3ee", "#22d3ee", 18);

  return true;
}

function drawPose(ctx, result, w, h, now) {
  ctx.clearRect(0, 0, w, h);
  hudCorners(ctx, w, h);
  scanLine(ctx, w, h, now);

  const poses = result?.landmarks;
  if (!poses?.length || !poses[0]) return false;

  const lms = poses[0];

  // Skeleton connections with gradient color per limb direction
  for (const [a, b] of POSE_LINKS) {
    const pa = lms[a], pb = lms[b];
    if (!pa || !pb) continue;
    if ((pa.visibility ?? 1) < 0.5 || (pb.visibility ?? 1) < 0.5) continue;
    gradLine(ctx, pa.x * w, pa.y * h, pb.x * w, pb.y * h,
      POSE_COLORS[a] ?? "#6366f1",
      POSE_COLORS[b] ?? "#06b6d4");
  }

  // Joint dots
  for (const [idxStr, color] of Object.entries(POSE_COLORS)) {
    const idx = +idxStr;
    const lm  = lms[idx];
    if (!lm || (lm.visibility ?? 1) < 0.5) continue;
    const x = lm.x * w, y = lm.y * h;
    const r = (idx === 11 || idx === 12) ? 5.5 : 4;
    glowDot(ctx, x, y, r, color, color, 14);
    // White center cap
    ctx.save();
    ctx.fillStyle   = "#fff";
    ctx.globalAlpha = 0.88;
    ctx.beginPath();
    ctx.arc(x, y, 1.8, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  // Animated pulse rings on wrists
  [15, 16].forEach(idx => {
    const lm = lms[idx];
    if (!lm || (lm.visibility ?? 1) < 0.5) return;
    pulseRing(ctx, lm.x * w, lm.y * h, 4, POSE_COLORS[idx], now);
  });

  return true;
}

// ── Detection loop ─────────────────────────────────────────────────────────────
function tick(now) {
  if (!active) return;
  rafId = requestAnimationFrame(tick);

  // Throttle to ~16fps to balance quality with CPU load
  if (now - lastTick < THROTTLE) return;
  lastTick = now;

  if (faceVideo && faceCanvas) {
    syncCanvasSize(faceCanvas, faceVideo);
    const w = faceCanvas.width, h = faceCanvas.height;
    let faceResult = null;
    if (faceDetector && faceVideo.readyState >= 2 && now !== lastFaceTs) {
      try { faceResult = faceDetector.detectForVideo(faceVideo, now); } catch (_) {}
      lastFaceTs = now;
    }
    const detected = drawFace(faceCanvas.getContext("2d"), faceResult, w, h, now);
    if (faceDetector) setBadge(faceBadge, detected ? "tracking" : "searching");
  }

  if (armVideo && armCanvas) {
    syncCanvasSize(armCanvas, armVideo);
    const w = armCanvas.width, h = armCanvas.height;
    let poseResult = null;
    if (poseDetector && armVideo.readyState >= 2 && now !== lastPoseTs) {
      try { poseResult = poseDetector.detectForVideo(armVideo, now); } catch (_) {}
      lastPoseTs = now;
    }
    const detected = drawPose(armCanvas.getContext("2d"), poseResult, w, h, now);
    if (poseDetector) setBadge(armBadge, detected ? "tracking" : "searching");
  }
}

// ── Public API (exposed on window for main.js) ────────────────────────────────
function startLandmarkOverlay() {
  if (active) return;
  active    = true;
  scanStart = null;   // reset scan sweep
  lastFaceTs = lastPoseTs = lastTick = -1;

  // Immediately clear canvases so they never show as a white block
  [faceCanvas, armCanvas].forEach(c => {
    if (c) c.getContext("2d").clearRect(0, 0, c.width || 640, c.height || 480);
  });

  rafId = requestAnimationFrame(tick);
}

function stopLandmarkOverlay() {
  active = false;
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
}

// ── Model loading ─────────────────────────────────────────────────────────────
async function loadModels() {
  setBadge(faceBadge, "loading");
  setBadge(armBadge,  "loading");
  try {
    const vision = await FilesetResolver.forVisionTasks(WASM);
    [faceDetector, poseDetector] = await Promise.all([
      FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: FACE_MODEL, delegate: "GPU" },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
      }),
      PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: POSE_MODEL, delegate: "GPU" },
        runningMode: "VIDEO",
        numPoses: 1,
        outputSegmentationMasks: false,
      }),
    ]);
    modelsReady = true;
    setBadge(faceBadge, "searching");
    setBadge(armBadge,  "searching");
  } catch (err) {
    console.warn("[LandmarkOverlay] Model unavailable:", err?.message ?? err);
    // Hide badges gracefully — nothing breaks in main flow
    if (faceBadge) faceBadge.hidden = true;
    if (armBadge)  armBadge.hidden  = true;
  }
}

// ── Init ───────────────────────────────────────────────────────────────────────
function init() {
  faceVideo  = document.getElementById("facePreview");
  armVideo   = document.getElementById("armPreview");
  faceCanvas = document.getElementById("faceLandmarkCanvas");
  armCanvas  = document.getElementById("armLandmarkCanvas");
  faceBadge  = document.getElementById("faceLandmarkBadge");
  armBadge   = document.getElementById("armLandmarkBadge");

  if (!faceVideo || !faceCanvas) return;   // elements not present, skip

  // Clear canvases immediately so they are never white before the camera starts
  [faceCanvas, armCanvas].forEach(c => {
    if (c) c.getContext("2d").clearRect(0, 0, c.width || 300, c.height || 150);
  });

  window.startLandmarkOverlay = startLandmarkOverlay;
  window.stopLandmarkOverlay  = stopLandmarkOverlay;

  loadModels();   // background load — doesn't block UI
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
