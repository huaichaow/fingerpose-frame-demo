import * as tf from '@tensorflow/tfjs-core';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import * as handPoseDetection from '@tensorflow-models/hand-pose-detection';

// Serve the WASM binaries from the local package (works under Vite dev).
setWasmPaths('/node_modules/@tensorflow/tfjs-backend-wasm/dist/');

const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const statusEl = document.getElementById('status');
const ctx = canvas.getContext('2d');

// Detection reads from a canvas snapshot of the video, NOT the video
// element: on this machine TF.js's HTMLVideoElement input path returns
// garbage (0 hands / NaN keypoints) while canvas input works correctly.
// Likewise the WebGL backend produces garbage on this GPU, so the WASM
// (CPU) backend is used with the lighter model variant.
const snapCanvas = document.createElement('canvas');
const snapCtx = snapCanvas.getContext('2d', { willReadFrequently: true });

// Persistent stroke layer; only cleared by the Clear box.
const drawCanvas = document.createElement('canvas');
const drawCtx = drawCanvas.getContext('2d');

const WRIST = 0;
const INDEX_TIP = 8;
const MIDDLE_MCP = 9;
const FINGER_TIPS_NO_THUMB = [8, 12, 16, 20];

// Fingers connected as chains from the wrist, used to draw the hand skeleton.
const FINGER_CHAINS = [
  [0, 1, 2, 3, 4],
  [0, 5, 6, 7, 8],
  [5, 9, 10, 11, 12],
  [9, 13, 14, 15, 16],
  [13, 17, 18, 19, 20],
  [0, 17],
];

// MediaPipe reports handedness assuming a mirrored (selfie) input image;
// the snapshot fed to the detector is NOT mirrored, so the labels are
// swapped: the user's real left hand arrives labeled 'Right'.
// If pen control ends up on the wrong hand on some setup, flip this.
const LEFT_HAND_LABEL = 'Right';

// Open/fist classification: a finger counts as extended when its tip is
// far from the wrist relative to hand size (wrist → middle MCP). The
// pen state only flips on a clear fist (≤1 extended) or a clear open
// hand (≥3 extended); anything between keeps the previous state, which
// acts as hysteresis against flicker mid-gesture.
const EXTENDED_RATIO = 1.6;
const FIST_MAX_EXTENDED = 1;
const OPEN_MIN_EXTENDED = 3;

// Speed-adaptive smoothing (One-Euro-style): near-still fingertips get
// heavy smoothing for stability, fast strokes follow almost raw so the
// drawing point doesn't lag behind the finger.
const EMA_ALPHA_MIN = 0.3;
const EMA_ALPHA_MAX = 0.95;
const EMA_SPEED_NORM = 60; // px per sample at which alpha reaches max
const DWELL_MS = 400;
const STROKE_WIDTH = 6;

const COLORS = [
  { label: 'Red', color: '#ff4444' },
  { label: 'Green', color: '#00e676' },
  { label: 'Blue', color: '#2196f3' },
  { label: 'Clear', action: 'clear' },
];

let detector = null;
let latestHands = [];

let activeColor = COLORS[0].color;
// Pen state, toggled by the left (controller) hand: fist = down, open = up.
let penDown = false;
let leftHandSeen = false;
let prevPoint = null;
// EMA-smoothed index tip of the drawing hand, in mirrored (screen) coords.
let smoothIndex = null;
// Dwell selection: which box the fingertip is in, since when, and
// whether it already fired (re-arms only after leaving the box).
let dwell = { boxIndex: -1, since: 0, fired: false };

function setStatus(text) {
  statusEl.textContent = text;
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720, facingMode: 'user' },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((resolve) => {
    video.onloadedmetadata = resolve;
  });
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  drawCanvas.width = video.videoWidth;
  drawCanvas.height = video.videoHeight;
}

async function createDetector() {
  await tf.setBackend('wasm');
  await tf.ready();
  return handPoseDetection.createDetector(
    handPoseDetection.SupportedModels.MediaPipeHands,
    { runtime: 'tfjs', maxHands: 2, modelType: 'lite' },
  );
}

function handHasFiniteKeypoints(hand) {
  return hand.keypoints.every(
    (kp) => Number.isFinite(kp.x) && Number.isFinite(kp.y),
  );
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

// Landmarks arrive in raw video coordinates; the video is drawn
// mirrored (selfie view), so mirror x once here and let all downstream
// logic work in screen coordinates.
function mirrorPoint(kp) {
  return { x: canvas.width - kp.x, y: kp.y };
}

function ema(prev, cur) {
  if (!prev) return cur;
  const speed = dist(prev, cur);
  const alpha = Math.min(
    EMA_ALPHA_MAX,
    EMA_ALPHA_MIN + (EMA_ALPHA_MAX - EMA_ALPHA_MIN) * (speed / EMA_SPEED_NORM),
  );
  return {
    x: prev.x + alpha * (cur.x - prev.x),
    y: prev.y + alpha * (cur.y - prev.y),
  };
}

function isLeftHand(hand) {
  return hand.handedness === LEFT_HAND_LABEL;
}

function countExtendedFingers(hand) {
  const kp = hand.keypoints;
  const handScale = dist(kp[WRIST], kp[MIDDLE_MCP]);
  if (handScale <= 0) return 0;
  return FINGER_TIPS_NO_THUMB.filter(
    (tip) => dist(kp[tip], kp[WRIST]) / handScale > EXTENDED_RATIO,
  ).length;
}

// Color boxes along the top edge, sized from the canvas so they scale.
function getPalette() {
  const w = canvas.width;
  const boxW = Math.round(w * 0.11);
  const boxH = Math.round(boxW * 0.62);
  const gap = Math.round(boxW * 0.25);
  const totalW = COLORS.length * boxW + (COLORS.length - 1) * gap;
  const startX = Math.round((w - totalW) / 2);
  const y = Math.round(boxH * 0.35);
  return COLORS.map((c, i) => ({
    ...c,
    x: startX + i * (boxW + gap),
    y,
    w: boxW,
    h: boxH,
  }));
}

// Everything below the boxes is drawable; inside this strip inking is
// suppressed so picking a color never leaves a mark.
function paletteZoneBottom(palette) {
  return palette[0].y + palette[0].h + 10;
}

function inBox(p, box) {
  return p.x >= box.x && p.x <= box.x + box.w && p.y >= box.y && p.y <= box.y + box.h;
}

function liftPen() {
  penDown = false;
  prevPoint = null;
}

function selectBox(box) {
  if (box.action === 'clear') {
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
  } else {
    activeColor = box.color;
  }
}

// Called once per detection sample (not per rendered frame) so each
// stroke segment corresponds to fresh landmarks.
//
// Roles: the LEFT hand is the pen controller — clench it into a fist to
// draw, open it to stop. The other hand draws with its index fingertip.
// Because the drawing fingertip is not part of the start/stop gesture,
// strokes end exactly where the fingertip was — no release tail.
function processHands(hands) {
  const leftHand = hands.find(isLeftHand) ?? null;
  const drawingHand = hands.find((h) => h !== leftHand) ?? null;
  leftHandSeen = !!leftHand;

  // Pen state from the controller hand, with a dead zone in between.
  if (!leftHand) {
    liftPen();
  } else {
    const extended = countExtendedFingers(leftHand);
    if (extended >= OPEN_MIN_EXTENDED) {
      liftPen();
    } else if (extended <= FIST_MAX_EXTENDED) {
      penDown = true;
    }
  }

  if (!drawingHand) {
    prevPoint = null;
    smoothIndex = null;
    dwell = { boxIndex: -1, since: 0, fired: false };
    return;
  }

  smoothIndex = ema(smoothIndex, mirrorPoint(drawingHand.keypoints[INDEX_TIP]));

  // Dwell-based color selection with the drawing hand's index fingertip.
  const palette = getPalette();
  const hoveredIndex = palette.findIndex((box) => inBox(smoothIndex, box));
  if (hoveredIndex !== dwell.boxIndex) {
    dwell = { boxIndex: hoveredIndex, since: performance.now(), fired: false };
  } else if (
    hoveredIndex >= 0 &&
    !dwell.fired &&
    performance.now() - dwell.since >= DWELL_MS
  ) {
    selectBox(palette[hoveredIndex]);
    dwell.fired = true;
  }

  // No inking while the fingertip is up in the palette strip.
  if (smoothIndex.y <= paletteZoneBottom(palette)) {
    prevPoint = null;
    return;
  }

  if (penDown) {
    if (prevPoint) {
      drawCtx.strokeStyle = activeColor;
      drawCtx.lineWidth = STROKE_WIDTH;
      drawCtx.lineCap = 'round';
      drawCtx.lineJoin = 'round';
      drawCtx.beginPath();
      drawCtx.moveTo(prevPoint.x, prevPoint.y);
      drawCtx.lineTo(smoothIndex.x, smoothIndex.y);
      drawCtx.stroke();
    }
    prevPoint = smoothIndex;
  } else {
    prevPoint = null;
  }
}

// Runs continuously, throttled so the WASM backend (which computes
// synchronously on the main thread) leaves time for rendering.
async function detectionLoop() {
  for (;;) {
    const started = performance.now();
    try {
      if (snapCanvas.width !== video.videoWidth) {
        snapCanvas.width = video.videoWidth;
        snapCanvas.height = video.videoHeight;
      }
      snapCtx.drawImage(video, 0, 0);
      const hands = await detector.estimateHands(snapCanvas);
      latestHands = hands.filter(handHasFiniteKeypoints);
      processHands(latestHands);
    } catch (err) {
      console.error(err);
      await new Promise((r) => setTimeout(r, 250));
    }

    // Yield briefly so the rAF render loop gets main-thread time (the
    // WASM backend computes synchronously), but don't mirror the full
    // inference duration — that halved the sample rate and doubled the
    // drawing-point latency.
    const elapsed = performance.now() - started;
    await new Promise((r) => setTimeout(r, Math.max(16, elapsed * 0.25)));
  }
}

// Selfie mirror is applied here (JS, not CSS) so strokes, palette and
// labels drawn afterwards stay in normal screen coordinates.
function drawVideoMirrored() {
  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  ctx.restore();
}

function drawHandSkeleton(hand) {
  const pts = hand.keypoints.map(mirrorPoint);
  const isController = isLeftHand(hand);
  ctx.lineWidth = 2;
  ctx.strokeStyle = isController && penDown
    ? 'rgba(0, 230, 118, 0.8)'
    : 'rgba(255, 255, 255, 0.6)';
  for (const chain of FINGER_CHAINS) {
    ctx.beginPath();
    chain.forEach((idx, i) => {
      const p = pts[idx];
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
  }

  for (const p of pts) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.fill();
  }
}

function drawPalette(palette) {
  const fontSize = Math.max(12, Math.round(palette[0].w * 0.2));
  ctx.font = `600 ${fontSize}px system-ui, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (const box of palette) {
    ctx.beginPath();
    ctx.roundRect(box.x, box.y, box.w, box.h, 10);
    if (box.action === 'clear') {
      ctx.fillStyle = 'rgba(40, 40, 40, 0.85)';
    } else {
      ctx.fillStyle = box.color;
    }
    ctx.fill();
    const isActive = box.color === activeColor;
    ctx.lineWidth = isActive ? 5 : 2;
    ctx.strokeStyle = isActive ? '#fff' : 'rgba(255, 255, 255, 0.5)';
    ctx.stroke();
    ctx.fillStyle = '#fff';
    ctx.fillText(box.label, box.x + box.w / 2, box.y + box.h / 2);
  }
}

function drawCursor() {
  if (!smoothIndex) return;

  // Dwell progress arc around the fingertip while hovering a box.
  if (dwell.boxIndex >= 0 && !dwell.fired) {
    const progress = Math.min(1, (performance.now() - dwell.since) / DWELL_MS);
    ctx.beginPath();
    ctx.arc(
      smoothIndex.x,
      smoothIndex.y,
      18,
      -Math.PI / 2,
      -Math.PI / 2 + progress * Math.PI * 2,
    );
    ctx.lineWidth = 4;
    ctx.strokeStyle = '#fff';
    ctx.stroke();
  }

  ctx.beginPath();
  ctx.arc(smoothIndex.x, smoothIndex.y, 8, 0, Math.PI * 2);
  ctx.fillStyle = activeColor;
  ctx.fill();
  ctx.lineWidth = penDown ? 4 : 2;
  ctx.strokeStyle = penDown ? '#fff' : 'rgba(255, 255, 255, 0.7)';
  ctx.stroke();
}

// Draws the video frame into the canvas, then the overlays on top.
// Compositing everything into one canvas avoids the Windows hardware
// video-overlay plane hiding absolutely-positioned siblings.
function renderLoop() {
  if (canvas.width !== video.videoWidth && video.videoWidth > 0) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    drawCanvas.width = video.videoWidth;
    drawCanvas.height = video.videoHeight;
  }
  drawVideoMirrored();
  ctx.drawImage(drawCanvas, 0, 0);
  for (const hand of latestHands) {
    drawHandSkeleton(hand);
  }
  drawPalette(getPalette());
  drawCursor();

  if (latestHands.length === 0) {
    setStatus('Show your hands!');
  } else if (!leftHandSeen) {
    setStatus('Show your left hand — clench it to draw, open it to stop');
  } else if (penDown) {
    setStatus('Drawing… (open your left hand to stop)');
  } else {
    setStatus('Left hand open = pen up. Clench it to draw with your other index finger');
  }

  requestAnimationFrame(renderLoop);
}

async function main() {
  try {
    setStatus('Loading hand model & requesting camera…');
    const detectorPromise = createDetector();
    await setupCamera();
    renderLoop();
    detector = await detectorPromise;
    setStatus('Show your hands!');
    detectionLoop();
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`);
  }
}

main();
