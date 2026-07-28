import * as tf from '@tensorflow/tfjs-core';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import * as handPoseDetection from '@tensorflow-models/hand-pose-detection';
import {
  Finger,
  FingerCurl,
  GestureDescription,
  GestureEstimator,
} from 'fingerpose';

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

const THUMB_TIP = 4;
const INDEX_TIP = 8;
const FINGER_TIPS = [4, 8, 12, 16, 20];
const GESTURE_MIN_SCORE = 8;

// Fingers connected as chains from the wrist, used to draw the hand skeleton.
const FINGER_CHAINS = [
  [0, 1, 2, 3, 4],
  [0, 5, 6, 7, 8],
  [5, 9, 10, 11, 12],
  [9, 13, 14, 15, 16],
  [13, 17, 18, 19, 20],
  [0, 17],
];

// "Frame corner" gesture: thumb + index extended, other fingers curled.
function buildFrameCornerGesture() {
  const gesture = new GestureDescription('frame_corner');

  gesture.addCurl(Finger.Thumb, FingerCurl.NoCurl, 1.0);
  gesture.addCurl(Finger.Index, FingerCurl.NoCurl, 2.0);

  for (const finger of [Finger.Middle, Finger.Ring, Finger.Pinky]) {
    gesture.addCurl(finger, FingerCurl.FullCurl, 1.0);
    gesture.addCurl(finger, FingerCurl.HalfCurl, 0.6);
  }

  return gesture;
}

const gestureEstimator = new GestureEstimator([buildFrameCornerGesture()]);

// Hidden picture revealed only inside the framing rectangle.
// Drop the image at public/reveal.png (served as /reveal.png).
const revealImg = new Image();
revealImg.src = '/reveal.png';
let revealReady = false;
revealImg.onload = () => {
  revealReady = true;
};
revealImg.onerror = () =>
  console.warn('reveal image missing — save it as public/reveal.png');

let detector = null;
let latestHands = [];

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

function matchesFrameCorner(hand) {
  // fingerpose expects 21 landmarks as [x, y, z] arrays.
  const landmarks = hand.keypoints.map((kp) => [kp.x, kp.y, 0]);
  const { gestures } = gestureEstimator.estimate(landmarks, GESTURE_MIN_SCORE);
  return gestures.some((g) => g.name === 'frame_corner');
}

function drawHand(hand, isFraming) {
  const pts = hand.keypoints;
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
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

  // Circles around all five fingertips; thumb + index get the gesture color.
  for (const idx of FINGER_TIPS) {
    const p = pts[idx];
    const isGestureTip = idx === THUMB_TIP || idx === INDEX_TIP;
    ctx.beginPath();
    ctx.arc(p.x, p.y, isGestureTip ? 16 : 12, 0, Math.PI * 2);
    ctx.lineWidth = 3;
    ctx.strokeStyle = isGestureTip
      ? isFraming
        ? '#00e676'
        : '#ffb300'
      : 'rgba(255, 255, 255, 0.9)';
    ctx.stroke();
  }
}

function drawFrameRect(hands) {
  const points = hands.flatMap((hand) => [
    hand.keypoints[THUMB_TIP],
    hand.keypoints[INDEX_TIP],
  ]);
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const width = Math.max(...xs) - minX;
  const height = Math.max(...ys) - minY;

  if (revealReady) {
    // Fit the image to the video size (keep aspect ratio, centered) and
    // show only the part inside the rectangle.
    const scale = Math.min(
      canvas.width / revealImg.naturalWidth,
      canvas.height / revealImg.naturalHeight,
    );
    const dw = revealImg.naturalWidth * scale;
    const dh = revealImg.naturalHeight * scale;
    const dx = (canvas.width - dw) / 2;
    const dy = (canvas.height - dh) / 2;

    ctx.save();
    ctx.beginPath();
    ctx.rect(minX, minY, width, height);
    ctx.clip();
    // The canvas is CSS-mirrored for the selfie view; pre-flip the image
    // so it appears un-mirrored on screen (dx stays valid: it's centered).
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(revealImg, dx, dy, dw, dh);
    ctx.restore();
  } else {
    ctx.fillStyle = 'rgba(0, 230, 118, 0.12)';
    ctx.fillRect(minX, minY, width, height);
  }

  ctx.lineWidth = 4;
  ctx.strokeStyle = '#00e676';
  ctx.strokeRect(minX, minY, width, height);
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
    } catch (err) {
      console.error(err);
      await new Promise((r) => setTimeout(r, 250));
    }

    const elapsed = performance.now() - started;
    await new Promise((r) => setTimeout(r, Math.max(33, elapsed)));
  }
}

// Draws the video frame into the canvas, then the overlays on top.
// Compositing everything into one canvas avoids the Windows hardware
// video-overlay plane hiding absolutely-positioned siblings.
function renderLoop() {
  if (canvas.width !== video.videoWidth && video.videoWidth > 0) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const framingHands = [];
  for (const hand of latestHands) {
    const isFraming = matchesFrameCorner(hand);
    if (isFraming) framingHands.push(hand);
    drawHand(hand, isFraming);
  }

  if (framingHands.length >= 2) {
    drawFrameRect(framingHands.slice(0, 2));
    setStatus('Framing! 📸');
  } else if (latestHands.length === 0) {
    setStatus('Show your hands!');
  } else {
    setStatus(
      `${latestHands.length} hand(s) detected — extend thumb + index on both hands`,
    );
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
