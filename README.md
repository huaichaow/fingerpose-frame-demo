# Hand Framing Demo — TensorFlow.js + fingerpose

A webcam demo that tracks both hands in real time. Make a "photo framing"
gesture with both hands — thumb + index finger extended, other fingers
curled — and a rectangle spanning the two hands is drawn. If a reveal
image is configured, the part of it inside the rectangle is disclosed,
like a window you move and resize with your hands.

## Features

- Tracks up to 2 hands (21 keypoints each) with skeleton overlay
- Circles around all five fingertips; thumb + index highlighted
  (amber → green when the framing gesture is recognized)
- Custom [fingerpose](https://github.com/andypotato/fingerpose) gesture
  ("frame corner"): thumb + index extended, middle/ring/pinky curled
- When both hands hold the gesture, draws the bounding rectangle of the
  four fingertips and reveals the hidden image inside it
- Mirrored selfie view; the revealed image is counter-flipped so it
  appears the right way around

## Stack

| Piece | Choice |
| --- | --- |
| Hand landmarks | `@tensorflow-models/hand-pose-detection` (MediaPipeHands, `runtime: 'tfjs'`, `maxHands: 2`) |
| Gesture recognition | `fingerpose` with a custom `GestureDescription` |
| TF.js backend | **WASM** (`@tensorflow/tfjs-backend-wasm`), lite model |
| Dev server / bundler | Vite |

## Run

```bash
npm install
npm run dev
```

Open http://localhost:5173 and allow camera access (localhost is a secure
context, so `getUserMedia` works without HTTPS).

### Reveal image

Save any image as `public/reveal.png`. It is scaled to the video size
(aspect ratio preserved, centered) and shown only inside the framing
rectangle. Without the file, the rectangle shows a translucent green fill.

## Implementation notes (hard-won)

These workarounds were required on the original development machine and
are good defensive practice generally:

1. **WASM backend instead of WebGL.** The TF.js WebGL backend produced
   garbage on this GPU/driver (phantom detections with NaN keypoints, or
   zero detections). The WASM backend is deterministic and correct.
2. **Canvas snapshot instead of the video element.** Even on WASM,
   passing the `HTMLVideoElement` to `estimateHands` returned broken
   results, while the identical frame copied to a canvas via `drawImage`
   worked perfectly. Each detection cycle snapshots the frame first.
3. **Single-canvas rendering.** The video frame is drawn into the same
   canvas as the overlays. Layering a transparent canvas over a `<video>`
   failed on Windows: the hardware video-overlay plane composited the
   video above the canvas, hiding all drawings.
4. **Throttled detection loop.** The WASM backend computes synchronously
   on the main thread; the loop idles at least as long as inference took
   (min 33 ms) so rendering stays smooth. Detection and rendering run as
   independent loops sharing the latest result.
5. **Avoid `runtime: 'mediapipe'` under Vite.** The legacy
   `@mediapipe/hands` package hangs forever on `wasm-instantiate` when
   pre-bundled by Vite.
