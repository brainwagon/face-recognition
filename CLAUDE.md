# CLAUDE.md — Face Recognition Project

## Project overview

Single-file, build-less web app (`index.html`). All JavaScript and CSS are inline. No npm, no bundler. Libraries and model weights are loaded from jsDelivr CDN at runtime.

Run with: `python3 -m http.server 8080` (must be localhost/HTTPS for `getUserMedia`).

## Architecture

Everything lives in `index.html`. Logical sections in order:

1. **CSS** — layout, panel styling, model selector, enrolled-faces list
2. **HTML** — `#scene` (video + canvas), status bar, enroll panel (model selector, name input, storage buttons, enrolled list), legend
3. **JavaScript** — all inline at bottom of `<body>`:
   - DOM refs
   - Model selection state (`currentModel`, `MODEL_PREF_KEY`)
   - Per-person colour palette (`PALETTE`, `colorFor()`)
   - Enrollment store (`enrollments` Map, `rebuildMatcher()`, `renderEnrolledList()`)
   - localStorage persistence (`storageKey()`, `saveToStorage()`, `loadFromStorage()`, `clearStorage()`)
   - Enrollment button handler
   - Geometry helpers (`mean()`, `dist()`, `rad2deg()`)
   - Head pose estimation (`estimatePose()`)
   - Canvas drawing (`drawArrow()`, `drawPoseAxes()`, `drawBoundingBox()`, `drawLabel()`, `drawLandmarks()`)
   - Model loading/switching (`loadModels()`, `switchModel()`)
   - Detection loop (`detectLoop()`)
   - Startup (`init()`)

## Key design decisions

### CDN-only, no local models
Library: `https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.js`
Models: `https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/`

The `@vladmandic/face-api` package includes model weights, so both library and weights come from jsDelivr. No local `models/` directory needed.

### CSS mirror (`transform: scaleX(-1)` on `#scene`)
The `#scene` container (which holds both `<video>` and `<canvas>`) is flipped horizontally so the display feels like a selfie mirror. This means:
- face-api.js detects in the unmirrored video coordinate space
- Canvas drawing coordinates are also unmirrored — positions align correctly after the CSS flip
- **Exception: text**. Text drawn on canvas is pixel-mirrored by the CSS flip. Fix: in `drawLabel()`, apply `ctx.translate(canvas.width, 0); ctx.scale(-1, 1)` before drawing text, then restore. This pre-mirrors the characters so the CSS flip un-mirrors them.

### Per-model localStorage keys
Enrollments are stored under `face-recognition-enrollments-tiny` and `face-recognition-enrollments-full` separately. Switching models loads the corresponding saved enrollments rather than wiping them permanently. This means switching back restores prior enrollments.

### Face descriptors are model-specific
The 128-dim face descriptor produced by `faceRecognitionNet` depends on the landmark model used for face alignment. Descriptors from the tiny and full landmark models are **not comparable**. Always re-enroll after switching models.

### Detection chain
```js
faceapi.detectAllFaces(video, detector)
  .withFaceLandmarks(useTiny)   // true = tiny model
  .withFaceDescriptors()
```
`faceRecognitionNet` is the same for both model modes — only the detector and landmark net change.

### Head pose estimation
Geometric approximation from 68-point landmarks (no OpenCV/solvePnP):
- **Roll**: `atan2` of the eye line
- **Pitch**: nose-tip fraction down the eye-to-chin axis (frontal ≈ 0.45)
- **Yaw**: nose-tip horizontal offset from jaw centre, normalised to jaw width

Adequate for a visual demo. If accuracy matters, upgrade to OpenCV.js with a full PnP solve.

## localStorage keys

| Key | Value |
|---|---|
| `face-recognition-model` | `"tiny"` or `"full"` |
| `face-recognition-enrollments-tiny` | JSON: `{ name: number[][] }` |
| `face-recognition-enrollments-full` | JSON: `{ name: number[][] }` |

Descriptors are serialised as `Array.from(Float32Array)` and deserialised back to `Float32Array` on load.

## face-api.js API notes

- `faceapi.nets.*` models are cached after first load — switching back to a previously loaded model is instant
- `FaceMatcher` threshold is `0.55` (lower = stricter). Adjust in `rebuildMatcher()` if false positives occur
- `faceapi.resizeResults()` scales detection coordinates to match canvas dimensions before drawing
