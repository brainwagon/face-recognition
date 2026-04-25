# Face Pose & Recognition Demo

A tutorial-style, single-file browser application demonstrating real-time face
detection, 68-point landmark overlay, head pose estimation, and face
recognition — entirely in the browser, with no server, no npm, and no build
step.

---

## Table of contents

1. [Running the demo](#running-the-demo)
2. [How it works — overview](#how-it-works--overview)
3. [The three neural networks](#the-three-neural-networks)
4. [The detection pipeline in detail](#the-detection-pipeline-in-detail)
5. [Face recognition: embeddings and matching](#face-recognition-embeddings-and-matching)
6. [Head pose estimation](#head-pose-estimation)
7. [The mirror trick and canvas text](#the-mirror-trick-and-canvas-text)
8. [Persistence with localStorage](#persistence-with-localstorage)
9. [Enrolling faces — practical tips](#enrolling-faces--practical-tips)
10. [Tradeoffs and tuning knobs](#tradeoffs-and-tuning-knobs)
11. [Potential improvements](#potential-improvements)

---

## Running the demo

```bash
python3 -m http.server 8080
```

Then open `http://localhost:8080` in Chrome or Firefox.

**Why a server?**  The browser `getUserMedia` API (webcam access) requires a
*secure context* — a page served over HTTPS or from `localhost`.  A bare
`file://` URL is not a secure context and will fail with a permission error.
Accessing from a second device on the same network (phone, tablet) also
requires HTTPS; a self-signed cert or a tunnel like `ngrok` works for that.

---

## How it works — overview

Every animation frame the app runs a three-stage neural network pipeline on
the live webcam image and then draws the results onto a transparent `<canvas>`
layered over the `<video>` element:

```
Webcam frame
    │
    ▼
┌─────────────────────────────┐
│  Stage 1: Face detection    │  SSD MobileNet v1
│  Input:  full video frame   │  → bounding box per face
└─────────────────────────────┘
    │  (crop each bounding box)
    ▼
┌─────────────────────────────┐
│  Stage 2: Landmark detection│  faceLandmark68Net
│  Input:  face crop          │  → 68 (x, y) keypoints
└─────────────────────────────┘
    │  (align face chip using landmarks)
    ▼
┌─────────────────────────────┐
│  Stage 3: Face embedding    │  faceRecognitionNet (InceptionResNet V1)
│  Input:  aligned face chip  │  → Float32Array(128) descriptor
└─────────────────────────────┘
    │
    ├──► FaceMatcher → identity label + confidence
    └──► estimatePose() → yaw / pitch / roll angles
```

The whole chain is expressed in face-api.js as a single fluent call:

```js
const detections = await faceapi
  .detectAllFaces(video, new faceapi.SsdMobilenetv1Options())
  .withFaceLandmarks()
  .withFaceDescriptors();
```

**Startup sequence** (runs once):

1. Download neural network weights from CDN (~12 MB, three model files)
2. Restore previously saved enrollments from localStorage
3. Request webcam permission via `navigator.mediaDevices.getUserMedia`
4. On the first playing frame, size the canvas to the camera resolution and
   start `detectLoop()` via `requestAnimationFrame`

---

## The three neural networks

### 1. SSD MobileNet v1 — face detection

**What it does:** scans the entire video frame and outputs a bounding box with
a confidence score for each face it finds.

**Architecture:** Single Shot MultiBox Detector (SSD) with a MobileNet v1
feature extractor.  SSD is a one-stage detector — it predicts bounding boxes
and class scores in a single forward pass, without a separate region-proposal
step.  MobileNet replaces standard convolutions with *depthwise-separable*
convolutions, which are roughly 8–9× cheaper in computation while retaining
most accuracy.

**Why this model?**  It handles multiple simultaneous faces, is robust to a
wide range of scales and lighting conditions, and runs at interactive speeds in
a browser (WebGL backend via TensorFlow.js).

**Typical latency:** 30–80 ms per frame on a laptop GPU.

---

### 2. faceLandmark68Net — landmark detection

**What it does:** given a cropped face image, locates 68 specific facial
keypoints and returns them as an array of `{x, y}` coordinates.

**Landmark layout (indices):**

```
 0–16   jaw line, left to right
17–21   left eyebrow
22–26   right eyebrow
27–30   nose bridge
31–35   nose tip / nostrils
36–41   left eye (6 points, clockwise)
42–47   right eye (6 points, clockwise)
48–59   outer mouth
60–67   inner mouth (lips)
```

**Why landmarks matter:**  They serve two purposes:
1. **Face alignment** — before passing the face to the recognition network,
   face-api.js uses the eye and mouth corner positions to crop and warp the
   face into a canonical upright pose.  Alignment is critical: unaligned
   inputs degrade recognition accuracy significantly.
2. **Head pose estimation** — the geometric pose algorithm uses eye centres,
   nose tip, chin, and jaw endpoints (see below).

---

### 3. faceRecognitionNet — face embedding (InceptionResNet V1)

**What it does:** takes a 160×160 aligned face chip and outputs a
128-dimensional floating-point vector called a *face descriptor* or
*embedding*.

**Architecture:** InceptionResNet V1, trained with a triplet loss on
VGGFace2 (a large-scale face dataset with ~3.3 million images of ~9000
identities).

**How recognition works:**  The network was trained so that embeddings of the
same person cluster together and embeddings of different people are far apart —
measured by Euclidean distance in the 128-dimensional space.  Recognition is
then a nearest-neighbour lookup: find the enrolled descriptor with the smallest
distance to the live descriptor.

A distance near 0 means "very likely the same person".  The threshold used here
is 0.55 — distances below that are reported as a match; above it, "unknown".

**The 128-dim descriptor is model-specific.**  It depends on the alignment
produced by the landmark model.  Descriptors computed with one version of the
landmark model are not comparable to those computed with another.  Always
re-enroll if you change models.

---

## The detection pipeline in detail

```js
const detections = await faceapi
  .detectAllFaces(video, new faceapi.SsdMobilenetv1Options())
  .withFaceLandmarks()
  .withFaceDescriptors();
```

This single `await` runs all three networks sequentially and returns an array
with one entry per detected face.  Each entry has:

| Field | Type | Contents |
|---|---|---|
| `detection` | `FaceDetection` | bounding box, confidence score |
| `landmarks` | `FaceLandmarks68` | 68 `{x,y}` positions, helper methods |
| `descriptor` | `Float32Array(128)` | face embedding |

Because the networks internally resize their inputs, the returned coordinates
are in the network's coordinate space, not the canvas's.  `faceapi.resizeResults()`
rescales all coordinates to match the actual canvas dimensions:

```js
const resized = faceapi.resizeResults(detections, { width: W, height: H });
```

**Why `requestAnimationFrame` instead of `setInterval`?**  Two reasons:

- The browser suspends `rAF` callbacks when the tab is hidden, saving CPU/GPU.
- Draws are synchronised to the display refresh cycle, preventing tearing.

The loop is naturally throttled by the `await` — the next frame is only
requested after all three networks finish, so throughput is limited by
inference time rather than the timer interval.

---

## Face recognition: embeddings and matching

### Enrollment

"Enrolling" a person means capturing their 128-dim descriptor from the live
feed and storing it under a name.  Multiple samples from different angles
and lighting conditions improve robustness because FaceMatcher compares the
live descriptor against *all* stored samples and picks the closest one.

**Practical advice:** enroll 3–6 samples per person: front, slight left turn,
slight right turn, and optionally different lighting.

### FaceMatcher

`faceapi.FaceMatcher` wraps the enrolled `LabeledFaceDescriptors` and exposes
`findBestMatch(descriptor)`.  Internally it computes the Euclidean distance
from the query descriptor to every enrolled sample, takes the minimum per
label, and returns the label whose minimum distance is smallest — provided
that minimum is below the threshold.

**Distance → confidence conversion used in the UI:**

```
confidence (%) = (1 − distance) × 100
```

This is a linear mapping: distance 0 → 100%, distance 0.55 → 45%.  It is not
a calibrated probability, just an intuitive display value.

### Threshold tuning (default: 0.55)

| Threshold | Effect |
|---|---|
| 0.4 | Strict — fewer false positives; enrolled person may show as "Unknown" at off-angles |
| 0.55 | Balanced default for a single webcam demo |
| 0.7 | Permissive — enrolled person rarely shows as "Unknown" but wrong matches increase |

Change the threshold in `rebuildMatcher()`:

```js
faceMatcher = new faceapi.FaceMatcher(labeled, 0.55);  // ← adjust here
```

---

## Head pose estimation

Head pose is derived geometrically from the 68 landmarks — no OpenCV, no
PnP solve required.

### Roll

```
roll = atan2(rightEye.y − leftEye.y,  rightEye.x − leftEye.x)
```

The angle of the line connecting both eye centres relative to horizontal.
Zero = eyes level.  Positive = head tilted clockwise.

### Pitch (up/down tilt)

```
pitch = (dist(eyeMidpoint, noseTip) / dist(eyeMidpoint, chin) − 0.45) × 2.2
```

When looking straight ahead the nose tip sits roughly 45% of the way from the
eye midpoint to the chin.  Looking up shortens this ratio; looking down
lengthens it.  The constant 2.2 stretches the range to approximately ±1 for
±30° of tilt.

### Yaw (left/right turn)

```
yaw = (noseTip.x − jawMidpoint.x) / (jawWidth × 0.5)
```

Turning the head moves the nose tip away from the jaw centreline toward the
near side.  Normalising by half the jaw width gives a dimensionless value that
is approximately ±1 at ±90°.

### Tradeoffs

| Approach | Pro | Con |
|---|---|---|
| Geometric (this demo) | Zero dependencies, <1 ms | Not perspective-correct; saturates beyond ~30° |
| OpenCV.js + solvePnP | Accurate at all angles | Adds ~7 MB library, more complex setup |
| ML-based (e.g. MediaPipe Face Mesh) | Accurate, no calibration | Separate model to load, replaces face-api |

The geometric approach is entirely adequate for a visual demo.

---

## The mirror trick and canvas text

Webcams produce a laterally inverted image relative to a mirror: raising your
right hand appears on the left side of the frame.  This feels unnatural when
the display is used as a selfie mirror.

**The fix:** apply `transform: scaleX(-1)` to the `#scene` container (which
holds both `<video>` and `<canvas>`).  Because both elements flip together,
the drawing coordinates do not need to change — face-api.js detects in the
unmirrored video coordinate space, and the canvas is drawn in that same space;
the CSS flip makes everything appear correctly.

**The text problem:** text drawn with `ctx.fillText()` is pixel data.  The CSS
flip mirrors those pixels, producing backwards text.  The fix is to pre-mirror
the text on the canvas so the CSS flip un-mirrors it:

```js
ctx.save();
ctx.translate(canvas.width, 0);   // move origin to right edge
ctx.scale(-1, 1);                  // flip horizontally
ctx.fillText(text, canvas.width - x - textWidth, y);  // mirror the x coord
ctx.restore();
```

---

## Persistence with localStorage

Enrolled face descriptors are saved under the key
`face-recognition-enrollments-full`.

`Float32Array` cannot be JSON-serialised directly.  The serialisation round-trip
is:

```js
// Save: Float32Array → plain Array → JSON string
data[name] = descs.map(d => Array.from(d));
localStorage.setItem(key, JSON.stringify(data));

// Load: JSON string → plain Array → Float32Array
enrollments.set(name, arrays.map(a => new Float32Array(a)));
```

**Storage size:** each descriptor is 128 floats × 4 bytes = 512 bytes.
A person with 10 samples uses ~5 KB.  localStorage's typical quota is 5–10 MB,
so hundreds of enrolled samples fit easily.

---

## Enrolling faces — practical tips

1. Type a name and click **Enroll Face** (or press Enter).  Only one face may
   be in frame — the app refuses to enroll when zero or multiple faces are
   visible.
2. Enroll **3–6 samples per person**: front-on, 20° left, 20° right, and
   optionally a different lighting setup.
3. Enrolled data is saved to localStorage automatically on each enrollment.
   Use **Save to Browser** / **Load from Browser** to manage it explicitly.
4. Recognition degrades at angles beyond ~40°, in very low light, and when
   the face is smaller than ~80×80 px in the frame.

---

## Tradeoffs and tuning knobs

### Accuracy vs. speed

The full SSD MobileNet + faceLandmark68Net combination is accurate and handles
off-angle faces well.  On a machine without a discrete GPU (using WebGL over
integrated graphics or falling back to CPU) frame rates may drop to 2–5 fps.
Options if speed is a bottleneck:

- Reduce the input video resolution via `getUserMedia` constraints:
  ```js
  navigator.mediaDevices.getUserMedia({ video: { width: 320, height: 240 } })
  ```
- Request a specific TF.js backend before loading models:
  ```js
  await faceapi.tf.setBackend('webgl');   // or 'wasm', 'cpu'
  ```

### FaceMatcher threshold

See the [threshold tuning table](#threshold-tuning-default-055) above.

### FPS smoothing

The EMA alpha in `detectLoop()` is 0.15 (new sample weight).  Higher values
make the display more responsive to instantaneous changes; lower values give a
more stable readout.

### Number of enrollment samples

FaceMatcher computes the minimum distance across all stored samples.  Beyond
~10 samples per person the marginal gain in recognition accuracy is small.
Too many samples (50+) slightly increases matching latency but is unlikely to
matter at single-digit fps.

---

## Potential improvements

| Feature | Approach |
|---|---|
| More accurate head pose | Replace `estimatePose()` with OpenCV.js + `solvePnP` using a generic 3-D face model |
| Better recognition at extreme angles | Enroll more samples; or switch to a model with explicit angle handling |
| Works offline | Cache the CDN assets with a Service Worker |
| Multiple cameras | Add a `<select>` populated with `enumerateDevices()` output, pass the chosen `deviceId` to `getUserMedia` |
| Export / import enrollments | Serialise `localStorage` data to a JSON file; restore by parsing and writing back |
| Liveness detection | Beyond the scope of face-api.js — requires a separate model or IR sensor |
