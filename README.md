# Face Pose & Recognition Demo

A build-less, single-file browser app that uses your webcam for real-time face detection, 68-point landmark overlay, head pose estimation, and face recognition with enrollment.

## Running

```bash
python3 -m http.server 8080
```

Then open `http://localhost:8080` in Chrome or Firefox.

> **Note:** `getUserMedia` requires a secure context. `file://` URLs will not work — you must serve via `localhost` or HTTPS. Accessing from a second machine (e.g. a phone or laptop on the same network) also requires HTTPS.

## Features

- **Live face detection** — bounding box drawn around each detected face
- **68-point landmarks** — facial feature points overlaid in real time
- **Head pose estimation** — yaw / pitch / roll axes drawn from the nose tip; angles shown in the status bar
- **Face recognition** — enroll people by name; recognised faces are labelled with name and confidence
- **Model switching** — toggle between Tiny (fast) and Full (accurate) landmark/detection models
- **Persistent enrollment** — enrolled faces are saved to browser localStorage per model; survive page reloads

## Enrolling a Face

1. Type a name in the input field
2. Make sure only one face is visible in frame
3. Click **Enroll Face** (or press Enter) — repeat from different angles for better accuracy
4. Enrolled faces are saved automatically; use **Save to Browser** / **Load from Browser** to manage them explicitly

## Models

| Setting | Detection | Landmarks | Notes |
|---|---|---|---|
| Tiny (fast) | TinyFaceDetector | faceLandmark68TinyNet | Default; good for real-time use |
| Full (accurate) | SsdMobilenetv1 | faceLandmark68Net | More accurate at off-angles; slower |

Switching models clears in-memory enrollments and loads any enrollments previously saved under that model. Enrollments are **not** compatible between models — re-enroll after switching.

## Libraries

All loaded from CDN — no build step or local install required.

- [`@vladmandic/face-api`](https://github.com/vladmandic/face-api) via jsDelivr
- Model weights also served from jsDelivr (`/model/` path in the same package)
