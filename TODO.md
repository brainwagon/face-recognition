# TODO

## Potential improvements

- **Constrain camera resolution to model** — `getUserMedia` currently requests `{ video: true }`, which lets the browser default to 720p or 1080p. Both models immediately downsample for detection (tiny: 416×416, full: 300×300), so the extra pixels are wasted. Ideal: 640×480 for tiny, 480×360 for full. Could be wired to the model selector so resolution switches automatically. Note: higher resolution only genuinely helps if enrolling at distance (small face in frame) where the 150×150 recognition crop benefits from more source pixels.
