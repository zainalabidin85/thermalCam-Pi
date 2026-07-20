# ThermalCam Pi

Flask app that runs on the Raspberry Pi, combining a USB camera with an MLX90614
infrared point sensor to produce a pseudo-thermal video feed, temperature
estimates, and a recording feature. Deployed on `pi4` as the systemd service
`thermalcam.service` (auto-starts on boot), serving on port `5000`.

## Hardware

- **Camera**: a plain USB UVC webcam (not a radiometric thermal sensor). It has
  no per-pixel temperature data of its own — the "thermal" look comes entirely
  from applying an OpenCV color map (inferno, hot, jet, etc.) to its grayscale
  video.
- **MLX90614**: an I2C infrared thermometer (address `0x5A`) that gives exactly
  **one** temperature reading per poll (object/center + ambient) — not an array.

Because of this, there is no true per-pixel radiometric data anywhere in this
app. Every "temperature at point X" value is an **estimate**: a linear
extrapolation from the MLX90614's single center-point reading across the
frame's normalized grayscale intensity range (colder pixels below the center
value, hotter pixels above it, scaled by how far ambient is from center). It's
useful for relative comparison (where's hotter/colder in frame) and rough
values, not for calibrated/precision temperature logging.

## Running

```
python3 thermal_overlay.py [--device /dev/video0] [--i2c-bus 1] [--color-map inferno] [--port 5000]
```

Color maps: `hot`, `inferno`, `magma`, `plasma`, `jet`, `rainbow`, `bone`, `none`.

## API

All endpoints are on port `5000`.

### Video

| Endpoint | Method | Description |
|---|---|---|
| `/video_feed?map=<name>` | GET | MJPEG stream (`multipart/x-mixed-replace`). Optional `map` query param overrides the color map for just this stream without changing the shared default. Live stream has no overlays. |
| `/` | GET | Web UI (`templates/thermal_index.html`) — view feed, change color map, record/download. |

### Temperature

| Endpoint | Method | Description |
|---|---|---|
| `/temperature_data` | GET | Raw sensor reading: `{"ambient", "center", "status", "fps"}`. `status` is `"OK"`/`"ERROR"` depending on whether the last MLX90614 poll succeeded. |
| `/pixel_temp?x=<0-1>&y=<0-1>` | GET | Estimated temperature at one normalized point in the frame: `{"temp": 31.4}` (or `{"temp": null}` if the sensor has no valid reading). This is the recommended way for an external consumer (e.g. the Jetson) to get a temperature at a specific location — it's computed server-side from the true raw sensor frame, not a compressed/color-mapped copy. |
| `/temp_range` | GET | Current frame's estimated min/max temp and where they are: `{"min", "max", "min_xy": {"x","y"}, "max_xy": {"x","y"}, "color_map", "status"}`. `min_xy`/`max_xy` are normalized (0-1) — multiply by your consumed frame's width/height for pixel coordinates. Meant to be polled periodically (thermal range changes slowly, no need per-frame), e.g. to invert the video's color map into an approximate per-pixel temperature. |
| `/system_status` | GET | Health check: `{"running", "mlx_connected", "camera_running", "frame_count", "fps", "color_map"}`. |

### Recording

Recordings are saved as `.mp4` (H.264/avc1, 15fps, 640x480) to `recordings/`, with a
color-scale legend (gradient bar + min/mid/max °C labels) burned into every
frame — this overlay only exists in recordings, never in the live stream.

| Endpoint | Method | Description |
|---|---|---|
| `/start_record` | POST | Start recording. Returns `{"success", "filename"}`. |
| `/stop_record` | POST | Stop recording. Returns `{"success", "filename", "path"}`. |
| `/recording_status` | GET | `{"recording", "filename"}`. |
| `/recordings` | GET | List saved recordings, newest first: `{"recordings": [{"filename", "size", "modified"}, ...]}`. |
| `/recordings/download/<filename>` | GET | Download a recording as an attachment. |
| `/recordings/delete/<filename>` | POST | Delete a recording. Refuses to delete the one currently being recorded. |

### Image tuning

| Endpoint | Method | Description |
|---|---|---|
| `/set_color_map` | POST | Body `{"map": "inferno"}`. Changes the shared default color map. |
| `/get_sharpen` | GET | `{"amount", "blur_size", "temporal_alpha"}`. |
| `/set_sharpen` | POST | Body with any of `amount` (0-3, unsharp mask strength), `blur_size` (3-15, odd), `temporal_alpha` (0.1-1.0, frame-blend smoothing — lower = more noise reduction, more motion blur). |

## Known limitation — MLX90614 I2C errors

The sensor throws frequent `Errno 5` I2C read errors (roughly 30-60% of
individual transactions), masked by retry + moving-average logic in
`mlx90614_robust.py`. Confirmed not a wiring/power fault (sensor detected fine
on the bus, `vcgencmd get_throttled` shows no undervoltage) — it's a bus
reliability quirk of this MLX90614 + Pi combination. Lowering the I2C clock
speed (the usual fix for clock-stretching issues) was tried and made it worse
(0% success at 10kHz, likely hitting the sensor's own SMBus timeout), so it's
been left at the default 100kHz. `/temperature_data`, `/pixel_temp`, and
`/temp_range` all return `status: "ERROR"` / `null` values on frames where the
last poll failed — callers should handle that rather than assume a reading is
always available.
