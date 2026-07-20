#!/usr/bin/env python3
"""
Thermal Camera with MLX90614 - High FPS Version
Serves pure video stream without overlays for maximum performance
"""

import os
import sys
import time
import threading
import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
import json

# Add paths if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your MLX90614 module
from mlx90614_robust import create_robust_mlx_reader


class ThermalSystem:
    def __init__(self, video_device="/dev/video0", i2c_bus=1, default_color_map="inferno"):
        self.video_device = video_device
        self.i2c_bus = i2c_bus
        self.color_map = default_color_map

        # Shared data with thread safety
        self.current_frame = None
        self.frame_lock = threading.Lock()

        # Temperature data from MLX90614
        self.ambient_temp = None
        self.center_temp = None
        self.mlx_ok = False
        self.temp_lock = threading.Lock()

        # System state
        self.running = False
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()

        # Flask app for web interface
        self.app = Flask(__name__)
        self.setup_routes()

        # Initialize components
        self.mlx_reader = None
        self.cap = None

        # Recording state
        self.is_recording = False
        self.video_writer = None
        self.record_lock = threading.Lock()
        self.recording_filename = None
        self.recordings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recordings')

        # Sharpening params (unsharp mask)
        self.sharpen_amount = 0.5
        self.sharpen_blur_size = 5

        # Temporal averaging (noise reduction)
        self.temporal_alpha = 0.6   # weight of current frame; 1.0 = off
        self.prev_frame = None

        # Color map dictionary
        self.color_map_dict = {
            "hot": cv2.COLORMAP_HOT,
            "inferno": cv2.COLORMAP_INFERNO,
            "magma": cv2.COLORMAP_MAGMA,
            "plasma": cv2.COLORMAP_PLASMA,
            "jet": cv2.COLORMAP_JET,
            "rainbow": cv2.COLORMAP_RAINBOW,
            "bone": cv2.COLORMAP_BONE,
            "none": None
        }

    def apply_color_map(self, frame, map_name=None):
        """Apply selected color map to frame with normalization for better contrast"""
        if map_name is None:
            map_name = self.color_map

        # Convert to grayscale if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Normalize for full contrast range (important for non-radiometric cameras)
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Upscale first — sharpening on more pixels reduces noise amplification
        upscaled = cv2.resize(normalized, (640, 480), interpolation=cv2.INTER_CUBIC)

        # Unsharp mask — params tunable via /set_sharpen API
        if self.sharpen_amount > 0:
            blur_k = self.sharpen_blur_size | 1  # ensure odd
            blurred = cv2.GaussianBlur(upscaled, (blur_k, blur_k), 1.0)
            upscaled = cv2.addWeighted(upscaled, 1 + self.sharpen_amount, blurred, -self.sharpen_amount, 0)
            upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)

        if map_name == "none" or map_name not in self.color_map_dict:
            return cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)

        map_type = self.color_map_dict[map_name]
        return cv2.applyColorMap(upscaled, map_type)

    def estimate_temp_range(self, gray):
        """Estimate the frame's min/max temp from the MLX90614 center-point reading,
        via the same linear extrapolation used by /pixel_temp. Also locates the
        coldest/hottest pixel, as normalized (0-1) x,y — same convention /pixel_temp
        takes as input, so a consumer multiplies by their frame's width/height.
        Returns (T_min, T_max, min_xy, max_xy), or (None, None, None, None) if the
        sensor has no valid reading right now."""
        with self.temp_lock:
            T_center, T_ambient, ok = self.center_temp, self.ambient_temp, self.mlx_ok

        if not ok or T_center is None or gray is None or not gray.size:
            return None, None, None, None

        gh, gw = gray.shape[:2]
        min_idx = np.unravel_index(np.argmin(gray), gray.shape)
        max_idx = np.unravel_index(np.argmax(gray), gray.shape)
        v_min = float(gray[min_idx])
        v_max = float(gray[max_idx])
        v_range = max(v_max - v_min, 1.0)
        v_center = float(gray[gh // 2, gw // 2])
        v_center_n = (v_center - v_min) / v_range
        T_span = max(abs((T_ambient if T_ambient is not None else T_center) - T_center), 1.0)
        T_top = T_center + (1.0 - v_center_n) * T_span
        T_bottom = T_center + (0.0 - v_center_n) * T_span

        min_xy = (min_idx[1] / max(gw - 1, 1), min_idx[0] / max(gh - 1, 1))
        max_xy = (max_idx[1] / max(gw - 1, 1), max_idx[0] / max(gh - 1, 1))
        return T_bottom, T_top, min_xy, max_xy

    def overlay_legend(self, colored, gray, map_name):
        """Burn a color-scale legend (gradient + temp labels) into a colored frame."""
        h, w = colored.shape[:2]
        bar_w = 12
        bar_h = int(h * 0.29)
        y0 = (h - bar_h) // 2
        y1 = y0 + bar_h
        x1 = w - 50
        x0 = x1 - bar_w
        label_x = x1 + 3

        panel_x0 = max(x0 - 6, 0)
        panel_y0 = max(y0 - 15, 0)
        panel_x1 = min(x1 + 42, w - 4)
        panel_y1 = min(y1 + 15, h)
        overlay = colored.copy()
        cv2.rectangle(overlay, (panel_x0, panel_y0), (panel_x1, panel_y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, colored, 0.55, 0, colored)

        # Gradient bar: top = hottest (matches apply_color_map's 0-255 normalization)
        grad = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(-1, 1)
        grad = np.repeat(grad, bar_w, axis=1)
        map_type = self.color_map_dict.get(map_name)
        grad_colored = cv2.applyColorMap(grad, map_type) if map_type is not None \
            else cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)
        colored[y0:y1, x0:x1] = grad_colored
        cv2.rectangle(colored, (x0, y0), (x1, y1), (255, 255, 255), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        T_bottom, T_top, _, _ = self.estimate_temp_range(gray)

        if T_top is not None:
            T_mid = (T_top + T_bottom) / 2.0

            cv2.putText(colored, f"{T_top:.1f}C", (label_x, y0 + 3), font, 0.25, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(colored, f"{T_mid:.1f}C", (label_x, y0 + bar_h // 2 + 2), font, 0.25, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(colored, f"{T_bottom:.1f}C", (label_x, y1 + 1), font, 0.25, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(colored, "N/A", (label_x, y0 + bar_h // 2), font, 0.25, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.putText(colored, (map_name or "").upper(), (x0 - 3, y0 - 6), font, 0.22, (255, 255, 255), 1, cv2.LINE_AA)
        return colored

    def setup_routes(self):
        """Setup Flask routes for web interface"""

        @self.app.route('/')
        def index():
            return render_template('thermal_index.html')

        @self.app.route('/video_feed')
        def video_feed():
            """Video feed endpoint - pure video with optional color map"""
            # Use requested color map per-stream without mutating shared state
            map_param = request.args.get('map', self.color_map)
            if map_param not in self.color_map_dict:
                map_param = self.color_map

            return Response(self.generate_frames(color_map=map_param),
                          mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/temperature_data')
        def temperature_data():
            """JSON endpoint for temperature data"""
            with self.temp_lock:
                data = {
                    'ambient': round(self.ambient_temp, 1) if self.ambient_temp is not None else None,
                    'center': round(self.center_temp, 1) if self.center_temp is not None else None,
                    'status': 'OK' if self.mlx_ok else 'ERROR',
                    'fps': round(self.fps, 1)
                }
            return jsonify(data)

        @self.app.route('/temp_range')
        def temp_range():
            """JSON endpoint: current frame's estimated min/max temp + color map,
            plus the normalized (0-1) x,y location of each, so a downstream consumer
            can invert the colormap to approximate per-pixel temperature from the
            video feed (see /pixel_temp for the same estimation applied to a single
            point). Multiply x,y by the consumed frame's width/height for pixel coords."""
            with self.frame_lock:
                gray = self.current_frame.copy() if self.current_frame is not None else None

            T_min, T_max, min_xy, max_xy = self.estimate_temp_range(gray)
            return jsonify({
                'min': round(T_min, 1) if T_min is not None else None,
                'max': round(T_max, 1) if T_max is not None else None,
                'min_xy': {'x': round(min_xy[0], 3), 'y': round(min_xy[1], 3)} if min_xy else None,
                'max_xy': {'x': round(max_xy[0], 3), 'y': round(max_xy[1], 3)} if max_xy else None,
                'color_map': self.color_map,
                'status': 'OK' if T_min is not None else 'ERROR'
            })

        @self.app.route('/system_status')
        def system_status():
            """System health check"""
            mlx_connected = False
            if self.mlx_reader is not None:
                try:
                    ambient, obj, ok = self.mlx_reader.get_latest()
                    mlx_connected = ok
                except Exception as e:
                    print(f"[STATUS] MLX check error: {e}")
                    mlx_connected = False

            return jsonify({
                'running': self.running,
                'mlx_connected': mlx_connected,
                'camera_running': self.cap is not None and self.cap.isOpened(),
                'frame_count': self.frame_count,
                'fps': round(self.fps, 1),
                'color_map': self.color_map
            })

        @self.app.route('/set_color_map', methods=['POST'])
        def set_color_map():
            """API endpoint to change color map"""
            data = request.get_json()
            map_name = data.get('map', 'inferno')

            if map_name in self.color_map_dict:
                self.color_map = map_name
                return jsonify({'success': True, 'color_map': map_name})
            else:
                return jsonify({'success': False, 'error': 'Invalid color map'}), 400

        @self.app.route('/start_record', methods=['POST'])
        def start_record():
            """Start recording the thermal video feed"""
            with self.record_lock:
                if self.is_recording:
                    return jsonify({'success': False, 'error': 'Already recording'}), 400

                os.makedirs(self.recordings_dir, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"thermal_{timestamp}.mp4"
                filepath = os.path.join(self.recordings_dir, filename)

                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                self.video_writer = cv2.VideoWriter(filepath, fourcc, 15.0, (640, 480))

                if not self.video_writer.isOpened():
                    self.video_writer = None
                    return jsonify({'success': False, 'error': 'Failed to open video writer'}), 500

                self.recording_filename = filename
                self.is_recording = True
                print(f"[REC] Recording started: {filename}")

            return jsonify({'success': True, 'filename': filename})

        @self.app.route('/stop_record', methods=['POST'])
        def stop_record():
            """Stop recording and save the file"""
            with self.record_lock:
                if not self.is_recording:
                    return jsonify({'success': False, 'error': 'Not currently recording'}), 400

                self.is_recording = False
                filename = self.recording_filename

                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None

                self.recording_filename = None
                print(f"[REC] Recording stopped: {filename}")

            return jsonify({'success': True, 'filename': filename, 'path': f'recordings/{filename}'})

        @self.app.route('/recording_status')
        def recording_status():
            """Return current recording state"""
            with self.record_lock:
                return jsonify({
                    'recording': self.is_recording,
                    'filename': self.recording_filename
                })

        @self.app.route('/recordings')
        def list_recordings():
            """List saved recordings, newest first"""
            os.makedirs(self.recordings_dir, exist_ok=True)
            files = []
            for fname in os.listdir(self.recordings_dir):
                fpath = os.path.join(self.recordings_dir, fname)
                if os.path.isfile(fpath):
                    stat = os.stat(fpath)
                    files.append({
                        'filename': fname,
                        'size': stat.st_size,
                        'modified': stat.st_mtime
                    })
            files.sort(key=lambda f: f['modified'], reverse=True)
            return jsonify({'recordings': files})

        @self.app.route('/recordings/download/<path:filename>')
        def download_recording(filename):
            """Download a saved recording"""
            safe_name = secure_filename(filename)
            if not safe_name or safe_name != filename:
                return jsonify({'error': 'Invalid filename'}), 400
            return send_from_directory(self.recordings_dir, safe_name, as_attachment=True)

        @self.app.route('/recordings/delete/<path:filename>', methods=['POST'])
        def delete_recording(filename):
            """Delete a saved recording"""
            safe_name = secure_filename(filename)
            if not safe_name or safe_name != filename:
                return jsonify({'success': False, 'error': 'Invalid filename'}), 400

            with self.record_lock:
                if self.is_recording and self.recording_filename == safe_name:
                    return jsonify({'success': False, 'error': 'Cannot delete the active recording'}), 400

            fpath = os.path.join(self.recordings_dir, safe_name)
            if not os.path.isfile(fpath):
                return jsonify({'success': False, 'error': 'Not found'}), 404

            os.remove(fpath)
            return jsonify({'success': True})

        @self.app.route('/get_sharpen')
        def get_sharpen():
            return jsonify({
                'amount': self.sharpen_amount,
                'blur_size': self.sharpen_blur_size,
                'temporal_alpha': self.temporal_alpha
            })

        @self.app.route('/set_sharpen', methods=['POST'])
        def set_sharpen():
            data = request.get_json()
            if 'amount' in data:
                self.sharpen_amount = max(0.0, min(3.0, float(data['amount'])))
            if 'blur_size' in data:
                self.sharpen_blur_size = max(3, min(15, int(data['blur_size'])))
            if 'temporal_alpha' in data:
                self.temporal_alpha = max(0.1, min(1.0, float(data['temporal_alpha'])))
                self.prev_frame = None  # reset on change
            return jsonify({'success': True, 'amount': self.sharpen_amount,
                            'blur_size': self.sharpen_blur_size, 'temporal_alpha': self.temporal_alpha})


        @self.app.route("/pixel_temp")
        def pixel_temp():
            try:
                x_norm = max(0.0, min(1.0, float(request.args.get("x", 0.5))))
                y_norm = max(0.0, min(1.0, float(request.args.get("y", 0.5))))
            except ValueError:
                return jsonify({"temp": None}), 400

            with self.frame_lock:
                if self.current_frame is None:
                    return jsonify({"temp": None})
                frame = self.current_frame.copy()

            h, w = frame.shape[:2]
            x_px = int(x_norm * (w - 1))
            y_px = int(y_norm * (h - 1))
            v_pixel = float(frame[y_px, x_px])
            v_center = float(frame[h // 2, w // 2])

            with self.temp_lock:
                T_center = self.center_temp
                T_ambient = self.ambient_temp
                ok = self.mlx_ok

            if T_center is None or not ok:
                return jsonify({"temp": None})

            import numpy as np
            v_min = float(frame.min())
            v_max = float(frame.max())
            v_range = max(v_max - v_min, 1.0)
            v_pixel_n = (v_pixel - v_min) / v_range
            v_center_n = (v_center - v_min) / v_range
            T_span = max(abs(T_ambient - T_center) * 1.0, 1.0) if T_ambient is not None else 20.0
            T_pixel = T_center + (v_pixel_n - v_center_n) * T_span
            return jsonify({"temp": round(T_pixel, 1)})

    def mlx_thread_func(self):
        """Background thread for MLX90614 readings"""
        print("[MLX] Starting MLX90614 reader thread")

        self.mlx_reader = create_robust_mlx_reader(
            window_size=3,
            poll_interval=0.3,
            i2c_bus=self.i2c_bus,
            max_retries=3
        )

        time.sleep(1)

        error_count = 0
        last_print_time = time.time()

        while self.running:
            try:
                ambient, obj, ok = self.mlx_reader.get_latest()

                with self.temp_lock:
                    self.ambient_temp = ambient
                    self.center_temp = obj
                    self.mlx_ok = ok

                if ok:
                    error_count = 0
                    # Print every 10 seconds using elapsed time, not modulo
                    now = time.time()
                    if now - last_print_time >= 10.0:
                        if ambient and obj:
                            print(f"[MLX] Ambient: {ambient:.1f}°C, Center: {obj:.1f}°C")
                        last_print_time = now
                else:
                    error_count += 1
                    if error_count > 10:
                        print(f"[MLX] Multiple consecutive errors ({error_count})")

            except Exception as e:
                error_count += 1
                print(f"[MLX] Error: {e}")
                with self.temp_lock:
                    self.mlx_ok = False

            time.sleep(0.1)

        if self.mlx_reader:
            self.mlx_reader.stop()

    def camera_thread_func(self):
        """Camera thread using OpenCV directly (more reliable)"""
        print(f"[CAM] Starting OpenCV camera thread with device {self.video_device}")

        device_num = 0
        if self.video_device.startswith('/dev/video'):
            try:
                device_num = int(self.video_device.replace('/dev/video', ''))
            except Exception:
                device_num = 0

        self.cap = None
        for attempt in range(3):
            print(f"[CAM] Attempt {attempt+1} to open camera {device_num}")
            self.cap = cv2.VideoCapture(device_num, cv2.CAP_V4L2)

            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                ret, test_frame = self.cap.read()
                if ret:
                    print(f"[CAM] Success! Frame shape: {test_frame.shape}")
                    actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                    print(f"[CAM] Camera FPS: {actual_fps}")
                    break
                else:
                    print(f"[CAM] Could not read test frame")
                    self.cap.release()
                    self.cap = None
            else:
                print(f"[CAM] Failed to open camera")
                if self.cap:
                    self.cap.release()
                    self.cap = None

            time.sleep(1)

        if self.cap is None or not self.cap.isOpened():
            print("[CAM] Failed to open camera after multiple attempts")
            return

        print("[CAM] OpenCV camera running - HIGH FPS MODE")

        consecutive_errors = 0

        while self.running:
            try:
                ret, frame = self.cap.read()

                if ret and frame is not None:
                    consecutive_errors = 0

                    if len(frame.shape) == 3:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = frame

                    with self.frame_lock:
                        if self.prev_frame is not None and self.temporal_alpha < 1.0:
                            blended = cv2.addWeighted(gray, self.temporal_alpha,
                                                      self.prev_frame, 1.0 - self.temporal_alpha, 0)
                            self.current_frame = blended
                            self.prev_frame = blended
                        else:
                            self.current_frame = gray.copy()
                            self.prev_frame = gray.copy()

                    # Write frame to recording if active (with burned-in color-scale legend)
                    with self.record_lock:
                        if self.is_recording and self.video_writer is not None:
                            colored = self.apply_color_map(gray, map_name=self.color_map)
                            colored = self.overlay_legend(colored, gray, self.color_map)
                            self.video_writer.write(colored)

                    self.frame_count += 1
                    if time.time() - self.last_fps_time >= 1.0:
                        self.fps = self.frame_count
                        self.frame_count = 0
                        self.last_fps_time = time.time()
                else:
                    consecutive_errors += 1
                    print(f"[CAM] Failed to read frame (error {consecutive_errors})")

                    if consecutive_errors > 10:
                        print("[CAM] Too many errors, restarting camera...")
                        self.cap.release()
                        time.sleep(2)
                        self.cap = cv2.VideoCapture(device_num, cv2.CAP_V4L2)
                        consecutive_errors = 0

                    time.sleep(0.1)

            except Exception as e:
                print(f"[CAM] Error: {e}")
                consecutive_errors += 1
                time.sleep(0.1)

            time.sleep(0.001)

        if self.cap:
            self.cap.release()
        print("[CAM] Camera thread stopped")

    def generate_frames(self, color_map=None):
        """Generate MJPEG frames for web streaming - PURE VIDEO, NO OVERLAYS"""
        frame_count = 0

        while self.running:
            with self.frame_lock:
                if self.current_frame is None:
                    time.sleep(0.001)
                    continue

                frame = self.current_frame.copy()

            colored_frame = self.apply_color_map(frame, map_name=color_map)

            ret, jpeg = cv2.imencode('.jpg', colored_frame, [
                cv2.IMWRITE_JPEG_QUALITY, 80,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                cv2.IMWRITE_JPEG_PROGRESSIVE, 0
            ])

            if not ret:
                continue

            frame_count += 1
            if frame_count % 500 == 0:
                print(f"[STREAM] Streamed {frame_count} frames ({self.fps} FPS)")

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    def start(self, port=5000):
        """Start the thermal system"""
        print("\n" + "="*50)
        print("Thermal System Starting - HIGH FPS MODE")
        print("="*50)
        print(f"Video device: {self.video_device}")
        print(f"I2C bus: {self.i2c_bus}")
        print(f"Default color map: {self.color_map}")
        print("="*50)

        self.running = True

        self.mlx_thread = threading.Thread(target=self.mlx_thread_func, daemon=True)
        self.mlx_thread.start()

        self.camera_thread = threading.Thread(target=self.camera_thread_func, daemon=True)
        self.camera_thread.start()

        time.sleep(2)

        print(f"\n[WEB] Starting Flask server on http://0.0.0.0:{port}")
        print("[WEB] Open this URL in your browser")
        print("[WEB] API endpoints:")
        print("      - GET  /video_feed?map=inferno  (video stream)")
        print("      - GET  /temperature_data        (temperature JSON)")
        print("      - GET  /temp_range              (estimated frame min/max temp)")
        print("      - GET  /system_status           (system info)")
        print("      - POST /set_color_map           (change color map)")
        print("      - POST /start_record            (start recording)")
        print("      - POST /stop_record             (stop recording)")
        print("      - GET  /recording_status        (recording state)")
        print("      - GET  /recordings              (list saved recordings)")
        print("      - GET  /recordings/download/<f> (download a recording)")
        print("      - POST /recordings/delete/<f>   (delete a recording)")
        print("[WEB] Press Ctrl+C to stop\n")

        try:
            self.app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        except KeyboardInterrupt:
            print("\n[SYSTEM] Shutting down...")
        finally:
            self.stop()

    def stop(self):
        """Clean shutdown"""
        print("[SYSTEM] Stopping...")
        self.running = False

        if self.cap:
            self.cap.release()

        with self.record_lock:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.is_recording = False

        if self.mlx_reader:
            self.mlx_reader.stop()

        print("[SYSTEM] Shutdown complete")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Thermal Camera with MLX90614 - High FPS Version')
    parser.add_argument('--device', default='/dev/video0',
                       help='Thermal camera video device (default: /dev/video0)')
    parser.add_argument('--i2c-bus', type=int, default=1,
                       help='I2C bus number (default: 1)')
    parser.add_argument('--color-map', default='inferno',
                   choices=['hot', 'inferno', 'magma', 'plasma', 'jet', 'rainbow', 'bone', 'none'],
                   help='Default color map for thermal display')
    parser.add_argument('--port', type=int, default=5000,
                       help='Web server port (default: 5000)')

    args = parser.parse_args()

    system = ThermalSystem(
        video_device=args.device,
        i2c_bus=args.i2c_bus,
        default_color_map=args.color_map
    )

    try:
        system.start(port=args.port)
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user")
    finally:
        system.stop()


if __name__ == "__main__":
    main()
