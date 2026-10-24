"""Rolling buffer that turns every alert into a clip with context
(PRE seconds before + POST seconds after). Frames are kept as JPEG bytes so
memory stays small even for 1080p CCTV."""

import os
import shutil
import subprocess
import threading
from collections import deque
from datetime import datetime
import cv2


class ClipRecorder:

    def __init__(self, out_dir, pre_s=5, post_s=10, fps=15):
        # Store settings
        self.out_dir = out_dir
        self.pre_s = pre_s
        self.post_s = post_s
        self.fps = fps

        # Store recent frames and alerts
        self.buf = deque()
        self.pending = []

        self._last_push = 0.0
        self._lock = threading.Lock()

    def push(self, frame, ts):
        # Save frames at the required FPS
        if ts - self._last_push < 1.0 / self.fps:
            return

        self._last_push = ts

        # Convert frame to JPEG to save memory
        success, jpg = cv2.imencode(".jpg", frame)

        if not success:
            return

        item = (ts, jpg)
        finished = []

        with self._lock:
            # Add the current frame
            self.buf.append(item)

            # Keep only the recent pre-event frames
            while self.buf and ts - self.buf[0][0] > self.pre_s:
                self.buf.popleft()

            # Collect frames for active alerts
            for alert in self.pending:
                alert["frames"].append(item)

                if ts >= alert["end_ts"]:
                    finished.append(alert)

            # Remove completed alerts
            for alert in finished:
                self.pending.remove(alert)

        # Save completed clips in the background
        for alert in finished:
            threading.Thread(
                target=self._write,
                args=(alert,),
                daemon=True
            ).start()

    def trigger(self, event_ts, name, on_saved=None):
        # Start recording a clip for an alert
        with self._lock:
            self.pending.append({
                "frames": list(self.buf),
                "end_ts": event_ts + self.post_s,
                "event_ts": event_ts,
                "name": name,
                "on_saved": on_saved
            })

    def flush(self):
        # Save clips that are still waiting for frames
        with self._lock:
            alerts = self.pending
            self.pending = []

        for alert in alerts:
            self._write(alert)

    def _write(self, alert):
        # Do not create an empty video
        if not alert["frames"]:
            return

        os.makedirs(self.out_dir, exist_ok=True)

        # Create a filename using the alert time
        timestamp = datetime.fromtimestamp(
            alert["event_ts"]
        ).strftime("%Y%m%d_%H%M%S")

        path = os.path.join(
            self.out_dir,
            f"{alert['name']}_{timestamp}.mp4"
        )

        # Get the video dimensions
        first_frame = cv2.imdecode(
            alert["frames"][0][1],
            cv2.IMREAD_COLOR
        )

        height, width = first_frame.shape[:2]

        # Create a temporary video
        temp_path = path + ".tmp.mp4"

        writer = cv2.VideoWriter(
            temp_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height)
        )

        # Add all saved frames to the video
        for timestamp, jpg in alert["frames"]:
            frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
            writer.write(frame)

        writer.release()

        # Convert the video to H.264 if FFmpeg is available
        if shutil.which("ffmpeg"):
            result = subprocess.run([
                "ffmpeg", "-y",
                "-loglevel", "error",
                "-i", temp_path,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                path
            ])

            if result.returncode == 0:
                os.remove(temp_path)
            else:
                os.replace(temp_path, path)
        else:
            os.replace(temp_path, path)

        # Notify the program when the clip is saved
        if alert["on_saved"]:
            alert["on_saved"](path)