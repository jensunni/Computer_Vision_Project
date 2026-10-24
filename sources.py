""" ITS WORKING
Step 1: Get video
Connect to a webcam or network CCTV camera.
Step 2: Read frames
Continuously capture images in a background thread.
Step 3: Provide frames to AI
Your detection model can request the latest available image."""
import cv2
import threading  # run multiple threads (lightweight, independent execution paths) at the same time within a single program.
import time
import os


class FrameSource:
    def __init__(self, uri):
        self.uri = int(uri) if str(uri).isdigit() else uri
        # Store as int if it is an integer else a string
        self.frame = None
        self.lock = threading.Lock()
        # Lock prevents conditions to ensure that only one data at a time
        self.stop_event = threading.Event()
        # Event is a boolean flag used for communication between threads

    def start(self):
        self.thread = threading.Thread(target=self.read_video, daemon=True)
        # `Rechord video` if needed and shutting down tread at the end of the program
        self.thread.start()
        return self

    def read_video(self):
        if isinstance(self.uri, str) and self.uri.startswith("rtsp"):
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                "rtsp_transport;tcp"
            )

        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(self.uri)

            if not cap.isOpened():
                print("Cannot open camera. Retrying...")
                cap.release()
                time.sleep(3)
                continue

            while not self.stop_event.is_set():
                success, frame = cap.read()

                if not success:
                    print("Cannot read frame. Reconnecting...")
                    break

                with self.lock:
                    self.frame = frame

            cap.release()
            time.sleep(3)

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.stop_event.set()