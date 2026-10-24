"""Central settings. Secrets (RTSP URL, mic name) come from environment variables."""
import os

# "0" = first webcam, or a full RTSP URL:  rtsp://user:pass@192.168.1.20:554/stream1
SOURCE = os.getenv("EXAM_SOURCE", "0")
EXAM_ID = os.getenv("EXAM_ID", "EXAM_001")
MIC_NAME = os.getenv("EXAM_MIC")          # e.g. "Microphone (USB Audio Device)" (Windows)

DB_PATH = "exam_events.db"
CLIP_DIR = "clips"
SNAP_DIR = "snapshots"
RECORDING_DIR = "recordings"

PRE_SECONDS = 5          # evidence before the event
POST_SECONDS = 10        # evidence after the event
CLIP_FPS = 15            # fps stored in evidence clips
ANALYSIS_FPS = 10        # AI sampling rate (plan suggests 5-10)

# Review-priority weights: a sorting aid for the reviewer, NOT a cheating probability
PRIORITY = {
    "NO_FACE": 40, "MULTIPLE_FACES": 70, "HEAD_AWAY": 30,
    "GAZE_AWAY": 20, "TALKING": 35, "PROHIBITED_OBJ": 60,
}