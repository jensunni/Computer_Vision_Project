# AI-Powered Exam Invigilator

A computer-vision system that watches a webcam or CCTV feed during an exam, **flags suspicious moments for a human reviewer**, and saves a snapshot and a short video clip as evidence for every flag.

> **The system never decides that someone is cheating.** Every alert is a prompt for human review. Looking away, glancing at a neighbor, or an unusual object can all have innocent explanations.

**Stack:** Python · OpenCV · MediaPipe · YOLO (Ultralytics) · SQLite · Streamlit · FFmpeg

---

## Screenshots

<!-- Add 1-2 screenshots to a "screenshots" folder, then delete the comment markers (the opening and closing arrows) and keep the image lines:
![Review dashboard](screenshots/dashboard.png)
![Webcam alerts](screenshots/webcam_alerts.png)
-->

---

## Quick start

```powershell
git clone https://github.com/jensunni/Computer_Vision_Project.git
cd Computer_Vision_Project
python -m venv exam_env
exam_env\Scripts\activate
python -m pip install -r requirements.txt
python main.py                 # webcam pipeline
streamlit run review_app.py    # review dashboard (in a second terminal)
```

Details for every mode are in [Installation](#installation) and [Usage](#usage).

---

## Features

| Mode | Alert | How it is detected |
|---|---|---|
| Webcam | `NO_FACE` | No face visible for 3 s |
| Webcam | `MULTIPLE_FACES` | More than one face for 2 s |
| Webcam | `HEAD_AWAY` | Head yaw/pitch (OpenCV `solvePnP`) beyond a limit from the student's calibrated baseline |
| Webcam | `GAZE_AWAY` | Iris position off-center while the head is frontal |
| Webcam | `TALKING` | Repeated lip opening and closing within a 3 s window |
| Webcam / Hall | `PROHIBITED_OBJ` | YOLO detects a phone or book |
| Hall (CCTV) | `LOOK_NEIGHBOR` | Head turned toward an occupied adjacent seat |
| Hall (CCTV) | `HEAD_AWAY` | Head turned toward a side with no occupied neighbor |
| Hall (CCTV) | `SEAT_EMPTY` | A seat occupied at calibration is empty for 20 s |
| Hall (CCTV) | `UNASSIGNED_PERSON` | Someone standing outside every seat |

Common to both modes:
- **Duration rules and cooldowns:** a condition must persist for N seconds before it alerts, and the same alert cannot repeat for a cooldown period.
- **Calibration:** head-pose limits are measured relative to each student's (or seat's) normal pose.
- **Evidence clips:** 5 s before and 10 s after each alert, saved from the original frames with no overlays.
- **SQLite event log** with a review-priority score, which is a sorting aid and not a probability of cheating.
- **Streamlit dashboard** to watch the clip, view the snapshot, and record a decision.

---

## Project structure

```
ai-exam-invigilator/
├── config.py             # all settings and thresholds
├── sources.py            # threaded webcam / RTSP reader with auto-reconnect
├── events_db.py          # SQLite: sessions, recording sources, events
├── clip_buffer.py        # rolling buffer -> pre/post-event evidence clips
├── ffmpeg_recorder.py    # independent CCTV + microphone recording (FFmpeg)
├── webcam_proctor.py     # standalone webcam prototype + detection helpers
├── main.py               # webcam pipeline with database and clips
├── cctv_demo.py          # RTSP stream + YOLO + ByteTrack demo
├── seats.py              # seat layout, neighbors, person/object -> seat assignment
├── seat_mapper.py        # draw seat rectangles once on the CCTV view
├── hall_rules.py         # hall rule logic (no camera dependencies, unit-testable)
├── hall_monitor.py       # CCTV hall pipeline
├── review_app.py         # Streamlit review dashboard
└── requirements.txt
```

Runtime folders are created automatically: `clips/`, `snapshots/`, `recordings/`, `alerts/` (standalone prototype), plus `exam_events.db`, `seats.json`, and `alerts_log.csv`.

---

## Pipeline

```
Webcam / CCTV (RTSP / video file)
        ↓
Threaded frame reader (always the latest frame, auto-reconnect)
        ↓
Detection:  YOLO (person, phone, book)  +  MediaPipe Face Mesh
        ↓
Hall mode only: ByteTrack IDs → seat assignment (R1C1, R1C2, ...)
        ↓
Rules: head pose vs. baseline, objects, empty seat, extra faces
        ↓
Alert gate: must last N seconds, then cooldown
        ↓
SQLite event + snapshot + evidence clip
        ↓
Streamlit dashboard → human decision
```

Recording is kept separate from AI analysis: the FFmpeg recorder runs as its own process, so a slow or crashed model does not stop the recording.

---

## Installation

**Requirements:** Python 3.9 to 3.12 (MediaPipe 0.10.14 does not support 3.13), a webcam or an RTSP-capable IP camera, and optionally FFmpeg on your `PATH` (needed for browser-playable clips and CCTV audio recording).

```powershell
python -m venv exam_env
exam_env\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the install:

```powershell
python -c "import cv2, mediapipe, ultralytics; print('OK', cv2.__version__)"
ffmpeg -version
```

The first run downloads the YOLO weights (`yolov8n.pt` or `yolov8s.pt`) automatically, so it needs internet access once. The weights are not stored in this repository.

---

## Usage

Run everything as scripts from a terminal, not from Jupyter. `cv2.imshow` often freezes notebook kernels.

### 1. Webcam prototype (single file, no database)

```powershell
python webcam_proctor.py
```

Look at the screen for 3 seconds while it calibrates. Alerts go to `alerts_log.csv` and `alerts/`. Press `q` to quit.

### 2. Webcam pipeline with database and clips

```powershell
python main.py
```

### 3. Review dashboard

```powershell
streamlit run review_app.py
```

Filter by status or seat, play the evidence clip, and save a decision (`false_alarm`, `needs_follow_up`, or `confirmed_violation`) with a note.

### 4. CCTV stream test

```powershell
$env:EXAM_SOURCE="rtsp://USER:PASSWORD@CAMERA_IP:554/STREAM_PATH"
python cctv_demo.py
```

Shows people, phones, and books with stream-local track IDs. Track IDs from one camera are not the same as IDs from another camera.

### 5. CCTV hall monitoring

```powershell
python seat_mapper.py        # draw one box per seat, then press s to save seats.json
python hall_monitor.py       # 10 s calibration, then monitoring
```

Seat mapper keys: drag = draw, `u` = undo, `c` = clear, `s` = save, `q` = quit.

Tips for drawing seats:
- Draw each box around the student's upper body and desk area.
- Boxes may touch but should not overlap much.
- Redo the mapping if the camera moves or the seating layout changes.

To test without being in a hall, point `EXAM_SOURCE` at a recorded video file. It loops automatically.

### Optional: independent FFmpeg recording (RTSP only)

```powershell
ffmpeg -list_devices true -f dshow -i dummy        # find your microphone name
$env:EXAM_MIC="Microphone (USB Audio Device)"
$env:EXAM_RECORD="1"
python hall_monitor.py
```

The continuous recording is saved to `recordings/` as `.mkv`. That format stays readable while it is still being written and survives crashes. A USB webcam cannot be opened by OpenCV and FFmpeg at the same time on Windows, so for a webcam use the clip buffer instead, which saves video without audio.

---

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `EXAM_SOURCE` | `0` for the first webcam, an RTSP URL, or a video file path | `0` |
| `EXAM_ID` | Exam identifier stored with each session | `EXAM_001` |
| `EXAM_MIC` | DirectShow microphone name for FFmpeg recording | unset |
| `EXAM_RECORD` | Set to `1` to record CCTV with FFmpeg in `hall_monitor.py` | unset |

Keep camera usernames and passwords in environment variables. Do not put them in code, notebooks, screenshots, or logs.

---

## Configuration

Settings live in `config.py`. Webcam thresholds live at the top of `webcam_proctor.py`.

| Setting | Meaning |
|---|---|
| `PRE_SECONDS`, `POST_SECONDS` | Evidence before and after an event |
| `ANALYSIS_FPS`, `HALL_ANALYSIS_FPS` | How often the AI samples frames |
| `HALL_MODEL`, `HALL_IMGSZ` | YOLO model and input size. Larger finds smaller people but is slower. |
| `HALL_YAW_LIMIT`, `HALL_OFFSET_LIMIT` | How far a head must turn before it counts |
| `HALL_DURATION`, `HALL_COOLDOWN` | Seconds a condition must last, and the repeat delay |
| `PRIORITY` | Review-priority weights, which only order the review queue |

Tune thresholds on your own test videos with different people, glasses, lighting, and seating.

---

## Database

SQLite file `exam_events.db`:

- `exam_sessions`: `session_id`, `exam_id`, `start_time`, `end_time`, `mode`, `recording_status`
- `recording_sources`: `source_id`, `session_id`, `source_type`, `source_name`, `start_time`, `end_time`, `file_path`, `audio_available`
- `events`: `event_id`, `session_id`, `source_id`, `track_id`, `seat_id`, `event_type`, `event_time`, `review_priority`, `detail`, `snapshot_path`, `clip_path`, `review_status`, `reviewer_decision`, `reviewer_note`

All writes use parameterized queries. Databases created by the earlier version get the `seat_id` column added automatically.

Example queries (open `exam_events.db` with any SQLite browser or the `sqlite3` CLI):

```sql
-- Alerts per type in one session
SELECT event_type, COUNT(*) AS alerts
FROM events
WHERE session_id = 1
GROUP BY event_type
ORDER BY alerts DESC;

-- Which seats raise the most alerts, and their average review priority
SELECT seat_id, COUNT(*) AS alerts, ROUND(AVG(review_priority), 1) AS avg_priority
FROM events
WHERE seat_id IS NOT NULL
GROUP BY seat_id
ORDER BY alerts DESC;
```

---

## Files not stored in Git

Add a `.gitignore` with these entries so generated and large files stay out of the repository:

```
.vscode/
__pycache__/
exam_env/
*.pt
clips/
snapshots/
recordings/
alerts/
exam_events.db
seats.json
alerts_log.csv
```

---

## Known limitations

- **Small faces on CCTV.** Head pose is unreliable when faces are small, turned away, or occluded. The camera should see students' faces, so front corners usually work better than the back of the room.
- **Hall rules assume a seat grid.** Neighbor logic only handles left and right neighbors in the same row, not front and back.
- **Direction is measured in the image.** "Left" and "right" mean image left and right.
- **Standard YOLO classes.** `yolov8` trained on COCO has no earphone class, and it can mislabel a notebook as a phone. Fine-tune it on your own exam-room images for better results.
- **Clip sync.** `extract_clip()` in `ffmpeg_recorder.py` estimates offsets from the wall clock. Verify audio and video alignment with a clap test before relying on it.
- **Lip-based talking detection** works for a webcam but not at CCTV distance, so it is not in hall mode.
- **Not tested end to end.** The database and clip recorder were checked with synthetic frames, and all files compile. Run the webcam, CCTV, and hall modes on your own hardware before depending on them.
- **No authentication on the dashboard.** Add access control before any real deployment.

---

## Privacy, fairness, and responsible use

- Get authorization and tell participants they are being recorded.
- Limit who can view recordings, and encrypt stored footage.
- Define retention and deletion rules before use. In India, check obligations under the DPDP Act 2023.
- Test across skin tones, glasses, head coverings, seating positions, and lighting. Face and pose models can perform unevenly.
- Never penalize automatically. Keep the human review step, document why an event was flagged, and allow false alerts to be dismissed or corrected.
- Do not add voice identification or transcription unless it is clearly justified and authorized.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | `python -m pip install opencv-python`, using the same environment as your Python |
| MediaPipe install fails or `mp.solutions` is missing | Use Python 3.9 to 3.12 and `mediapipe==0.10.14` |
| Window freezes in Jupyter | Run the scripts from a terminal |
| RTSP stream does not open | Check the URL and path, test it in VLC, and use a wired connection when possible |
| Clips do not play in the dashboard | Install FFmpeg so clips are re-encoded to H.264 |
| Notebook detected as "cell phone" | Use `yolov8s.pt` or larger, improve lighting, or fine-tune on your own images |
| Too many head-turn alerts | Raise `HALL_YAW_LIMIT` or `HALL_OFFSET_LIMIT`, or increase the durations |
| `seats.json not found` | Run `python seat_mapper.py` first |

---

## Roadmap

1. Test and tune hall mode on real recorded exam footage.
2. Fine-tune YOLO for phones, notebooks, loose papers, and earphones.
3. Add an LSTM or Transformer over pose and head-angle sequences to replace fixed rules.
4. Add dashboard login and role-based access.
5. Run multiple streams (CCTV plus webcams) with per-source health monitoring.
6. Add per-behavior evaluation (precision and recall) on labeled mock exams.

---

## Author

**Jensunni Shaji** · [GitHub](https://github.com/jensunni) · [LinkedIn](https://linkedin.com/in/jensunni-shaji-b24185241) · jensunnixyz@gmail.com
