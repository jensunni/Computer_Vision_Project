"""
Webcam Exam Proctor - malpractice detection (webcam only)

Detects:
    NO_FACE         - student not visible
    MULTIPLE_FACES  - more than one person in frame
    HEAD_AWAY       - head turned / tilted away from screen
    GAZE_AWAY       - eyes looking sideways while head is frontal
    TALKING         - sustained mouth movement
    PROHIBITED_OBJ  - phone / book detected (YOLO)

Every alert is written to alerts_log.csv and a snapshot is saved to alerts/.
Alerts are FLAGS FOR HUMAN REVIEW, not automatic verdicts.

Run:    python webcam_proctor.py
Keys:   q = quit
"""
# --------------IMPORTING LIBRARIES--------------------------------------
import csv # To store Malpratice details
import os # For System Operations
import time # for time series data
from collections import deque # add and remove elements from lists ends extremely quickly.
from datetime import datetime # for date operations

import cv2 # OpenCV
import mediapipe as mp # detect postures and guestors
import numpy as np
from ultralytics import YOLO # real-time object detection system

class Web_Detection:
    # ----------------CONFIGARATION-------------------------------------------
    CAMERA_INDEX = 0
    CALIBRATION_SECONDS = 3          # student looks straight at screen at start

    # thresholds (degrees / ratios / seconds) - tune on your own test videos
    YAW_LIMIT = 30                   # left/right head turn
    PITCH_LIMIT = 25                 # up/down head tilt
    GAZE_LOW, GAZE_HIGH = 0.35, 0.65 # iris ratio; 0.5 = centered
    MOUTH_OPEN_RATIO = 0.03          # lip gap / face height
    TALK_WINDOW_SEC = 3
    TALK_OPEN_FRACTION = 0.4         # share of window with mouth open

    DURATION = {                     # seconds a condition must persist to alert
        "NO_FACE": 3.0,
        "MULTIPLE_FACES": 2.0,
        "HEAD_AWAY": 3.0,
        "GAZE_AWAY": 4.0,
        "TALKING": 1.0,
        "PROHIBITED_OBJ": 1.5,
    }
    ALERT_COOLDOWN = 10.0            # seconds before the same alert can re-fire

    YOLO_MODEL = "yolov8n.pt"        # swap with your fine-tuned weights later
    YOLO_CLASSES = {67: "Moble_Phone", 73: "Book"}   # COCO ids
    YOLO_CONF = 0.45
    YOLO_EVERY_N_FRAMES = 5          # run detector every N frames for speed

    LOG_FILE = "alerts_log.csv"
    SNAP_DIR = "alerts"
    # -------------------------------------------------------------------------

    # 3D reference face model (generic) for solvePnP
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),            # nose tip
        (0.0, -63.6, -12.5),        # chin
        (-43.3, 32.7, -26.0),       # eye outer corner (subject's right)
        (43.3, 32.7, -26.0),        # eye outer corner (subject's left)
        (-28.9, -28.9, -24.1),      # mouth corner (subject's right)
        (28.9, -28.9, -24.1),       # mouth corner (subject's left)
    ], dtype=np.float64)

    PNP_IDX = [1, 152, 33, 263, 61, 291]   # FaceMesh landmark ids
    #------------------------------------------------------------------------
    @staticmethod
    def head_pose(lms, w, h):
        # Step 1: Get the facial landmark coordinates
        points = np.array([(lms[i].x * w, lms[i].y * h) for i in Web_Detection.PNP_IDX ], dtype=np.float64)

        # Step 2: Create the camera matrix
        focal_length = w

        # Here focal_length reprecent lens focal_length and w represent image width
        # Currently they are same value but in real time use we can change focal_length
        # without affecting the width(detected object) based on locations constraints

        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1] ], dtype=np.float64)
        
        # Step 3: Estimate the head's rotation
        success, rotation_vector, _ = cv2.solvePnP(
            Web_Detection.MODEL_POINTS,
            points,
            camera_matrix,
            np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE )

        # Step 4: Check whether the calculation succeeded
        if not success:
            return None

        # Step 5: Convert rotation vector into a rotation matrix
        rotation_matrix = cv2.Rodrigues(rotation_vector)[0]

        # Step 6: Extract the rotation angles
        angles = cv2.RQDecomp3x3(rotation_matrix)[0]

        pitch, yaw, roll = angles

        # Step 7: Correct the pitch angle
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180

        # Step 8: Return yaw and pitch
        return yaw, pitch
    
    @staticmethod
    def gaze_ratio(lms, w):
        # Calculate iris position inside one eye
        def calculate_ratio(iris, eye_start, eye_end):

            iris_x = lms[iris].x * w
            start_x = lms[eye_start].x * w
            end_x = lms[eye_end].x * w

            ratio = (iris_x - start_x) / (end_x - start_x + 1e-6)

            return ratio

        # Calculate the ratio for both eyes
        right_eye = calculate_ratio(468, 33, 133)
        left_eye = calculate_ratio(473, 362, 263)

        # Find the average ratio
        average_gaze = (right_eye + left_eye) / 2

        return average_gaze
    
    @staticmethod
    def mouth_ratio(lms, h):

        # Step 1: Calculate the distance between the lips
        mouth_gap = abs(lms[14].y - lms[13].y) * h

        # Step 2: Calculate the height of the face
        face_height = abs(lms[152].y - lms[10].y) * h

        # Step 3: Calculate mouth opening relative to face height
        ratio = mouth_gap / (face_height + 1e-6)

        return ratio
    
    @staticmethod
    def log_alert(name, detail, frame):
        
        # Step 1: Create the folder to store alert images
        os.makedirs(Web_Detection.SNAP_DIR, exist_ok=True)

        # Step 2: Get the current date and time
        ts = datetime.now()

        # Step 3: Create a filename for the snapshot
        snap = os.path.join( Web_Detection.SNAP_DIR, f"{name}_{ts:%Y%m%d_%H%M%S}.jpg")

        # Step 4: Save the webcam image
        cv2.imwrite(snap, frame)

        # Step 5: Check whether the CSV file already exists
        new = not os.path.exists(Web_Detection.LOG_FILE)

        # Step 6: Open the CSV file in append mode
        with open(Web_Detection.LOG_FILE, "a", newline="") as f:

            writer = csv.writer(f)

            # Add column headings if the file is new
            if new:
                writer.writerow(["timestamp", "alert", "detail", "snapshot"])

            # Save the alert information as a new row
            writer.writerow([
                ts.isoformat(timespec="seconds"),
                name,
                detail,
                snap
            ])

        # Step 7: Display the alert in the terminal
        print(f"[ALERT] {ts:%H:%M:%S} {name} {detail}")

    def main(self):
        # ------Connecting to Webcam---------------
        cap = cv2.VideoCapture(self.CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=3, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        yolo = YOLO(self.YOLO_MODEL)
        tracker = AlertTracker() 

        mouth_hist = deque()            # (time, is_open)
        detections = []                 # latest YOLO boxes
        frame_i = 0
        calib = []                      # (yaw, pitch) samples
        base_yaw = base_pitch = 0.0
        t0 = time.time()
        calibrated = False
        recent_alerts = deque(maxlen=4)  # on-screen messages

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()
            frame_i += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            faces = res.multi_face_landmarks or []
            n_faces = len(faces)

            yaw = pitch = gaze = mouth = None
            if n_faces >= 1:
                lms = faces[0].landmark
                hp = self.head_pose(lms, w, h)
                if hp:
                    yaw, pitch = hp
                gaze = self.gaze_ratio(lms, w)
                mouth = self.mouth_ratio(lms, h)

            # ---------------- calibration phase ----------------
            if not calibrated:
                if yaw is not None and n_faces == 1:
                    calib.append((yaw, pitch))
                elapsed = now - t0
                cv2.putText(frame, f"Calibrating... look at the screen ({elapsed:.0f}/{self.CALIBRATION_SECONDS}s)",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if elapsed >= self.CALIBRATION_SECONDS and len(calib) > 10:
                    base_yaw = float(np.median([c[0] for c in calib]))
                    base_pitch = float(np.median([c[1] for c in calib]))
                    calibrated = True
                    print(f"Calibrated: yaw={base_yaw:.1f}, pitch={base_pitch:.1f}")
                cv2.imshow("Proctor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # ---------------- object detection ----------------(Moble And Book Haves Some Issues Here )
            if frame_i % self.YOLO_EVERY_N_FRAMES == 0:
                detections = []
                r = yolo(frame, conf=self.YOLO_CONF, classes=list(self.YOLO_CLASSES), verbose=False)[0]
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    detections.append((self.YOLO_CLASSES[int(b.cls[0])], float(b.conf[0]), (x1, y1, x2, y2)))
            for label, conf, (x1, y1, x2, y2) in detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # ---------------- rule conditions ----------------
            head_away = False
            if yaw is not None:
                head_away = (abs(yaw - base_yaw) > self.YAW_LIMIT) or (abs(pitch - base_pitch) > self.PITCH_LIMIT)

            gaze_away = (gaze is not None and not head_away
                        and (gaze < self.GAZE_LOW or gaze > self.GAZE_HIGH))

            if mouth is not None:
                mouth_hist.append((now, mouth > self.MOUTH_OPEN_RATIO))
            while mouth_hist and now - mouth_hist[0][0] > self.TALK_WINDOW_SEC:
                mouth_hist.popleft()
            talking = False
            if len(mouth_hist) > 15:
                frac = sum(o for _, o in mouth_hist) / len(mouth_hist)
                # require both open and closed states (movement), not a held-open mouth
                talking = self.TALK_OPEN_FRACTION < frac < 0.9

            conditions = {
                "NO_FACE": n_faces == 0,
                "MULTIPLE_FACES": n_faces > 1,
                "HEAD_AWAY": head_away,
                "GAZE_AWAY": gaze_away,
                "TALKING": talking,
                "PROHIBITED_OBJ": len(detections) > 0,
            }

            for name, active in conditions.items():
                if tracker.update(name, active, now):
                    detail = ""
                    if name == "PROHIBITED_OBJ":
                        detail = ",".join(d[0] for d in detections)
                    elif name == "HEAD_AWAY":
                        detail = f"yaw={yaw - base_yaw:.0f} pitch={pitch - base_pitch:.0f}"
                    elif name == "GAZE_AWAY":
                        detail = f"gaze={gaze:.2f}"
                    self.log_alert(name, detail, frame.copy())
                    recent_alerts.append(f"{datetime.now():%H:%M:%S} {name}")

            # ---------------- overlay ----------------
            y = 25
            for name, active in conditions.items():
                color = (0, 0, 255) if active else (0, 200, 0)
                cv2.putText(frame, f"{name}: {'YES' if active else 'ok'}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                y += 22
            if yaw is not None:
                cv2.putText(frame, f"yaw {yaw - base_yaw:+.0f}  pitch {pitch - base_pitch:+.0f}  gaze {gaze:.2f}",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            for i, msg in enumerate(recent_alerts):
                cv2.putText(frame, msg, (w - 280, 25 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            cv2.imshow("Proctor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

class AlertTracker:

    def __init__(self):
        # Store when each condition first became active
        self.start = {}

        # Create a web_dectection object
        self.obj=Web_Detection()

        # Store when each alert was last triggered
        self.last_fired = {}

    def update(self, name, active, now):

        # Step 1: If the condition is inactive, reset its timer
        if not active:
            self.start.pop(name, None)
            return False

        # Step 2: Record the starting time if not already recorded
        if name not in self.start:
            self.start[name] = now

        # Step 3: Check whether the condition lasted long enough
        persisted = now - self.start[name] >= self.obj.DURATION[name]

        # Step 4: Check whether the cooldown period has passed
        last_alert = self.last_fired.get(name, -1e9) # -1e9 = -10^9
        cooled = now - last_alert >= self.obj.ALERT_COOLDOWN

        # Step 5: Trigger the alert if both conditions are satisfied
        if persisted and cooled:
            self.last_fired[name] = now
            return True

        # Otherwise, do not trigger an alert
        return False


if __name__ == "__main__":
    item=Web_Detection()
    item.main()