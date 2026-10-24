"""
Proctor v2 - your webcam_proctor.py logic + sessions, SQLite events and evidence clips.

  Webcam:   python main.py
  CCTV:     set EXAM_SOURCE=rtsp://user:pass@ip:554/stream   (PowerShell: $env:EXAM_SOURCE="...")
            python main.py

The face/gaze/talk rules are for ONE student per camera (webcam). For a hall with
many students use cctv_demo.py (person tracking) and per-seat crops.
"""

import time
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

import config as cfg
from webcam_proctor import Web_Detection as wd
from sources import FrameSource


def get_face_data(frame, face_mesh):
    height, width = frame.shape[:2]

    result = face_mesh.process(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    faces = result.multi_face_landmarks or []

    yaw = pitch = gaze = None

    if faces:
        landmarks = faces[0].landmark
        pose = wd.head_pose(landmarks, width, height)

        if pose:
            yaw, pitch = pose

        gaze, mouth = wd.gaze_ratio(landmarks, width), wd.mouth_ratio(
            landmarks, height
        )
    else:
        mouth = None

    return faces, yaw, pitch, gaze, mouth


def main():
    is_webcam = str(cfg.SOURCE).isdigit()

    camera = FrameSource(cfg.SOURCE, "CAM_01").start()

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=3,
        refine_landmarks=True
    )

    model = YOLO(wd.YOLO_MODEL)

    calibration = []
    calibrated = False
    start_time = time.time()
    sequence = -1
    last_analysis = 0

    base_yaw = base_pitch = 0

    try:
        while True:
            frame, timestamp, sequence = camera.read(sequence)

            if frame is None:
                time.sleep(0.01)
                continue

            if timestamp - last_analysis < 1 / cfg.ANALYSIS_FPS:
                continue

            last_analysis = timestamp

            if is_webcam:
                frame = cv2.flip(frame, 1)

            faces, yaw, pitch, gaze, mouth = get_face_data(
                frame, face_mesh
            )

            # Learn the student's normal head position
            if not calibrated:
                if yaw is not None and len(faces) == 1:
                    calibration.append((yaw, pitch))

                elapsed = time.time() - start_time

                if elapsed >= wd.CALIBRATION_SECONDS and len(calibration) > 10:
                    base_yaw = np.median([item[0] for item in calibration])
                    base_pitch = np.median([item[1] for item in calibration])
                    calibrated = True
                else:
                    cv2.putText(
                        frame, "Calibrating...",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2
                    )

                    cv2.imshow("Proctor", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                    continue

            # Detect prohibited objects
            result = model(
                frame,
                conf=wd.YOLO_CONF,
                classes=list(wd.YOLO_CLASSES),
                verbose=False
            )[0]

            objects_found = len(result.boxes) > 0

            # Check head movement
            head_away = False

            if yaw is not None and pitch is not None:
                head_away = (
                    abs(yaw - base_yaw) > wd.YAW_LIMIT
                    or abs(pitch - base_pitch) > wd.PITCH_LIMIT
                )

            # Check gaze direction
            gaze_away = (
                gaze is not None
                and not head_away
                and (
                    gaze < wd.GAZE_LOW
                    or gaze > wd.GAZE_HIGH
                )
            )

            # Display monitoring results
            alerts = {
                "NO_FACE": len(faces) == 0,
                "MULTIPLE_FACES": len(faces) > 1,
                "HEAD_AWAY": head_away,
                "GAZE_AWAY": gaze_away,
                "PROHIBITED_OBJ": objects_found
            }

            for name, active in alerts.items():
                print(f"{name}: {active}")

            cv2.imshow("Proctor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.stop()
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()