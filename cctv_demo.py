"""Milestones 1A-1C: open a CCTV (RTSP) stream, detect people/phones/books, track IDs.
  PowerShell:  $env:EXAM_SOURCE="rtsp://user:pass@ip:554/stream"; python cctv_demo.py
Track IDs are local to this camera stream. Press q to quit."""
import time

import cv2
from ultralytics import YOLO

import config as cfg
from sources import FrameSource

CLASSES = {0: "person", 67: "cell phone", 73: "book"}



import cv2
import time
from ultralytics import YOLO

def main():
    # 1. Start the CCTV camera
    camera = FrameSource(cfg.SOURCE, "CCTV_01").start()

    # 2. Load the YOLO model
    model = YOLO("yolov8s.pt")

    sequence = -1

    # 3. Read CCTV frames continuously
    while True:
        frame, timestamp, sequence = camera.read(sequence)

        # Skip if no frame is available
        if frame is None:
            time.sleep(0.01)
            continue

        # 4. Detect and track objects
        result = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=list(CLASSES),
            conf=0.4,
            verbose=False )[0]

        # 5. Draw boxes around detected objects
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            class_id = int(box.cls[0])
            object_name = model.names[class_id]
            confidence = float(box.conf[0])

            # Get tracking ID if available
            track_id = ""
            if result.boxes.id is not None:
                track_id = f" #{int(result.boxes.id[list(result.boxes).index(box)])}"

            label = f"{object_name}{track_id} {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, label, (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        # 6. Show the CCTV video
        cv2.imshow("CCTV", frame)

        # 7. Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 8. Close the camera and windows
    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()