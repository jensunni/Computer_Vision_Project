"""Draw one rectangle around each seated student on the CCTV view (once per room layout).

  python seat_mapper.py                 uses EXAM_SOURCE (RTSP / webcam / video file)
  python seat_mapper.py hall_photo.jpg  uses a saved image

Mouse: click-drag = draw seat    u = undo    c = clear    s = save seats.json    q = quit
Draw each box around the student's upper body AND the desk area, so the box covers
where that student normally sits. Boxes may touch but should not overlap much.
1. Capture an image
CCTV, webcam, video source, or saved photo

2. Draw seat rectangles
Click and drag with the mouse

3. Label and preview seats
R1C1, R1C2, R2C1, etc.

4. Save to seats.json
The seat layout can be reused by your AI system"""

import sys # Interact directly with the Python interpreter and the environment in which your script is running
import time
import cv2

import config as cfg
import seats as seatlib
from sources import FrameSource


# Step 1: Get an image from a file or camera
def grab_frame():
    if len(sys.argv) > 1:
        frame = cv2.imread(sys.argv[1])

        if frame is None:
            raise SystemExit("Cannot read image")

        return frame

    camera = FrameSource(cfg.SOURCE, "MAP").start()
    start_time = time.time()

    while time.time() - start_time < 30:
        frame, timestamp, sequence = camera.read()

        if frame is not None:
            camera.stop()
            return frame

        time.sleep(0.05)

    camera.stop()
    raise SystemExit("No camera frame received")


# Step 2: Draw and save seats
def main():
    frame = grab_frame()
    height, width = frame.shape[:2]

    scale = min(1.0, 1280 / width)
    rectangles = []

    start_point = None
    end_point = None

    # Handle mouse actions
    def draw_seat(event, x, y, flags, param):
        nonlocal start_point, end_point

        x = int(x / scale)
        y = int(y / scale)

        if event == cv2.EVENT_LBUTTONDOWN:
            start_point = (x, y)
            end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and start_point is not None:
            end_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and start_point is not None:
            x1, y1 = start_point

            left = min(x1, x)
            top = min(y1, y)
            right = max(x1, x)
            bottom = max(y1, y)

            if right - left > 20 and bottom - top > 20:
                rectangles.append((left, top, right, bottom))

            start_point = None
            end_point = None

    cv2.namedWindow("Seat Mapper")
    cv2.setMouseCallback("Seat Mapper", draw_seat)

    while True:
        view = frame.copy()

        # Draw saved seat rectangles
        seats = seatlib.label_seats(rectangles)

        for seat in seats:
            x1, y1, x2, y2 = seat["rect"]

            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                view, seat["id"], (x1 + 4, y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        # Draw the rectangle while dragging
        if start_point is not None and end_point is not None:
            cv2.rectangle(
                view, start_point, end_point, (0, 255, 255), 2
            )

        # Show instructions
        message = f"Seats: {len(rectangles)} | u=undo c=clear s=save q=quit"

        cv2.putText(
            view, message, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

        cv2.imshow(
            "Seat Mapper",
            cv2.resize(view, None, fx=scale, fy=scale)
        )

        key = cv2.waitKey(30) & 0xFF

        if key == ord("u") and rectangles:
            rectangles.pop()

        elif key == ord("c"):
            rectangles.clear()

        elif key == ord("s") and rectangles:
            seats = seatlib.save_seats(
                rectangles, (width, height), cfg.SEATS_FILE
            )
            print("Seats saved:", [seat["id"] for seat in seats])
            break

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()