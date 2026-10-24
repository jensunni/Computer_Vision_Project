"""
Hall monitor (CCTV, many students):  YOLO + ByteTrack -> seats -> per-seat head crop
-> head pose -> neighbor-aware rules -> SQLite events + evidence clips.

Steps:
  1. python seat_mapper.py          draw the seats once
  2. python hall_monitor.py         (EXAM_SOURCE = RTSP URL, webcam index, or a video file)
     For a recorded test video the file simply loops.
  Optional: EXAM_RECORD=1 also starts an independent FFmpeg recording (RTSP only).

Detected per seat:  LOOK_NEIGHBOR, HEAD_AWAY, PROHIBITED_OBJ, SEAT_EMPTY
Hall-wide:          UNASSIGNED_PERSON (someone standing outside every seat)
All are flags for human review, never verdicts.
"""
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp
from ultralytics import YOLO

import config as cfg
import events_db as db
import hall_rules as rules
import seats as seatlib
import webcam_proctor as wp                      # reuse head_pose()
from clip_buffer import ClipRecorder
from sources import FrameSource

PERSON = 0
OBJ_NAMES = {67: "cell phone", 73: "book"}


def analyze_head(face_mesh, frame, box):
    """Crop the head/shoulder region of one tracked person and measure head pose.
    Returns (yaw_deg, nose_offset) or None if no face is found."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx1, cx2 = max(0, int(x1 - 0.15 * bw)), min(W, int(x2 + 0.15 * bw))
    cy1, cy2 = max(0, int(y1 - 0.05 * bh)), min(H, int(y1 + cfg.HALL_HEAD_FRACTION * bh))
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.shape[0] < 20 or crop.shape[1] < 20:
        return None
    if crop.shape[1] < cfg.HALL_MIN_CROP:          # small CCTV faces: upscale
        s = cfg.HALL_MIN_CROP / crop.shape[1]
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    ch, cw = crop.shape[:2]
    res = face_mesh.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    lms = res.multi_face_landmarks[0].landmark
    hp = wp.head_pose(lms, cw, ch)
    if hp is None:
        return None
    left, right = sorted((lms[234].x, lms[454].x))        # face edges
    offset = (lms[1].x - (left + right) / 2) / (right - left + 1e-6)   # + = nose toward image right
    return hp[0], offset


def main():
    if not os.path.exists(cfg.SEATS_FILE):
        raise SystemExit("seats.json not found - run: python seat_mapper.py")

    src = FrameSource(cfg.SOURCE, "CCTV_01").start()
    frame, seq = None, -1
    t_wait = time.time()
    while frame is None:
        frame, _, seq = src.read(seq)
        if time.time() - t_wait > 30:
            raise SystemExit("No frames from source")
        time.sleep(0.05)
    H, W = frame.shape[:2]
    seats = seatlib.load_seats(cfg.SEATS_FILE, (W, H))
    states = {s["id"]: rules.SeatState(s) for s in seats}
    print(f"{len(seats)} seats loaded")

    conn = db.connect(cfg.DB_PATH)
    session_id = db.start_session(conn, cfg.EXAM_ID, "cctv_hall")
    rec = None
    rec_path, audio = None, False
    if str(cfg.SOURCE).startswith("rtsp") and os.getenv("EXAM_RECORD") == "1":
        from ffmpeg_recorder import start_recording
        rec, rec_path, _ = start_recording(cfg.SOURCE, cfg.RECORDING_DIR, cfg.MIC_NAME)
        audio = bool(cfg.MIC_NAME)
    source_id = db.add_source(conn, session_id, "cctv", "CCTV_01", rec_path, audio)

    clips = ClipRecorder(cfg.CLIP_DIR, cfg.PRE_SECONDS, cfg.POST_SECONDS, cfg.CLIP_FPS)
    model = YOLO(cfg.HALL_MODEL)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.4)
    gate = rules.Gate()
    classes = [PERSON] + list(OBJ_NAMES)
    t0, last_tick, calibrated = time.time(), 0.0, False

    def raise_alert(name, seat_id, detail, ts, track_id=None):
        snap_img = frame.copy()
        seat = next((s for s in seats if s["id"] == seat_id), None)
        if seat:
            x1, y1, x2, y2 = seat["rect"]
            cv2.rectangle(snap_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(snap_img, f"{seat_id} {name}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        os.makedirs(cfg.SNAP_DIR, exist_ok=True)
        snap = os.path.join(cfg.SNAP_DIR, f"{seat_id or 'HALL'}_{name}_{datetime.fromtimestamp(ts):%Y%m%d_%H%M%S}.jpg")
        cv2.imwrite(snap, snap_img)
        eid = db.add_event(conn, session_id, source_id, name, ts, cfg.PRIORITY.get(name, 10),
                           detail, snap, track_id=track_id, seat_id=seat_id)
        clips.trigger(ts, f"{seat_id or 'HALL'}_{name}",
                      on_saved=lambda path, e=eid: db.set_clip(conn, e, path))
        print(f"[ALERT] event {eid}: {seat_id or 'HALL'} {name} {detail}")

    try:
        while True:
            raw, ts, seq = src.read(seq)
            if raw is None:
                time.sleep(0.005)
                continue
            clips.push(raw, ts)                          # evidence = original frames
            if ts - last_tick < 1.0 / cfg.HALL_ANALYSIS_FPS:
                continue
            last_tick = ts
            frame = raw

            # ---------- detect + track ----------
            r = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=classes,
                            conf=cfg.HALL_CONF, imgsz=cfg.HALL_IMGSZ, verbose=False)[0]
            ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
            persons, objects = [], []
            for b, tid in zip(r.boxes, ids):
                cls, conf = int(b.cls[0]), float(b.conf[0])
                box = tuple(map(int, b.xyxy[0]))
                if cls == PERSON:
                    persons.append({"box": box, "id": tid, "conf": conf})
                elif conf >= cfg.HALL_OBJ_CONF:
                    objects.append({"label": OBJ_NAMES[cls], "box": box, "conf": conf})

            by_seat, unassigned = seatlib.assign_persons(seats, persons)
            obj_by_seat = seatlib.assign_objects(seats, objects)

            # ---------- per-seat measurements ----------
            for seat in seats:
                st = states[seat["id"]]
                st.total_ticks += 1
                person = by_seat.get(seat["id"])
                if person:
                    st.occupied_ticks += 1
                    st.last_seen, st.track_id, st.person_box = ts, person["id"], person["box"]
                    if ts - st.last_face_try >= cfg.HALL_FACE_INTERVAL:
                        st.last_face_try = ts
                        res = analyze_head(face_mesh, frame, person["box"])
                        if res:
                            st.yaw, st.off, st.face_ts = res[0], res[1], ts
                            if not calibrated:
                                st.samples.append(res)
                else:
                    st.person_box = None

            # ---------- calibration ----------
            if not calibrated:
                left = cfg.HALL_CALIBRATION_SECONDS - (time.time() - t0)
                cv2.putText(frame, f"Calibrating seats... {max(0, left):.0f}s (students look forward)",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                if left <= 0:
                    rules.finish_calibration(states)
                    calibrated = True
                    occ = [sid for sid, s in states.items() if s.expected]
                    print(f"Calibrated. Occupied seats ({len(occ)}): {occ}")
            else:
                # ---------- rules ----------
                for seat in seats:
                    sid = seat["id"]
                    st = states[sid]
                    person = by_seat.get(sid)
                    st.active = set()

                    name, detail = rules.head_event(st, seats, states, ts)
                    for n in ("LOOK_NEIGHBOR", "HEAD_AWAY"):
                        active = name == n
                        if active:
                            st.active.add(n)
                        if gate.update(f"{sid}:{n}", active, ts, cfg.HALL_DURATION[n], cfg.HALL_COOLDOWN):
                            raise_alert(n, sid, detail, ts, st.track_id)

                    objs = obj_by_seat.get(sid, [])
                    if objs:
                        st.active.add("PROHIBITED_OBJ")
                    if gate.update(f"{sid}:OBJ", bool(objs), ts, cfg.HALL_DURATION["PROHIBITED_OBJ"], cfg.HALL_COOLDOWN):
                        raise_alert("PROHIBITED_OBJ", sid, ",".join(sorted({o["label"] for o in objs})), ts, st.track_id)

                    empty = st.expected and person is None
                    if empty:
                        st.active.add("SEAT_EMPTY")
                    if gate.update(f"{sid}:EMPTY", empty, ts, cfg.HALL_DURATION["SEAT_EMPTY"], cfg.HALL_COOLDOWN * 4):
                        raise_alert("SEAT_EMPTY", sid, f"empty since {datetime.fromtimestamp(st.last_seen):%H:%M:%S}", ts)

                if gate.update("HALL:UNASSIGNED", len(unassigned) > 0, ts,
                               cfg.HALL_DURATION["UNASSIGNED_PERSON"], cfg.HALL_COOLDOWN * 2):
                    raise_alert("UNASSIGNED_PERSON", None,
                                "track ids: " + ",".join(str(p["id"]) for p in unassigned), ts)

            # ---------- overlay ----------
            for p in persons:
                x1, y1, x2, y2 = p["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 160, 0), 1)
                if p["id"] is not None:
                    cv2.putText(frame, f"#{p['id']}", (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 160, 0), 1)
            for o in objects:
                x1, y1, x2, y2 = o["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, o["label"], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            for seat in seats:
                st = states[seat["id"]]
                x1, y1, x2, y2 = seat["rect"]
                color = (0, 0, 255) if st.active else ((0, 200, 0) if (st.expected or not calibrated) else (120, 120, 120))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                tag = seat["id"]
                if st.active:
                    tag += " " + "/".join(sorted(st.active))
                cv2.putText(frame, tag, (x1 + 3, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            scale = min(1.0, 1280 / frame.shape[1])
            cv2.imshow("Hall monitor", cv2.resize(frame, None, fx=scale, fy=scale))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        src.stop()
        clips.flush()
        if rec is not None:
            from ffmpeg_recorder import stop_recording
            stop_recording(rec)
        db.end_session(conn, session_id)
        cv2.destroyAllWindows()
        print("Session saved:", session_id)


if __name__ == "__main__":
    main()