"""SQLite storage: sessions, recording sources, events (parameterized queries only)."""
import sqlite3
import threading
from datetime import datetime

_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS exam_sessions(
    session_id TEXT PRIMARY KEY, exam_id TEXT, start_time TEXT, end_time TEXT,
    mode TEXT, recording_status TEXT);
CREATE TABLE IF NOT EXISTS recording_sources(
    source_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, source_type TEXT,
    source_name TEXT, start_time TEXT, end_time TEXT, file_path TEXT, audio_available INTEGER);
CREATE TABLE IF NOT EXISTS events(
    event_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, source_id INTEGER,
    track_id INTEGER, event_type TEXT, event_time TEXT, review_priority INTEGER,
    detail TEXT, snapshot_path TEXT, clip_path TEXT,
    review_status TEXT DEFAULT 'pending', reviewer_decision TEXT, reviewer_note TEXT);
"""


def _now():
    return datetime.now().isoformat(timespec="seconds")


def connect(path):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.executescript(SCHEMA)
    return conn


def start_session(conn, exam_id, mode):
    sid = f"{exam_id}_{datetime.now():%Y%m%d_%H%M%S}"
    with _LOCK:
        conn.execute("INSERT INTO exam_sessions VALUES (?,?,?,?,?,?)",
                     (sid, exam_id, _now(), None, mode, "recording"))
        conn.commit()
    return sid


def end_session(conn, session_id):
    with _LOCK:
        conn.execute("UPDATE exam_sessions SET end_time=?, recording_status='finished' "
                     "WHERE session_id=?", (_now(), session_id))
        conn.commit()


def add_source(conn, session_id, source_type, source_name, file_path, audio_available):
    with _LOCK:
        cur = conn.execute(
            "INSERT INTO recording_sources(session_id, source_type, source_name, start_time, "
            "file_path, audio_available) VALUES (?,?,?,?,?,?)",
            (session_id, source_type, source_name, _now(), file_path, int(audio_available)))
        conn.commit()
        return cur.lastrowid


def add_event(conn, session_id, source_id, event_type, event_ts, priority,
              detail="", snapshot_path=None, track_id=None):
    when = datetime.fromtimestamp(event_ts).isoformat(timespec="seconds")
    with _LOCK:
        cur = conn.execute(
            "INSERT INTO events(session_id, source_id, track_id, event_type, event_time, "
            "review_priority, detail, snapshot_path) VALUES (?,?,?,?,?,?,?,?)",
            (session_id, source_id, track_id, event_type, when, priority, detail, snapshot_path))
        conn.commit()
        return cur.lastrowid


def set_clip(conn, event_id, clip_path):
    with _LOCK:
        conn.execute("UPDATE events SET clip_path=? WHERE event_id=?", (clip_path, event_id))
        conn.commit()