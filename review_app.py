"""Human review dashboard.   Run:  streamlit run review_app.py"""

import os
import sqlite3
import pandas as pd
import streamlit as st
import config as cfg


# 1. Connect to the database
def connect_database():
    return sqlite3.connect(cfg.DB_PATH)


# 2. Load events based on their status
def load_events(conn, status):
    query = "SELECT * FROM events"
    params = ()

    if status != "all":
        query += " WHERE review_status = ?"
        params = (status,)

    query += " ORDER BY review_priority DESC, event_time"

    return pd.read_sql(query, conn, params=params)


# 3. Save the reviewer's decision
def save_decision(conn, event_id, decision, note):
    conn.execute(
        """
        UPDATE events
        SET review_status = 'reviewed',
            reviewer_decision = ?,
            reviewer_note = ?
        WHERE event_id = ?
        """,
        (decision, note, int(event_id))
    )

    conn.commit()


# 4. Create the dashboard
def main():
    st.set_page_config(page_title="Exam Review", layout="wide")
    st.title("Exam malpractice review")

    conn = connect_database()

    status = st.sidebar.selectbox(
        "Status", ["pending", "reviewed", "all"])

    df = load_events(conn, status)

    if df.empty:
        st.info("No events.")
        conn.close()
        st.stop()

    columns = [
        "event_id", "event_time", "event_type",
        "review_priority", "detail", "review_status"
    ]

    st.dataframe(
        df[columns],
        use_container_width=True,
        hide_index=True
    )

    event_id = st.selectbox("Open event", df["event_id"].tolist())
    event = df[df["event_id"] == event_id].iloc[0]

    left, right = st.columns(2)

    # Show event information and video
    with left:
        st.subheader(f"#{event_id} - {event['event_type']}")
        st.write(f"Time: {event['event_time']}")
        st.write(f"Priority: {event['review_priority']}")
        st.write(f"Details: {event['detail']}")

        clip = event["clip_path"]

        if pd.notna(clip) and os.path.exists(clip):
            st.video(clip)
        else:
            st.warning("Video clip not available.")

    # Show snapshot and review controls
    with right:
        snapshot = event["snapshot_path"]

        if pd.notna(snapshot) and os.path.exists(snapshot):
            st.image(snapshot)

        decision = st.radio(
            "Decision",
            [
                "false_alarm",
                "needs_follow_up",
                "confirmed_violation"
            ]
        )

        note = st.text_area("Reviewer note")

        if st.button("Save decision"):
            save_decision(conn, event_id, decision, note)
            st.success("Decision saved.")
            st.rerun()

    conn.close()


if __name__ == "__main__":
    main()