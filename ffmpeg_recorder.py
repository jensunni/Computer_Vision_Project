"""Independent FFmpeg recording (CCTV + separate USB microphone).
Runs as its own process, so if the AI crashes the recording continues.

Uses .mkv: the file stays readable while it is still being written and survives
crashes (a half-written .mp4 is often unplayable).
NOTE: for a USB *webcam*, OpenCV and FFmpeg cannot open the device at the same
time on Windows. Use ClipRecorder for webcam; use this for RTSP CCTV.
"""

import os
import shutil
import subprocess
import time
from datetime import datetime


def start_recording(rtsp_url, output_folder, mic_name=None):

    # 1. Check whether FFmpeg is installed
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Please install FFmpeg first.")

    # 2. Create the output folder
    os.makedirs(output_folder, exist_ok=True)

    # 3. Create a recording filename
    filename = f"cctv_{datetime.now():%Y%m%d_%H%M%S}.mkv"
    recording_path = os.path.join(output_folder, filename)

    # 4. Tell FFmpeg to read the CCTV stream
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url
    ]

    # 5. Add microphone audio if configured
    if mic_name:
        command += [
            "-f", "dshow",
            "-i", f"audio={mic_name}",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "48000"
        ]
    else:
        # Record video only
        command += [
            "-map", "0:v:0",
            "-c:v", "copy"
        ]

    # 6. Save the recording
    command.append(recording_path)

    # 7. Start FFmpeg as a separate process
    log_path = os.path.join(output_folder, "ffmpeg.log")
    log_file = open(log_path, "ab")

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=log_file,
        stderr=log_file
    )

    # Return process, filename and approximate start time
    return process, recording_path, time.time()


def stop_recording(process):

    # Send FFmpeg its quit command
    try:
        process.stdin.write(b"q")
        process.stdin.flush()
        process.wait(timeout=15)

    except Exception:
        process.terminate()


def extract_clip(
    recording_path,
    recording_start_time,
    event_time,
    output_path,
    pre_seconds=5,
    post_seconds=10
):

    # Find where the event occurs in the recording
    start_seconds = max(
        0,
        event_time - recording_start_time - pre_seconds
    )

    # Extract a shorter video
    command = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-ss", f"{start_seconds:.2f}",
        "-i", recording_path,
        "-t", str(pre_seconds + post_seconds),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-c:a", "aac",
        output_path
    ]

    subprocess.run(command, check=True)