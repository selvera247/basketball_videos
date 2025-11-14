import os
from pathlib import Path

import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips

RAW_FOLDER = Path("obs_raw")
OUTPUT_FOLDER = Path("obs_clean")

# --- TUNABLE PARAMETERS -----------------------------------------------------

START_BUFFER_SEC = 5.0      # trim first 5 seconds
END_BUFFER_SEC = 5.0        # trim last 5 seconds

FRAME_SAMPLE_RATE = 2.0     # how many frames per second we analyze (e.g. 2 FPS)
MOTION_RATIO_THRESHOLD = 0.005  # fraction of pixels that must change to count as "motion"
MAX_GAP_SEC = 0.75          # max gap between motion timestamps to still be same segment
SEGMENT_PADDING_SEC = 0.25  # pad each segment start/end to avoid chopping mid-move
MIN_SEGMENT_LENGTH_SEC = 1.0  # drop very tiny segments

# ---------------------------------------------------------------------------


def detect_motion_segments(video_path: Path):
    """Return list of (start_sec, end_sec) segments where motion is detected."""
    print(f"\n[INFO] Analyzing motion in: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"[INFO] FPS: {fps:.2f}, frames: {frame_count}, duration: {duration:.2f}s")

    # Calculate which timestamps we actually care about (after buffers)
    effective_start = START_BUFFER_SEC
    effective_end = max(duration - END_BUFFER_SEC, effective_start + 1.0)

    sample_step = max(int(round(fps / FRAME_SAMPLE_RATE)), 1)

    motion_times = []
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only sample every Nth frame to save time
        if frame_idx % sample_step != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / fps

        # Ignore frames outside effective window
        if timestamp < effective_start or timestamp > effective_end:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray
            frame_idx += 1
            continue

        diff = cv2.absdiff(prev_gray, gray)
        # Blur & threshold to reduce noise
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Motion ratio = changed pixels / total pixels
        changed_pixels = np.sum(thresh == 255)
        total_pixels = thresh.size
        motion_ratio = changed_pixels / float(total_pixels)

        if motion_ratio > MOTION_RATIO_THRESHOLD:
            motion_times.append(timestamp)

        prev_gray = gray
        frame_idx += 1

    cap.release()

    if not motion_times:
        print("[WARN] No motion segments detected (after buffers).")
        return []

    # Group motion_times into continuous segments
    motion_times.sort()
    segments = []
    seg_start = motion_times[0]
    prev_t = motion_times[0]

    for t in motion_times[1:]:
        if t - prev_t <= MAX_GAP_SEC:
            # still same segment
            prev_t = t
        else:
            # segment ended at prev_t
            start = max(seg_start - SEGMENT_PADDING_SEC, effective_start)
            end = min(prev_t + SEGMENT_PADDING_SEC, effective_end)
            if end - start >= MIN_SEGMENT_LENGTH_SEC:
                segments.append((start, end))
            seg_start = t
            prev_t = t

    # close final segment
    start = max(seg_start - SEGMENT_PADDING_SEC, effective_start)
    end = min(prev_t + SEGMENT_PADDING_SEC, effective_end)
    if end - start >= MIN_SEGMENT_LENGTH_SEC:
        segments.append((start, end))

    print(f"[INFO] Detected {len(segments)} motion segment(s).")
    for i, (s, e) in enumerate(segments, 1):
        print(f"  Segment {i}: {s:.2f}s → {e:.2f}s ({e - s:.2f}s)")

    return segments


def build_clean_video(video_path: Path, segments):
    """Cut video into motion segments and concatenate into a clean clip."""
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    with VideoFileClip(str(video_path)) as clip:
        subclips = []
        for (start, end) in segments:
            # Safety check
            if end <= start:
                continue
            sub = clip.subclip(start, end)
            subclips.append(sub)

        if not subclips:
            print("[WARN] No valid segments to export. Skipping.")
            return None

        final = concatenate_videoclips(subclips)

        output_name = video_path.stem + "_clean.mp4"
        output_path = OUTPUT_FOLDER / output_name

        print(f"[INFO] Writing cleaned video → {output_path}")
        final.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            threads=4,
            verbose=False,
            logger=None,
        )

    print(f"[DONE] Saved: {output_path}")
    return output_path


def process_one_video(video_path: Path):
    segments = detect_motion_segments(video_path)
    if not segments:
        # Fallback: just trim start/end buffers
        print("[INFO] Fallback to simple buffer trim only.")
        with VideoFileClip(str(video_path)) as clip:
            duration = clip.duration
            start = START_BUFFER_SEC
            end = max(duration - END_BUFFER_SEC, start + 1.0)
            trimmed = clip.subclip(start, end)

            OUTPUT_FOLDER.mkdir(exist_ok=True)
            output_name = video_path.stem + "_trimmed.mp4"
            output_path = OUTPUT_FOLDER / output_name

            trimmed.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                threads=4,
                verbose=False,
                logger=None,
            )
        print(f"[DONE] Saved trimmed-only video: {output_path}")
        return

    build_clean_video(video_path, segments)


def process_all_in_folder():
    RAW_FOLDER.mkdir(exist_ok=True)
    videos = [p for p in RAW_FOLDER.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".mkv"}]

    if not videos:
        print(f"[INFO] No videos found in {RAW_FOLDER.resolve()}")
        return

    print(f"[INFO] Found {len(videos)} video(s) to process.")
    for vid in videos:
        process_one_video(vid)


if __name__ == "__main__":
    process_all_in_folder()
