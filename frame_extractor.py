"""
Frame Extractor for Screen Share Videos

Extracts keyframes from screen share (SS) Panopto recordings at scene-change
boundaries.  The result is a set of PNG frames + an alignment JSON that maps
transcript segments to frames by timestamp — replacing the traditional
slide-based alignment when the video itself IS the slides.

Pipeline:
  1. Detect scene changes in the video using ffmpeg scene filter
  2. Extract one representative frame per scene
  3. Build alignment JSON mapping transcript segments → frames by timestamp
  4. Frames are saved to <course_id>/frames/<video_stem>/frame_NNN.png

Usage:
  python frame_extractor.py --video path/to/video.mp4
  python frame_extractor.py --video path/to/video.mp4 --caption path/to/caption.json
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
_AUTO_NOTE_DIR = Path.home() / ".auto_note"
if os.environ.get("AUTONOTE_DATA_DIR"):
    DATA_DIR = Path(os.environ["AUTONOTE_DATA_DIR"])
elif getattr(sys, "frozen", False) or PROJECT_DIR == _AUTO_NOTE_DIR / "scripts":
    DATA_DIR = _AUTO_NOTE_DIR
else:
    DATA_DIR = PROJECT_DIR

_cfg_file = DATA_DIR / "config.json"
_fe_config: dict = json.loads(_cfg_file.read_text(encoding="utf-8")) if _cfg_file.exists() else {}
_out_dir = _fe_config.get("OUTPUT_DIR", "").strip()
COURSE_DATA_DIR = Path(_out_dir) if _out_dir else DATA_DIR

# Prevent console windows flashing on Windows
_SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

# ── Tunable constants ────────────────────────────────────────────────────────

# Scene detection threshold: lower = more sensitive (more frames extracted).
# Range 0.0-1.0.  0.3 works well for slide transitions; use 0.2 for subtle changes.
SCENE_THRESHOLD = 0.3

# Number of sample frames to use for screen-vs-camera classification.
CLASSIFY_SAMPLE_FRAMES = 6

# Minimum seconds between extracted frames (avoids near-duplicate frames from
# quick animations or cursor flickers).
MIN_SCENE_GAP = 2.0

# Maximum number of frames to extract (safety limit for very long recordings).
MAX_FRAMES = 500


# ── Screen vs Camera auto-detection ──────────────────────────────────────────

def classify_video(video_path: Path) -> str:
    """Classify a video as 'screen' or 'camera' by analyzing sample frames.

    Extracts a few evenly-spaced frames and uses heuristics:
    - Screen recordings: sharp edges, high contrast, uniform backgrounds,
      lots of text/UI elements, low color variance in large regions
    - Camera recordings: smooth gradients, natural colors, motion blur,
      faces/bodies, varied lighting

    Returns 'screen' or 'camera'.
    """
    import tempfile

    duration = get_video_duration(video_path)
    if duration <= 0:
        return "camera"  # can't determine, default to camera

    # Sample frames evenly across the video (skip first/last 10%)
    start = duration * 0.1
    end = duration * 0.9
    n = CLASSIFY_SAMPLE_FRAMES
    timestamps = [start + i * (end - start) / (n - 1) for i in range(n)]

    tmp_dir = Path(tempfile.mkdtemp(prefix="classify_"))
    frame_paths = []
    for i, ts in enumerate(timestamps):
        png = tmp_dir / f"sample_{i}.png"
        subprocess.run(
            ["ffmpeg", "-ss", f"{ts:.1f}", "-i", str(video_path),
             "-frames:v", "1", "-q:v", "2", str(png), "-y"],
            capture_output=True, timeout=30,
            creationflags=_SUBPROCESS_FLAGS,
        )
        if png.exists():
            frame_paths.append(png)

    if not frame_paths:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return "camera"

    # Analyze frames with image heuristics
    screen_votes = 0
    try:
        from PIL import Image as PILImage
        import statistics

        for fp in frame_paths:
            img = PILImage.open(fp).convert("RGB")
            w, h = img.size

            # Heuristic 1: Edge density (screen recordings have sharp edges)
            # Use a simple Laplacian-like approach via pixel differences
            pixels = list(img.getdata())
            row_diffs = 0
            for y in range(0, h - 1, 4):
                for x in range(0, w - 1, 4):
                    idx = y * w + x
                    idx_r = idx + 1
                    if idx_r < len(pixels):
                        diff = sum(abs(a - b) for a, b in zip(pixels[idx], pixels[idx_r]))
                        if diff > 100:  # sharp edge threshold
                            row_diffs += 1

            total_samples = (h // 4) * (w // 4)
            edge_ratio = row_diffs / max(total_samples, 1)

            # Heuristic 2: Color uniformity (screen recordings have large
            # uniform regions — backgrounds, toolbars)
            # Sample a grid and check how many blocks have low variance
            block_size = 32
            uniform_blocks = 0
            total_blocks = 0
            for by in range(0, h - block_size, block_size):
                for bx in range(0, w - block_size, block_size):
                    block = img.crop((bx, by, bx + block_size, by + block_size))
                    block_pixels = list(block.getdata())
                    r_vals = [p[0] for p in block_pixels]
                    g_vals = [p[1] for p in block_pixels]
                    b_vals = [p[2] for p in block_pixels]
                    total_blocks += 1
                    # Low variance = uniform color
                    if (statistics.stdev(r_vals) < 15 and
                        statistics.stdev(g_vals) < 15 and
                        statistics.stdev(b_vals) < 15):
                        uniform_blocks += 1

            uniformity = uniform_blocks / max(total_blocks, 1)

            # Heuristic 3: Brightness distribution (screens tend toward
            # high brightness with white backgrounds)
            brightness = [sum(p) / 3 for p in pixels[::16]]
            avg_brightness = statistics.mean(brightness)
            bright_ratio = sum(1 for b in brightness if b > 200) / len(brightness)

            # Vote: screen if sharp edges + uniform regions + bright
            is_screen = (
                (edge_ratio > 0.08 and uniformity > 0.4) or
                (uniformity > 0.55 and bright_ratio > 0.3) or
                (edge_ratio > 0.12 and bright_ratio > 0.4)
            )
            if is_screen:
                screen_votes += 1

    except ImportError:
        # PIL not available — fall back to OpenAI vision API
        pass
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    result = "screen" if screen_votes > len(frame_paths) / 2 else "camera"
    print(f"  Video classification: {result} ({screen_votes}/{len(frame_paths)} frames voted screen)")
    return result


# ── Scene detection via ffmpeg ───────────────────────────────────────────────

def detect_scenes(video_path: Path, threshold: float = SCENE_THRESHOLD,
                  min_gap: float = MIN_SCENE_GAP) -> list[float]:
    """Detect scene change timestamps in a video using ffmpeg's scene filter.

    Returns a sorted list of timestamps (seconds) where scene changes occur.
    The first timestamp is always 0.0 (start of video).
    """
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"select='gt(scene\\,{threshold})',showinfo",
        "-vsync", "vfr",
        "-f", "null", "-"
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
        creationflags=_SUBPROCESS_FLAGS,
    )

    # Parse timestamps from ffmpeg showinfo output
    timestamps = [0.0]  # Always include the start
    pattern = re.compile(r"pts_time:(\d+\.?\d*)")

    for line in result.stderr.splitlines():
        if "showinfo" in line and "pts_time" in line:
            m = pattern.search(line)
            if m:
                ts = float(m.group(1))
                # Enforce minimum gap
                if ts - timestamps[-1] >= min_gap:
                    timestamps.append(ts)

    print(f"  Detected {len(timestamps)} scene changes (threshold={threshold})")
    return timestamps[:MAX_FRAMES]


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=30,
        creationflags=_SUBPROCESS_FLAGS,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


# ── Frame extraction ─────────────────────────────────────────────────────────

def extract_frames(video_path: Path, timestamps: list[float],
                   out_dir: Path) -> dict[int, Path]:
    """Extract frames at the given timestamps and save as PNGs.

    Returns a mapping of frame_index (0-based) → PNG path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[int, Path] = {}

    for i, ts in enumerate(timestamps):
        png = out_dir / f"frame_{i + 1:03d}.png"
        if png.exists():
            mapping[i] = png
            continue

        cmd = [
            "ffmpeg", "-ss", f"{ts:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(png), "-y"
        ]
        subprocess.run(
            cmd, capture_output=True, timeout=60,
            creationflags=_SUBPROCESS_FLAGS,
        )
        if png.exists():
            mapping[i] = png

    print(f"  Extracted {len(mapping)} frames → {out_dir}")
    return mapping


# ── Build alignment from timestamps ──────────────────────────────────────────

def build_frame_alignment(
    caption_path: Path,
    timestamps: list[float],
    video_duration: float,
    video_stem: str,
    frame_dir: Path,
) -> dict:
    """Build alignment JSON mapping transcript segments to extracted frames.

    Each segment is assigned to the frame whose scene interval contains the
    segment's midpoint.  The result matches the format of semantic_alignment.py
    output so downstream code (alignment_parser, note_generation) works unchanged.
    """
    with open(caption_path, encoding="utf-8") as f:
        caption = json.load(f)

    segments: list[dict] = caption.get("segments", [])
    if not segments:
        return {}

    # Build scene intervals: each scene runs from timestamps[i] to timestamps[i+1]
    intervals: list[tuple[float, float]] = []
    for i in range(len(timestamps)):
        start = timestamps[i]
        end = timestamps[i + 1] if i + 1 < len(timestamps) else video_duration
        intervals.append((start, end))

    # Assign each transcript segment to a frame
    aligned_segments: list[dict] = []
    for seg in segments:
        mid = (seg["start"] + seg["end"]) / 2.0

        # Find the interval containing this midpoint
        frame_idx = 0
        for fi, (istart, iend) in enumerate(intervals):
            if istart <= mid < iend:
                frame_idx = fi
                break
        else:
            # Past last interval → assign to last frame
            frame_idx = len(intervals) - 1

        aligned_segments.append({
            "id": seg.get("id", 0),
            "start": seg["start"],
            "end": seg["end"],
            "text": seg.get("text", ""),
            "slide": frame_idx + 1,  # 1-based for compatibility
            "slide_label": f"Frame {frame_idx + 1}",
            "similarity": 1.0,  # exact timestamp match
            "off_slide": False,
        })

    # Build timeline (collapse consecutive same-frame segments)
    timeline: list[dict] = []
    cur_frame = None
    cur_start = 0.0
    cur_end = 0.0

    for aseg in aligned_segments:
        fi = aseg["slide"]
        if cur_frame is None:
            cur_frame = fi
            cur_start = aseg["start"]
            cur_end = aseg["end"]
        elif fi == cur_frame:
            cur_end = aseg["end"]
        else:
            timeline.append({
                "slide": cur_frame,
                "start": round(cur_start, 3),
                "end": round(cur_end, 3),
                "label": f"Frame {cur_frame}",
            })
            cur_frame = fi
            cur_start = aseg["start"]
            cur_end = aseg["end"]

    if cur_frame is not None:
        timeline.append({
            "slide": cur_frame,
            "start": round(cur_start, 3),
            "end": round(cur_end, 3),
            "label": f"Frame {cur_frame}",
        })

    return {
        "lecture": video_stem,
        "slide_file": f"frames/{video_stem}",
        "source": "screenshare",
        "total_slides": len(timestamps),
        "total_segments": len(segments),
        "off_slide_count": 0,
        "duration": caption.get("duration", video_duration),
        "language": caption.get("language", ""),
        "segments": aligned_segments,
        "timeline": timeline,
    }


# ── High-level entry point ───────────────────────────────────────────────────

def extract_and_align(
    video_path: Path,
    caption_path: Path | None,
    course_dir: Path,
    skip_classify: bool = False,
) -> tuple[Path | None, Path | None]:
    """Full pipeline: classify video → detect scenes → extract frames → build alignment.

    If the video is detected as a camera recording (not screen), returns
    (None, None) so the caller can fall back to slide-based alignment.
    Use skip_classify=True to force frame extraction regardless.

    Returns (frame_dir, alignment_path) or (None, None) on failure/skip.
    """
    video_stem = video_path.stem
    frame_dir = course_dir / "frames" / video_stem
    align_dir = course_dir / "alignment"
    align_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Frame extraction: {video_stem}")
    print(f"{'=' * 70}")

    # Auto-classify: screen recording or camera?
    if not skip_classify:
        video_type = classify_video(video_path)
        if video_type == "camera":
            print(f"  Video classified as CAMERA recording — skipping frame extraction.")
            print(f"  Use slide-based alignment instead.")
            return None, None
        print(f"  Video classified as SCREEN recording — extracting frames.")

    # Step 1: Detect scenes
    print("  Detecting scene changes...")
    timestamps = detect_scenes(video_path)
    if len(timestamps) < 2:
        print("  [warn] Too few scene changes detected, using uniform sampling")
        duration = get_video_duration(video_path)
        if duration > 0:
            # Sample every 30 seconds
            timestamps = [i * 30.0 for i in range(int(duration / 30) + 1)]
        else:
            print("  [error] Cannot determine video duration")
            return None, None

    # Step 2: Extract frames
    print(f"  Extracting {len(timestamps)} frames...")
    frame_map = extract_frames(video_path, timestamps, frame_dir)
    if not frame_map:
        print("  [error] No frames extracted")
        return None, None

    # Step 3: Build alignment (if caption exists)
    alignment_path = None
    if caption_path and caption_path.exists():
        duration = get_video_duration(video_path)
        alignment = build_frame_alignment(
            caption_path, timestamps, duration, video_stem, frame_dir
        )
        alignment_path = align_dir / f"{video_stem}.json"
        with open(alignment_path, "w", encoding="utf-8") as f:
            json.dump(alignment, f, ensure_ascii=False, indent=2)
        print(f"  Alignment → {alignment_path}")

    print(f"  Done: {len(frame_map)} frames, alignment={'yes' if alignment_path else 'pending caption'}")
    return frame_dir, alignment_path


# ── Course-level auto-discovery ──────────────────────────────────────────────

def process_course(course_id: str, base_dir: Path) -> int:
    """Auto-discover videos for a course, classify them, and extract frames.

    Reads the manifest to find all downloaded videos for the course.
    For each video:
      - If stream_tag is "SS" → always extract frames (known screen share)
      - Otherwise → auto-classify the video content as screen or camera
      - Screen recordings → extract frames + build alignment
      - Camera recordings → skip (use slide-based alignment instead)

    Returns the number of screen-recording videos processed.
    """
    manifest_file = DATA_DIR / "manifest.json"
    if not manifest_file.exists():
        print("[info] No manifest found — nothing to process.")
        return 0

    with open(manifest_file, encoding="utf-8") as f:
        manifest = json.load(f)

    course_dir = base_dir / course_id
    processed = 0

    for key, entry in manifest.items():
        if entry.get("status") != "done":
            continue

        video_path = Path(entry["path"])
        if not video_path.exists():
            continue

        # Check if this video belongs to the requested course
        # Video path pattern: <base>/<course_id>/videos/<name>.mp4
        try:
            if video_path.parent.parent.name != course_id:
                continue
        except Exception:
            continue

        video_stem = video_path.stem
        frame_dir = course_dir / "frames" / video_stem
        align_file = course_dir / "alignment" / f"{video_stem}.json"

        # Skip if already processed
        if align_file.exists():
            print(f"  [skip] Already extracted: {video_stem}")
            processed += 1
            continue

        # Find matching caption
        caption_path = course_dir / "captions" / f"{video_stem}.json"
        if not caption_path.exists():
            print(f"  [skip] No caption for: {video_stem} (transcribe first)")
            continue

        # For SS-tagged streams, skip classification (known screen share)
        skip_classify = (entry.get("stream_tag", "").upper() == "SS")
        result = extract_and_align(video_path, caption_path, course_dir,
                                   skip_classify=skip_classify)
        if result[0] is not None:
            processed += 1

    if processed == 0:
        print("[info] No screen-recording videos found for this course.")
        print("       All videos appear to be camera recordings — use slide-based alignment.")
    return processed


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Extract frames from screen share videos")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Path to a single video MP4")
    group.add_argument("--course", help="Course ID — auto-discover SS videos from manifest")
    parser.add_argument("--caption", help="Path to caption JSON (for --video mode)")
    parser.add_argument("--course-dir", help="Course directory (default: inferred)")
    parser.add_argument("--path", help="Base output directory (for --course mode)")
    parser.add_argument("--threshold", type=float, default=SCENE_THRESHOLD,
                        help=f"Scene detection threshold (default: {SCENE_THRESHOLD})")
    args = parser.parse_args()

    _update_threshold(args.threshold)

    if args.course:
        base_dir = Path(args.path) if args.path else COURSE_DATA_DIR
        process_course(args.course, base_dir)
    else:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"[error] Video not found: {video_path}")
            sys.exit(1)

        caption_path = Path(args.caption) if args.caption else None

        if args.course_dir:
            course_dir = Path(args.course_dir)
        else:
            course_dir = video_path.parent.parent

        extract_and_align(video_path, caption_path, course_dir)


def _update_threshold(val: float) -> None:
    global SCENE_THRESHOLD
    SCENE_THRESHOLD = val


if __name__ == "__main__":
    main()
