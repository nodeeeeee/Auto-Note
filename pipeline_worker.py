"""
Pipelined per-video processing: transcribe → extract frames → align.

Uses a producer-consumer pattern with two threads so that while video N+1
is being transcribed (GPU), video N can be frame-extracted and aligned
concurrently.  This overlaps the stages and reduces total wall-clock time
compared to the sequential approach (transcribe ALL → frame-extract ALL →
align ALL).

Usage:
  python pipeline_worker.py --course 85397 [--force] [--path ~/AutoNote]

Replaces the three separate calls to:
  extract_caption.py / frame_extractor.py / semantic_alignment.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from queue import Queue

PROJECT_DIR = Path(__file__).parent
_AUTO_NOTE_DIR = Path.home() / ".auto_note"
if os.environ.get("AUTONOTE_DATA_DIR"):
    DATA_DIR = Path(os.environ["AUTONOTE_DATA_DIR"])
elif getattr(sys, "frozen", False) or PROJECT_DIR == _AUTO_NOTE_DIR / "scripts":
    DATA_DIR = _AUTO_NOTE_DIR
else:
    DATA_DIR = PROJECT_DIR

try:
    _cfg_file = DATA_DIR / "config.json"
    _cfg: dict = json.loads(_cfg_file.read_text(encoding="utf-8")) if _cfg_file.exists() else {}
except Exception:
    _cfg = {}
_out = _cfg.get("OUTPUT_DIR", "").strip()
COURSE_DATA_DIR = Path(_out) if _out else Path.home() / "AutoNote"

MANIFEST_FILE = DATA_DIR / "manifest.json"

# Detect scripts location
SCRIPTS_DIR = _AUTO_NOTE_DIR / "scripts"
if not SCRIPTS_DIR.exists():
    SCRIPTS_DIR = PROJECT_DIR

PYTHON = sys.executable

# Prevent console flashing on Windows
_CREATIONFLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _script(name: str) -> str:
    p = SCRIPTS_DIR / name
    if p.exists():
        return str(p)
    return str(PROJECT_DIR / name)


def _run(cmd: list[str], label: str) -> bool:
    """Run a subprocess, streaming output to stdout."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}", flush=True)
    proc = subprocess.run(
        cmd,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        creationflags=_CREATIONFLAGS,
    )
    return proc.returncode == 0


def get_videos(course_id: str, base_dir: Path) -> list[dict]:
    """Get list of downloaded videos for a course from manifest."""
    if not MANIFEST_FILE.exists():
        return []
    with open(MANIFEST_FILE, encoding="utf-8") as f:
        manifest = json.load(f)

    videos = []
    for key, entry in manifest.items():
        if entry.get("status") != "done":
            continue
        vpath = entry.get("path", "")
        if not vpath or not Path(vpath).exists():
            continue
        try:
            if Path(vpath).parent.parent.name != course_id:
                continue
        except Exception:
            continue
        videos.append({
            "path": Path(vpath),
            "stem": Path(vpath).stem,
            "stream_tag": entry.get("stream_tag", ""),
            "course_dir": base_dir / course_id,
        })
    return videos


def transcribe_one(video: dict, force: bool) -> bool:
    """Transcribe a single video."""
    caption = video["course_dir"] / "captions" / f"{video['stem']}.json"
    if not force and caption.exists():
        print(f"  [skip] Already transcribed: {video['stem']}")
        return True
    cmd = [PYTHON, _script("extract_caption.py"), "--video", str(video["path"])]
    if force:
        cmd.append("--force")
    return _run(cmd, f"Transcribe: {video['stem']}")


def extract_frames_one(video: dict, force_screen: bool = False) -> bool:
    """Extract frames from a single video (if screenshare).

    When ``force_screen`` is True the camera/screen auto-classifier is
    bypassed so frames are extracted unconditionally — used when the user
    explicitly chose "video screenshots" in the UI and expects images even
    from camera-style recordings.
    """
    caption = video["course_dir"] / "captions" / f"{video['stem']}.json"
    if not caption.exists():
        return True  # no caption yet, skip frame extraction
    align_file = video["course_dir"] / "alignment" / f"{video['stem']}.json"
    # Skip if already has screenshare alignment
    if align_file.exists():
        try:
            data = json.loads(align_file.read_text(encoding="utf-8"))
            if data.get("source") == "screenshare":
                print(f"  [skip] Frames already extracted: {video['stem']}")
                return True
        except Exception:
            pass

    cmd = [PYTHON, _script("frame_extractor.py"),
           "--video", str(video["path"]),
           "--caption", str(caption),
           "--course-dir", str(video["course_dir"])]
    if force_screen:
        cmd.append("--force-screen")
    return _run(cmd, f"Extract frames: {video['stem']}")


def align_one(video: dict, force: bool) -> bool:
    """Align a single caption to slides."""
    caption = video["course_dir"] / "captions" / f"{video['stem']}.json"
    if not caption.exists():
        return True  # no caption, skip

    # Skip if screenshare alignment exists (frame_extractor handled it)
    align_file = video["course_dir"] / "alignment" / f"{video['stem']}.json"
    if not force and align_file.exists():
        try:
            data = json.loads(align_file.read_text(encoding="utf-8"))
            if data.get("source") == "screenshare":
                print(f"  [skip] Screenshare alignment exists: {video['stem']}")
                return True
        except Exception:
            pass

    # For slide-based alignment, use the course-level command
    # (it handles matching captions to slides)
    if not force and align_file.exists():
        print(f"  [skip] Already aligned: {video['stem']}")
        return True

    cmd = [PYTHON, _script("semantic_alignment.py"),
           "--caption", str(caption)]

    # Find matching slides — use the course-level discovery
    # Just run for the whole course; it will skip already-aligned captions
    cmd = [PYTHON, _script("semantic_alignment.py"),
           "--course", video["course_dir"].name]
    if force:
        cmd.append("--force")
    return _run(cmd, f"Align: {video['stem']}")


def pipeline_sequential(videos: list[dict], force: bool,
                        skip_frames: bool = False,
                        force_screen: bool = False) -> None:
    """Simple sequential pipeline: transcribe → frames → align per video."""
    for i, video in enumerate(videos, 1):
        print(f"\n{'═' * 60}")
        print(f"  Video {i}/{len(videos)}: {video['stem']}")
        print(f"{'═' * 60}", flush=True)

        transcribe_one(video, force)
        if not skip_frames:
            extract_frames_one(video, force_screen=force_screen)

    # Alignment is best done course-level (batch BGE-M3 matching)
    if videos:
        course_dir = videos[0]["course_dir"]
        cmd = [PYTHON, _script("semantic_alignment.py"),
               "--course", course_dir.name]
        if force:
            cmd.append("--force")
        _run(cmd, "Align all transcripts")


def pipeline_threaded(videos: list[dict], force: bool,
                      skip_frames: bool = False,
                      force_screen: bool = False) -> None:
    """Threaded pipeline: transcribe and frame-extract/align overlap.

    Thread 1 (transcriber): transcribes videos one at a time, pushes
    completed videos into a queue.
    Thread 2 (processor): takes transcribed videos and extracts frames.
    After all videos are processed, alignment runs once (batch mode).
    """
    if not videos:
        return

    queue: Queue[dict | None] = Queue()
    errors: list[str] = []

    def transcriber():
        for i, video in enumerate(videos, 1):
            print(f"\n{'═' * 60}")
            print(f"  [{i}/{len(videos)}] Transcribing: {video['stem']}")
            print(f"{'═' * 60}", flush=True)
            ok = transcribe_one(video, force)
            if not ok:
                errors.append(f"Transcribe failed: {video['stem']}")
            queue.put(video)
        queue.put(None)  # sentinel

    def frame_processor():
        while True:
            video = queue.get()
            if video is None:
                break
            if skip_frames:
                continue  # frames disabled — drain the queue only
            print(f"\n  Processing frames: {video['stem']}", flush=True)
            extract_frames_one(video, force_screen=force_screen)

    t1 = threading.Thread(target=transcriber, name="transcriber")
    t2 = threading.Thread(target=frame_processor, name="frame-processor")

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Alignment runs after all transcriptions + frame extractions are done
    # (needs BGE-M3 model which conflicts with Whisper GPU usage)
    course_dir = videos[0]["course_dir"]
    cmd = [PYTHON, _script("semantic_alignment.py"),
           "--course", course_dir.name]
    if force:
        cmd.append("--force")
    _run(cmd, "Align all transcripts")

    if errors:
        print(f"\n[warn] {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipelined per-video processing: transcribe → frames → align")
    parser.add_argument("--course", required=True, help="Course ID")
    parser.add_argument("--path", help="Base output directory")
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if output files exist")
    parser.add_argument("--sequential", action="store_true",
                        help="Disable threading (debug mode)")
    parser.add_argument("--skip-frames", action="store_true",
                        help="Skip frame extraction + per-frame vision "
                             "description. Use this when you plan to generate "
                             "notes with --image-source slides — the frames "
                             "and their descriptions are never read.")
    parser.add_argument("--force-screen", action="store_true",
                        help="Bypass the camera/screen auto-classifier in the "
                             "frame extractor and always extract frames. Pass "
                             "this when the user has explicitly chosen video "
                             "screenshots so camera-style recordings still "
                             "produce images instead of falling through to "
                             "missing slide PDFs.")
    args = parser.parse_args()

    base_dir = Path(args.path) if args.path else COURSE_DATA_DIR
    videos = get_videos(args.course, base_dir)

    if not videos:
        print(f"No downloaded videos found for course {args.course}.")
        print("Run 'Download videos' first.")
        return

    print(f"Found {len(videos)} video(s) for course {args.course}.")

    if args.sequential:
        pipeline_sequential(videos, args.force, skip_frames=args.skip_frames,
                            force_screen=args.force_screen)
    else:
        pipeline_threaded(videos, args.force, skip_frames=args.skip_frames,
                          force_screen=args.force_screen)

    print(f"\n✓ Pipeline complete: {len(videos)} video(s) processed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[info] Interrupted by user.")
        sys.exit(0)
