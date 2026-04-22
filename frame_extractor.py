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
COURSE_DATA_DIR = Path(_out_dir) if _out_dir else Path.home() / "AutoNote"

# Prevent console windows flashing on Windows
_SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


# ── ffmpeg/ffprobe resolution ─────────────────────────────────────────────────
# macOS users without Homebrew get ffmpeg via the `imageio-ffmpeg` wheel, which
# does NOT ship `ffprobe`. Without these resolvers, every subprocess call below
# raised FileNotFoundError and produced zero frames — silently, because the
# code only checks whether the output PNG exists afterwards. That turned into
# "screen recording not captured" in generated notes on EE4802 test runs.

_FFMPEG_BIN: str | None = None
_FFPROBE_BIN: str | None = None  # None sentinel means "resolved; unavailable"
_FFPROBE_RESOLVED = False


def _resolve_ffmpeg() -> str:
    """Locate an ffmpeg executable.

    Order: system PATH → imageio-ffmpeg bundled binary → auto-install. Raises
    RuntimeError if none works. Result is cached so resolution cost is paid
    once per process.
    """
    global _FFMPEG_BIN
    if _FFMPEG_BIN:
        return _FFMPEG_BIN

    from shutil import which
    sys_ff = which("ffmpeg")
    if sys_ff:
        _FFMPEG_BIN = sys_ff
        return sys_ff

    def _try_imageio() -> str | None:
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            return None
        except Exception as e:
            print(f"  [warn] imageio-ffmpeg get_ffmpeg_exe failed: {e}")
            return None

    ff = _try_imageio()
    if ff:
        _FFMPEG_BIN = ff
        return ff

    print("  ffmpeg not found locally — installing imageio-ffmpeg fallback…")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "--disable-pip-version-check", "imageio-ffmpeg"],
            check=True, timeout=180,
        )
    except Exception as e:
        raise RuntimeError(f"ffmpeg unavailable and auto-install failed: {e}") from e

    ff = _try_imageio()
    if not ff:
        raise RuntimeError("ffmpeg unavailable after imageio-ffmpeg install")
    _FFMPEG_BIN = ff
    return ff


def _resolve_ffprobe() -> str | None:
    """Locate ffprobe on system PATH. Returns None when unavailable.

    imageio-ffmpeg doesn't bundle ffprobe, so when only it is available we
    fall back to parsing `ffmpeg -i` stderr for duration.
    """
    global _FFPROBE_BIN, _FFPROBE_RESOLVED
    if _FFPROBE_RESOLVED:
        return _FFPROBE_BIN
    from shutil import which
    _FFPROBE_BIN = which("ffprobe")
    _FFPROBE_RESOLVED = True
    return _FFPROBE_BIN


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def _parse_ffmpeg_duration(stderr: str | None) -> float | None:
    """Parse 'Duration: HH:MM:SS.ss' from ffmpeg -i stderr."""
    if not stderr:
        return None
    m = _DURATION_RE.search(stderr)
    if not m:
        return None
    h, mi, s = m.group(1), m.group(2), m.group(3)
    return int(h) * 3600 + int(mi) * 60 + float(s)


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

# Perceptual-hash threshold for considering two frames as the same slide page.
# dHash is 16×16 = 256 bits. Higher = merge more aggressively.
# 35 bits ≈ 14% difference — merges incremental bullet reveals on the same
# slide while still keeping genuinely different pages.
PAGE_SIMILARITY_THRESHOLD = 35

# Vision-description patterns that indicate a junk frame (desktop, loading,
# unreadable). Frames matching these are deleted after description.
_JUNK_DESC_RE = re.compile(
    r"\b(windows 11|windows 10|desktop (background|interface)|taskbar|"
    r"file explorer|loading screen|blank screen|black screen|entirely black|"
    r"unable to (view|describe|access|process)|"
    r"cannot (describe|view|generate|provide|access)|"
    r"i'?m sorry|\bsorry\b|"
    r"abstract blue wave|swirling blue|"
    r"(humorous|funny|joke) (meme|image|comic|cartoon|scene|depiction|strip|illustration|individuals|panel)|"
    r"(meme|comic) (format|image|strip|panel)|"
    r"xkcd|four-panel comic|meme structure)\b",
    re.IGNORECASE,
)


def _get_pixels(img):
    """Get pixel data from a PIL Image, compatible with Pillow 14+."""
    if hasattr(img, 'get_flattened_data'):
        return list(img.get_flattened_data())
    return _get_pixels(img)


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
    ff_bin = _resolve_ffmpeg()
    for i, ts in enumerate(timestamps):
        png = tmp_dir / f"sample_{i}.png"
        subprocess.run(
            [ff_bin, "-ss", f"{ts:.1f}", "-i", str(video_path),
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
            pixels = _get_pixels(img)
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
                    block_pixels = _get_pixels(block)
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


# ── Perceptual hashing for frame deduplication ───────────────────────────────

def _perceptual_hash(img, hash_size: int = 16) -> int:
    """Compute a difference hash (dHash) for a PIL Image.

    Resizes to (hash_size+1, hash_size) grayscale, then compares adjacent
    pixels to produce a hash_size² bit integer. Two similar images will
    have hashes with low Hamming distance.
    """
    from PIL import Image as PILImage
    small = img.convert("L").resize((hash_size + 1, hash_size), PILImage.LANCZOS)
    pixels = _get_pixels(small)
    w = hash_size + 1
    bits = 0
    for y in range(hash_size):
        for x in range(hash_size):
            bits = (bits << 1) | (1 if pixels[y * w + x] < pixels[y * w + x + 1] else 0)
    return bits


def _hamming(a: int, b: int) -> int:
    """Hamming distance between two integers (number of differing bits)."""
    return bin(a ^ b).count("1")


def _information_score(img) -> int:
    """Score an image by visual information content (edge/detail density).

    Computes the sum of horizontal and vertical pixel-intensity gradients on a
    small grayscale thumbnail.  Frames with more text, diagrams, or revealed
    bullets score higher than sparse or blank versions of the same slide.
    """
    from PIL import Image as PILImage
    small = img.convert("L").resize((160, 120), PILImage.LANCZOS)
    pixels = _get_pixels(small)
    w, h = 160, 120
    score = 0
    for y in range(h):
        row = y * w
        for x in range(w - 1):
            score += abs(pixels[row + x] - pixels[row + x + 1])
    for y in range(h - 1):
        row = y * w
        for x in range(w):
            score += abs(pixels[row + x] - pixels[row + w + x])
    return score


# ── Intelligent scene detection ──────────────────────────────────────────────

def detect_scenes(video_path: Path, threshold: float = SCENE_THRESHOLD,
                  min_gap: float = MIN_SCENE_GAP,
                  stream_tag: str = "") -> list[float]:
    """Detect unique slide/screen changes in a video.

    Strategy (three-pass):
      1. Use ffmpeg scene filter to find raw scene-change timestamps
      2. Add periodic samples (always for OBJECT/SS streams; fallback for others)
      3. Extract a candidate frame at each timestamp, compute perceptual
         hashes (dHash) and information scores.  Group consecutive frames
         that belong to the same slide page (incremental reveals, animations)
         and keep only the most informative frame from each group.

    For OBJECT/SS streams (stable slide recordings with subtle text-only
    changes), a lower scene threshold and periodic sampling are required
    because the scene filter alone misses gradual transitions.
    """
    duration = get_video_duration(video_path)
    tag = (stream_tag or "").upper()
    is_screen_stream = tag in ("SS", "OBJECT")

    # Screen streams: more sensitive detection + denser periodic fallback.
    if is_screen_stream:
        threshold = min(threshold, 0.15)
        sample_interval = 30.0
        always_periodic = True
    else:
        sample_interval = 10.0
        always_periodic = False

    # ── Pass 1: ffmpeg scene detection ───────────────────────────────────────
    ff_bin = _resolve_ffmpeg()
    cmd = [
        ff_bin, "-i", str(video_path),
        "-vf", f"select='gt(scene\\,{threshold})',showinfo",
        "-vsync", "vfr",
        "-f", "null", "-"
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
        creationflags=_SUBPROCESS_FLAGS,
    )

    raw_timestamps = [0.0]
    pattern = re.compile(r"pts_time:(\d+\.?\d*)")
    for line in result.stderr.splitlines():
        if "showinfo" in line and "pts_time" in line:
            m = pattern.search(line)
            if m:
                ts = float(m.group(1))
                if ts - raw_timestamps[-1] >= min_gap:
                    raw_timestamps.append(ts)

    print(f"  Scene filter: {len(raw_timestamps)} raw candidates "
          f"(threshold={threshold}, stream={tag or 'auto'})")

    # ── Pass 2: periodic samples (always for screen streams, fallback otherwise) ──
    need_periodic = always_periodic or (len(raw_timestamps) < 5 and duration > 60)
    if need_periodic and duration > 60:
        reason = "screen stream — always sampling" if always_periodic else "too few scene changes"
        print(f"  {reason} — adding periodic samples every {sample_interval}s")
        periodic = [t for t in
                    (i * sample_interval for i in range(int(duration / sample_interval) + 1))
                    if t < duration]
        combined = sorted(set(raw_timestamps + periodic))
        merged: list[float] = [combined[0]]
        for ts in combined[1:]:
            if ts - merged[-1] >= min_gap:
                merged.append(ts)
        raw_timestamps = merged
        print(f"  After merge: {len(raw_timestamps)} candidates")

    # ── Pass 3: group frames by slide page, pick the most informative frame ──
    #
    # Multiple scene-change frames may come from the same slide page (e.g.
    # incremental bullet reveals, animations, cursor movements).  We cluster
    # consecutive frames whose perceptual hashes are similar (same page) and
    # keep only the frame with the highest visual information score — typically
    # the most "complete" version of that slide.
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="scene_dedup_"))

    try:
        from PIL import Image as PILImage

        # Extract candidate frames and compute hashes + info scores
        candidates: list[tuple[float, int, int]] = []  # (timestamp, hash, info_score)
        print(f"  Deduplicating {len(raw_timestamps)} candidates via perceptual hash...")

        for ts in raw_timestamps[:MAX_FRAMES * 2]:
            png = tmp_dir / f"cand_{ts:.1f}.png"
            subprocess.run(
                [ff_bin, "-ss", f"{ts:.3f}", "-i", str(video_path),
                 "-frames:v", "1", "-q:v", "3", str(png), "-y"],
                capture_output=True, timeout=30,
                creationflags=_SUBPROCESS_FLAGS,
            )
            if not png.exists():
                continue

            img = PILImage.open(png)
            h = _perceptual_hash(img)
            score = _information_score(img)
            candidates.append((ts, h, score))
            png.unlink()  # free disk space

        # Group consecutive frames into slide pages.  A new page starts when
        # the frame differs from BOTH the group anchor (first frame) and the
        # previous frame by >= PAGE_SIMILARITY_THRESHOLD bits.
        groups: list[list[tuple[float, int, int]]] = []
        current: list[tuple[float, int, int]] = []

        for cand in candidates:
            ts, h, score = cand
            if not current:
                current.append(cand)
            else:
                anchor_h = current[0][1]
                prev_h   = current[-1][1]
                # Same page if similar to anchor OR similar to previous frame
                if (_hamming(h, anchor_h) < PAGE_SIMILARITY_THRESHOLD or
                        _hamming(h, prev_h) < PAGE_SIMILARITY_THRESHOLD):
                    current.append(cand)
                else:
                    groups.append(current)
                    current = [cand]
        if current:
            groups.append(current)

        # From each page group, pick the frame with the highest info score
        unique_timestamps: list[float] = []
        for grp in groups:
            best = max(grp, key=lambda c: c[2])  # highest info score
            unique_timestamps.append(best[0])

        print(f"  {len(candidates)} candidates → {len(groups)} slide page(s) "
              f"(threshold={PAGE_SIMILARITY_THRESHOLD})")

    except ImportError:
        print("  [warn] PIL not available — skipping deduplication")
        unique_timestamps = raw_timestamps

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not unique_timestamps:
        unique_timestamps = [0.0]

    return unique_timestamps[:MAX_FRAMES]


def get_video_duration(video_path: Path) -> float:
    """Return video duration in seconds.

    Uses ffprobe when available, falls back to parsing ffmpeg's stderr
    ('Duration: HH:MM:SS.ss'). The fallback is the path macOS users hit
    when they only have the imageio-ffmpeg wheel — which bundles ffmpeg
    but not ffprobe.
    """
    ffprobe = _resolve_ffprobe()
    if ffprobe:
        result = subprocess.run(
            [ffprobe, "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=30,
            creationflags=_SUBPROCESS_FLAGS,
        )
        try:
            return float(result.stdout.strip())
        except (ValueError, AttributeError):
            return 0.0

    try:
        ff = _resolve_ffmpeg()
    except RuntimeError:
        return 0.0
    result = subprocess.run(
        [ff, "-hide_banner", "-i", str(video_path)],
        capture_output=True, text=True, timeout=30,
        creationflags=_SUBPROCESS_FLAGS,
    )
    dur = _parse_ffmpeg_duration(result.stderr)
    return dur if dur is not None else 0.0


# ── Per-frame screen/camera classification ────────────────────────────────────

def _is_screen_frame(img) -> bool:
    """Classify a single frame as screen recording (True) or camera (False).

    Uses the same heuristics as classify_video but on a single image.
    Screen recordings have: sharp edges, uniform backgrounds, high brightness.
    Camera shots have: natural colors, gradients, low uniformity.
    """
    import statistics
    w, h = img.size
    pixels = _get_pixels(img)

    # Edge density
    row_diffs = 0
    for y in range(0, h - 1, 4):
        for x in range(0, w - 1, 4):
            idx = y * w + x
            idx_r = idx + 1
            if idx_r < len(pixels):
                diff = sum(abs(a - b) for a, b in zip(pixels[idx], pixels[idx_r]))
                if diff > 100:
                    row_diffs += 1
    total_samples = (h // 4) * (w // 4)
    edge_ratio = row_diffs / max(total_samples, 1)

    # Color uniformity (32px blocks)
    block_size = 32
    uniform_blocks = 0
    total_blocks = 0
    for by in range(0, h - block_size, block_size):
        for bx in range(0, w - block_size, block_size):
            block = img.crop((bx, by, bx + block_size, by + block_size))
            bp = _get_pixels(block)
            r_vals = [p[0] for p in bp]
            g_vals = [p[1] for p in bp]
            b_vals = [p[2] for p in bp]
            total_blocks += 1
            if (statistics.stdev(r_vals) < 15 and
                statistics.stdev(g_vals) < 15 and
                statistics.stdev(b_vals) < 15):
                uniform_blocks += 1
    uniformity = uniform_blocks / max(total_blocks, 1)

    # Brightness
    brightness = [sum(p) / 3 for p in pixels[::16]]
    bright_ratio = sum(1 for b in brightness if b > 200) / len(brightness)

    # Screen recordings: high brightness (>70%) + high uniformity (>55%)
    # because slides have clean solid backgrounds filling the entire frame.
    # Camera shots: even with a white projection screen, overall brightness
    # is lower (<50%) due to dark surroundings (curtains, floor, equipment).
    return (bright_ratio > 0.70 and uniformity > 0.55)


def filter_camera_frames(frame_map: dict[int, Path],
                         timestamps: list[float]) -> tuple[dict[int, Path], list[float]]:
    """Remove camera frames from extracted frames, keeping only screen recordings.

    Returns filtered (frame_map, timestamps).
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        return frame_map, timestamps

    kept_map: dict[int, Path] = {}
    kept_ts: list[float] = []
    removed = 0

    for idx in sorted(frame_map.keys()):
        png = frame_map[idx]
        try:
            img = PILImage.open(png).convert("RGB")
            if _is_screen_frame(img):
                kept_map[idx] = png
                kept_ts.append(timestamps[idx])
            else:
                removed += 1
                png.unlink(missing_ok=True)  # delete camera frame
        except Exception:
            kept_map[idx] = png  # keep on error
            kept_ts.append(timestamps[idx])

    if removed:
        print(f"  Filtered out {removed} camera frame(s), kept {len(kept_map)} screen frame(s)")

        # Renumber frames sequentially
        new_map: dict[int, Path] = {}
        for new_idx, old_idx in enumerate(sorted(kept_map.keys())):
            old_path = kept_map[old_idx]
            new_path = old_path.parent / f"frame_{new_idx + 1:03d}.png"
            if old_path != new_path:
                old_path.rename(new_path)
            new_map[new_idx] = new_path
        return new_map, kept_ts

    return frame_map, timestamps


def _is_blank_frame(img) -> bool:
    """Detect mostly-black or mostly-white frames (filler/transition frames).

    Samples pixels on a coarse grid; a frame is "blank" if >95% of samples are
    either very dark (<15) or very bright (>240).
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        return False
    small = img.convert("L").resize((64, 36), PILImage.LANCZOS)
    pixels = _get_pixels(small)
    total = len(pixels)
    if total == 0:
        return False
    dark = sum(1 for p in pixels if p < 15)
    bright = sum(1 for p in pixels if p > 240)
    return (dark / total > 0.95) or (bright / total > 0.95)


def _describe_frames(frame_dir: Path, frame_map: dict[int, Path]) -> list[int]:
    """Use GPT-4o-mini vision to describe each frame's content.
    Saves descriptions to a .image_cache.json file next to the frame dir.
    """
    cache_file = frame_dir / "image_cache.json"
    cache = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
        except Exception:
            pass

    # Use the ImageDescriber from semantic_alignment
    try:
        from semantic_alignment import ImageDescriber
        from PIL import Image as PILImage
    except ImportError:
        return

    describer = ImageDescriber()
    described = 0
    junked: list[int] = []
    for idx in sorted(frame_map.keys()):
        key = f"page_{idx}"
        if key in cache:
            if _JUNK_DESC_RE.search(cache[key]):
                junked.append(idx)
            continue
        try:
            img = PILImage.open(frame_map[idx]).convert("RGB")
            # Cheap pre-check: near-black or near-white frames carry no info
            # and often trigger vision-API refusals.  Skip them outright.
            if _is_blank_frame(img):
                junked.append(idx)
                continue
            desc = describer.describe_slide_image(img)
            if not desc:
                continue
            if _JUNK_DESC_RE.search(desc):
                junked.append(idx)
                continue
            cache[key] = desc
            described += 1
        except Exception:
            continue

    if described:
        cache_file.write_text(json.dumps(cache, indent=2))
        print(f"  Described {described} frame(s) via vision API")
    return junked


def _drop_junk_and_renumber(
    frame_map: dict[int, Path],
    timestamps: list[float],
    junk_indices: list[int],
    frame_dir: Path,
) -> tuple[dict[int, Path], list[float]]:
    """Delete junk frames, update cache, renumber remaining frames contiguously."""
    if not junk_indices:
        return frame_map, timestamps

    junk_set = set(junk_indices)
    for idx in junk_indices:
        p = frame_map.get(idx)
        if p:
            p.unlink(missing_ok=True)

    # Rebuild map + timestamps preserving only non-junk, then renumber.
    kept_pairs = [(i, frame_map[i]) for i in sorted(frame_map) if i not in junk_set]
    kept_ts = [timestamps[i] for i in sorted(frame_map) if i not in junk_set and i < len(timestamps)]

    cache_file = frame_dir / "image_cache.json"
    cache = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
        except Exception:
            pass

    new_map: dict[int, Path] = {}
    new_cache: dict = {}
    for new_idx, (old_idx, old_path) in enumerate(kept_pairs):
        new_path = old_path.parent / f"frame_{new_idx + 1:03d}.png"
        if old_path != new_path and old_path.exists():
            old_path.rename(new_path)
        new_map[new_idx] = new_path
        old_key = f"page_{old_idx}"
        if old_key in cache:
            new_cache[f"page_{new_idx}"] = cache[old_key]

    cache_file.write_text(json.dumps(new_cache, indent=2))
    print(f"  Filtered {len(junk_indices)} junk frame(s); {len(new_map)} remain")
    return new_map, kept_ts


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
            _resolve_ffmpeg(), "-ss", f"{ts:.3f}",
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
    stream_tag: str = "",
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

    # Step 1: Detect unique scenes (scene filter + periodic fallback + dedup)
    print("  Detecting scene changes...")
    timestamps = detect_scenes(video_path, stream_tag=stream_tag)

    # Step 2: Extract frames
    print(f"  Extracting {len(timestamps)} frames...")
    frame_map = extract_frames(video_path, timestamps, frame_dir)
    if not frame_map:
        print("  [error] No frames extracted")
        return None, None

    # Note: per-frame camera filtering is disabled. The video-level classifier
    # already decided this is a screen recording. Slides can have any background
    # color (dark, light, colored), so brightness-based filtering is unreliable.

    # Step 2c: Describe each frame using GPT-4o-mini vision so the note LLM
    # knows what each frame actually shows (diagrams, text, charts, etc.)
    junk = _describe_frames(frame_dir, frame_map)

    # Step 2d: Drop junk frames (desktop wallpapers, loading screens, vision
    # refusals) and renumber the remaining frames + timestamps contiguously.
    if junk:
        frame_map, timestamps = _drop_junk_and_renumber(
            frame_map, timestamps, junk, frame_dir
        )

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

        # Skip if already has frame-based alignment
        if align_file.exists():
            try:
                with open(align_file, encoding="utf-8") as _af:
                    if json.load(_af).get("source") == "screenshare":
                        print(f"  [skip] Already extracted: {video_stem}")
                        processed += 1
                        continue
                # Existing alignment is slide-based — re-extract frames
            except Exception:
                pass

        # Find matching caption
        caption_path = course_dir / "captions" / f"{video_stem}.json"
        if not caption_path.exists():
            print(f"  [skip] No caption for: {video_stem} (transcribe first)")
            continue

        # For SS/OBJECT-tagged streams, skip classification (known screen content)
        tag = entry.get("stream_tag", "").upper()
        skip_classify = (tag in ("SS", "OBJECT"))
        result = extract_and_align(video_path, caption_path, course_dir,
                                   skip_classify=skip_classify,
                                   stream_tag=tag)
        if result[0] is not None:
            processed += 1
            # Record classification in manifest so the app knows
            entry["video_type"] = "screen"
        else:
            entry["video_type"] = "camera"
        # Save manifest after each classification
        with open(manifest_file, "w", encoding="utf-8") as _mf:
            json.dump(manifest, _mf, indent=2)

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
