"""
Caption Extraction Pipeline
For each downloaded video:
  1. Transcribe directly with faster-whisper large-v3 (GPU)
     faster-whisper accepts video files natively via ffmpeg decoding.
  2. Save word-level timestamped JSON to [course_id]/captions/

Directory layout:
  [project]/[course_id]/videos/[title].mp4    <- input
  [project]/[course_id]/captions/[title].json <- final transcript

Usage:
  python extract_caption.py              # process all pending videos
  python extract_caption.py --video PATH # process a single video file
"""

import gc
import json
import sys
from pathlib import Path

import torch

PROJECT_DIR   = Path(__file__).parent
MANIFEST_FILE = PROJECT_DIR / "manifest.json"

# ── Tunable constants ─────────────────────────────────────────────────────────

WHISPER_MODEL_SIZE = "large-v3"
WHISPER_BEAM_SIZE  = 5
WHISPER_LANGUAGE   = None       # None = auto-detect


# ── GPU enforcement ───────────────────────────────────────────────────────────

def require_gpu() -> None:
    """
    Called once at program startup.
    Raises RuntimeError if no CUDA GPU is available so the pipeline never
    falls back to CPU silently.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. This pipeline requires a CUDA-capable GPU.\n"
            "Verify that 'nvidia-smi' works and PyTorch was installed with CUDA support."
        )
    name  = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
    free  = torch.cuda.mem_get_info(0)[0] // (1024 ** 2)
    print(f"[GPU] {name}  total={total} MiB  free={free} MiB")


def free_gpu() -> None:
    """Release all unreferenced CUDA tensors and return VRAM to the pool."""
    gc.collect()
    torch.cuda.empty_cache()


# ── Manifest helpers ──────────────────────────────────────────────────────────

def load_manifest() -> dict:
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict) -> None:
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


# ── Transcribe ────────────────────────────────────────────────────────────────

def transcribe(video_path: Path, caption_path: Path) -> bool:
    """
    Transcribe a video file directly with faster-whisper large-v3 on GPU.
    faster-whisper uses ffmpeg internally to decode audio from the video.
    A tqdm progress bar tracks segments as they are decoded.
    """
    if caption_path.exists():
        print(f"  [skip] Caption already exists: {caption_path.name}")
        return True

    from faster_whisper import WhisperModel
    from tqdm import tqdm

    device, compute_type = "cuda", "float16"
    print(f"  Loading Whisper {WHISPER_MODEL_SIZE} ({device}/{compute_type})...")
    model = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)

    print(f"  Transcribing: {video_path.name}")
    segments_gen, info = model.transcribe(
        str(video_path),
        beam_size=WHISPER_BEAM_SIZE,
        language=WHISPER_LANGUAGE,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    print(f"  Language: {info.language} (p={info.language_probability:.2f}), "
          f"duration: {info.duration:.0f}s")

    result = {
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration": round(info.duration, 3),
        "segments": [],
    }

    total_duration = info.duration or 1.0
    bar = tqdm(
        total=int(total_duration),
        unit="s",
        unit_scale=False,
        desc="  transcribing",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total}s [{elapsed}<{remaining}]",
        dynamic_ncols=True,
    )
    last_pos = 0

    for seg in segments_gen:
        words = []
        if seg.words:
            for w in seg.words:
                words.append({
                    "word":  w.word,
                    "start": round(w.start, 3),
                    "end":   round(w.end, 3),
                    "prob":  round(w.probability, 3),
                })
        result["segments"].append({
            "id":    seg.id,
            "start": round(seg.start, 3),
            "end":   round(seg.end, 3),
            "text":  seg.text.strip(),
            "words": words,
        })
        advance = int(seg.end) - last_pos
        if advance > 0:
            bar.update(advance)
            last_pos = int(seg.end)

    bar.update(int(total_duration) - last_pos)
    bar.close()

    del model
    free_gpu()

    caption_path.parent.mkdir(parents=True, exist_ok=True)
    with open(caption_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    n_seg  = len(result["segments"])
    n_word = sum(len(s["words"]) for s in result["segments"])
    print(f"  Saved: {n_seg} segments / {n_word} words -> {caption_path}")
    return True


# ── Full pipeline for one video ───────────────────────────────────────────────

def process_video(video_path: Path, manifest: dict, manifest_key: str | None) -> bool:
    video_path   = Path(video_path)
    course_dir   = video_path.parent.parent     # [project]/[course_id]/
    caption_path = course_dir / "captions" / f"{video_path.stem}.json"

    if caption_path.exists():
        print(f"  [skip] Already captioned: {video_path.name}")
        if manifest_key and manifest_key in manifest:
            manifest[manifest_key]["caption"] = str(caption_path)
        return True

    print(f"\n{'='*70}")
    print(f"Processing: {video_path.name}")
    print(f"{'='*70}")

    if not transcribe(video_path, caption_path):
        return False

    if manifest_key and manifest_key in manifest:
        manifest[manifest_key]["caption"] = str(caption_path)
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def get_pending(manifest: dict) -> list[tuple[str, str]]:
    """Return (key, video_path) for downloaded videos not yet captioned."""
    pending = []
    for key, entry in manifest.items():
        if entry.get("status") != "done":
            continue
        vpath = entry.get("path")
        if not vpath or not Path(vpath).exists():
            continue
        caption = Path(vpath).parent.parent / "captions" / f"{Path(vpath).stem}.json"
        if not caption.exists():
            pending.append((key, vpath))
    return pending


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Extract captions from Canvas lecture videos")
    parser.add_argument("--video", metavar="PATH",
                        help="Process a single video file (ignores manifest)")
    args = parser.parse_args()

    require_gpu()

    manifest = load_manifest()

    if args.video:
        vp = Path(args.video)
        if not vp.exists():
            print(f"[error] File not found: {vp}")
            sys.exit(1)
        process_video(vp, manifest, manifest_key=None)
        save_manifest(manifest)
        return

    pending = get_pending(manifest)
    if not pending:
        print("All videos already captioned (or none downloaded).")
        done = [(k, v) for k, v in manifest.items()
                if v.get("status") == "done" and v.get("caption")]
        if done:
            print(f"\nCaptioned ({len(done)}):")
            for k, v in done:
                print(f"  {Path(v['path']).name}  ->  {Path(v['caption']).name}")
        return

    print(f"Found {len(pending)} video(s) to caption.\n")
    ok = 0
    for key, vpath in pending:
        if process_video(Path(vpath), manifest, key):
            ok += 1
        save_manifest(manifest)

    print(f"\nDone: {ok}/{len(pending)} videos captioned successfully.")


if __name__ == "__main__":
    main()
