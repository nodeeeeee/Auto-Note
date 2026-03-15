"""
Caption Extraction Pipeline
For each downloaded video:
  1. Transcribe with faster-whisper large-v3 (GPU) if available,
     otherwise fall back to OpenAI Whisper API (whisper-1).
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
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_DIR   = Path(__file__).parent
MANIFEST_FILE = PROJECT_DIR / "manifest.json"

# ── Tunable constants ─────────────────────────────────────────────────────────

WHISPER_MODEL_SIZE = "large-v3"
WHISPER_BEAM_SIZE  = 5
WHISPER_LANGUAGE   = None       # None = auto-detect

# Audio chunk size for OpenAI API (minutes). 60 min @ 32 kbps mono ≈ 14 MB,
# well under the 25 MB per-request limit.
_API_CHUNK_MINUTES = 60

# OpenAI API key: read from openai_api.txt or OPENAI_API_KEY env var
_openai_key_file = PROJECT_DIR / "openai_api.txt"
_OPENAI_API_KEY  = (
    _openai_key_file.read_text().strip()
    if _openai_key_file.exists() else
    os.environ.get("OPENAI_API_KEY", "")
)

# Full language-name → ISO 639-1 code (OpenAI returns full names)
_LANG_NAMES = {
    "afrikaans": "af", "arabic": "ar", "armenian": "hy", "azerbaijani": "az",
    "belarusian": "be", "bosnian": "bs", "bulgarian": "bg", "catalan": "ca",
    "chinese": "zh", "croatian": "hr", "czech": "cs", "danish": "da",
    "dutch": "nl", "english": "en", "estonian": "et", "finnish": "fi",
    "french": "fr", "galician": "gl", "german": "de", "greek": "el",
    "hebrew": "he", "hindi": "hi", "hungarian": "hu", "icelandic": "is",
    "indonesian": "id", "italian": "it", "japanese": "ja", "kannada": "kn",
    "kazakh": "kk", "korean": "ko", "latvian": "lv", "lithuanian": "lt",
    "macedonian": "mk", "malay": "ms", "marathi": "mr", "maori": "mi",
    "nepali": "ne", "norwegian": "no", "persian": "fa", "polish": "pl",
    "portuguese": "pt", "romanian": "ro", "russian": "ru", "serbian": "sr",
    "slovak": "sk", "slovenian": "sl", "spanish": "es", "swahili": "sw",
    "swedish": "sv", "tagalog": "tl", "tamil": "ta", "thai": "th",
    "turkish": "tr", "ukrainian": "uk", "urdu": "ur", "vietnamese": "vi",
    "welsh": "cy",
}


# ── GPU helpers (only used by local backend) ──────────────────────────────────

def require_gpu() -> None:
    import torch
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
    import torch
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


# ── OpenAI API helpers ────────────────────────────────────────────────────────

def _video_duration(video_path: Path) -> float:
    """Return video duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def _extract_audio(video_path: Path, out_path: Path,
                   start: float = 0.0, duration: float | None = None) -> None:
    """Extract a low-bitrate mono mp3 clip suitable for the OpenAI API."""
    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    if start > 0:
        cmd += ["-ss", str(start)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-vn", "-ar", "16000", "-ac", "1", "-b:a", "32k", str(out_path)]
    subprocess.run(cmd, capture_output=True, check=True)


def _api_segments_to_schema(api_segs: list, time_offset: float = 0.0) -> list:
    """Convert OpenAI verbose_json segments to our internal schema."""
    out = []
    for seg in api_segs:
        words = []
        for w in (seg.get("words") or []):
            words.append({
                "word":  w["word"],
                "start": round(w["start"] + time_offset, 3),
                "end":   round(w["end"]   + time_offset, 3),
                "prob":  round(w.get("probability", 1.0), 3),
            })
        out.append({
            "id":    seg["id"],
            "start": round(seg["start"] + time_offset, 3),
            "end":   round(seg["end"]   + time_offset, 3),
            "text":  seg["text"].strip(),
            "words": words,
        })
    return out


def transcribe_api(video_path: Path, caption_path: Path) -> bool:
    """
    Transcribe using OpenAI Whisper API (whisper-1).
    Extracts audio as 32 kbps mono mp3 in chunks ≤ _API_CHUNK_MINUTES,
    calls the API per chunk, then merges results.
    """
    if caption_path.exists():
        print(f"  [skip] Caption already exists: {caption_path.name}")
        return True

    if not _OPENAI_API_KEY:
        print("  [error] No OpenAI API key found. "
              "Set openai_api.txt or OPENAI_API_KEY env var.")
        return False

    try:
        from openai import OpenAI
    except ImportError:
        print("  [error] 'openai' package not installed. Run: pip install openai")
        return False

    client = OpenAI(api_key=_OPENAI_API_KEY)

    print(f"  Transcribing via OpenAI Whisper API: {video_path.name}")
    total_dur = _video_duration(video_path)
    print(f"  Duration: {total_dur:.0f}s")

    chunk_sec = _API_CHUNK_MINUTES * 60
    offsets   = [i * chunk_sec for i in range(int(total_dur // chunk_sec) + 1)
                 if i * chunk_sec < total_dur]

    all_segments: list = []
    detected_lang = "en"
    lang_prob     = 1.0

    with tempfile.TemporaryDirectory() as tmp:
        for i, start in enumerate(offsets):
            dur = min(chunk_sec, total_dur - start)
            audio_file = Path(tmp) / f"chunk_{i:03d}.mp3"

            print(f"  Extracting audio chunk {i+1}/{len(offsets)} "
                  f"({start:.0f}s – {start+dur:.0f}s)...")
            _extract_audio(video_path, audio_file, start=start, duration=dur)

            print(f"  Calling Whisper API (chunk {i+1}/{len(offsets)})...")
            with open(audio_file, "rb") as f:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"],
                    language=WHISPER_LANGUAGE,   # None = auto-detect
                )

            # First chunk determines language
            if i == 0:
                lang_full    = getattr(response, "language", "english") or "english"
                detected_lang = _LANG_NAMES.get(lang_full.lower(),
                                                 lang_full[:2].lower())
                print(f"  Language: {lang_full} ({detected_lang})")

            segs = _api_segments_to_schema(
                response.model_dump().get("segments", []),
                time_offset=start,
            )
            # Re-number segment IDs to be globally unique
            base_id = len(all_segments)
            for j, s in enumerate(segs):
                s["id"] = base_id + j
            all_segments.extend(segs)

    result = {
        "language":             detected_lang,
        "language_probability": lang_prob,
        "duration":             round(total_dur, 3),
        "segments":             all_segments,
    }

    caption_path.parent.mkdir(parents=True, exist_ok=True)
    with open(caption_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    n_seg  = len(result["segments"])
    n_word = sum(len(s["words"]) for s in result["segments"])
    print(f"  Saved: {n_seg} segments / {n_word} words -> {caption_path}")
    return True


# ── Local faster-whisper backend ──────────────────────────────────────────────

def transcribe_local(video_path: Path, caption_path: Path) -> bool:
    """
    Transcribe a video file directly with faster-whisper large-v3 on GPU.
    faster-whisper uses ffmpeg internally to decode audio from the video.
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


# ── Dispatcher ────────────────────────────────────────────────────────────────

def _local_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def transcribe(video_path: Path, caption_path: Path) -> bool:
    """
    Auto-select backend: use local faster-whisper if available (GPU + package
    installed), otherwise fall back to OpenAI Whisper API.
    """
    if _local_available():
        require_gpu()
        return transcribe_local(video_path, caption_path)
    else:
        print("  [info] faster-whisper not available — using OpenAI Whisper API.")
        return transcribe_api(video_path, caption_path)


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
