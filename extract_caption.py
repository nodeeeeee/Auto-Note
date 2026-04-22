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

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Prevent console windows flashing on Windows when spawning ffprobe/ffmpeg
_SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

PROJECT_DIR   = Path(__file__).parent
_AUTO_NOTE_DIR = Path.home() / ".auto_note"
if os.environ.get("AUTONOTE_DATA_DIR"):
    DATA_DIR = Path(os.environ["AUTONOTE_DATA_DIR"])
elif getattr(sys, "frozen", False) or PROJECT_DIR == _AUTO_NOTE_DIR / "scripts":
    DATA_DIR = _AUTO_NOTE_DIR
else:
    DATA_DIR = PROJECT_DIR
MANIFEST_FILE = DATA_DIR / "manifest.json"

# ── Tunable constants ─────────────────────────────────────────────────────────

# Transcription backend: "auto" | "gpu" | "api"
# "auto"  → use GPU only if faster-whisper is installed AND VRAM ≥ 16 GB
# "gpu"   → always attempt GPU (fall back to API if unavailable)
# "api"   → always use OpenAI Whisper API
WHISPER_BACKEND   = os.environ.get("AUTONOTE_WHISPER_BACKEND", "auto")
VRAM_THRESHOLD_MIB = 15 * 1024   # ~15 GB nominal — accepts 16 GB cards (which report ~15.4 GB)
FORCE_REGEN       = False        # set via --force CLI flag

WHISPER_MODEL_SIZE = "large-v3"
WHISPER_BEAM_SIZE  = 5
WHISPER_LANGUAGE   = None       # None = auto-detect

# Normalise: the Settings UI stores "None" as the bare Python keyword, but guard
# against old scripts that may have written the string "None" or empty string.
if WHISPER_LANGUAGE in ("None", "", "null"):
    WHISPER_LANGUAGE = None

# Audio chunk size for OpenAI API (minutes). 20 min @ 32 kbps mono ≈ 4.8 MB,
# well under the 25 MB per-request limit with a comfortable safety margin.
_API_CHUNK_MINUTES = 20

# OpenAI API key: read from openai_api.txt or OPENAI_API_KEY env var
_openai_key_file = DATA_DIR / "openai_api.txt"
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


# ── ffmpeg/ffprobe resolution ─────────────────────────────────────────────────

def _resolve_ffmpeg() -> str:
    """Locate an ffmpeg executable.

    Order: system PATH → imageio-ffmpeg bundled binary → auto-install
    imageio-ffmpeg and retry. Raises RuntimeError if none are available.
    Mirrors downloader._resolve_ffmpeg for consistency.
    """
    from shutil import which
    sys_ff = which("ffmpeg")
    if sys_ff:
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
    return ff


def _resolve_ffprobe() -> str | None:
    """Locate ffprobe if available on system PATH.

    Returns None if ffprobe isn't installed — callers should fall back to
    parsing ffmpeg output. imageio-ffmpeg does not bundle ffprobe, so there
    is no Python-package fallback for it.
    """
    from shutil import which
    return which("ffprobe")


# ── OpenAI API helpers ────────────────────────────────────────────────────────

_DURATION_RE = None


def _parse_ffmpeg_duration(stderr: str) -> float | None:
    """Parse 'Duration: HH:MM:SS.ss' from ffmpeg -i stderr output."""
    global _DURATION_RE
    if _DURATION_RE is None:
        import re
        _DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
    m = _DURATION_RE.search(stderr or "")
    if not m:
        return None
    h, mi, s = m.group(1), m.group(2), m.group(3)
    return int(h) * 3600 + int(mi) * 60 + float(s)


def _video_duration(video_path: Path) -> float:
    """Return video duration in seconds.

    Prefers ffprobe (faster, structured output), falls back to parsing
    ffmpeg's stderr when ffprobe isn't installed — which is common on
    macOS users who get ffmpeg via imageio-ffmpeg but have no separate
    ffprobe binary.
    """
    ffprobe = _resolve_ffprobe()
    if ffprobe:
        result = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json",
             "-show_format", str(video_path)],
            capture_output=True, text=True, check=True,
            creationflags=_SUBPROCESS_FLAGS,
        )
        return float(json.loads(result.stdout)["format"]["duration"])

    # Fallback: ffmpeg prints "Duration: HH:MM:SS.ss" to stderr when
    # given -i with no output. Exit code is non-zero (no output file)
    # so we don't use check=True here.
    ff = _resolve_ffmpeg()
    result = subprocess.run(
        [ff, "-hide_banner", "-i", str(video_path)],
        capture_output=True, text=True, check=False,
        creationflags=_SUBPROCESS_FLAGS,
    )
    dur = _parse_ffmpeg_duration(result.stderr)
    if dur is None:
        raise RuntimeError(
            f"Could not determine duration of {video_path.name} — "
            f"ffmpeg output did not contain a Duration line."
        )
    return dur


def _extract_audio(video_path: Path, out_path: Path,
                   start: float = 0.0, duration: float | None = None,
                   desc: str | None = None, total_sec: float | None = None) -> None:
    """Extract a low-bitrate mono mp3 clip suitable for the OpenAI API.

    When *desc* is given, a tqdm progress bar is shown via ffmpeg-progress-yield.
    *total_sec* (or *duration*) is used as the known duration for the bar.
    """
    ff = _resolve_ffmpeg()
    cmd = [ff, "-y", "-i", str(video_path)]
    if start > 0:
        cmd += ["-ss", str(start)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-vn", "-ar", "16000", "-ac", "1", "-b:a", "32k", str(out_path)]

    if desc:
        dur = duration or total_sec
        try:
            from ffmpeg_progress_yield import FfmpegProgress
            from tqdm import tqdm
            ff  = FfmpegProgress(cmd)
            bar = tqdm(
                total=100, unit="%", desc=desc,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]",
                dynamic_ncols=True,
            )
            last_pct = 0
            for pct in ff.run_command_with_progress(
                popen_kwargs={"creationflags": _SUBPROCESS_FLAGS},
                duration_override=dur,
            ):
                p = int(pct)
                if p > last_pct:
                    bar.update(p - last_pct)
                    last_pct = p
            bar.update(100 - last_pct)
            bar.close()
            return
        except Exception:
            pass  # fall through to silent mode
    subprocess.run(cmd, capture_output=True, check=True, creationflags=_SUBPROCESS_FLAGS)


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


def _filter_api_segments(api_segs: list) -> tuple[list, int]:
    """
    Remove hallucinated or silent segments returned by the OpenAI Whisper API.
    Uses the same thresholds as the local faster-whisper backend.
    Returns (filtered_list, n_dropped).
    """
    good = []
    dropped = 0
    for seg in api_segs:
        text = (seg.get("text") or "").strip()
        if not text:
            dropped += 1
            continue
        if seg.get("no_speech_prob", 0.0) > 0.6:
            dropped += 1
            continue
        if seg.get("compression_ratio", 1.0) > 2.4:
            dropped += 1
            continue
        good.append(seg)
    return good, dropped


def transcribe_api(video_path: Path, caption_path: Path) -> bool:
    """
    Transcribe using OpenAI Whisper API (whisper-1).
    Step 1 — extract full audio from video to [course_dir]/audio/[stem].mp3
             (reused on retry if already present).
    Step 2 — split audio into ≤ _API_CHUNK_MINUTES chunks and call the API.
    """
    import time as _time

    if not FORCE_REGEN and caption_path.exists():
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

    # ── Step 1: extract full audio ────────────────────────────────────────────
    audio_dir  = video_path.parent.parent / "audio"
    audio_path = audio_dir / (video_path.stem + ".mp3")
    audio_dir.mkdir(parents=True, exist_ok=True)

    total_dur = _video_duration(video_path)

    if audio_path.exists():
        print(f"  Audio already extracted: {audio_path.name}  ({total_dur:.0f}s)")
    else:
        print(f"  Extracting audio from video ({total_dur:.0f}s)...")
        _extract_audio(video_path, audio_path,
                       desc="  extracting audio", total_sec=total_dur)
        size_mb = audio_path.stat().st_size / (1024 ** 2)
        print(f"  Audio saved: {audio_path.name} ({size_mb:.1f} MB)")

    # ── Step 2: chunk & transcribe ────────────────────────────────────────────
    chunk_sec = _API_CHUNK_MINUTES * 60
    offsets   = [i * chunk_sec for i in range(int(total_dur // chunk_sec) + 1)
                 if i * chunk_sec < total_dur]
    n_chunks  = len(offsets)

    print(f"  Transcribing via Whisper API: {n_chunks} chunk(s) × "
          f"≤{_API_CHUNK_MINUTES} min  (total {total_dur:.0f}s)")

    all_segments: list = []
    detected_lang: str | None = WHISPER_LANGUAGE
    lang_prob     = 1.0
    total_dropped = 0

    with tempfile.TemporaryDirectory() as tmp:
        # ── Language detection: probe from mid-audio ─────────────────────────
        # Whisper's auto-detect can misidentify accented English as another
        # language (e.g. Malay, Welsh).  We probe from the middle of the
        # recording (avoiding intro music / silence) and, when the result is
        # not English, re-probe with language="en" and compare avg_logprob.
        # The model's own confidence reliably distinguishes correct from
        # hallucinated transcriptions.
        if not WHISPER_LANGUAGE:
            _PROBE_SEC = 30
            probe_start = max(0, total_dur / 2 - _PROBE_SEC / 2)
            probe_dur   = min(_PROBE_SEC, total_dur - probe_start)
            probe_file  = Path(tmp) / "lang_probe.mp3"
            print(f"  Detecting language from mid-audio "
                  f"({probe_start:.0f}s–{probe_start + probe_dur:.0f}s)...",
                  end="", flush=True)
            _extract_audio(audio_path, probe_file,
                           start=probe_start, duration=probe_dur)
            with open(probe_file, "rb") as f:
                probe_resp = client.audio.transcriptions.create(
                    model="whisper-1", file=f,
                    response_format="verbose_json",
                )
            lang_full     = getattr(probe_resp, "language", "english") or "english"
            detected_lang = _LANG_NAMES.get(lang_full.lower(), lang_full[:2].lower())
            print(f"  '{detected_lang}'", end="", flush=True)

            # If auto-detect chose a non-English language, verify by comparing
            # model confidence (avg_logprob) between the two.
            if detected_lang != "en":
                auto_segs = (probe_resp.model_dump().get("segments") or [])
                auto_lp   = (sum(s.get("avg_logprob", 0) for s in auto_segs)
                             / max(len(auto_segs), 1))
                with open(probe_file, "rb") as f:
                    en_resp = client.audio.transcriptions.create(
                        model="whisper-1", file=f,
                        response_format="verbose_json",
                        language="en",
                    )
                en_segs = (en_resp.model_dump().get("segments") or [])
                en_lp   = (sum(s.get("avg_logprob", 0) for s in en_segs)
                           / max(len(en_segs), 1))
                if en_lp > auto_lp:
                    print(f" → English wins (en={en_lp:.3f} vs {detected_lang}={auto_lp:.3f})")
                    detected_lang = "en"
                else:
                    print(f" → confirmed (en={en_lp:.3f} vs {detected_lang}={auto_lp:.3f})")

        print(f"  Using language: '{detected_lang}'")

        # ── Transcribe chunks ────────────────────────────────────────────────
        for i, start in enumerate(offsets):
            dur        = min(chunk_sec, total_dur - start)
            chunk_file = Path(tmp) / f"chunk_{i:03d}.mp3"

            print(f"  Chunk {i+1}/{n_chunks}: {start:.0f}s – {start+dur:.0f}s  "
                  f"extracting...", end="", flush=True)
            _extract_audio(audio_path, chunk_file, start=start, duration=dur)
            chunk_mb = chunk_file.stat().st_size / (1024 ** 2)

            print(f"  {chunk_mb:.1f} MB  "
                  f"sending to API...", end="", flush=True)

            t0 = _time.monotonic()
            with open(chunk_file, "rb") as f:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"],
                    language=detected_lang,
                )
            elapsed = _time.monotonic() - t0

            resp_dict = response.model_dump()
            api_segs  = resp_dict.get("segments", [])
            api_words = resp_dict.get("words", [])

            # The API returns words at the top level, not nested inside segments.
            # Distribute word-level timestamps into their parent segment.
            if api_words:
                w_idx = 0
                for seg in api_segs:
                    seg["words"] = []
                    while w_idx < len(api_words) and api_words[w_idx]["start"] < seg["end"]:
                        seg["words"].append(api_words[w_idx])
                        w_idx += 1

            # Filter hallucinated / silent segments (same thresholds as local backend)
            api_segs, n_dropped = _filter_api_segments(api_segs)
            total_dropped += n_dropped

            segs = _api_segments_to_schema(api_segs, time_offset=start)
            # Re-number segment IDs to be globally unique
            base_id = len(all_segments)
            for j, s in enumerate(segs):
                s["id"] = base_id + j
            all_segments.extend(segs)

            drop_note = f"  ({n_dropped} dropped)" if n_dropped else ""
            print(f"  done in {elapsed:.0f}s  {len(segs)} segs{drop_note}")

    result = {
        "language":             detected_lang,
        "language_probability": lang_prob,
        "duration":             round(total_dur, 3),
        "segments":             all_segments,
    }

    n_seg  = len(all_segments)
    n_word = sum(len(s["words"]) for s in all_segments)

    # Quality check: flag likely wrong/empty recordings so alignment skips them
    dur = total_dur or 1.0
    wpm = (n_word / dur) * 60
    if n_word < 50 or wpm < 10:
        result["quality"] = "low"
        print(f"  [warn] Very sparse transcript ({n_word} words, {wpm:.0f} wpm) — "
              f"flagged as low quality, alignment will be skipped.")
    else:
        result["quality"] = "ok"

    caption_path.parent.mkdir(parents=True, exist_ok=True)
    with open(caption_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    drop_note = f"  ({total_dropped} hallucinated segments removed)" if total_dropped else ""
    print(f"  Saved: {n_seg} segments / {n_word} words -> {caption_path}{drop_note}")
    return True


# ── Local faster-whisper backend ──────────────────────────────────────────────

def transcribe_local(video_path: Path, caption_path: Path) -> bool:
    """
    Transcribe a video file directly with faster-whisper large-v3 on GPU.
    faster-whisper uses ffmpeg internally to decode audio from the video.
    """
    if not FORCE_REGEN and caption_path.exists():
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
        # Prevents hallucination cascades: without this, one hallucinated segment
        # is fed as context to the next, causing runaway gibberish / wrong-language output.
        condition_on_previous_text=False,
        # Discard segments where the model is not confident there is speech.
        no_speech_threshold=0.6,
        # Discard highly repetitive segments (a hallucination signature).
        compression_ratio_threshold=2.4,
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

    # Quality check: flag likely wrong/empty recordings so alignment skips them
    dur    = result["duration"] or 1.0
    wpm    = (n_word / dur) * 60
    if n_word < 50 or wpm < 10:
        result["quality"] = "low"
        print(f"  [warn] Very sparse transcript ({n_word} words, {wpm:.0f} wpm) — "
              f"flagged as low quality, alignment will be skipped.")
    else:
        result["quality"] = "ok"

    with open(caption_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  Saved: {n_seg} segments / {n_word} words -> {caption_path}")
    return True


# ── Dispatcher ────────────────────────────────────────────────────────────────

def _gpu_vram_ok() -> bool:
    """Return True if a CUDA GPU with ≥ VRAM_THRESHOLD_MIB VRAM is present."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        total_mib = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        return total_mib >= VRAM_THRESHOLD_MIB
    except Exception:
        return False


def _local_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return _gpu_vram_ok()
    except ImportError:
        return False


def transcribe(video_path: Path, caption_path: Path) -> bool:
    """
    Select backend according to WHISPER_BACKEND (env AUTONOTE_WHISPER_BACKEND):
      "auto" — GPU if faster-whisper installed and VRAM ≥ 16 GB, else API
      "gpu"  — always GPU (fall back to API if unavailable)
      "api"  — always OpenAI Whisper API
    """
    backend = WHISPER_BACKEND
    use_gpu = False
    if backend == "gpu":
        if _gpu_vram_ok():
            try:
                import faster_whisper  # noqa: F401
                use_gpu = True
            except ImportError:
                print("  [warn] GPU backend selected but faster-whisper not installed "
                      "— falling back to API.")
        else:
            print("  [warn] GPU backend selected but CUDA / VRAM check failed "
                  "— falling back to API.")
    elif backend == "auto":
        use_gpu = _local_available()
        if not use_gpu:
            print("  [info] Auto-select: faster-whisper unavailable or VRAM < 16 GB "
                  "— using OpenAI Whisper API.")
    # backend == "api": use_gpu stays False

    if use_gpu:
        require_gpu()
        return transcribe_local(video_path, caption_path)
    else:
        return transcribe_api(video_path, caption_path)


# ── Full pipeline for one video ───────────────────────────────────────────────

def process_video(video_path: Path, manifest: dict, manifest_key: str | None) -> bool:
    video_path   = Path(video_path)
    course_dir   = video_path.parent.parent     # [project]/[course_id]/
    caption_path = course_dir / "captions" / f"{video_path.stem}.json"

    if not FORCE_REGEN and caption_path.exists():
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
    """Return (key, video_path) for downloaded videos not yet captioned.

    When FORCE_REGEN is True, returns ALL downloaded videos regardless of
    whether captions already exist.
    """
    pending = []
    for key, entry in manifest.items():
        if entry.get("status") != "done":
            continue
        vpath = entry.get("path")
        if not vpath or not Path(vpath).exists():
            continue
        if FORCE_REGEN:
            pending.append((key, vpath))
        else:
            caption = Path(vpath).parent.parent / "captions" / f"{Path(vpath).stem}.json"
            if not caption.exists():
                pending.append((key, vpath))
    return pending


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Extract captions from Canvas lecture videos")
    parser.add_argument("--video", metavar="PATH",
                        help="Process a single video file (ignores manifest)")
    parser.add_argument("--force", action="store_true",
                        help="Re-transcribe all videos even if captions already exist")
    args = parser.parse_args()

    global FORCE_REGEN
    if args.force:
        FORCE_REGEN = True

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

    total = len(pending)
    print(f"Found {total} video(s) to caption.\n")
    ok = 0
    for i, (key, vpath) in enumerate(pending, 1):
        print(f"[{i}/{total}] {Path(vpath).name}")
        if process_video(Path(vpath), manifest, key):
            ok += 1
        save_manifest(manifest)

    print(f"\nDone: {ok}/{len(pending)} videos captioned successfully.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[info] Interrupted by user.")
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as _exc:
        import traceback
        print(f"\n[error] Unexpected error: {_exc}")
        traceback.print_exc()
        sys.exit(1)
