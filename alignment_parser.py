"""
Alignment Parser
Converts the full semantic_alignment.py JSON output into a compact
per-slide representation for token-efficient LLM prompting.

Full alignment JSON: ~2434 segments × ~120 bytes = ~300 KB
Compact output:      ~73 slides  × ~400 bytes = ~30 KB  (10× smaller)

Usage:
  python alignment_parser.py 85427/alignment/lecture.json
  python alignment_parser.py 85427/alignment/lecture.json --out compact.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _clean_transcript(text: str) -> str:
    """Remove filler words and normalise whitespace."""
    # Strip common ASR fillers that add no information
    fillers = re.compile(
        r"\b(uh+|um+|er+|ah+|okay so|so yeah|you know|right so|alright)\b",
        re.IGNORECASE,
    )
    text = fillers.sub("", text)
    # Collapse whitespace
    return re.sub(r"\s{2,}", " ", text).strip()


def parse(alignment_path: Path) -> dict:
    """
    Convert full alignment JSON → compact representation.

    Returns dict with keys:
      lecture, slide_file, duration, language,
      slides: list of {slide, label, start, end, duration, transcript},
      off_slide: {start_times, transcript}   (may be absent)
    """
    with open(alignment_path, encoding="utf-8") as f:
        data = json.load(f)

    # ── Group segment text by slide index ─────────────────────────────────────
    slide_texts: dict[int, list[str]] = {}   # 1-based slide → [text, ...]
    off_texts:   list[str]            = []
    off_starts:  list[float]          = []

    for seg in data.get("segments", []):
        t = _clean_transcript(seg.get("text", "")).strip()
        if not t:
            continue
        if seg.get("off_slide"):
            off_texts.append(t)
            off_starts.append(seg["start"])
        else:
            s = seg.get("slide") or 1
            slide_texts.setdefault(s, []).append(t)

    # ── Build compact slides list from timeline ────────────────────────────────
    # Use timeline for timing (already merged), but re-attach full transcript.
    compact_slides: list[dict] = []
    seen_slides: set[int] = set()

    for entry in data.get("timeline", []):
        s     = entry["slide"]
        start = entry["start"]
        end   = entry["end"]

        if s in seen_slides:
            # Slide revisited — merge into existing entry
            for cs in compact_slides:
                if cs["slide"] == s:
                    cs["end"]      = max(cs["end"], end)
                    cs["duration"] = round(cs["end"] - cs["start"], 1)
                    break
            continue

        seen_slides.add(s)
        transcript = " ".join(slide_texts.get(s, []))
        compact_slides.append({
            "slide":      s,
            "label":      entry["label"],
            "start":      round(start, 1),
            "end":        round(end, 1),
            "duration":   round(end - start, 1),
            "transcript": transcript,
        })

    # Sort by first appearance
    compact_slides.sort(key=lambda x: x["start"])

    # ── Build output ──────────────────────────────────────────────────────────
    result: dict = {
        "lecture":    data.get("lecture", ""),
        "slide_file": data.get("slide_file", ""),
        "duration":   data.get("duration", 0),
        "language":   data.get("language", ""),
        "total_slides": data.get("total_slides", 0),
        "slides":     compact_slides,
    }

    if off_texts:
        result["off_slide"] = {
            "transcript": " ".join(off_texts),
            "start_times": off_starts[:5],   # first few timestamps only
        }

    return result


def save(compact: dict, out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)


def parse_and_save(alignment_path: Path, out_path: Path | None = None) -> Path:
    compact  = parse(alignment_path)
    if out_path is None:
        out_path = alignment_path.with_suffix(".compact.json")
    save(compact, out_path)

    orig_kb    = alignment_path.stat().st_size / 1024
    compact_kb = out_path.stat().st_size / 1024
    print(f"  Parsed: {len(compact['slides'])} slides, "
          f"{orig_kb:.0f} KB → {compact_kb:.0f} KB "
          f"({compact_kb/orig_kb*100:.0f}% of original)")
    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Compact alignment JSON for LLM prompting")
    parser.add_argument("alignment", metavar="PATH", help="Alignment JSON path")
    parser.add_argument("--out",     metavar="PATH", help="Output path (default: *.compact.json)")
    args = parser.parse_args()

    ap = Path(args.alignment)
    if not ap.exists():
        print(f"[error] Not found: {ap}"); sys.exit(1)

    out = Path(args.out) if args.out else None
    parse_and_save(ap, out)


if __name__ == "__main__":
    main()
