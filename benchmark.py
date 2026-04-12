"""
Benchmark note quality on three metrics:
  1. Content coverage  — % of key transcript concepts that appear in the note
  2. Image density     — % of content-rich frames/slides actually used
  3. Logic coherency   — structural quality: section flow, transitions, no orphan images

Usage:
  python benchmark.py --course 85397
  python benchmark.py --note path/to/note.md --transcript path/to/caption.json
  python benchmark.py --course 85397 --verbose   # show per-metric breakdown
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


# ── Content coverage ─────────────────────────────────────────────────────────

def _extract_key_terms(text: str, min_freq: int = 2) -> set[str]:
    """Extract significant terms (multi-word capitalized phrases, technical acronyms)."""
    # Multi-word capitalized phrases: "Network Layer", "Address Resolution Protocol"
    phrases = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b", text)
    # Acronyms: TCP, UDP, HTTP, MAC
    acronyms = re.findall(r"\b[A-Z]{2,6}\b", text)
    # Hyphenated technical terms
    hyphenated = re.findall(r"\b[a-z]+(?:-[a-z]+)+\b", text.lower())

    # Count frequency, keep terms appearing at least min_freq times (reduces noise)
    from collections import Counter
    all_terms = phrases + acronyms + hyphenated
    counts = Counter(t.lower() for t in all_terms)
    return {t for t, c in counts.items() if c >= min_freq and len(t) >= 3}


def content_coverage(note_text: str, transcript_text: str) -> tuple[float, dict]:
    """Return (coverage_score_0_to_10, breakdown).

    Compares key terms in the transcript to terms in the note. A high score means
    the note covers most of the concepts the lecturer discussed.
    """
    transcript_terms = _extract_key_terms(transcript_text)
    note_text_lower = note_text.lower()

    if not transcript_terms:
        return 10.0, {"hit": 0, "total": 0, "missed": []}

    hits = [t for t in transcript_terms if t in note_text_lower]
    missed = sorted(transcript_terms - set(hits))
    score = len(hits) / len(transcript_terms) * 10

    return score, {
        "hit": len(hits),
        "total": len(transcript_terms),
        "ratio": len(hits) / len(transcript_terms),
        "missed_sample": missed[:10],
    }


# ── Image density ────────────────────────────────────────────────────────────

_IMG_REF = re.compile(r"!\[(?:Slide|Frame)\s+\d+\]\(([^)]+)\)")

# Keywords that suggest a cached description is content-rich (not a blank/loading)
_CONTENT_KEYWORDS = re.compile(
    r"\b(diagram|chart|graph|figure|flowchart|table|formula|equation|"
    r"architecture|layout|structure|matrix|tree|network|circuit|timeline|"
    r"plot|drawing|schematic|visual|slide|algorithm|code|protocol|frame|"
    r"header|packet|address|node|link|layer|buffer|stack|queue|message)\b",
    re.IGNORECASE,
)


def _content_rich(description: str) -> bool:
    """Heuristic: a cached image description indicates real content."""
    if not description or len(description) < 40:
        return False
    # Exclude loading screens and pure UI descriptions
    if re.search(r"\b(windows 11|desktop background|taskbar|file explorer|"
                 r"loading screen|blank|empty|unable to view)\b",
                 description, re.IGNORECASE):
        return False
    return bool(_CONTENT_KEYWORDS.search(description))


def image_density(note_text: str, image_cache: dict) -> tuple[float, dict]:
    """Return (density_score_0_to_10, breakdown).

    Compares the number of images inserted to the number of content-rich
    images available.  100% usage = 10/10.
    """
    used = len(set(_IMG_REF.findall(note_text)))
    available = [k for k, v in image_cache.items() if _content_rich(v)]

    if not available:
        # No content-rich images available; any used is good
        return 10.0 if used == 0 else 10.0, {
            "used": used, "available": 0, "ratio": 1.0,
        }

    ratio = used / len(available)
    score = min(10.0, ratio * 10)
    return score, {
        "used": used,
        "available": len(available),
        "ratio": ratio,
    }


# ── Logic coherency ─────────────────────────────────────────────────────────

def logic_coherency(note_text: str) -> tuple[float, dict]:
    """Return (coherency_score_0_to_10, breakdown).

    Checks structural quality:
      - Section headings present and well-distributed
      - No orphan images (images without surrounding text)
      - No consecutive image lines (clustering)
      - Paragraphs flow (no abrupt truncation)
      - No placeholder text or artifacts
    """
    score = 10.0
    issues = []

    lines = note_text.splitlines()
    headings = [l for l in lines if l.startswith("### ")]
    subheadings = [l for l in lines if l.startswith("#### ")]

    # Check 1: section headings present
    if len(headings) == 0:
        score -= 2.0
        issues.append("No section headings (### )")

    # Check 2: excessive image clustering (3+ consecutive images without
    # intervening prose). Pairs of related images with captions are a normal
    # grouping pattern; only sequences of 3+ back-to-back images count as a
    # layout problem.
    img_lines = [i for i, l in enumerate(lines) if _IMG_REF.match(l.strip())]
    clusters = 0
    run = 1
    for i in range(len(img_lines) - 1):
        between = lines[img_lines[i]+1:img_lines[i+1]]
        # Treat italic captions immediately below an image as part of the
        # image, not "text between".
        text_lines = [l for l in between
                      if l.strip()
                      and not l.startswith("#")
                      and not l.strip().startswith("*(")]
        if not text_lines:
            run += 1
            if run == 3:
                clusters += 1
        else:
            run = 1
    if clusters > 0:
        penalty = min(3.0, clusters * 0.5)
        score -= penalty
        issues.append(f"{clusters} long image cluster(s) (3+ in a row)")

    # Check 3: orphan images (image with no surrounding context). Look
    # further back (6 lines) and ignore other image lines + their captions
    # when searching for preceding prose, so images in a group share credit
    # for the paragraph that precedes the group.
    orphans = 0
    for i in img_lines:
        before_lines = []
        for l in lines[max(0, i-6):i]:
            s = l.strip()
            if not s or s.startswith("#"):
                continue
            if _IMG_REF.match(s) or s.startswith("*("):
                continue
            before_lines.append(s)
        if not before_lines:
            orphans += 1
    if orphans > 0:
        penalty = min(2.0, orphans * 0.3)
        score -= penalty
        issues.append(f"{orphans} orphan image(s) without preceding paragraph")

    # Check 4: truncation signs (ends mid-word)
    last_line = ""
    for l in reversed(lines):
        if l.strip():
            last_line = l.strip()
            break
    if last_line and re.search(r"[a-zA-Z\u4e00-\u9fff]$", last_line):
        # Ends with a letter, likely truncated mid-sentence
        if not re.search(r"[.!?:)\]\"'」』。？！]$", last_line):
            score -= 2.0
            issues.append(f"May end mid-sentence: '{last_line[-40:]}'")

    # Check 5: artifacts
    artifacts = []
    if "APPROVED" in note_text and re.search(r"^APPROVED$", note_text, re.MULTILINE):
        artifacts.append("APPROVED leak")
    if "NUS Confidential" in note_text:
        artifacts.append("NUS Confidential leak")
    if "© CS" in note_text or "(c) CS" in note_text:
        artifacts.append("copyright marker leak")
    if "no source material" in note_text.lower() or "cannot be generated" in note_text.lower():
        artifacts.append("LLM refusal message")
    if artifacts:
        score -= min(3.0, len(artifacts) * 1.5)
        issues.append(f"Artifacts: {', '.join(artifacts)}")

    return max(0.0, score), {
        "headings": len(headings),
        "subheadings": len(subheadings),
        "image_lines": len(img_lines),
        "clusters": clusters,
        "orphans": orphans,
        "issues": issues,
    }


# ── Main benchmark runner ────────────────────────────────────────────────────

def benchmark_note(note_path: Path, transcript_text: str,
                   image_cache: dict) -> dict:
    """Run all three benchmarks on a note and return scores + breakdown."""
    note_text = note_path.read_text(encoding="utf-8")

    cov_score, cov_info = content_coverage(note_text, transcript_text)
    img_score, img_info = image_density(note_text, image_cache)
    coh_score, coh_info = logic_coherency(note_text)

    overall = (cov_score * 0.4 + img_score * 0.3 + coh_score * 0.3)

    return {
        "note": note_path.name,
        "overall": round(overall, 2),
        "coverage": round(cov_score, 2),
        "image_density": round(img_score, 2),
        "coherency": round(coh_score, 2),
        "details": {
            "coverage": cov_info,
            "image_density": img_info,
            "coherency": coh_info,
        },
    }


def _load_transcript(caption_path: Path) -> str:
    """Load transcript text from caption JSON."""
    if not caption_path.exists():
        return ""
    with open(caption_path, encoding="utf-8") as f:
        data = json.load(f)
    segs = data.get("segments", [])
    # Filter Whisper dot hallucinations
    texts = [s["text"] for s in segs
             if not re.fullmatch(r"[\s.]+", s.get("text", ""))]
    return " ".join(texts)


def _load_image_cache_for_note(note_path: Path) -> dict:
    """Find image cache for a note by inspecting image refs in the note."""
    note_text = note_path.read_text(encoding="utf-8")
    refs = _IMG_REF.findall(note_text)
    if not refs:
        return {}

    # Extract LXX directory from image path
    dirs = {ref.rsplit("/", 1)[0] for ref in refs if "/" in ref}
    if not dirs:
        return {}

    course_dir = note_path.parent
    combined: dict = {}
    for d in dirs:
        img_dir = course_dir / d
        if not img_dir.exists():
            continue
        # Look for image_cache.json in frames dir (screenshare) or adjacent
        l_name = d.split("/")[-1]  # e.g., "L10"
        # Find the frame dir via images/LXX/ → frames/<stem>/image_cache.json
        # The cache file is stored alongside frames; mapping requires lookup.
        # For this benchmark, we'll check the course frames dir for matching caches.
        pass  # We'll load via video stem instead

    return combined


def benchmark_course(course_dir: Path, verbose: bool = False) -> list[dict]:
    """Benchmark all notes in a course directory."""
    notes_dir = course_dir / "notes"
    captions_dir = course_dir / "captions"
    frames_dir = course_dir / "frames"

    if not notes_dir.exists():
        print(f"No notes dir: {notes_dir}")
        return []

    results = []
    for note_file in sorted(notes_dir.glob("*_notes.md")):
        # Skip score files
        if note_file.name.endswith(".score.json"):
            continue
        stem = note_file.stem.replace("_notes", "")

        # Find matching caption
        caption = captions_dir / f"{stem}.json"
        transcript = _load_transcript(caption) if caption.exists() else ""

        # Find matching image cache
        img_cache = {}
        # Try screenshare frames cache
        frame_cache = frames_dir / stem / "image_cache.json"
        if frame_cache.exists():
            try:
                img_cache = json.loads(frame_cache.read_text())
            except Exception:
                pass

        result = benchmark_note(note_file, transcript, img_cache)
        results.append(result)

    return results


def print_report(results: list[dict], verbose: bool = False) -> None:
    """Print a summary table of benchmark results."""
    if not results:
        print("No notes to benchmark.")
        return

    # Header
    print()
    print(f"{'Note':<55} {'Overall':>8} {'Cov':>6} {'Img':>6} {'Coh':>6}")
    print("─" * 85)

    for r in sorted(results, key=lambda x: -x["overall"]):
        name = r["note"][:53]
        print(f"{name:<55} {r['overall']:>7.2f}  {r['coverage']:>5.2f}  "
              f"{r['image_density']:>5.2f}  {r['coherency']:>5.2f}")

    # Averages
    print("─" * 85)
    n = len(results)
    avg_overall = sum(r["overall"] for r in results) / n
    avg_cov = sum(r["coverage"] for r in results) / n
    avg_img = sum(r["image_density"] for r in results) / n
    avg_coh = sum(r["coherency"] for r in results) / n
    print(f"{'AVERAGE':<55} {avg_overall:>7.2f}  {avg_cov:>5.2f}  "
          f"{avg_img:>5.2f}  {avg_coh:>5.2f}")
    print()

    if verbose:
        print("\n── Per-note details ──\n")
        for r in sorted(results, key=lambda x: x["overall"]):
            print(f"▸ {r['note']}  (overall {r['overall']})")
            d = r["details"]
            cov = d["coverage"]
            print(f"    Coverage {r['coverage']}/10: {cov['hit']}/{cov['total']} terms"
                  f" ({cov.get('ratio', 0)*100:.0f}%)")
            if cov.get("missed_sample"):
                print(f"      Missed: {', '.join(cov['missed_sample'][:5])}…")
            img = d["image_density"]
            print(f"    Images   {r['image_density']}/10: {img['used']}/{img['available']} "
                  f"content-rich available")
            coh = d["coherency"]
            print(f"    Coherency {r['coherency']}/10: {coh['headings']} sections, "
                  f"{coh['image_lines']} images, {coh['clusters']} clusters, "
                  f"{coh['orphans']} orphans")
            if coh["issues"]:
                for issue in coh["issues"]:
                    print(f"      - {issue}")
            print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark generated notes")
    parser.add_argument("--course", metavar="ID",
                        help="Course ID to benchmark (auto-finds notes)")
    parser.add_argument("--note", metavar="PATH",
                        help="Benchmark a single note file")
    parser.add_argument("--transcript", metavar="PATH",
                        help="Transcript JSON for --note mode")
    parser.add_argument("--image-cache", metavar="PATH",
                        help="Image cache JSON for --note mode")
    parser.add_argument("--path", metavar="DIR",
                        help="Base output directory (default: ~/AutoNote)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-metric breakdown")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON instead of table")
    args = parser.parse_args()

    base_dir = Path(args.path) if args.path else Path.home() / "AutoNote"

    if args.course:
        course_dir = base_dir / args.course
        results = benchmark_course(course_dir, verbose=args.verbose)
    elif args.note:
        note_path = Path(args.note)
        transcript = ""
        if args.transcript:
            transcript = _load_transcript(Path(args.transcript))
        image_cache: dict = {}
        if args.image_cache:
            try:
                image_cache = json.loads(Path(args.image_cache).read_text())
            except Exception:
                pass
        results = [benchmark_note(note_path, transcript, image_cache)]
    else:
        parser.error("Provide --course or --note")

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print_report(results, verbose=args.verbose)


if __name__ == "__main__":
    main()
