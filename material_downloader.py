"""
Canvas Course Material Downloader

For each active course:
  1. Enumerate all files via Canvas API (preserving folder structure)
  2. Estimate total size
     - If < 1 GB  → download everything
     - If >= 1 GB → use Claude AI to identify lecture notes & tutorials only
  3. Download selected files into [course_id]/materials/<subfolder>/
  4. Maintain a per-course download log so already-downloaded files are skipped

Directory layout:
  [project]/[course_id]/materials/<canvas_subfolder>/<filename>
  [project]/[course_id]/materials/download_log.json

Usage:
  python material_downloader.py           # process all active courses
  python material_downloader.py -c 85427  # process one course by id
  python material_downloader.py --list    # list courses + estimated sizes only
"""

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from canvasapi import Canvas

# ── Configuration ────────────────────────────────────────────────────────────

CANVAS_URL   = "https://canvas.nus.edu.sg"
CANVAS_TOKEN = "21450~VhMGPzTKzT9wABF3VF76k8YE9LKUwfL93nU2LMc3xcfKPMhTkHJYaG3vZ34mfUke"
ANTHROPIC_API_KEY = None   # set below or via env

PROJECT_DIR  = Path(__file__).parent
SIZE_LIMIT   = 1 * 1024 ** 3   # 1 GB

# Skip training / non-academic courses by name keywords
SKIP_COURSE_KEYWORDS = [
    "training", "pdp", "rmcpdp", "osa", "soct", "travel", "essentials",
    "respect", "consent",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _headers() -> dict:
    return {"Authorization": f"Bearer {CANVAS_TOKEN}"}


def sanitize(name: str) -> str:
    """Make a string safe for use as a file/directory name."""
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()


def load_log(log_path: Path) -> dict:
    if log_path.exists():
        with open(log_path) as f:
            return json.load(f)
    return {}


def save_log(log_path: Path, log: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


# ── Canvas file enumeration ───────────────────────────────────────────────────

def get_course_files(course) -> list[dict]:
    """
    Return all files in a course as a list of dicts:
      id, display_name, size, url, folder_path, mime_class
    folder_path is relative (strips leading "course files/").
    """
    # Build folder_id → relative path map
    folder_map: dict[int, str] = {}
    try:
        for folder in course.get_folders():
            full = folder.full_name  # e.g. "course files/LectureNotes"
            rel  = full.removeprefix("course files").lstrip("/")
            folder_map[folder.id] = rel
    except Exception as e:
        print(f"    [warn] folders: {e}")

    files = []
    try:
        for f in course.get_files():
            folder_rel = folder_map.get(f.folder_id, "")
            files.append({
                "id":           f.id,
                "display_name": f.display_name,
                "filename":     f.filename,
                "size":         getattr(f, "size", 0) or 0,
                "url":          f.url,
                "mime_class":   getattr(f, "mime_class", ""),
                "folder_path":  folder_rel,   # e.g. "LectureNotes"
                "updated_at":   getattr(f, "updated_at", ""),
            })
    except Exception as e:
        print(f"    [warn] files: {e}")

    return files


# ── AI classification ─────────────────────────────────────────────────────────

def classify_with_ai(files: list[dict], course_name: str) -> list[dict]:
    """
    Ask Claude to identify which files are lecture notes or tutorial materials.
    Returns the filtered list.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Build a compact listing for the prompt
    lines = []
    for f in files:
        path = f["folder_path"]
        name = f["display_name"]
        mb   = f["size"] / 1024 / 1024
        lines.append(f'  id={f["id"]}  folder="{path}"  name="{name}"  size={mb:.1f}MB')

    file_listing = "\n".join(lines)

    prompt = f"""You are helping organize university course materials for: {course_name}

Here is a list of all available files:
{file_listing}

Task: identify which files are LECTURE NOTES or TUTORIAL materials.
- Lecture notes: slides, lecture PDFs, notes, handouts, readings
- Tutorials: tutorial sheets, problem sets, worksheets, exercises, lab guides

Return ONLY a JSON array of the integer file IDs to download. No explanation.
Example: [123, 456, 789]"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    # Extract JSON array from response
    match = re.search(r'\[[\d,\s]+\]', text)
    if not match:
        print("    [warn] AI returned unexpected format; downloading all files")
        return files

    selected_ids = set(json.loads(match.group()))
    selected = [f for f in files if f["id"] in selected_ids]
    print(f"    AI selected {len(selected)}/{len(files)} files (lecture notes + tutorials)")
    return selected


# ── File download ─────────────────────────────────────────────────────────────

def download_file(file_info: dict, course_id: int, log: dict) -> bool:
    """
    Download a single file to [course_id]/materials/<folder_path>/<filename>.
    Returns True on success (or already-downloaded skip).
    Updates log in-place.
    """
    fid  = str(file_info["id"])
    name = file_info["display_name"]

    # Skip if already downloaded and file still exists
    if fid in log:
        existing = Path(log[fid]["path"])
        if existing.exists():
            print(f"    [skip] {name}")
            return True

    # Build destination path
    folder_rel = file_info["folder_path"]
    dest_dir   = PROJECT_DIR / str(course_id) / "materials"
    if folder_rel:
        dest_dir = dest_dir / Path(*[sanitize(p) for p in folder_rel.split("/")])
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / sanitize(name)

    # Download with streaming
    try:
        r = requests.get(
            file_info["url"],
            headers=_headers(),
            stream=True,
            timeout=60,
            allow_redirects=True,
        )
        r.raise_for_status()

        size    = file_info["size"] or 0
        written = 0
        with open(dest_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    fh.write(chunk)
                    written += len(chunk)
                    if size:
                        pct = written * 100 // size
                        print(f"\r    Downloading {name[:50]:<50} {pct:3d}%", end="", flush=True)

        print(f"\r    Downloaded  {name[:50]:<50} {written//1024:>6} KB")

        log[fid] = {
            "display_name": name,
            "folder":       folder_rel,
            "size":         written,
            "path":         str(dest_path),
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        }
        return True

    except Exception as e:
        print(f"\n    [error] {name}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


# ── Per-course orchestration ──────────────────────────────────────────────────

def process_course(course, force_all: bool = False) -> tuple[int, int]:
    """
    Download materials for one course.
    Returns (downloaded, skipped) counts.
    """
    cid   = course.id
    cname = course.name
    print(f"\n{'='*70}")
    print(f"Course: {cname}  (id={cid})")
    print(f"{'='*70}")

    log_path = PROJECT_DIR / str(cid) / "materials" / "download_log.json"
    log      = load_log(log_path)

    print("  Enumerating files...")
    files = get_course_files(course)
    if not files:
        print("  No files found.")
        return 0, 0

    total_size = sum(f["size"] for f in files)
    total_mb   = total_size / 1024 / 1024
    print(f"  {len(files)} file(s)  |  estimated size: {total_mb:.1f} MB")

    # Decide which files to download
    if force_all or total_size < SIZE_LIMIT:
        if total_size >= SIZE_LIMIT:
            print(f"  Size {total_mb:.0f} MB >= 1 GB limit — but force_all requested")
        else:
            print(f"  Size < 1 GB — downloading all files")
        to_download = files
    else:
        print(f"  Size {total_mb:.0f} MB >= 1 GB — using AI to select lecture notes & tutorials")
        to_download = classify_with_ai(files, cname)

    downloaded = skipped = errors = 0
    for f in to_download:
        fid = str(f["id"])
        if fid in log and Path(log[fid]["path"]).exists():
            skipped += 1
            print(f"    [skip] {f['display_name']}")
            continue

        if download_file(f, cid, log):
            downloaded += 1
        else:
            errors += 1

        # Save log after every file so crashes don't lose progress
        save_log(log_path, log)

    save_log(log_path, log)
    print(f"\n  Done: {downloaded} downloaded, {skipped} skipped, {errors} errors")
    return downloaded, skipped


# ── Main entry point ──────────────────────────────────────────────────────────

def is_academic_course(course) -> bool:
    """Filter out training / non-academic Canvas shells."""
    try:
        name = course.name.lower()
    except AttributeError:
        return False
    return not any(kw in name for kw in SKIP_COURSE_KEYWORDS)


def main() -> None:
    import argparse
    import os

    global ANTHROPIC_API_KEY
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
    if not ANTHROPIC_API_KEY:
        # Try to read from project dir
        key_file = PROJECT_DIR / "anthropic_key.txt"
        if key_file.exists():
            ANTHROPIC_API_KEY = key_file.read_text().strip()

    parser = argparse.ArgumentParser(description="Canvas Course Material Downloader")
    parser.add_argument("-c", "--course", type=int, metavar="ID",
                        help="Process a single course by Canvas course ID")
    parser.add_argument("--list", action="store_true",
                        help="List courses and estimated sizes, then exit")
    parser.add_argument("--all", dest="force_all", action="store_true",
                        help="Download all files regardless of size limit")
    args = parser.parse_args()

    canvas  = Canvas(CANVAS_URL, CANVAS_TOKEN)
    courses = [c for c in canvas.get_courses(enrollment_state="active")
               if is_academic_course(c)]

    if args.course:
        courses = [c for c in courses if c.id == args.course]
        if not courses:
            # Fetch directly
            try:
                courses = [canvas.get_course(args.course)]
            except Exception as e:
                print(f"[error] Course {args.course} not found: {e}")
                sys.exit(1)

    if args.list:
        print(f"{'ID':<10} {'Size':>10}  Course")
        print("-" * 70)
        for c in courses:
            try:
                files = get_course_files(c)
                sz    = sum(f["size"] for f in files)
                flag  = ">" if sz >= SIZE_LIMIT else " "
                print(f"{c.id:<10} {flag}{sz/1024/1024:>8.1f} MB  {c.name}")
            except Exception as e:
                print(f"{c.id:<10} {'?':>10}  {c.name}  ({e})")
        return

    total_dl = total_sk = 0
    for course in courses:
        dl, sk = process_course(course, force_all=args.force_all)
        total_dl += dl
        total_sk += sk

    print(f"\n{'='*70}")
    print(f"All done: {total_dl} downloaded, {total_sk} skipped across {len(courses)} course(s)")


if __name__ == "__main__":
    main()
