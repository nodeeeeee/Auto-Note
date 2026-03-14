"""
downloader.py — Unified Canvas Downloader

Combines video_downloader.py and material_downloader.py into a single CLI.

Usage:
  python downloader.py --course-list
  python downloader.py --video-list
  python downloader.py --video-list --course 85427
  python downloader.py --download-video 1 3 5
  python downloader.py --download-video 2 4 --course 85427
  python downloader.py --download-video-all
  python downloader.py --download-video-all --course 85427 --secretly
  python downloader.py --material-list --course 85427
  python downloader.py --download-material "L02-slides.pdf" --course 85427
  python downloader.py --download-material-all --course 85427
  python downloader.py --download-material-all --secretly --path /data/courses
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from canvasapi import Canvas
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────

CANVAS_URL    = "https://canvas.nus.edu.sg"
CANVAS_TOKEN  = "21450~VhMGPzTKzT9wABF3VF76k8YE9LKUwfL93nU2LMc3xcfKPMhTkHJYaG3vZ34mfUke"
PANOPTO_HOST  = "mediaweb.ap.panopto.com"
PROJECT_DIR   = Path(__file__).parent
MANIFEST_FILE = PROJECT_DIR / "manifest.json"
SIZE_LIMIT    = 1 * 1024 ** 3   # 1 GB

SKIP_COURSE_KEYWORDS = [
    "training", "pdp", "rmcpdp", "osa", "soct", "travel", "essentials",
    "respect", "consent",
]

# Stealth-mode delay ranges (seconds)
_SECRETLY_VIDEO_MIN = 5  * 60    # 5 min
_SECRETLY_VIDEO_MAX = 15 * 60    # 15 min
_SECRETLY_DIR_MIN   = 2  * 60    # 2 min
_SECRETLY_DIR_MAX   = 5  * 60    # 5 min


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _sanitize(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()


def _canvas_headers() -> dict:
    return {"Authorization": f"Bearer {CANVAS_TOKEN}"}


def _is_academic(course) -> bool:
    try:
        name = course.name.lower()
    except AttributeError:
        return False
    return not any(kw in name for kw in SKIP_COURSE_KEYWORDS)


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _secretly_wait_video() -> None:
    delay = random.uniform(_SECRETLY_VIDEO_MIN, _SECRETLY_VIDEO_MAX)
    mins  = delay / 60
    for _ in tqdm(range(int(delay)), desc=f"  [secretly] Next video in {mins:.1f} min",
                  unit="s", bar_format="{desc} |{bar}| {n}/{total}s", leave=False):
        time.sleep(1)


def _secretly_wait_dir() -> None:
    delay = random.uniform(_SECRETLY_DIR_MIN, _SECRETLY_DIR_MAX)
    mins  = delay / 60
    for _ in tqdm(range(int(delay)), desc=f"  [secretly] Next folder in {mins:.1f} min",
                  unit="s", bar_format="{desc} |{bar}| {n}/{total}s", leave=False):
        time.sleep(1)


# ── Canvas course listing ──────────────────────────────────────────────────────

def get_academic_courses(canvas: Canvas) -> list:
    """Return all active academic courses."""
    courses = []
    for c in canvas.get_courses(enrollment_state="active"):
        if _is_academic(c):
            courses.append(c)
    return courses


def get_course_by_id(canvas: Canvas, course_id: int):
    """Fetch a single course by ID (even if not in active list)."""
    return canvas.get_course(course_id)


# ══════════════════════════════════════════════════════════════════════════════
#  VIDEO SECTION
# ══════════════════════════════════════════════════════════════════════════════

def _find_panopto_items(canvas: Canvas, course) -> list[dict]:
    """Scan all modules in a course for Panopto ExternalTool items."""
    videos = []
    try:
        for module in course.get_modules():
            for item in module.get_module_items():
                if item.type == "ExternalTool":
                    url = getattr(item, "external_url", "") or ""
                    if "panopto" in url.lower():
                        videos.append({
                            "course_id":   course.id,
                            "course_name": course.name,
                            "module_name": module.name,
                            "title":       item.title,
                            "lti_url":     url,
                            "item_id":     item.id,
                        })
    except Exception as e:
        tqdm.write(f"  [warn] course {course.id}: {e}")
    return videos


def discover_videos(canvas: Canvas, course_id: int | None = None) -> list[dict]:
    """
    Discover all Panopto videos.
    If course_id is given, only scan that course.
    Each entry gets a 'global_num' (1-based across all courses) and
    'course_num' (1-based within its course).
    """
    if course_id:
        courses = [get_course_by_id(canvas, course_id)]
    else:
        courses = get_academic_courses(canvas)

    all_videos: list[dict] = []
    course_counters: dict[int, int] = {}

    with tqdm(courses, desc="Scanning courses", unit="course") as bar:
        for course in bar:
            bar.set_postfix_str(course.name[:40])
            videos = _find_panopto_items(canvas, course)
            cid    = course.id
            course_counters.setdefault(cid, 0)
            for v in videos:
                course_counters[cid] += 1
                v["course_num"] = course_counters[cid]
            all_videos.extend(videos)

    # Assign global numbers
    for i, v in enumerate(all_videos, start=1):
        v["global_num"] = i

    return all_videos


def print_video_list(videos: list[dict], manifest: dict, by_course: bool = False) -> None:
    """Print a numbered video listing."""
    num_key = "course_num" if by_course else "global_num"
    print()
    print(f"  {'#':>3}  {'Status':<10}  {'Course':<20}  {'Module':<22}  Title")
    print(f"  {'-'*3}  {'-'*10}  {'-'*20}  {'-'*22}  {'-'*40}")
    for v in videos:
        status = manifest.get(str(v["item_id"]), {}).get("status", "pending")
        num    = v[num_key]
        course = v["course_name"][:20]
        module = v["module_name"][:22]
        title  = v["title"][:50]
        mark   = "✓" if status == "done" else " "
        print(f"  {num:>3}  [{mark}]{status:<8}  {course:<20}  {module:<22}  {title}")
    print()
    print(f"  Total: {len(videos)} video(s)")
    print()


def _get_sessionless_launch_url(course_id: int, item_id: int) -> str | None:
    r = requests.get(
        f"{CANVAS_URL}/api/v1/courses/{course_id}/external_tools/sessionless_launch"
        f"?launch_type=module_item&module_item_id={item_id}",
        headers=_canvas_headers(),
        timeout=30,
    )
    return r.json().get("url") if r.status_code == 200 else None


def _get_panopto_session(launch_url: str) -> tuple[str | None, list[dict]]:
    from playwright.sync_api import sync_playwright
    session_id, cookies = None, []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx     = browser.new_context()
        page    = ctx.new_page()
        try:
            page.goto(launch_url, wait_until="networkidle", timeout=60000)
            time.sleep(2)
            m = re.search(r"[?&]id=([0-9a-f-]{36})", page.url, re.IGNORECASE)
            if m:
                session_id = m.group(1)
            cookies = [c for c in ctx.cookies()
                       if "panopto" in c.get("domain", "").lower()]
        except Exception as e:
            tqdm.write(f"  [warn] Playwright: {e}")
        finally:
            browser.close()
    return session_id, cookies


def _get_stream_url(session_id: str, cookies: list[dict]) -> str | None:
    sess = requests.Session()
    for ck in cookies:
        sess.cookies.set(ck["name"], ck["value"], domain=ck.get("domain", ""))
    r = sess.post(
        f"https://{PANOPTO_HOST}/Panopto/Pages/Viewer/DeliveryInfo.aspx",
        data={
            "deliveryId": session_id, "isLiveNotes": "false",
            "refreshAuthCookie": "true", "isActiveBroadcast": "false",
            "isEditing": "false", "isKollectiveAgentInstalled": "false",
            "isEmbed": "true", "responseType": "json",
        },
        timeout=30,
    )
    if r.status_code != 200:
        return None
    streams = r.json().get("Delivery", {}).get("Streams", [])
    for tag in ("DV", "OBJECT", None):
        for s in streams:
            url = s.get("StreamUrl", "")
            if url and (tag is None or s.get("Tag") == tag):
                return url
    return None


def download_video(video: dict, manifest: dict, base_dir: Path) -> bool:
    """Download a single Panopto video with a tqdm progress bar."""
    import PanoptoDownloader

    item_id   = video["item_id"]
    course_id = video["course_id"]
    title     = video["title"]
    key       = str(item_id)

    if manifest.get(key, {}).get("status") == "done":
        tqdm.write(f"  [skip] Already downloaded: {title}")
        return True

    out_dir  = base_dir / str(course_id) / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_sanitize(title)}.mp4"

    if out_path.exists():
        tqdm.write(f"  [skip] File exists: {out_path}")
        manifest[key] = {"status": "done", "path": str(out_path)}
        return True

    tqdm.write(f"  Fetching launch URL for: {title}")
    launch_url = _get_sessionless_launch_url(course_id, item_id)
    if not launch_url:
        tqdm.write(f"  [error] Could not get launch URL")
        manifest[key] = {"status": "error", "title": title}
        return False

    tqdm.write(f"  Launching Panopto session...")
    session_id, cookies = _get_panopto_session(launch_url)
    if not session_id:
        tqdm.write(f"  [error] Could not get Panopto session ID")
        manifest[key] = {"status": "error", "title": title}
        return False

    stream_url = _get_stream_url(session_id, cookies)
    if not stream_url:
        tqdm.write(f"  [error] Could not get stream URL")
        manifest[key] = {"status": "error", "title": title}
        return False

    bar = tqdm(total=100, desc=f"  {title[:50]}", unit="%",
               bar_format="{desc} |{bar}| {n:3d}/{total}%", leave=True)

    def progress_cb(pct: float) -> None:
        bar.n = int(pct)
        bar.refresh()

    try:
        PanoptoDownloader.download(stream_url, str(out_path), progress_cb)
        bar.n = 100
        bar.refresh()
        bar.close()
        tqdm.write(f"  Saved → {out_path}")
        manifest[key] = {
            "status": "done", "path": str(out_path),
            "title": title, "session_id": session_id,
        }
        return True
    except Exception as e:
        bar.close()
        tqdm.write(f"  [error] Download failed: {e}")
        if out_path.exists():
            out_path.unlink()
        manifest[key] = {"status": "error", "title": title, "error": str(e)}
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  MATERIAL SECTION
# ══════════════════════════════════════════════════════════════════════════════

def get_course_files(course) -> list[dict]:
    """Return all files in a course with folder paths."""
    folder_map: dict[int, str] = {}
    try:
        for folder in course.get_folders():
            rel = folder.full_name.removeprefix("course files").lstrip("/")
            folder_map[folder.id] = rel
    except Exception as e:
        tqdm.write(f"    [warn] folders: {e}")

    files = []
    try:
        for f in course.get_files():
            files.append({
                "id":           f.id,
                "display_name": f.display_name,
                "filename":     f.filename,
                "size":         getattr(f, "size", 0) or 0,
                "url":          f.url,
                "mime_class":   getattr(f, "mime_class", ""),
                "folder_path":  folder_map.get(f.folder_id, ""),
                "updated_at":   getattr(f, "updated_at", ""),
                "course_id":    course.id,
                "course_name":  course.name,
            })
    except Exception as e:
        tqdm.write(f"    [warn] files: {e}")
    return files


def discover_materials(canvas: Canvas, course_id: int | None = None) -> list[dict]:
    """
    Discover all course materials (optionally filtered to one course).
    Each file gets a 'global_num' (1-based).
    """
    if course_id:
        courses = [get_course_by_id(canvas, course_id)]
    else:
        courses = get_academic_courses(canvas)

    all_files: list[dict] = []
    with tqdm(courses, desc="Scanning courses", unit="course") as bar:
        for course in bar:
            bar.set_postfix_str(course.name[:40])
            all_files.extend(get_course_files(course))

    for i, f in enumerate(all_files, start=1):
        f["global_num"] = i

    return all_files


def _classify_with_ai(files: list[dict], course_name: str) -> list[dict]:
    """Use Claude to select lecture-note and tutorial files from a large course."""
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        kf = PROJECT_DIR / "anthropic_key.txt"
        if kf.exists():
            key = kf.read_text().strip()
    client = anthropic.Anthropic(api_key=key)

    lines = [
        f'  id={f["id"]}  folder="{f["folder_path"]}"  '
        f'name="{f["display_name"]}"  size={f["size"]/1024/1024:.1f}MB'
        for f in files
    ]
    prompt = (
        f"You are helping organize university course materials for: {course_name}\n\n"
        f"Here is a list of all available files:\n" + "\n".join(lines) + "\n\n"
        "Task: identify which files are LECTURE NOTES or TUTORIAL materials.\n"
        "- Lecture notes: slides, lecture PDFs, notes, handouts, readings\n"
        "- Tutorials: tutorial sheets, problem sets, worksheets, exercises, lab guides\n\n"
        "Return ONLY a JSON array of the integer file IDs to download. No explanation.\n"
        "Example: [123, 456, 789]"
    )
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    text  = resp.content[0].text.strip()
    match = re.search(r'\[[\d,\s]+\]', text)
    if not match:
        tqdm.write("    [warn] AI returned unexpected format; using all files")
        return files
    selected_ids = set(json.loads(match.group()))
    selected     = [f for f in files if f["id"] in selected_ids]
    tqdm.write(f"    AI selected {len(selected)}/{len(files)} files")
    return selected


def print_material_list(files: list[dict], logs: dict[int, dict]) -> None:
    """Print a numbered material listing."""
    print()
    print(f"  {'#':>4}  {'Status':<10}  {'Course':<20}  {'Folder':<22}  {'Size':>8}  Name")
    print(f"  {'-'*4}  {'-'*10}  {'-'*20}  {'-'*22}  {'-'*8}  {'-'*40}")
    for f in files:
        log      = logs.get(f["course_id"], {})
        status   = "done" if (str(f["id"]) in log and
                              Path(log[str(f["id"])]["path"]).exists()) else "pending"
        mark     = "✓" if status == "done" else " "
        course   = f["course_name"][:20]
        folder   = (f["folder_path"] or "/")[:22]
        size_kb  = f"{f['size']//1024:>6} KB" if f["size"] else "     ? KB"
        name     = f["display_name"][:50]
        print(f"  {f['global_num']:>4}  [{mark}]{status:<8}  {course:<20}  {folder:<22}  {size_kb}  {name}")
    print()
    print(f"  Total: {len(files)} file(s)")
    print()


def download_material(file_info: dict, log: dict, base_dir: Path) -> bool:
    """Download a single course file with a tqdm progress bar."""
    fid  = str(file_info["id"])
    name = file_info["display_name"]
    cid  = file_info["course_id"]

    if fid in log and Path(log[fid]["path"]).exists():
        tqdm.write(f"  [skip] {name}")
        return True

    folder_rel = file_info["folder_path"]
    dest_dir   = base_dir / str(cid) / "materials"
    if folder_rel:
        dest_dir = dest_dir / Path(*[_sanitize(p) for p in folder_rel.split("/")])
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / _sanitize(name)

    try:
        r = requests.get(
            file_info["url"],
            headers=_canvas_headers(),
            stream=True, timeout=60, allow_redirects=True,
        )
        r.raise_for_status()

        file_size = file_info["size"] or 0
        bar = tqdm(
            total=file_size if file_size else None,
            desc=f"  {name[:50]}",
            unit="B", unit_scale=True, unit_divisor=1024,
            bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
            leave=True,
        )
        written = 0
        with open(dest_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    fh.write(chunk)
                    written += len(chunk)
                    bar.update(len(chunk))
        bar.close()

        log[fid] = {
            "display_name":  name,
            "folder":        folder_rel,
            "size":          written,
            "path":          str(dest_path),
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        }
        tqdm.write(f"  Saved → {dest_path}")
        return True

    except Exception as e:
        tqdm.write(f"  [error] {name}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def _load_all_logs(courses: list, base_dir: Path) -> dict[int, dict]:
    """Load download logs for all given courses, keyed by course_id."""
    logs = {}
    for c in courses:
        log_path = base_dir / str(c.id) / "materials" / "download_log.json"
        logs[c.id] = _load_json(log_path)
    return logs


def _save_log(course_id: int, log: dict, base_dir: Path) -> None:
    path = base_dir / str(course_id) / "materials" / "download_log.json"
    _save_json(path, log)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Canvas downloader for videos and course materials",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Scope ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--course", metavar="ID", type=int,
        help="Restrict all operations to a single course by Canvas ID",
    )
    parser.add_argument(
        "--path", metavar="PATH",
        help="Base directory for downloads (default: project directory)",
    )
    parser.add_argument(
        "--secretly", action="store_true",
        help="Wait 5–15 min between videos and 2–5 min between folders "
             "to avoid triggering rate-limits or admin alerts",
    )

    # ── Course listing ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--course-list", action="store_true",
        help="List all active academic courses",
    )

    # ── Video commands ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--video-list", action="store_true",
        help="List all available videos (combine with --course to filter)",
    )
    parser.add_argument(
        "--download-video", metavar="N", nargs="+", type=int,
        help="Download specific video(s) by number. Numbers are course-local "
             "when --course is given, global otherwise.",
    )
    parser.add_argument(
        "--download-video-all", action="store_true",
        help="Download all pending videos (combine with --course to filter)",
    )

    # ── Material commands ──────────────────────────────────────────────────────
    parser.add_argument(
        "--material-list", action="store_true",
        help="List all available course files (combine with --course to filter)",
    )
    parser.add_argument(
        "--download-material", metavar="FILENAME", nargs="+",
        help="Download material(s) by filename. Combine with --course to "
             "disambiguate files with the same name across courses.",
    )
    parser.add_argument(
        "--download-material-all", action="store_true",
        help="Download all pending materials (combine with --course to filter)",
    )

    args = parser.parse_args()

    # At least one action required
    actions = [
        args.course_list, args.video_list, args.download_video,
        args.download_video_all, args.material_list,
        args.download_material, args.download_material_all,
    ]
    if not any(actions):
        parser.print_help()
        sys.exit(0)

    base_dir = Path(args.path) if args.path else PROJECT_DIR
    canvas   = Canvas(CANVAS_URL, CANVAS_TOKEN)

    # ── --course-list ──────────────────────────────────────────────────────────
    if args.course_list:
        print("\nActive academic courses:")
        print(f"  {'ID':<10}  Course")
        print(f"  {'-'*10}  {'-'*50}")
        for c in get_academic_courses(canvas):
            print(f"  {c.id:<10}  {c.name}")
        print()
        return

    # ── --video-list ───────────────────────────────────────────────────────────
    if args.video_list:
        print(f"\nDiscovering videos{' for course ' + str(args.course) if args.course else ''}...")
        videos   = discover_videos(canvas, args.course)
        manifest = _load_json(MANIFEST_FILE)
        print_video_list(videos, manifest, by_course=bool(args.course))
        return

    # ── --download-video N [N ...] ─────────────────────────────────────────────
    if args.download_video:
        print(f"\nDiscovering videos...")
        videos   = discover_videos(canvas, args.course)
        manifest = _load_json(MANIFEST_FILE)
        num_key  = "course_num" if args.course else "global_num"
        targets  = {v for v in videos if v[num_key] in args.download_video}
        missing  = set(args.download_video) - {v[num_key] for v in targets}
        if missing:
            print(f"  [warn] No video(s) found for number(s): {sorted(missing)}")
        if not targets:
            sys.exit(1)

        print(f"\nDownloading {len(targets)} video(s):")
        for i, video in enumerate(sorted(targets, key=lambda v: v[num_key])):
            print(f"\n[{i+1}/{len(targets)}] {video['title']}")
            download_video(video, manifest, base_dir)
            _save_json(MANIFEST_FILE, manifest)
            if args.secretly and i < len(targets) - 1:
                _secretly_wait_video()

        print(f"\nDone.")
        return

    # ── --download-video-all ───────────────────────────────────────────────────
    if args.download_video_all:
        print(f"\nDiscovering videos{' for course ' + str(args.course) if args.course else ''}...")
        videos   = discover_videos(canvas, args.course)
        manifest = _load_json(MANIFEST_FILE)
        pending  = [v for v in videos
                    if manifest.get(str(v["item_id"]), {}).get("status") != "done"]

        print(f"\nDownloading {len(pending)} pending video(s) "
              f"(of {len(videos)} total):")

        for i, video in enumerate(pending):
            print(f"\n[{i+1}/{len(pending)}] {video['title']}")
            download_video(video, manifest, base_dir)
            _save_json(MANIFEST_FILE, manifest)
            if args.secretly and i < len(pending) - 1:
                _secretly_wait_video()

        print(f"\nDone.")
        return

    # ── --material-list ────────────────────────────────────────────────────────
    if args.material_list:
        print(f"\nDiscovering materials{' for course ' + str(args.course) if args.course else ''}...")
        files = discover_materials(canvas, args.course)
        if args.course:
            courses = [get_course_by_id(canvas, args.course)]
        else:
            courses = get_academic_courses(canvas)
        logs = _load_all_logs(courses, base_dir)
        print_material_list(files, logs)
        return

    # ── --download-material FILENAME [FILENAME ...] ────────────────────────────
    if args.download_material:
        print(f"\nDiscovering materials{' for course ' + str(args.course) if args.course else ''}...")
        files = discover_materials(canvas, args.course)

        targets: list[dict] = []
        for query in args.download_material:
            matches = [f for f in files
                       if query.lower() in f["display_name"].lower()]
            if not matches:
                print(f"  [warn] No file matching '{query}' found")
            else:
                if len(matches) > 1:
                    print(f"  [info] '{query}' matched {len(matches)} file(s) — downloading all:")
                    for m in matches:
                        print(f"    {m['course_name']} / {m['folder_path']} / {m['display_name']}")
                targets.extend(matches)

        if not targets:
            sys.exit(1)

        # Group by course for log management
        by_course: dict[int, list[dict]] = {}
        for f in targets:
            by_course.setdefault(f["course_id"], []).append(f)

        prev_cid: int | None = None
        for cid, cfiles in by_course.items():
            log_path = base_dir / str(cid) / "materials" / "download_log.json"
            log      = _load_json(log_path)
            print(f"\nCourse {cid}:")
            for i, f in enumerate(cfiles):
                download_material(f, log, base_dir)
                _save_json(log_path, log)
            if args.secretly and prev_cid is not None:
                _secretly_wait_dir()
            prev_cid = cid

        print(f"\nDone.")
        return

    # ── --download-material-all ────────────────────────────────────────────────
    if args.download_material_all:
        if args.course:
            courses = [get_course_by_id(canvas, args.course)]
        else:
            courses = get_academic_courses(canvas)

        total_dl = total_sk = total_err = 0

        for ci, course in enumerate(courses):
            print(f"\n{'='*70}")
            print(f"Course: {course.name}  (id={course.id})")
            print(f"{'='*70}")

            log_path = base_dir / str(course.id) / "materials" / "download_log.json"
            log      = _load_json(log_path)

            print("  Enumerating files...")
            files = get_course_files(course)
            if not files:
                print("  No files found.")
                continue

            total_size = sum(f["size"] for f in files)
            total_mb   = total_size / 1024 / 1024
            print(f"  {len(files)} file(s)  |  {total_mb:.1f} MB total")

            # AI filtering for large courses
            if total_size >= SIZE_LIMIT:
                print(f"  Size ≥ 1 GB — using AI to select lecture notes & tutorials...")
                to_download = _classify_with_ai(files, course.name)
            else:
                to_download = files

            # Track previous folder for secretly waits
            prev_folder: str | None = None
            dl = sk = err = 0

            for f in to_download:
                fid = str(f["id"])
                if fid in log and Path(log[fid]["path"]).exists():
                    sk += 1
                    tqdm.write(f"  [skip] {f['display_name']}")
                    continue

                # Secretly: wait between folders
                cur_folder = f["folder_path"]
                if args.secretly and prev_folder is not None and cur_folder != prev_folder:
                    _secretly_wait_dir()
                prev_folder = cur_folder

                if download_material(f, log, base_dir):
                    dl += 1
                else:
                    err += 1
                _save_json(log_path, log)

            print(f"\n  Done: {dl} downloaded, {sk} skipped, {err} errors")
            total_dl += dl; total_sk += sk; total_err += err

            # Secretly: wait between courses
            if args.secretly and ci < len(courses) - 1:
                _secretly_wait_dir()

        print(f"\n{'='*70}")
        print(f"All done: {total_dl} downloaded, {total_sk} skipped, "
              f"{total_err} errors across {len(courses)} course(s)")


if __name__ == "__main__":
    main()
