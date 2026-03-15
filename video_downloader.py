"""
Canvas Lecture Video Downloader
Downloads Panopto lecture videos from Canvas and organizes them into course folders.

Structure: [project_folder]/[course_id]/videos/[video_title].mp4

Requirements (auto-note conda env):
  canvasapi, requests, playwright, PanoptoDownloader

Usage:
  python video_downloader.py              # list all videos
  python video_downloader.py -d 1        # download 1 video
  python video_downloader.py -d 5        # download 5 videos
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import requests
from canvasapi import Canvas
from playwright.sync_api import sync_playwright

# ── Configuration ────────────────────────────────────────────────────────────
CANVAS_URL = "https://canvas.nus.edu.sg"
PANOPTO_HOST = "mediaweb.ap.panopto.com"
PROJECT_DIR = Path(__file__).parent
_canvas_token_file = PROJECT_DIR / "canvas_token.txt"
CANVAS_TOKEN = (
    _canvas_token_file.read_text().strip()
    if _canvas_token_file.exists() else
    os.environ.get("CANVAS_TOKEN", "")
)
MANIFEST_FILE = PROJECT_DIR / "manifest.json"

# ── Manifest helpers ──────────────────────────────────────────────────────────

def load_manifest() -> dict:
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict):
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

# ── Canvas video discovery ────────────────────────────────────────────────────

def get_active_courses(canvas: Canvas) -> list[dict]:
    """Return all active courses the user is enrolled in."""
    courses = []
    for c in canvas.get_courses(enrollment_state="active"):
        try:
            courses.append({"id": c.id, "name": c.name})
        except AttributeError:
            pass
    return courses


def find_panopto_items_in_course(canvas: Canvas, course_id: int) -> list[dict]:
    """Scan all modules in a course for Panopto ExternalTool items."""
    videos = []
    try:
        course = canvas.get_course(course_id)
        for module in course.get_modules():
            for item in module.get_module_items():
                if item.type == "ExternalTool":
                    url = getattr(item, "external_url", "") or ""
                    if "panopto" in url.lower():
                        videos.append({
                            "course_id": course_id,
                            "course_name": course.name,
                            "module_name": module.name,
                            "title": item.title,
                            "lti_url": url,
                            "item_id": item.id,
                        })
    except Exception as e:
        print(f"  [warn] course {course_id}: {e}")
    return videos


def list_all_videos(canvas: Canvas) -> list[dict]:
    """Discover all Panopto videos across all active courses."""
    courses = get_active_courses(canvas)
    print(f"Found {len(courses)} active courses.")
    all_videos = []
    for c in courses:
        print(f"  Scanning course {c['id']}: {c['name']}")
        videos = find_panopto_items_in_course(canvas, c["id"])
        if videos:
            print(f"    -> {len(videos)} Panopto video(s) found")
        all_videos.extend(videos)
    return all_videos


# ── Panopto stream discovery via Canvas sessionless launch ────────────────────

def get_sessionless_launch_url(course_id: int, item_id: int) -> str | None:
    """Get a Canvas sessionless launch URL for a module item (no browser login needed)."""
    r = requests.get(
        f"{CANVAS_URL}/api/v1/courses/{course_id}/external_tools/sessionless_launch"
        f"?launch_type=module_item&module_item_id={item_id}",
        headers={"Authorization": f"Bearer {CANVAS_TOKEN}"},
        timeout=30,
    )
    if r.status_code != 200:
        return None
    return r.json().get("url")


def get_panopto_session_id_and_cookies(launch_url: str) -> tuple[str | None, list[dict]]:
    """
    Use Playwright to follow the Canvas sessionless launch URL (no SSO required).
    Returns (panopto_session_id, panopto_cookies).
    """
    session_id = None
    panopto_cookies = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context()
        page = ctx.new_page()
        try:
            page.goto(launch_url, wait_until="networkidle", timeout=60000)
            time.sleep(2)

            # Extract session ID from final Panopto URL
            final_url = page.url
            match = re.search(r"[?&]id=([0-9a-f-]{36})", final_url, re.IGNORECASE)
            if match:
                session_id = match.group(1)

            # Collect Panopto cookies
            all_cookies = ctx.cookies()
            panopto_cookies = [
                c for c in all_cookies
                if "panopto" in c.get("domain", "").lower()
            ]
        except Exception as e:
            print(f"  [warn] Playwright error: {e}")
        finally:
            browser.close()

    return session_id, panopto_cookies


def get_stream_url(session_id: str, panopto_cookies: list[dict]) -> str | None:
    """
    Call Panopto's DeliveryInfo endpoint to get the HLS master.m3u8 URL.
    Returns the primary (DV / camera) stream URL.
    """
    sess = requests.Session()
    for ck in panopto_cookies:
        sess.cookies.set(ck["name"], ck["value"], domain=ck.get("domain", ""))

    r = sess.post(
        f"https://{PANOPTO_HOST}/Panopto/Pages/Viewer/DeliveryInfo.aspx",
        data={
            "deliveryId": session_id,
            "isLiveNotes": "false",
            "refreshAuthCookie": "true",
            "isActiveBroadcast": "false",
            "isEditing": "false",
            "isKollectiveAgentInstalled": "false",
            "isEmbed": "true",
            "responseType": "json",
        },
        timeout=30,
    )
    if r.status_code != 200:
        return None

    info = r.json()
    streams = info.get("Delivery", {}).get("Streams", [])

    # Prefer "DV" (camera) tag, fall back to first stream
    for preferred_tag in ("DV", "OBJECT", None):
        for s in streams:
            url = s.get("StreamUrl", "")
            if url and (preferred_tag is None or s.get("Tag") == preferred_tag):
                return url
    return None


# ── Download orchestration ────────────────────────────────────────────────────

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()


def download_video(video: dict, manifest: dict) -> bool:
    """Download a single video. Returns True on success."""
    import PanoptoDownloader

    item_id = video["item_id"]
    course_id = video["course_id"]
    title = video["title"]
    manifest_key = str(item_id)

    if manifest.get(manifest_key, {}).get("status") == "done":
        print(f"  [skip] Already downloaded: {title}")
        return True

    # Create output directory
    out_dir = PROJECT_DIR / str(course_id) / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_title = sanitize_filename(title)
    out_path = out_dir / f"{safe_title}.mp4"

    if out_path.exists():
        print(f"  [skip] File exists: {out_path}")
        manifest[manifest_key] = {"status": "done", "path": str(out_path)}
        return True

    # Step A: get sessionless launch URL (no SSO)
    print(f"  Getting launch URL for item {item_id}...")
    launch_url = get_sessionless_launch_url(course_id, item_id)
    if not launch_url:
        print(f"  [error] Could not get launch URL")
        manifest[manifest_key] = {"status": "error", "title": title}
        return False

    # Step B: launch in Playwright, get Panopto session + cookies
    print(f"  Launching Panopto session...")
    session_id, panopto_cookies = get_panopto_session_id_and_cookies(launch_url)
    if not session_id:
        print(f"  [error] Could not get Panopto session ID")
        manifest[manifest_key] = {"status": "error", "title": title}
        return False
    print(f"  Panopto session: {session_id}")

    # Step C: get HLS stream URL
    stream_url = get_stream_url(session_id, panopto_cookies)
    if not stream_url:
        print(f"  [error] Could not get stream URL")
        manifest[manifest_key] = {"status": "error", "title": title}
        return False
    print(f"  Stream: {stream_url[:80]}...")

    # Step D: download with PanoptoDownloader (ffmpeg HLS)
    print(f"  Downloading -> {out_path}")

    def progress_cb(pct):
        print(f"\r    {int(pct):3d}%", end="", flush=True)

    try:
        PanoptoDownloader.download(stream_url, str(out_path), progress_cb)
        print()
        print(f"  Done: {out_path}")
        manifest[manifest_key] = {
            "status": "done",
            "path": str(out_path),
            "title": title,
            "session_id": session_id,
        }
        return True
    except Exception as e:
        print(f"\n  [error] Download failed: {e}")
        if out_path.exists():
            out_path.unlink()
        manifest[manifest_key] = {"status": "error", "title": title, "error": str(e)}
        return False


# ── Main entry point ──────────────────────────────────────────────────────────

def main(max_downloads: int = 0):
    canvas = Canvas(CANVAS_URL, CANVAS_TOKEN)
    manifest = load_manifest()

    # ── Step 1: List all videos ───────────────────────────────────────────────
    print("\n=== Step 1: Discovering Panopto videos on Canvas ===")
    videos = list_all_videos(canvas)
    print(f"\nTotal Panopto videos found: {len(videos)}")
    for i, v in enumerate(videos, 1):
        status = manifest.get(str(v["item_id"]), {}).get("status", "pending")
        print(f"  [{i:2d}] [{status}] {v['course_name']} | {v['module_name']} | {v['title']}")

    if max_downloads == 0:
        print("\n[info] Listing only. Use -d N to download N videos.")
        return

    # ── Step 2–3: Download videos ─────────────────────────────────────────────
    pending = [v for v in videos if manifest.get(str(v["item_id"]), {}).get("status") != "done"]
    print(f"\n=== Downloading up to {max_downloads} of {len(pending)} pending video(s) ===")

    downloaded = 0
    for video in pending:
        if downloaded >= max_downloads:
            break
        print(f"\n[{downloaded+1}/{max_downloads}] {video['title']}")
        if download_video(video, manifest):
            downloaded += 1
        save_manifest(manifest)

    print(f"\n=== Done: {downloaded} video(s) downloaded ===")
    print("Files saved in:")
    for v in videos:
        key = str(v["item_id"])
        if manifest.get(key, {}).get("status") == "done":
            print(f"  {manifest[key]['path']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Canvas Lecture Video Downloader")
    parser.add_argument(
        "--download", "-d", type=int, default=0, metavar="N",
        help="Number of videos to download (0 = list only)"
    )
    args = parser.parse_args()
    main(max_downloads=args.download)
