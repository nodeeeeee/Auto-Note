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
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
    from canvasapi import Canvas
    from tqdm import tqdm
except ImportError as _e:
    print(f"[error] Missing dependency: {_e}")
    print("[error] Please install the ML environment from Settings → ML Environment in the AutoNote app.")
    sys.exit(1)

# ── Configuration ──────────────────────────────────────────────────────────────

PROJECT_DIR   = Path(__file__).parent
MANIFEST_FILE = None   # set after DATA_DIR is resolved below

# Persistent data dir: ~/.auto_note/ when installed (AppImage or scripts/ copy),
# else the project directory (dev mode).
_AUTO_NOTE_DIR = Path.home() / ".auto_note"
# AUTONOTE_DATA_DIR env var lets the Electron app (or any launcher) explicitly
# set the data directory so config/credentials are always found regardless of
# where the script file lives (dev mode vs installed).
if os.environ.get("AUTONOTE_DATA_DIR"):
    DATA_DIR = Path(os.environ["AUTONOTE_DATA_DIR"])
elif getattr(sys, "frozen", False) or PROJECT_DIR == _AUTO_NOTE_DIR / "scripts":
    DATA_DIR = _AUTO_NOTE_DIR
else:
    DATA_DIR = PROJECT_DIR

MANIFEST_FILE = DATA_DIR / "manifest.json"

# ── User-configurable connection settings (set via GUI Settings page) ──────────
_config_file = DATA_DIR / "config.json"
_config: dict = json.load(open(_config_file, encoding="utf-8")) if _config_file.exists() else {}

CANVAS_URL   = _config.get("CANVAS_URL",   "").strip().rstrip("/")
if CANVAS_URL and not CANVAS_URL.startswith(("http://", "https://")):
    CANVAS_URL = "https://" + CANVAS_URL
PANOPTO_HOST = _config.get("PANOPTO_HOST", "")
# Strip protocol prefix if user accidentally stored a full URL (e.g. "https://mediaweb.ap.panopto.com")
if PANOPTO_HOST.startswith(("https://", "http://")):
    PANOPTO_HOST = re.sub(r'^https?://', '', PANOPTO_HOST).split('/')[0]

_canvas_token_file = DATA_DIR / "canvas_token.txt"
CANVAS_TOKEN = (
    # strip() covers leading/trailing whitespace; split()[0] takes the first
    # token only, guarding against files that accidentally contain two tokens
    # separated by newlines (e.g. copy-paste from two accounts).
    (_canvas_token_file.read_text().split()[0] if _canvas_token_file.exists() else None)
    or os.environ.get("CANVAS_TOKEN", "")
)
SIZE_LIMIT    = 1 * 1024 ** 3   # 1 GB

SKIP_COURSE_KEYWORDS = [
    "training", "pdp", "rmcpdp", "osa", "soct", "travel", "essentials",
    "respect", "consent",
]

# Slack-mode delay ranges (seconds)
_SECRETLY_VIDEO_MIN = 5  * 60    # 5 min
_SECRETLY_VIDEO_MAX = 15 * 60    # 15 min
_SECRETLY_DIR_MIN   = 2  * 60    # 2 min
_SECRETLY_DIR_MAX   = 5  * 60    # 5 min


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _sanitize(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()


def _canvas_headers() -> dict:
    return {"Authorization": f"Bearer {CANVAS_TOKEN}"}


_playwright_checked = False

def _ensure_playwright_browsers() -> None:
    """Auto-install Chromium browser for Playwright if not already present.

    Called once per process — checks if the Chromium binary exists, and
    if not, runs `playwright install chromium` automatically so the user
    never has to do it themselves.
    """
    global _playwright_checked
    if _playwright_checked:
        return
    _playwright_checked = True

    try:
        from playwright._impl._driver import compute_driver_executable
        driver_exec = compute_driver_executable()
        import subprocess as _sp
        result = _sp.run(
            [str(driver_exec), "install", "--dry-run", "chromium"],
            capture_output=True, text=True, timeout=10,
        )
        # If dry-run succeeds without mentioning "not installed", we're good
        if result.returncode == 0:
            return
    except Exception:
        pass

    # Check by actually trying to find the executable path
    try:
        from playwright.sync_api import sync_playwright as _sp_check
        with _sp_check() as p:
            p.chromium.executable_path  # noqa: B018
        return  # browser exists
    except Exception:
        pass

    # Browser not found — install it
    tqdm.write("  [playwright] Chromium not found — installing automatically…")
    import subprocess as _sp
    try:
        proc = _sp.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode == 0:
            tqdm.write("  [playwright] Chromium installed successfully.")
        else:
            tqdm.write(f"  [playwright] Install exited with code {proc.returncode}")
            if proc.stderr:
                tqdm.write(f"  [playwright] {proc.stderr.strip()[:200]}")
    except Exception as e:
        tqdm.write(f"  [playwright] Auto-install failed: {e}")
        tqdm.write("  [playwright] Run manually: python -m playwright install chromium")


def _is_academic(course) -> bool:
    try:
        name = course.name.lower()
    except AttributeError:
        return False
    return not any(kw in name for kw in SKIP_COURSE_KEYWORDS)


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── Concurrent transcription helper ───────────────────────────────────────────

_transcribe_procs: list[subprocess.Popen] = []


def _spawn_transcribe(video_path: str) -> None:
    """Spawn transcription for a video in the background.

    Runs extract_caption.py --video <path> as a subprocess.
    The download loop continues immediately without waiting.
    """
    script = PROJECT_DIR / "extract_caption.py"
    # Also check installed scripts dir
    installed = Path.home() / ".auto_note" / "scripts" / "extract_caption.py"
    if installed.exists():
        script = installed

    cmd = [sys.executable, str(script), "--video", video_path]
    tqdm.write(f"  [transcribe] Starting background transcription: {Path(video_path).name}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "AUTONOTE_DATA_DIR": str(DATA_DIR)},
    )
    _transcribe_procs.append(proc)


def _wait_transcriptions() -> None:
    """Wait for all background transcription processes to finish."""
    if not _transcribe_procs:
        return
    tqdm.write(f"\n  Waiting for {len(_transcribe_procs)} background transcription(s) to finish...")
    for proc in _transcribe_procs:
        for line in proc.stdout:
            tqdm.write(f"  [transcribe] {line.rstrip()}")
        proc.wait()
        if proc.returncode == 0:
            tqdm.write(f"  [transcribe] Done (exit 0)")
        else:
            tqdm.write(f"  [transcribe] Failed (exit {proc.returncode})")
    _transcribe_procs.clear()


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
    """Return all academic courses the user has any enrollment in.

    Intentionally does NOT filter by enrollment_state="active" so that
    courses with TA, auditor, design-experience, or cross-listed enrollments
    are included — these often have an enrollment state other than "active".
    """
    courses = []
    for c in canvas.get_courses():
        if _is_academic(c):
            courses.append(c)
    return courses


def get_course_by_id(canvas: Canvas, course_id: int):
    """Fetch a single course by ID (even if not in active list)."""
    return canvas.get_course(course_id)


# ══════════════════════════════════════════════════════════════════════════════
#  VIDEO SECTION
# ══════════════════════════════════════════════════════════════════════════════

# Canvas external-tool ID for the "Videos/Panopto" course navigation tab.
# Used as a fallback; the actual ID is discovered per-course via
# _resolve_panopto_tool_id() because Canvas assigns different tool IDs
# per-institution and sometimes per-course.
_PANOPTO_TAB_TOOL_ID_FALLBACK = 128

# Stream-tag preference order. Can be overridden at runtime via
# AUTONOTE_PREFER_STREAM or the --prefer-stream CLI flag. Default keeps the
# existing screen-share-first behaviour, which is right for most lectures.
_PREFER_STREAM_ORDER: tuple[str, ...] = tuple(
    os.environ.get("AUTONOTE_PREFER_STREAM", "SS,OBJECT,DV").split(",")
)

# Per-course cache for resolved Panopto tool IDs.
_PANOPTO_TOOL_ID_CACHE: dict[int, int] = {}


def _resolve_panopto_tool_id(course_id: int) -> int | None:
    """Discover the Canvas external-tool ID of the Panopto/Videos tab for a
    specific course.

    Canvas assigns different external-tool IDs per-institution and sometimes
    per-course, so the old hardcoded ID (128) only worked for the author's
    default token. For a different token or course (e.g. EE4802), a different
    tool ID or tool set may apply — and Strategy 2 silently returned zero.

    Resolution order:
      1. /api/v1/courses/<id>/tabs — works even when the LTI tool is account-
         scoped (not "installed" at the course level). The tab id is of the
         form "context_external_tool_<N>", so we parse N directly. This is
         what the UI itself uses to render the left-hand navigation.
      2. /api/v1/courses/<id>/external_tools — works for tools explicitly
         installed in the course. Matches on name/domain/url.
    Results are cached.
    """
    if course_id in _PANOPTO_TOOL_ID_CACHE:
        return _PANOPTO_TOOL_ID_CACHE[course_id]

    _VIDEO_LABELS = ("videos/panopto", "videos", "video", "lecture videos", "panopto")

    # ── Strategy A: course tabs (most reliable for account-scoped tools) ─────
    try:
        r = requests.get(
            f"{CANVAS_URL}/api/v1/courses/{course_id}/tabs",
            headers=_canvas_headers(),
            timeout=30,
        )
        if r.status_code == 200:
            for tab in r.json():
                tid_str = (tab.get("id") or "")
                label   = (tab.get("label") or "").lower()
                if not tid_str.startswith("context_external_tool_"):
                    continue
                if "panopto" in label or label in _VIDEO_LABELS:
                    try:
                        tid = int(tid_str.rsplit("_", 1)[-1])
                        _PANOPTO_TOOL_ID_CACHE[course_id] = tid
                        return tid
                    except ValueError:
                        continue
    except Exception as e:
        tqdm.write(f"  [warn] _resolve_panopto_tool_id tabs({course_id}): {e}")

    # ── Strategy B: external_tools (explicit course-level installs) ──────────
    try:
        r = requests.get(
            f"{CANVAS_URL}/api/v1/courses/{course_id}/external_tools",
            headers=_canvas_headers(),
            params={"per_page": 100},
            timeout=30,
        )
        if r.status_code == 200:
            for tool in r.json():
                name   = (tool.get("name")   or "").lower()
                domain = (tool.get("domain") or "").lower()
                url    = (tool.get("url")    or "").lower()
                if ("panopto" in name or "panopto" in domain or "panopto" in url
                        or name in _VIDEO_LABELS):
                    tid = int(tool["id"])
                    _PANOPTO_TOOL_ID_CACHE[course_id] = tid
                    return tid
    except Exception as e:
        tqdm.write(f"  [warn] _resolve_panopto_tool_id tools({course_id}): {e}")

    return None


def _find_panopto_items(canvas: Canvas, course) -> list[dict]:
    """
    Discover all Panopto sessions for a course using three strategies:

    1. Module ExternalTool items  (e.g. CS3210 — each lecture linked in a module)
    2. Panopto folder API         (e.g. CS2105 — sessions stored in a Panopto folder)
    3. Canvas Pages scan          (e.g. CS2101 — Viewer.aspx links embedded in pages)

    All three are merged and de-duplicated by session UUID.
    """
    seen_ids:    set[str] = set()
    seen_titles: set[str] = set()   # title-based dedup across strategies
    videos:      list[dict] = []

    def _add(v: dict) -> None:
        uid = str(v["item_id"])
        if uid in seen_ids:
            return
        # Cross-strategy dedup: Panopto "delivery" IDs (LTI) differ from
        # "session" IDs (folder API), so fall back to title matching.
        title_key = v.get("title", "").strip().lower()
        if title_key and title_key in seen_titles:
            return
        seen_ids.add(uid)
        if title_key:
            seen_titles.add(title_key)
        videos.append(v)

    # ── Strategy 1: module ExternalTool items ─────────────────────────────────
    s1_count = 0
    try:
        for module in course.get_modules():
            for item in module.get_module_items():
                if item.type == "ExternalTool":
                    url = getattr(item, "external_url", "") or ""
                    if "panopto" in url.lower():
                        _add({
                            "course_id":   course.id,
                            "course_name": course.name,
                            "module_name": module.name,
                            "title":       item.title,
                            "lti_url":     url,
                            "item_id":     item.id,
                            "viewer_url":  None,
                        })
                        s1_count += 1
    except Exception as e:
        tqdm.write(f"  [warn] Strategy 1 (modules) {course.id}: {e}")
    tqdm.write(f"  [info] Strategy 1 (modules):      {s1_count} video(s)")

    # ── Strategy 2: Panopto folder API (Videos/Panopto tab) ───────────────────
    s2_count = 0
    try:
        tqdm.write(f"  [info] Strategy 2 (Panopto tab): launching browser…")
        folder_id, panopto_cookies, bearer_token = _get_panopto_tab_folder(course.id)
        if folder_id:
            tqdm.write(f"  [info] Strategy 2: folder_id={folder_id}, scanning…")
            for s in _iter_panopto_folder(folder_id, panopto_cookies, bearer_token=bearer_token):
                viewer_url = (f"https://{PANOPTO_HOST}/Panopto/Pages/Viewer.aspx"
                              f"?id={s['session_id']}")
                _add({
                    "course_id":   course.id,
                    "course_name": course.name,
                    "module_name": s.get("folder_name", "Videos/Panopto"),
                    "title":       s["name"],
                    "lti_url":     None,
                    "item_id":     s["session_id"],
                    "viewer_url":  viewer_url,
                    "_panopto_cookies": panopto_cookies,
                    "_bearer_token":    bearer_token,
                })
                s2_count += 1
        else:
            tqdm.write(f"  [warn] Strategy 2: could not find Panopto folder (browser may have failed)")
    except Exception as e:
        tqdm.write(f"  [warn] Strategy 2 (Panopto folder) {course.id}: {e}")
    tqdm.write(f"  [info] Strategy 2 (Panopto tab):   {s2_count} video(s)")

    # ── Strategy 3: Canvas Pages scan (fallback only) ─────────────────────────
    if not videos:
        tqdm.write(f"  [info] Strategy 3 (pages scan):   trying…")
        try:
            s3_before = len(videos)
            for v in _find_panopto_in_pages(course):
                _add(v)
            tqdm.write(f"  [info] Strategy 3 (pages scan):   {len(videos) - s3_before} video(s)")
        except Exception as e:
            tqdm.write(f"  [warn] Strategy 3 (pages scan) {course.id}: {e}")

    return videos


def _get_panopto_tab_folder(course_id: int) -> tuple[str | None, list[dict], str | None]:
    """
    Launch the Videos/Panopto tab via Canvas sessionless LTI, navigate with
    Playwright, capture the root Panopto folder ID, auth cookies, and OAuth
    Bearer token.
    Returns (folder_id, panopto_cookies, bearer_token).
    """
    global PANOPTO_HOST  # may be auto-detected from Playwright request URLs
    _ensure_playwright_browsers()
    from playwright.sync_api import sync_playwright

    tool_id = _resolve_panopto_tool_id(course_id) or _PANOPTO_TAB_TOOL_ID_FALLBACK
    r = requests.get(
        f"{CANVAS_URL}/api/v1/courses/{course_id}/external_tools/sessionless_launch"
        f"?id={tool_id}&launch_type=course_navigation",
        headers=_canvas_headers(),
        timeout=30,
    )
    if r.status_code != 200:
        tqdm.write(f"  [warn] Panopto tab launch failed (tool_id={tool_id}, status={r.status_code})")
        return None, [], None
    launch_url = r.json().get("url")
    if not launch_url:
        return None, [], None

    folder_id:      str | None = None
    bearer_token:   str | None = None
    detected_host:  str | None = None
    cookies:        list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx     = browser.new_context()
        page    = ctx.new_page()

        def _on_request(req: object) -> None:
            nonlocal folder_id, bearer_token, detected_host
            url = req.url
            m = re.search(r"folderID=([0-9a-f-]{36})", url, re.IGNORECASE)
            if m and folder_id is None:
                folder_id = m.group(1)
            # Auto-detect Panopto hostname from any Panopto request URL.
            if detected_host is None and "panopto" in url.lower():
                from urllib.parse import urlparse
                netloc = urlparse(url).netloc
                if netloc and "panopto" in netloc.lower():
                    detected_host = netloc
            # Capture the OAuth Bearer token issued to the embedded page.
            # The WCF GetSessions endpoint requires it to return Subfolders.
            auth = req.headers.get("authorization", "")
            if auth.startswith("Bearer ") and bearer_token is None:
                bearer_token = auth[len("Bearer "):]

        page.on("request", _on_request)
        try:
            page.goto(launch_url, wait_until="networkidle", timeout=60000)
            time.sleep(3)
            cookies = [c for c in ctx.cookies()
                       if "panopto" in c.get("domain", "").lower()]
            # Also try cookies domain as a fallback host source
            if detected_host is None and cookies:
                domain = cookies[0].get("domain", "").lstrip(".")
                if domain:
                    detected_host = domain
        except Exception as e:
            tqdm.write(f"  [warn] Playwright (Panopto tab) course {course_id}: {e}")
        finally:
            browser.close()

    # Persist the detected host so all subsequent API calls use it.
    if detected_host:
        if not PANOPTO_HOST:
            PANOPTO_HOST = detected_host
            tqdm.write(f"  [info] Auto-detected PANOPTO_HOST: {PANOPTO_HOST}")
            # Save to config so future sessions don't need to re-detect.
            try:
                cfg = json.load(open(_config_file, encoding="utf-8")) if _config_file.exists() else {}
                cfg["PANOPTO_HOST"] = PANOPTO_HOST
                with open(_config_file, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
            except Exception:
                pass

    return folder_id, cookies, bearer_token


def _iter_panopto_folder(
    folder_id:    str,
    cookies:      list[dict],
    folder_name:  str = "Videos/Panopto",
    bearer_token: str | None = None,
    _sess:        "requests.Session | None" = None,
) -> list[dict]:
    """
    Recursively list all sessions in a Panopto folder via the WCF GetSessions
    endpoint.

    Key requirements discovered by request inspection:
    - The WCF endpoint requires an OAuth Bearer token (captured by Playwright)
      to return the Subfolders list; cookie-only requests receive Subfolders=None.
    - getFolderData=true must be set to receive the Subfolders field at all.
    - Duration filter: entries with no Duration are navigation placeholders
      (e.g. "Home") not actual recordings.

    Returns list of {session_id, name, folder_name}.
    """
    if _sess is None:
        _sess = requests.Session()
        for ck in cookies:
            _sess.cookies.set(ck["name"], ck["value"], domain=ck.get("domain", ""))

    headers = {"Content-Type": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    results:    list[dict] = []
    subfolders: list[dict] = []   # collected from first page (getFolderData)
    PAGE_SIZE = 250
    start_idx = 0

    while True:
        r = _sess.post(
            f"https://{PANOPTO_HOST}/Panopto/Services/Data.svc/GetSessions",
            json={"queryParameters": {
                "folderID":                   folder_id,
                "startIndex":                 start_idx,
                "maxResults":                 PAGE_SIZE,
                "sortColumn":                 1,
                "sortAscending":              True,
                "getFolderData":              start_idx == 0,  # only needed on first page
                "includeArchived":            True,
                "includePlaylists":           True,
                "includePlaceholderSessions": False,
            }},
            headers=headers,
            timeout=30,
        )
        if r.status_code != 200:
            tqdm.write(f"  [warn] GetSessions HTTP {r.status_code} for folder {folder_id}")
            break

        d = r.json().get("d") or {}
        page_results = d.get("Results") or []

        for s in page_results:
            sid      = s.get("DeliveryID") or s.get("SessionID")
            name     = s.get("SessionName", "")
            duration = s.get("Duration")
            # Skip placeholder/empty sessions that have no video content.
            # These appear as folder-level navigation entries (Duration=None)
            # and return ErrorCode=6 from DeliveryInfo.aspx.
            if duration is None:
                continue
            if sid and name:
                results.append({
                    "session_id":  sid,
                    "name":        name,
                    "folder_name": folder_name,
                })

        if start_idx == 0:
            subfolders = d.get("Subfolders") or []

        start_idx += len(page_results)
        total = d.get("TotalNumberOfResults") or 0
        if not page_results or start_idx >= total:
            break

    # Recurse into subfolders
    for sf in subfolders:
        sf_id   = sf.get("ID") or sf.get("FolderID") or sf.get("Id")
        sf_name = sf.get("Name") or folder_name
        if sf_id:
            tqdm.write(f"  [info] Scanning subfolder: {sf_name}")
            results.extend(
                _iter_panopto_folder(sf_id, cookies, sf_name, bearer_token, _sess)
            )

    return results


def _find_panopto_in_pages(course) -> list[dict]:
    """
    Scan Canvas Pages for embedded Panopto Viewer.aspx links.
    Returns video dicts for each unique session UUID found.
    """
    global PANOPTO_HOST  # may be auto-detected from page bodies
    videos: list[dict] = []
    seen: set[str] = set()

    try:
        for page_stub in course.get_pages():
            page = course.get_page(page_stub.url)
            body = getattr(page, "body", "") or ""
            if "panopto" not in body.lower():
                continue
            # Auto-detect PANOPTO_HOST from embedded URLs if not yet known.
            if not PANOPTO_HOST:
                m = re.search(
                    r"https?://([\w.-]*panopto[\w.-]*)/Panopto/",
                    body, re.IGNORECASE,
                )
                if m:
                    PANOPTO_HOST = m.group(1)
                    tqdm.write(f"  [info] Auto-detected PANOPTO_HOST from page: {PANOPTO_HOST}")
                    try:
                        cfg = json.load(open(_config_file, encoding="utf-8")) if _config_file.exists() else {}
                        cfg["PANOPTO_HOST"] = PANOPTO_HOST
                        with open(_config_file, "w", encoding="utf-8") as f:
                            json.dump(cfg, f, indent=2)
                    except Exception:
                        pass
            # Match ?id=UUID  (Viewer.aspx or Embed.aspx links)
            for uid in re.findall(
                r"[?&]id=([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}"
                r"-[0-9a-f]{4}-[0-9a-f]{12})",
                body, re.IGNORECASE,
            ):
                uid = uid.lower()
                if uid in seen:
                    continue
                seen.add(uid)
                viewer_url = (f"https://{PANOPTO_HOST}/Panopto/Pages/Viewer.aspx"
                              f"?id={uid}")
                videos.append({
                    "course_id":   course.id,
                    "course_name": course.name,
                    "module_name": page_stub.title,
                    "title":       page_stub.title,
                    "lti_url":     None,
                    "item_id":     uid,   # UUID string
                    "viewer_url":  viewer_url,
                })
    except Exception as e:
        tqdm.write(f"  [warn] pages scan {course.id}: {e}")

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
    _ensure_playwright_browsers()
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


def _download_authenticated(
    url: str,
    out_path: "Path",
    headers: dict,
    progress_cb: "callable",
) -> None:
    """Stream-download a URL that requires auth headers, with progress callback."""
    r = requests.get(url, headers=headers, stream=True, timeout=300, allow_redirects=True)
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0))
    downloaded = 0
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    progress_cb(downloaded / total * 100)
    if not total:
        progress_cb(100)


def _get_stream_candidates(
    session_id: str,
    cookies: list[dict],
    bearer_token: str | None = None,
    course_id:    int | None = None,
) -> list[tuple[str, dict, str]]:
    """Return all HLS stream candidates for a Panopto session in preference
    order. Each item is (stream_url, auth_headers, stream_tag). Empty list on
    failure. download_video() probes each candidate for audio and uses the
    first one that has it."""
    singles = _get_stream_url(
        session_id, cookies,
        bearer_token=bearer_token, course_id=course_id,
        _return_all=True,
    )
    return singles or []


def _get_stream_url(
    session_id: str,
    cookies: list[dict],
    bearer_token: str | None = None,
    course_id:    int | None = None,
    _return_all: bool = False,
) -> tuple[str, dict] | list[tuple[str, dict, str]] | None:
    """Get the HLS stream URL for a Panopto session.

    Returns (stream_url, auth_headers, stream_tag) on success, None on failure.
    If _return_all=True, returns a list of all candidate tuples in preference
    order so callers can fall back (audio-less stream → try next).

    auth_headers is always {} (HLS streams are CDN-public once the URL is known).
    stream_tag is the Panopto stream tag ("SS" for screen share, "DV" for camera, etc.).

    Priority:
      1. DeliveryInfo.aspx POST with Bearer token → HLS master.m3u8
      2. Playwright LTI re-launch → navigate to Viewer.aspx → intercept DeliveryInfo
    """
    def _extract_streams(body: dict) -> list[tuple[str, str]]:
        """Return a list of (stream_url, tag) candidates in preference order.
        Caller can try each until one produces a video with an audio track —
        some Panopto recordings store audio only in the DV stream while SS /
        OBJECT are video-only for screen recordings.

        Tag matching is case-insensitive: Panopto returns either upper-case
        ("OBJECT", "DV") or lower-case ("object", "dv") tags depending on
        the session's recording setup. A case-sensitive match would fall
        through to "any stream first" — which is DV (the camera), so the
        downloaded video would be a lecture-hall camera shot instead of
        the screen recording.
        """
        streams = (body.get("Delivery") or {}).get("Streams") or []
        # Preference: SS (screen-share) > OBJECT (screen recording) > DV (camera).
        # We still yield *all* streams so the caller can fall back if the
        # preferred one lacks audio.
        order = tuple(t.upper() for t in _PREFER_STREAM_ORDER)
        out: list[tuple[str, str]] = []
        seen_urls: set[str] = set()
        for tag in order + (None,):
            for s in streams:
                surl = s.get("StreamUrl", "")
                if not surl or surl in seen_urls:
                    continue
                stream_tag = (s.get("Tag", "") or "").upper()
                if tag is None or stream_tag == tag:
                    seen_urls.add(surl)
                    out.append((surl, s.get("Tag", "unknown")))
        return out

    def _extract_stream(body: dict) -> tuple[str, str] | None:
        streams = _extract_streams(body)
        return streams[0] if streams else None

    sess = requests.Session()
    for ck in cookies:
        sess.cookies.set(ck["name"], ck["value"], domain=ck.get("domain", ""))
    auth_hdrs: dict = {}
    if bearer_token:
        auth_hdrs["Authorization"] = f"Bearer {bearer_token}"

    # ── 1. DeliveryInfo.aspx POST ─────────────────────────────────────────────
    # The bearer token in Authorization header allows the server to validate
    # the LTI enrollment context and return the HLS stream URL.
    delivery_url = f"https://{PANOPTO_HOST}/Panopto/Pages/Viewer/DeliveryInfo.aspx"
    try:
        r = sess.post(
            delivery_url,
            data={
                "deliveryId": session_id, "isLiveNotes": "false",
                "refreshAuthCookie": "true", "isActiveBroadcast": "false",
                "isEditing": "false", "isKollectiveAgentInstalled": "false",
                "isEmbed": "false", "responseType": "json",
            },
            headers=auth_hdrs,
            timeout=30,
        )
        if r.status_code == 200:
            body = r.json()
            if not body.get("ErrorCode"):
                if _return_all:
                    all_streams = _extract_streams(body)
                    if all_streams:
                        return [(u, {}, t) for (u, t) in all_streams]
                result = _extract_stream(body)
                if result:
                    surl, stag = result
                    return surl, {}, stag
    except Exception:
        pass

    # ── 3. Playwright LTI re-launch → viewer intercept ───────────────────────
    if course_id is None:
        return None

    tqdm.write(f"  DeliveryInfo unavailable; re-launching via Canvas LTI…")
    tool_id = _resolve_panopto_tool_id(course_id) or _PANOPTO_TAB_TOOL_ID_FALLBACK
    r2 = requests.get(
        f"{CANVAS_URL}/api/v1/courses/{course_id}/external_tools/sessionless_launch"
        f"?id={tool_id}&launch_type=course_navigation",
        headers=_canvas_headers(), timeout=30,
    )
    if r2.status_code != 200:
        return None
    launch_url = r2.json().get("url")
    if not launch_url:
        return None

    _ensure_playwright_browsers()
    from playwright.sync_api import sync_playwright

    viewer_page_url = f"https://{PANOPTO_HOST}/Panopto/Pages/Viewer.aspx?id={session_id}"
    captured: list[tuple[str, dict, str]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx     = browser.new_context()
        page    = ctx.new_page()

        def _on_response(resp) -> None:
            if "DeliveryInfo.aspx" in resp.url and not captured:
                try:
                    body = resp.json()
                    if _return_all:
                        for surl, stag in _extract_streams(body):
                            captured.append((surl, {}, stag))
                    else:
                        result = _extract_stream(body)
                        if result:
                            surl, stag = result
                            captured.append((surl, {}, stag))
                except Exception:
                    pass

        page.on("response", _on_response)
        try:
            page.goto(launch_url, wait_until="networkidle", timeout=60000)
            time.sleep(2)
            page.goto(viewer_page_url, wait_until="networkidle", timeout=60000)
            time.sleep(3)
        except Exception as e:
            tqdm.write(f"  [warn] Playwright viewer: {e}")
        finally:
            browser.close()

    if _return_all:
        return list(captured) if captured else []
    return captured[0] if captured else None


def _resolve_ffmpeg() -> str | None:
    """Locate an ffmpeg executable.

    Order:
      1. System ffmpeg in PATH (best — usually fastest, tracks user updates)
      2. imageio-ffmpeg's bundled binary (cross-platform fallback for macOS
         and any system without ffmpeg installed; pip-installable, no admin
         rights needed). If imageio-ffmpeg isn't present (e.g. the user has
         a venv from an older AutoNote release that pre-dates this fallback),
         install it on demand — it's a single platform-specific wheel under
         30 MB and the binary is cached for future runs.
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
            tqdm.write(f"  [warn] imageio-ffmpeg get_ffmpeg_exe failed: {e}")
            return None

    ff = _try_imageio()
    if ff:
        return ff

    # Auto-install on first need. Quiet, fast, idempotent.
    tqdm.write("  ffmpeg not found locally — installing imageio-ffmpeg fallback…")
    try:
        import subprocess as _sp
        _sp.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "imageio-ffmpeg"],
            check=True, timeout=180,
        )
    except Exception as e:
        tqdm.write(f"  [warn] auto-install imageio-ffmpeg failed: {e}")
        return None

    return _try_imageio()


def _hls_has_audio(stream_url: str, ff_bin: str | None = None) -> bool:
    """Return True when an HLS master playlist references an audio track.
    Called before download so we can skip audio-less variants (some Panopto
    recordings only pack audio in the DV stream, not OBJECT/SS)."""
    if ff_bin is None:
        ff_bin = _resolve_ffmpeg()
    if not ff_bin:
        return True  # can't check — assume yes and let caller try
    ffprobe = ff_bin.replace("/ffmpeg", "/ffprobe")
    if not Path(ffprobe).exists():
        # imageio-ffmpeg bundles only ffmpeg; probe via ffmpeg -i
        probe = subprocess.run(
            [ff_bin, "-hide_banner", "-i", stream_url],
            capture_output=True, text=True, timeout=30,
        )
        return "Audio:" in (probe.stderr or "")
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_name",
             "-of", "default=nw=1:nk=1", stream_url],
            capture_output=True, text=True, timeout=30,
        )
        return bool((r.stdout or "").strip())
    except Exception:
        return True  # don't block download on probe failure


def _run_ffmpeg_hls_merge(video_url: str, audio_url: str,
                          out_path: Path, progress_cb) -> None:
    """Merge a video-only HLS stream with an audio-only HLS stream.

    Used when the screen-recording stream (OBJECT/SS) has no audio track and
    the audio lives in a separate DV/AUDIO stream. Produces one MP4 with the
    screen video and the lecturer's audio.
    """
    ff_bin = _resolve_ffmpeg()
    if not ff_bin:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`."
        )
    cmd = [
        ff_bin, "-y", "-loglevel", "error",
        "-i", video_url,
        "-i", audio_url,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c", "copy",
        "-shortest",
        str(out_path),
    ]
    try:
        from ffmpeg_progress_yield import FfmpegProgress
        ff = FfmpegProgress(cmd)
        for pct in ff.run_command_with_progress():
            progress_cb(pct)
        if not out_path.exists() or out_path.stat().st_size < 1024:
            raise RuntimeError(
                f"ffmpeg produced no output for {out_path.name} — "
                f"video/audio merge failed."
            )
    except ImportError:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            tail = (result.stderr or "").strip().splitlines()[-10:]
            raise RuntimeError(f"ffmpeg merge failed:\n" + "\n".join(tail))
        progress_cb(100)


def _run_ffmpeg_hls(stream_url: str, out_path: Path, progress_cb) -> None:
    """Download an HLS master.m3u8 stream to *out_path* via ffmpeg.

    Uses the bundled `imageio-ffmpeg` binary when no system ffmpeg is
    present, so macOS users without Homebrew get a working downloader.

    Dropping `-f hls` lets ffmpeg auto-detect from the URL — this is more
    robust when the master playlist references a mixed-protocol variant set.
    `-map 0` keeps all streams from the selected variant (video + audio).
    Non-zero exit codes are surfaced as RuntimeError so partial/empty files
    don't silently get written to disk.
    """
    ff_bin = _resolve_ffmpeg()
    if not ff_bin:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg (e.g. `brew install ffmpeg` on "
            "macOS or `choco install ffmpeg` on Windows), or `pip install "
            "imageio-ffmpeg` for a bundled fallback."
        )
    # -y            : overwrite partial leftovers from a prior failed attempt
    # -loglevel error: only surface real errors (progress plugin reads stderr)
    # -map 0        : keep every stream of the selected variant (a/v)
    # -c copy       : no re-encoding; lossless + fast
    cmd = [
        ff_bin, "-y", "-loglevel", "error",
        "-i", stream_url,
        "-map", "0",
        "-c", "copy",
        str(out_path),
    ]
    try:
        from ffmpeg_progress_yield import FfmpegProgress
        ff = FfmpegProgress(cmd)
        for pct in ff.run_command_with_progress():
            progress_cb(pct)
        # FfmpegProgress doesn't raise on non-zero exit; inspect the
        # resulting file to catch silent failures.
        if not out_path.exists() or out_path.stat().st_size < 1024:
            raise RuntimeError(
                f"ffmpeg produced no output for {out_path.name} — "
                f"check network connectivity or stream auth."
            )
    except ImportError:
        tqdm.write("  (ffmpeg-progress-yield not available, running ffmpeg without progress)")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            tail = (result.stderr or "").strip().splitlines()[-10:]
            raise RuntimeError(
                f"ffmpeg failed (exit {result.returncode}):\n"
                + "\n".join(tail)
            )
        progress_cb(100)


def download_video(video: dict, manifest: dict, base_dir: Path) -> bool | None:
    """Download a single Panopto video with a tqdm progress bar.

    Returns:
      True  — video was actually downloaded now
      None  — video was skipped (already done or file existed)
      False — download failed

    Handles two source types:
      • Module ExternalTool (item_id is an integer): LTI sessionless launch →
        Playwright → extract session_id + cookies → stream URL.
      • Folder/Page session (item_id is a UUID string, viewer_url is set):
        use cached Panopto cookies from discovery, or re-authenticate via the
        Videos/Panopto tab, then call stream URL directly.
    """
    try:
        import PanoptoDownloader
    except ImportError:
        PanoptoDownloader = None

    item_id    = video["item_id"]
    course_id  = video["course_id"]
    title      = video["title"]
    viewer_url = video.get("viewer_url")
    key        = str(item_id)

    if manifest.get(key, {}).get("status") == "done":
        tqdm.write(f"  [skip] Already downloaded: {title}")
        return None

    out_dir  = base_dir / str(course_id) / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_sanitize(title)}.mp4"

    if out_path.exists():
        tqdm.write(f"  [skip] File exists: {out_path}")
        manifest[key] = {"status": "done", "path": str(out_path)}
        return None

    # ── Resolve session_id + cookies ──────────────────────────────────────────
    bearer_token: str | None = None
    if viewer_url:
        # Non-module session: session_id is the UUID stored as item_id.
        session_id = str(item_id)
        # Use cookies cached during discovery if available, otherwise re-auth.
        cookies      = video.get("_panopto_cookies") or []
        bearer_token = video.get("_bearer_token")
        if not cookies:
            tqdm.write(f"  Re-authenticating via Videos/Panopto tab…")
            _, cookies, bearer_token = _get_panopto_tab_folder(course_id)
        if not cookies:
            tqdm.write(f"  [error] Could not obtain Panopto auth cookies")
            manifest[key] = {"status": "error", "title": title}
            return False
    else:
        # Module ExternalTool: classic LTI launch path.
        tqdm.write(f"  Fetching launch URL for: {title}")
        launch_url = _get_sessionless_launch_url(course_id, item_id)
        if not launch_url:
            tqdm.write(f"  [error] Could not get launch URL")
            manifest[key] = {"status": "error", "title": title}
            return False
        tqdm.write(f"  Launching Panopto session…")
        session_id, cookies = _get_panopto_session(launch_url)
        if not session_id:
            tqdm.write(f"  [error] Could not get Panopto session ID")
            manifest[key] = {"status": "error", "title": title}
            return False

    candidates = _get_stream_candidates(
        session_id, cookies,
        bearer_token=bearer_token,
        course_id=course_id if viewer_url else None,
    )
    if not candidates:
        tqdm.write(f"  [error] Could not get stream URL")
        manifest[key] = {"status": "error", "title": title}
        return False

    # Record which tags existed so the frame extractor can tell a true camera
    # recording from a split-stream case where the audio is in DV but the
    # screen video is in OBJECT/SS.
    available_tags = [ct for (_, _, ct) in candidates]
    # Tags from Panopto are sometimes lower-case ("object"/"dv"). Normalise
    # before matching so the "screen-stream-first" preference still fires.
    has_screen_stream = any((t or "").upper() in ("SS", "OBJECT")
                            for t in available_tags)

    # Classify each candidate as "video-with-audio" or "video-only" up front.
    # Panopto screen recordings often store SS/OBJECT (screen content, no
    # audio) alongside DV (camera + audio). When both exist, we want the
    # screen video AND the DV audio — merged by ffmpeg — so notes see slides
    # as frames AND the transcript carries lecture audio.
    video_cand = audio_cand = None   # (url, headers, tag, is_m3u8)
    tried = []
    for (cu, ch, ct) in candidates:
        tried.append(ct)
        is_m3u8 = "master.m3u8" in cu
        has_audio = (not is_m3u8) or _hls_has_audio(cu)
        ct_upper = (ct or "").upper()
        # Preferred screen stream (first one we see that is SS/OBJECT) becomes
        # the chosen VIDEO source — regardless of whether it has audio.
        if video_cand is None and ct_upper in ("SS", "OBJECT"):
            video_cand = (cu, ch, ct, is_m3u8, has_audio)
        # First stream with audio becomes the audio source.
        if audio_cand is None and has_audio:
            audio_cand = (cu, ch, ct, is_m3u8, has_audio)

    # If we never saw a screen stream, fall back to the audio-bearing stream
    # (treat its video as the video source too).
    if video_cand is None:
        if audio_cand is None:
            tqdm.write(
                f"  [error] No stream with audio found for {title} "
                f"(tried: {', '.join(tried)}) — skipping; the recording "
                f"itself appears to be video-only.")
            manifest[key] = {
                "status": "error", "title": title,
                "error": "no-audio-track",
            }
            raise RuntimeError(f"No audio track in any Panopto stream for '{title}'")
        video_cand = audio_cand
    if audio_cand is None:
        # Screen stream exists but nothing has audio — still skip.
        tqdm.write(
            f"  [error] Video streams exist but none has audio for {title} "
            f"(tried: {', '.join(tried)}) — skipping.")
        manifest[key] = {
            "status": "error", "title": title,
            "error": "no-audio-track",
        }
        raise RuntimeError(f"No audio track in any Panopto stream for '{title}'")

    merge_needed = (
        video_cand[0] != audio_cand[0]        # different stream URLs
        and video_cand[3] and audio_cand[3]   # both HLS master playlists
    )
    stream_url    = video_cand[0]
    dl_headers    = video_cand[1]
    stream_tag    = video_cand[2]
    audio_url     = audio_cand[0] if merge_needed else None
    audio_tag     = audio_cand[2] if merge_needed else None
    if merge_needed:
        tqdm.write(
            f"  Stream type: {stream_tag} video + {audio_tag} audio (merging)")
    else:
        tqdm.write(f"  Stream type: {stream_tag}")

    bar = tqdm(total=100, desc=f"  {title[:50]}", unit="%",
               bar_format="{desc} |{bar}| {n:3d}/{total}%", leave=True)

    def progress_cb(pct: float) -> None:
        bar.n = int(pct)
        bar.refresh()

    try:
        if dl_headers:
            # Direct authenticated download (e.g. REST API DownloadUrl)
            _download_authenticated(stream_url, out_path, dl_headers, progress_cb)
        elif audio_url:
            # Split-stream case: screen video (no audio) + separate audio stream.
            _run_ffmpeg_hls_merge(stream_url, audio_url, out_path, progress_cb)
        elif "master.m3u8" in stream_url:
            # HLS stream — always use ffmpeg. PanoptoDownloader's HLS path
            # is broken for URLs with query strings (its endswith('master.m3u8')
            # check fails on the ?InvocationID=… suffix, raising NotAVideo —
            # which is what macOS users hit when no system ffmpeg is in PATH).
            _run_ffmpeg_hls(stream_url, out_path, progress_cb)
        elif PanoptoDownloader:
            PanoptoDownloader.download(stream_url, str(out_path), progress_cb)
        else:
            raise RuntimeError("Non-HLS stream and PanoptoDownloader not installed")
        bar.n = 100
        bar.refresh()
        bar.close()
        tqdm.write(f"  Saved → {out_path}")
        manifest[key] = {
            "status": "done", "path": str(out_path),
            "title": title, "session_id": session_id,
            "stream_tag": stream_tag,
            "has_screen_stream": has_screen_stream,
            "available_tags": available_tags,
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
                "folder_id":    f.folder_id,
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
        kf = DATA_DIR / "anthropic_key.txt"
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


def _download_folder_zip(
    folder_id: int,
    folder_path: str,
    course_id: int,
    files_in_folder: list[dict],
    log: dict,
    base_dir: Path,
) -> tuple[int, int, int]:
    """
    Download a Canvas folder as a single ZIP request, then extract.
    Returns (n_downloaded, n_skipped, n_errors).
    Falls back to individual download_material() calls if the ZIP
    endpoint is unavailable or returns bad data.
    """
    import io, zipfile

    pending = [f for f in files_in_folder
               if not (str(f["id"]) in log
                       and Path(log[str(f["id"])]["path"]).exists())]
    skipped = len(files_in_folder) - len(pending)
    if not pending:
        return 0, skipped, 0

    dest_dir = base_dir / str(course_id) / "materials"
    if folder_path:
        dest_dir = dest_dir / Path(*[_sanitize(p) for p in folder_path.split("/")])
    dest_dir.mkdir(parents=True, exist_ok=True)

    folder_label = folder_path or "/"
    tqdm.write(f"\n  Folder: {folder_label}  ({len(pending)} pending / {skipped} already done)")

    # ── Request folder ZIP ─────────────────────────────────────────────────────
    r = requests.get(
        f"{CANVAS_URL}/api/v1/folders/{folder_id}/download",
        headers=_canvas_headers(),
        allow_redirects=True,
        stream=True,
        timeout=300,
    )

    def _fallback() -> tuple[int, int, int]:
        dl = err = 0
        for f in pending:
            if download_material(f, log, base_dir):
                dl += 1
            else:
                err += 1
        return dl, skipped, err

    if r.status_code != 200:
        tqdm.write(f"    [warn] ZIP not available (HTTP {r.status_code}), falling back to individual downloads")
        return _fallback()

    content_type = r.headers.get("Content-Type", "")
    if "zip" not in content_type and "octet-stream" not in content_type:
        tqdm.write(f"    [warn] Unexpected content-type '{content_type}', falling back")
        return _fallback()

    # ── Stream the ZIP ─────────────────────────────────────────────────────────
    total_bytes = int(r.headers.get("Content-Length", 0))
    bar = tqdm(
        total=total_bytes or None,
        desc=f"  [{folder_label}] zip",
        unit="B", unit_scale=True, unit_divisor=1024,
        bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
        leave=True,
    )
    buf = io.BytesIO()
    for chunk in r.iter_content(chunk_size=65536):
        if chunk:
            buf.write(chunk)
            bar.update(len(chunk))
    bar.close()
    buf.seek(0)

    # ── Extract ────────────────────────────────────────────────────────────────
    try:
        zf = zipfile.ZipFile(buf)
    except zipfile.BadZipFile as e:
        tqdm.write(f"    [warn] Bad ZIP ({e}), falling back to individual downloads")
        return _fallback()

    # Build name → member map (strip internal directory prefixes)
    name_to_member: dict[str, str] = {}
    with zf:
        for member in zf.namelist():
            fname = Path(member).name
            if fname:                            # skip directory entries
                name_to_member[fname] = member

        # ── Match each pending file to a ZIP entry and write to disk ──────────
        now = datetime.now(timezone.utc).isoformat()
        dl = err = 0
        for f in pending:
            sanitized = _sanitize(f["display_name"])
            member = name_to_member.get(sanitized) or name_to_member.get(f["display_name"])
            if member is None:
                tqdm.write(f"    [warn] Not found in ZIP: {f['display_name']} — downloading individually")
                if download_material(f, log, base_dir):
                    dl += 1
                else:
                    err += 1
                continue

            target = dest_dir / sanitized
            try:
                target.write_bytes(zf.read(member))
            except Exception as e:
                tqdm.write(f"    [error] Failed to write {target.name}: {e}")
                err += 1
                continue
            tqdm.write(f"    Extracted → {target.name}")
            log[str(f["id"])] = {
                "display_name":  f["display_name"],
                "folder":        folder_path,
                "size":          target.stat().st_size,
                "path":          str(target),
                "downloaded_at": now,
            }
            dl += 1

    return dl, skipped, err


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

    # ── Pipeline options ──────────────────────────────────────────────────────
    parser.add_argument(
        "--transcribe", action="store_true",
        help="Auto-transcribe each video right after it downloads "
             "(runs concurrently: next download + previous transcription overlap)",
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

    if not CANVAS_URL:
        print("[error] Canvas URL is not configured.")
        print("[error] Enter your Canvas URL in Settings → Connection in the AutoNote app.")
        sys.exit(1)
    if not CANVAS_TOKEN:
        print("[error] Canvas token is not configured.")
        print("[error] Enter your Canvas API token in Settings → API Keys in the AutoNote app.")
        sys.exit(1)

    base_dir = Path(args.path) if args.path else DATA_DIR
    try:
        canvas = Canvas(CANVAS_URL, CANVAS_TOKEN)
    except Exception as e:
        print(f"[error] Failed to connect to Canvas: {e}")
        sys.exit(1)

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
        targets  = [v for v in videos if v[num_key] in args.download_video]
        missing  = set(args.download_video) - {v[num_key] for v in targets}
        if missing:
            print(f"  [warn] No video(s) found for number(s): {sorted(missing)}")
        if not targets:
            sys.exit(1)

        print(f"\nDownloading {len(targets)} video(s):")
        for i, video in enumerate(sorted(targets, key=lambda v: v[num_key])):
            print(f"\n[{i+1}/{len(targets)}] {video['title']}")
            result = download_video(video, manifest, base_dir)
            _save_json(MANIFEST_FILE, manifest)
            if args.transcribe and result is True:
                _spawn_transcribe(manifest[str(video["item_id"])]["path"])
            if args.secretly and result is True and i < len(targets) - 1:
                _secretly_wait_video()

        if args.transcribe:
            _wait_transcriptions()
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
            try:
                result = download_video(video, manifest, base_dir)
            except RuntimeError as _exc:
                tqdm.write(f"  [skip] {_exc}")
                _save_json(MANIFEST_FILE, manifest)
                continue
            _save_json(MANIFEST_FILE, manifest)
            if args.transcribe and result is True:
                _spawn_transcribe(manifest[str(video["item_id"])]["path"])
            if args.secretly and result is True and i < len(pending) - 1:
                _secretly_wait_video()

        if args.transcribe:
            _wait_transcriptions()
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

    # ── --download-material NUMBER_OR_FILENAME [NUMBER_OR_FILENAME ...] ────────
    if args.download_material:
        print(f"\nDiscovering materials{' for course ' + str(args.course) if args.course else ''}...")
        files = discover_materials(canvas, args.course)

        targets: list[dict] = []
        for query in args.download_material:
            # Try as a 1-based list number first (matches the # column from --material-list)
            try:
                idx = int(query)
                if 1 <= idx <= len(files):
                    targets.append(files[idx - 1])
                    print(f"  #{idx}: {files[idx - 1]['display_name']}")
                    continue
                else:
                    print(f"  [warn] Number {idx} out of range (1–{len(files)})")
                    continue
            except ValueError:
                pass
            # Fall back to filename substring matching
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

        prev_downloaded = False
        for cid, cfiles in by_course.items():
            log_path = base_dir / str(cid) / "materials" / "download_log.json"
            log      = _load_json(log_path)
            print(f"\nCourse {cid}:")
            if args.secretly and prev_downloaded:
                _secretly_wait_dir()
            course_dl = 0
            for i, f in enumerate(cfiles):
                if download_material(f, log, base_dir):
                    course_dl += 1
                _save_json(log_path, log)
            prev_downloaded = course_dl > 0

        print(f"\nDone.")
        return

    # ── --download-material-all ────────────────────────────────────────────────
    if args.download_material_all:
        if args.course:
            courses = [get_course_by_id(canvas, args.course)]
        else:
            courses = get_academic_courses(canvas)

        total_dl = total_sk = 0

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

            # Group pending files by folder for bulk ZIP downloads
            by_folder: dict[tuple[int, str], list[dict]] = {}
            for f in to_download:
                key = (f.get("folder_id", 0), f["folder_path"])
                by_folder.setdefault(key, []).append(f)

            dl = sk = err = 0
            folders = list(by_folder.items())
            for fi, ((folder_id, folder_path), folder_files) in enumerate(folders):
                n_dl, n_sk, n_err = _download_folder_zip(
                    folder_id, folder_path, course.id,
                    folder_files, log, base_dir,
                )
                dl += n_dl; sk += n_sk; err += n_err
                _save_json(log_path, log)

                # Only wait if this folder actually downloaded something
                if args.secretly and n_dl > 0 and fi < len(folders) - 1:
                    _secretly_wait_dir()

            print(f"\n  Done: {dl} downloaded, {sk} skipped, {err} errors "
                  f"across {len(by_folder)} folder(s)")
            total_dl += dl; total_sk += sk

            # Only wait between courses if this course actually downloaded something
            if args.secretly and dl > 0 and ci < len(courses) - 1:
                _secretly_wait_dir()

        print(f"\n{'='*70}")
        print(f"All done: {total_dl} downloaded, {total_sk} skipped "
              f"across {len(courses)} course(s)")


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
