#!/usr/bin/env python3
"""
gui.py — Material Design desktop GUI for the auto_note pipeline.

Requires: pip install flet
Usage:    python gui.py
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import flet as ft

# ── Project layout ────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).parent
if getattr(sys, "frozen", False):
    import shutil as _shutil
    PYTHON = _shutil.which("python3") or _shutil.which("python") or "python3"
else:
    PYTHON = sys.executable

SCRIPTS = {
    "downloader": PROJECT_DIR / "downloader.py",
    "transcribe": PROJECT_DIR / "extract_caption.py",
    "align":      PROJECT_DIR / "semantic_alignment.py",
    "generate":   PROJECT_DIR / "note_generation.py",
}

# User data directory: persistent across app restarts.
# When running as a PyInstaller bundle (AppImage / .exe), __file__ resolves to
# a temporary extraction folder that is deleted on exit — any files written
# there are lost.  Use ~/.auto_note/ instead so credentials and config survive.
if getattr(sys, "frozen", False):
    DATA_DIR = Path.home() / ".auto_note"
else:
    DATA_DIR = PROJECT_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)

_DEFAULT_PYTHON = PYTHON   # auto-detected fallback; may be overridden by user config

ML_VENV_DIR    = DATA_DIR / "venv"
ML_VENV_PYTHON = str(ML_VENV_DIR / (
    "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
))

# ML packages needed by the pipeline scripts (GUI requirements are bundled separately)
_ML_PACKAGES = [
    "tqdm",
    "faster-whisper",
    "sentence-transformers",
    "faiss-cpu",
    "pymupdf",
    "python-pptx",
    "python-docx",
    "openai",
    "anthropic",
    "google-generativeai",
    "requests",
    "pillow",
    "httpx",
    "playwright",
    "canvasapi",
]


def _detect_cuda() -> tuple[int, int] | None:
    """Return (major, minor) CUDA version from nvidia-smi, or None if no GPU."""
    try:
        r = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0:
            m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", r.stdout)
            if m:
                return (int(m.group(1)), int(m.group(2)))
            return (12, 0)   # nvidia-smi works but version unreadable → assume 12.x
    except Exception:
        pass
    return None


def _torch_index_url(cuda: tuple[int, int] | None) -> str | None:
    """Return the PyTorch extra-index-url for the detected CUDA version."""
    if cuda is None:
        return None          # CPU build from PyPI
    major, minor = cuda
    version = major * 10 + minor   # 12.8 → 128
    if version >= 128:
        return "https://download.pytorch.org/whl/cu128"
    if version >= 126:
        return "https://download.pytorch.org/whl/cu126"
    if version >= 124:
        return "https://download.pytorch.org/whl/cu124"
    return "https://download.pytorch.org/whl/cu121"


def _load_python_from_config() -> None:
    """Override PYTHON global with the user-configured path or the managed venv."""
    global PYTHON
    config_file = DATA_DIR / "config.json"
    if config_file.exists():
        try:
            cfg = json.load(open(config_file))
            p = cfg.get("PYTHON_PATH", "").strip()
            if p:
                PYTHON = p
                return
        except Exception:
            pass
    # Auto-use managed venv if it exists and no explicit path is configured
    if Path(ML_VENV_PYTHON).exists():
        PYTHON = ML_VENV_PYTHON


COURSES: dict[int, str] = {}   # populated from Canvas API after token is entered

_SKIP_KEYWORDS = [
    "training", "pdp", "rmcpdp", "osa", "soct", "travel",
    "essentials", "respect", "consent", "osh",
]


def _load_courses_from_canvas() -> str:
    """Fetch active courses from Canvas and update the global COURSES dict.
    Returns "" on success, or a human-readable error string on failure."""
    COURSES.clear()
    token_file  = DATA_DIR / "canvas_token.txt"
    config_file = DATA_DIR / "config.json"
    token = token_file.read_text().strip() if token_file.exists() else ""
    if not token:
        return "Canvas token not saved — enter it in Settings → API Keys."
    cfg        = json.load(open(config_file)) if config_file.exists() else {}
    canvas_url = cfg.get("CANVAS_URL", "").strip().rstrip("/")
    if canvas_url and not canvas_url.startswith(("http://", "https://")):
        canvas_url = "https://" + canvas_url
    if not canvas_url:
        return "Canvas URL not saved — enter it in Settings → Connection."
    try:
        import requests
        resp = requests.get(
            f"{canvas_url}/api/v1/courses",
            headers={"Authorization": f"Bearer {token}"},
            params={"enrollment_state": "active", "per_page": 100},
            timeout=10,
        )
        if resp.status_code == 401:
            return "401 Unauthorized — Canvas token is invalid or expired. Generate a new one in Canvas → Account → Settings → New Access Token."
        resp.raise_for_status()
        for c in resp.json():
            name = c.get("name") or c.get("course_code") or ""
            if not name:
                continue
            if any(kw in name.lower() for kw in _SKIP_KEYWORDS):
                continue
            COURSES[c["id"]] = name
        return ""
    except Exception as exc:
        return str(exc)

# ── Palette ───────────────────────────────────────────────────────────────────

C_PRIMARY   = ft.Colors.CYAN_400
C_SECONDARY = ft.Colors.AMBER_400
C_SUCCESS   = ft.Colors.GREEN_400
C_ERROR     = ft.Colors.RED_400
C_WARN      = ft.Colors.ORANGE_400
C_SURFACE   = "#1E2A2A"
C_CARD      = "#1A2626"
C_RAIL      = "#111E1E"
C_OUTPUT_BG = "#0D1515"
MONO        = "Courier New"

# ── Pipeline state helpers ────────────────────────────────────────────────────

def _manifest() -> dict:
    p = DATA_DIR / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}

def _video_status(course_id: int) -> tuple[int, int]:
    m = _manifest()
    items = [v for v in m.values() if str(course_id) in v.get("path", "")]
    done  = sum(1 for v in items if v.get("status") == "done")
    return done, len(items)

def _caption_count(course_id: int) -> int:
    d = DATA_DIR / str(course_id) / "captions"
    return len(list(d.glob("*.json"))) if d.exists() else 0

def _alignment_count(course_id: int) -> int:
    d = DATA_DIR / str(course_id) / "alignment"
    return len([f for f in d.glob("*.json")
                if "compact" not in f.name]) if d.exists() else 0

def _notes_path(course_id: int) -> Path | None:
    d = DATA_DIR / str(course_id) / "notes"
    if d.exists():
        mds = list(d.glob("*.md"))
        return mds[0] if mds else None
    return None

def _course_name_from_notes(course_id: int) -> str:
    base = COURSES.get(course_id, f"Course {course_id}")
    d    = PROJECT_DIR / str(course_id) / "notes"
    if d.exists():
        for md in d.glob("*.md"):
            stem = md.stem.replace("_notes", "").replace("_", " ")
            if stem:
                return stem
    return base

def _read_constant(script_key: str, name: str) -> str:
    src = SCRIPTS[script_key].read_text(errors="ignore")
    m   = re.search(rf"^{name}\s*=\s*(.+)", src, re.MULTILINE)
    if m:
        return re.sub(r"\s*#.*$", "", m.group(1)).strip().strip('"')
    return "?"


def _write_constant(script_key: str, name: str, new_display_val: str) -> bool:
    """Write a constant back to its source file with the correct Python literal type."""
    path = SCRIPTS[script_key]
    try:
        src = path.read_text(errors="ignore")

        def _replacer(m: re.Match) -> str:
            # None keyword → bare None
            if new_display_val == "None":
                return m.group(1) + "None"
            # Numeric → bare (int or float)
            try:
                float(new_display_val)
                return m.group(1) + new_display_val
            except ValueError:
                pass
            # String → always quoted
            return m.group(1) + f'"{new_display_val}"'

        new_src, n = re.subn(
            rf'^({name}\s*=\s*)([^\n#]+)',
            _replacer, src, count=1, flags=re.MULTILINE,
        )
        if n == 0:
            return False
        path.write_text(new_src)
        return True
    except Exception:
        return False

# ── Shared state ──────────────────────────────────────────────────────────────

class AppState:
    def __init__(self) -> None:
        self.running = False
        self.proc: subprocess.Popen | None = None

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.running = False

state = AppState()

# ── Output console ────────────────────────────────────────────────────────────

class OutputConsole:
    """
    Pinned-bottom terminal panel.
    - Streams subprocess stdout+stderr live
    - Stop button kills the running process
    - Expand fills all available vertical space
    """

    def __init__(self, page: ft.Page) -> None:
        self.page = page

        self._lines = ft.ListView(
            expand=True,
            spacing=0,
            auto_scroll=True,
            padding=ft.Padding.symmetric(horizontal=8, vertical=6),
        )

        self._badge = ft.Text("", size=11, color=C_PRIMARY)

        self._stop_btn = ft.IconButton(
            icon=ft.Icons.STOP_CIRCLE_OUTLINED,
            icon_color=C_ERROR,
            tooltip="Stop process",
            visible=False,
            icon_size=18,
            on_click=self._on_stop,
        )
        self._clear_btn = ft.IconButton(
            icon=ft.Icons.DELETE_SWEEP_OUTLINED,
            icon_color=ft.Colors.with_opacity(0.45, ft.Colors.WHITE),
            tooltip="Clear output",
            icon_size=16,
            on_click=lambda _: self.clear(),
        )

        self.container = ft.Column(
            controls=[
                # Header bar
                ft.Container(
                    content=ft.Row(
                        controls=[
                            ft.Icon(ft.Icons.TERMINAL, color=C_PRIMARY, size=14),
                            ft.Text(" Output", size=12, color=C_PRIMARY,
                                    weight=ft.FontWeight.BOLD),
                            ft.Container(expand=True),
                            self._badge,
                            self._stop_btn,
                            self._clear_btn,
                        ],
                    ),
                    padding=ft.Padding.only(top=8, bottom=4),
                ),
                # Scrollable text area — fixed height so form area gets the rest
                ft.Container(
                    content=self._lines,
                    height=220,
                    bgcolor=C_OUTPUT_BG,
                    border_radius=6,
                    border=ft.border.all(
                        1, ft.Colors.with_opacity(0.12, ft.Colors.WHITE)
                    ),
                ),
            ],
            spacing=0,
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _on_stop(self, _) -> None:
        state.stop()
        self.write("\n[stopped by user]", color=C_WARN)
        self._stop_btn.visible = False
        self.set_status("■ stopped", C_WARN)
        self.page.update()

    # ── public API ─────────────────────────────────────────────────────────────

    def write(self, text: str, color: str | None = None) -> None:
        for line in text.splitlines():
            self._lines.controls.append(
                ft.Text(
                    line,
                    size=11,
                    font_family=MONO,
                    color=color or ft.Colors.with_opacity(0.88, ft.Colors.WHITE),
                    no_wrap=False,
                    selectable=True,
                )
            )
        self.page.update()

    def clear(self) -> None:
        self._lines.controls.clear()
        self._badge.value = ""
        self.page.update()

    def set_status(self, msg: str, color: str = C_PRIMARY) -> None:
        self._badge.value = msg
        self._badge.color = color
        self.page.update()

    def run(self, cmd: list[str], on_done: callable | None = None) -> None:
        """Run cmd, streaming stdout+stderr into the console."""
        if state.running:
            self.write("⚠  Already running — stop it first.", color=C_WARN)
            return

        state.running = True
        self.clear()
        self._stop_btn.visible = True
        self.set_status("● running…", C_WARN)

        # Show full command (script name + all args)
        display_cmd = " ".join(
            str(c) for c in cmd[1:]   # skip python path, keep script + args
        )
        self.write(f"$ {display_cmd}\n",
                   color=ft.Colors.with_opacity(0.40, ft.Colors.WHITE))

        # Line queue: (text, color). _flush_thread drains it every 50 ms.
        _line_q: queue.Queue = queue.Queue()
        _FLUSH_INTERVAL = 0.05   # seconds between UI updates

        def _flush_thread() -> None:
            """Drain the line queue and do a single page.update() per interval."""
            default_color = ft.Colors.with_opacity(0.88, ft.Colors.WHITE)
            while True:
                batch: list[tuple[str, str | None]] = []
                try:
                    # Block until the first item arrives
                    batch.append(_line_q.get(timeout=1.0))
                except queue.Empty:
                    if not state.running:
                        break
                    continue
                # Drain everything else that arrived in this window
                deadline = time.monotonic() + _FLUSH_INTERVAL
                while time.monotonic() < deadline:
                    try:
                        batch.append(_line_q.get_nowait())
                    except queue.Empty:
                        break
                # Append all lines, one page.update() for the whole batch
                for text, color in batch:
                    for line in text.splitlines():
                        self._lines.controls.append(
                            ft.Text(
                                line, size=11, font_family=MONO,
                                color=color or default_color,
                                no_wrap=False, selectable=True,
                            )
                        )
                self.page.update()

        def _worker() -> None:
            try:
                state.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=PROJECT_DIR,
                    bufsize=1,
                )
                for line in state.proc.stdout:
                    # Filter out frozen importlib noise
                    if "<frozen importlib" in line or "OpenSSL 3" in line:
                        continue
                    _line_q.put((line.rstrip(), None))
                state.proc.wait()
                rc = state.proc.returncode

                if rc == 0:
                    _line_q.put(("\n✓  Completed successfully.", C_SUCCESS))
                    self.set_status("✓ done", C_SUCCESS)
                elif rc == -15:   # SIGTERM from our stop button
                    pass          # already handled in _on_stop
                else:
                    _line_q.put((f"\n✗  Exited with code {rc}.", C_ERROR))
                    self.set_status(f"✗ code {rc}", C_ERROR)
            except Exception as exc:
                _line_q.put((f"\n✗  {exc}", C_ERROR))
                self.set_status("✗ error", C_ERROR)
            finally:
                state.running       = False
                state.proc          = None
                self._stop_btn.visible = False
                self.page.update()
                if on_done:
                    on_done()

        threading.Thread(target=_flush_thread, daemon=True).start()
        threading.Thread(target=_worker, daemon=True).start()

# ── UI helpers ────────────────────────────────────────────────────────────────

def _card(content: ft.Control, padding: int = 16) -> ft.Card:
    return ft.Card(
        content=ft.Container(content=content, padding=padding, bgcolor=C_CARD),
        elevation=2,
    )

def _section_title(text: str, icon: str | None = None) -> ft.Row:
    controls: list[ft.Control] = []
    if icon:
        controls += [ft.Icon(icon, color=C_PRIMARY, size=18), ft.Container(width=8)]
    controls.append(
        ft.Text(text, size=15, weight=ft.FontWeight.BOLD, color=C_PRIMARY)
    )
    return ft.Row(controls=controls)

def _label(text: str) -> ft.Text:
    return ft.Text(text, size=12, color=ft.Colors.with_opacity(0.60, ft.Colors.WHITE))

def _chip(label: str, color: str) -> ft.Container:
    return ft.Container(
        content=ft.Text(label, size=10, color=color, weight=ft.FontWeight.BOLD),
        bgcolor=ft.Colors.with_opacity(0.12, color),
        border_radius=12,
        padding=ft.Padding.symmetric(horizontal=8, vertical=3),
    )

def _status_chip(done: int, total: int) -> ft.Container:
    if total == 0:
        return _chip("○  none", ft.Colors.with_opacity(0.35, ft.Colors.WHITE))
    if done >= total:
        return _chip(f"✓  {done}/{total}", C_SUCCESS)
    return _chip(f"◐  {done}/{total}", C_WARN)

def _course_dropdown(value: str, on_select: callable,
                     include_all: bool = False) -> ft.Dropdown:
    """
    Bug fix: use `on_select` (Flet 0.82+) and read value from `e.data`
    rather than `e.control.value` (which may not be updated yet).
    """
    options = []
    if include_all:
        options.append(ft.dropdown.Option(key="0", text="All courses"))
    for cid, name in COURSES.items():
        options.append(ft.dropdown.Option(key=str(cid), text=f"{name}  ({cid})"))
    if not options:
        options.append(ft.dropdown.Option(
            key="", text="— no courses, add Canvas token in Settings —"))
    return ft.Dropdown(
        options=options,
        value=value if COURSES else None,
        on_select=on_select,
        bgcolor=C_SURFACE,
        border_color=ft.Colors.with_opacity(0.25, ft.Colors.WHITE),
        focused_border_color=C_PRIMARY,
        color=ft.Colors.WHITE,
        label="Course",
        label_style=ft.TextStyle(color=C_PRIMARY),
        expand=True,
    )

def _text_field(label: str, value: str = "", hint: str = "",
                expand: bool | int = True) -> ft.TextField:
    return ft.TextField(
        label=label,
        value=value,
        hint_text=hint,
        bgcolor=C_SURFACE,
        border_color=ft.Colors.with_opacity(0.25, ft.Colors.WHITE),
        focused_border_color=C_PRIMARY,
        color=ft.Colors.WHITE,
        label_style=ft.TextStyle(color=C_PRIMARY),
        cursor_color=C_PRIMARY,
        expand=expand,
    )

def _run_btn(text: str, icon: str, on_click: callable) -> ft.FilledButton:
    return ft.FilledButton(
        content=ft.Row(
            controls=[ft.Icon(icon, size=16), ft.Text(text, size=13)],
            tight=True, spacing=6,
        ),
        style=ft.ButtonStyle(
            bgcolor=C_PRIMARY,
            color=ft.Colors.BLACK,
            padding=ft.Padding.symmetric(horizontal=16, vertical=10),
        ),
        on_click=on_click,
    )

def _outlined_btn(text: str, icon: str, on_click: callable) -> ft.OutlinedButton:
    return ft.OutlinedButton(
        content=ft.Row(
            controls=[ft.Icon(icon, size=16, color=C_PRIMARY),
                      ft.Text(text, size=13, color=C_PRIMARY)],
            tight=True, spacing=6,
        ),
        style=ft.ButtonStyle(
            side=ft.BorderSide(1, C_PRIMARY),
            padding=ft.Padding.symmetric(horizontal=14, vertical=10),
        ),
        on_click=on_click,
    )

def _page_layout(scroll_controls: list[ft.Control]) -> ft.Column:
    """Scrollable form area that fills its allotted space."""
    return ft.Column(
        controls=scroll_controls,
        spacing=12,
        scroll=ft.ScrollMode.AUTO,
        expand=True,
    )

# ── Page: Dashboard ───────────────────────────────────────────────────────────

def build_dashboard(page: ft.Page, console: OutputConsole,
                    navigate: callable | None = None,
                    on_refresh: callable | None = None) -> ft.Column:

    def _course_card(cid: int, name: str) -> ft.Card:
        vd, vt  = _video_status(cid)
        caps    = _caption_count(cid)
        aligns  = _alignment_count(cid)
        notes_p = _notes_path(cid)
        short   = name.split()[0]
        note_info = (
            ft.Text(notes_p.name, size=10,
                    color=ft.Colors.with_opacity(0.5, ft.Colors.WHITE))
            if notes_p else
            ft.Text("no notes yet", size=10,
                    color=ft.Colors.with_opacity(0.3, ft.Colors.WHITE))
        )
        return ft.Card(
            content=ft.Container(
                content=ft.Column(controls=[
                    ft.Row(controls=[
                        ft.Text(short, size=18, weight=ft.FontWeight.BOLD,
                                color=C_PRIMARY),
                        ft.Container(expand=True),
                        _chip("✓ notes", C_SUCCESS) if notes_p
                        else _chip("○ pending",
                                   ft.Colors.with_opacity(0.4, ft.Colors.WHITE)),
                    ]),
                    ft.Text(name, size=11,
                            color=ft.Colors.with_opacity(0.55, ft.Colors.WHITE)),
                    ft.Divider(height=10,
                               color=ft.Colors.with_opacity(0.08, ft.Colors.WHITE)),
                    ft.Row(controls=[
                        ft.Column(controls=[_label("Videos"),   _status_chip(vd, vt)],   spacing=4),
                        ft.Column(controls=[_label("Captions"), _status_chip(caps, max(caps, vd))], spacing=4),
                        ft.Column(controls=[_label("Aligned"),  _status_chip(aligns, max(aligns, caps))], spacing=4),
                    ], spacing=20),
                    ft.Container(height=4),
                    note_info,
                ], spacing=6),
                padding=16,
                bgcolor=C_CARD,
            ),
            elevation=3,
            expand=True,
        )

    def _quick(label: str, icon: str, idx: int) -> ft.ElevatedButton:
        def _go(_):
            if navigate:
                navigate(idx)
        return ft.ElevatedButton(
            content=ft.Row(
                controls=[ft.Icon(icon, size=15), ft.Text(label, size=12)],
                tight=True, spacing=6,
            ),
            style=ft.ButtonStyle(
                bgcolor=ft.Colors.with_opacity(0.08, ft.Colors.WHITE),
                color=ft.Colors.WHITE,
                side=ft.BorderSide(1, ft.Colors.with_opacity(0.12, ft.Colors.WHITE)),
                padding=ft.Padding.symmetric(horizontal=12, vertical=8),
            ),
            on_click=_go,
        )

    if COURSES:
        items = list(COURSES.items())
        course_rows = [
            ft.Row(controls=[_course_card(cid, name)
                              for cid, name in items[i:i+2]], spacing=12)
            for i in range(0, len(items), 2)
        ]
    else:
        course_rows = [_card(ft.Column(controls=[
            ft.Container(height=8),
            ft.Icon(ft.Icons.SCHOOL_OUTLINED, size=52,
                    color=ft.Colors.with_opacity(0.25, ft.Colors.WHITE)),
            ft.Text("No courses loaded", size=15,
                    color=ft.Colors.with_opacity(0.45, ft.Colors.WHITE),
                    weight=ft.FontWeight.W_500),
            ft.Text(
                "Go to Settings → enter your Canvas URL and API token,\n"
                "then save to load your courses automatically.",
                size=12, text_align=ft.TextAlign.CENTER,
                color=ft.Colors.with_opacity(0.35, ft.Colors.WHITE),
            ),
            ft.Container(height=8),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10))]

    refresh_btn = ft.IconButton(
        icon=ft.Icons.REFRESH,
        tooltip="Refresh courses from Canvas",
        icon_color=C_PRIMARY,
        on_click=lambda _: on_refresh() if on_refresh else None,
    )

    scroll_content = [
        ft.Row(controls=[
            _section_title("Course Overview", ft.Icons.DASHBOARD_OUTLINED),
            ft.Container(expand=True),
            refresh_btn,
        ]),
        *course_rows,
        ft.Container(height=4),
        _section_title("Quick Actions", ft.Icons.BOLT_OUTLINED),
        ft.Row(controls=[
            _quick("Full Pipeline",  ft.Icons.PLAY_CIRCLE_OUTLINE, 1),
            _quick("Download",       ft.Icons.DOWNLOAD_OUTLINED,   2),
            _quick("Transcribe",     ft.Icons.MIC_NONE,            3),
            _quick("Align",          ft.Icons.LINK_OUTLINED,       4),
            _quick("Generate Notes", ft.Icons.ARTICLE_OUTLINED,    5),
        ], spacing=8, wrap=True),
    ]
    return _page_layout(scroll_content)

# ── Page: Full Pipeline ───────────────────────────────────────────────────────

def build_pipeline(page: ft.Page, console: OutputConsole) -> ft.Column:
    course_val = {"v": str(next(iter(COURSES), ""))}

    course_dd = _course_dropdown(
        value=course_val["v"],
        # Bug fix: read e.data, not e.control.value
        on_select=lambda e: course_val.update({"v": e.data}),
    )

    step_checks = {
        "dl_material": ft.Checkbox(label="Download materials",   value=True,  fill_color=C_PRIMARY),
        "dl_video":    ft.Checkbox(label="Download videos",      value=True,  fill_color=C_PRIMARY),
        "transcribe":  ft.Checkbox(label="Transcribe videos",    value=True,  fill_color=C_PRIMARY),
        "align":       ft.Checkbox(label="Align transcripts",    value=True,  fill_color=C_PRIMARY),
        "generate":    ft.Checkbox(label="Generate study notes", value=True,  fill_color=C_PRIMARY),
    }

    secretly_sw   = ft.Switch(label="Stealth mode for downloads",
                               value=False, active_color=C_PRIMARY)
    course_name_f = _text_field("Course name for notes")
    detail_label  = ft.Text("7", size=22, weight=ft.FontWeight.BOLD, color=C_PRIMARY)
    detail_slider = ft.Slider(
        min=0, max=10, value=7, divisions=10,
        active_color=C_PRIMARY,
        on_change=lambda e: (
            setattr(detail_label, "value", str(int(e.control.value))),
            page.update(),
        ),
    )
    lec_filter_f = _text_field("Lecture filter", hint="1-5  or  1,3,5  (blank=all)")
    force_sw     = ft.Switch(label="Force regenerate", value=False, active_color=C_PRIMARY)

    def _run(_):
        cid  = int(course_val["v"])
        name = course_name_f.value.strip() or _course_name_from_notes(cid)
        steps = [k for k, cb in step_checks.items() if cb.value]
        if not steps:
            console.write("No steps selected.", color=C_WARN)
            return

        cmds: list[tuple[str, list[str]]] = []
        if "dl_material" in steps:
            c = [PYTHON, str(SCRIPTS["downloader"]),
                 "--course", str(cid), "--download-material-all"]
            if secretly_sw.value:
                c.append("--secretly")
            cmds.append(("Download materials", c))

        if "dl_video" in steps:
            c = [PYTHON, str(SCRIPTS["downloader"]),
                 "--course", str(cid), "--download-video-all"]
            if secretly_sw.value:
                c.append("--secretly")
            cmds.append(("Download videos", c))

        if "transcribe" in steps:
            cmds.append(("Transcribe", [PYTHON, str(SCRIPTS["transcribe"])]))

        if "align" in steps:
            cmds.append(("Align", [PYTHON, str(SCRIPTS["align"]),
                                   "--course", str(cid)]))

        if "generate" in steps:
            c = [PYTHON, str(SCRIPTS["generate"]),
                 "--course",      str(cid),
                 "--course-name", name,
                 "--detail",      str(int(detail_slider.value))]
            if lec_filter_f.value.strip():
                c += ["--lectures", lec_filter_f.value.strip()]
            if force_sw.value:
                c.append("--force")
            cmds.append(("Generate notes", c))

        def _chain(idx: int = 0) -> None:
            if idx >= len(cmds):
                return
            label, cmd = cmds[idx]
            console.write(
                f"\n{'─'*50}\n▶  Step {idx+1}/{len(cmds)}: {label}\n",
                color=C_SECONDARY,
            )
            console.run(cmd, on_done=lambda: _chain(idx + 1))

        _chain()

    scroll_content = [
        _section_title("Full Pipeline Wizard", ft.Icons.ACCOUNT_TREE_OUTLINED),
        _card(ft.Row(controls=[
            ft.Column(controls=[_label("Course"), course_dd], spacing=6, expand=True),
        ])),
        ft.Row(controls=[
            _card(ft.Column(controls=[
                _label("Steps to execute"),
                *step_checks.values(),
            ], spacing=4), padding=16),
            _card(ft.Column(controls=[
                _label("Download"),
                secretly_sw,
                ft.Divider(height=10,
                           color=ft.Colors.with_opacity(0.08, ft.Colors.WHITE)),
                _label("Note generation"),
                course_name_f,
                ft.Row(controls=[
                    detail_label,
                    ft.Column(controls=[detail_slider], expand=True),
                ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER,
                   expand=True),
                lec_filter_f,
                force_sw,
            ], spacing=8), padding=16),
        ], spacing=12, wrap=True),
        ft.Row(controls=[_run_btn("Run Pipeline", ft.Icons.PLAY_ARROW, _run)]),
    ]
    return _page_layout(scroll_content)

# ── Page: Download ────────────────────────────────────────────────────────────

def build_download(page: ft.Page, console: OutputConsole) -> ft.Column:
    course_val  = {"v": "0"}
    course_dd   = _course_dropdown(
        value="0",
        # Bug fix: use e.data
        on_select=lambda e: course_val.update({"v": e.data}),
        include_all=True,
    )
    secretly_sw  = ft.Switch(label="Stealth mode", value=False, active_color=C_PRIMARY)
    video_nums_f = _text_field("Video numbers", hint="e.g. 1 3 5", expand=2)
    mat_names_f  = _text_field("Filename(s)", hint="partial names, space-separated",
                                expand=True)

    def _course_args() -> list[str]:
        cid = course_val["v"]
        return ["--course", cid] if cid != "0" else []

    def _secretly_args() -> list[str]:
        return ["--secretly"] if secretly_sw.value else []

    def _go(extra: list[str]) -> None:
        console.run([PYTHON, str(SCRIPTS["downloader"])]
                    + _course_args() + extra + _secretly_args())

    scroll_content = [
        _section_title("Download", ft.Icons.DOWNLOAD_OUTLINED),
        _card(ft.Row(controls=[
            ft.Column(controls=[_label("Course filter"), course_dd], spacing=6, expand=True),
            ft.Column(controls=[_label("Options"), secretly_sw], spacing=6),
        ], spacing=24)),

        _card(ft.Column(controls=[
            ft.Row(controls=[
                ft.Icon(ft.Icons.VIDEOCAM_OUTLINED, color=C_PRIMARY, size=16),
                ft.Text("Videos", size=13, weight=ft.FontWeight.BOLD,
                        color=ft.Colors.WHITE),
            ], spacing=8),
            ft.Row(controls=[
                _outlined_btn("List videos", ft.Icons.LIST,
                              lambda _: _go(["--video-list"])),
                _outlined_btn("Download all pending", ft.Icons.DOWNLOAD,
                              lambda _: _go(["--download-video-all"])),
            ], spacing=8, wrap=True),
            ft.Row(controls=[
                video_nums_f,
                _run_btn("Download selected", ft.Icons.DOWNLOAD,
                         lambda _: _go(["--download-video"]
                                       + video_nums_f.value.strip().split()
                                       if video_nums_f.value.strip() else [])),
            ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.END),
        ], spacing=10)),

        _card(ft.Column(controls=[
            ft.Row(controls=[
                ft.Icon(ft.Icons.FOLDER_OUTLINED, color=C_PRIMARY, size=16),
                ft.Text("Course materials", size=13, weight=ft.FontWeight.BOLD,
                        color=ft.Colors.WHITE),
            ], spacing=8),
            ft.Row(controls=[
                _outlined_btn("List materials", ft.Icons.LIST,
                              lambda _: _go(["--material-list"])),
                _outlined_btn("Download all pending", ft.Icons.DOWNLOAD,
                              lambda _: _go(["--download-material-all"])),
            ], spacing=8, wrap=True),
            ft.Row(controls=[
                mat_names_f,
                _run_btn("Download selected", ft.Icons.DOWNLOAD,
                         lambda _: _go(["--download-material"]
                                       + mat_names_f.value.strip().split()
                                       if mat_names_f.value.strip() else [])),
            ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.END),
        ], spacing=10)),
    ]
    return _page_layout(scroll_content)

# ── Page: Transcribe ──────────────────────────────────────────────────────────

def build_transcribe(page: ft.Page, console: OutputConsole) -> ft.Column:
    single_path_f = _text_field(
        "Single video path",
        hint="Leave blank to process all pending videos in manifest",
    )

    def _run(_) -> None:
        cmd = [PYTHON, str(SCRIPTS["transcribe"])]
        if single_path_f.value.strip():
            cmd += ["--video", single_path_f.value.strip()]
        console.run(cmd)

    model = _read_constant("transcribe", "WHISPER_MODEL_SIZE")
    lang  = _read_constant("transcribe", "WHISPER_LANGUAGE")

    scroll_content = [
        _section_title("Transcribe Videos", ft.Icons.MIC_NONE),
        _card(ft.Column(controls=[
            ft.Row(controls=[
                ft.Icon(ft.Icons.INFO_OUTLINE, color=C_PRIMARY, size=15),
                ft.Text(f"Model: {model}   Language: {lang or 'auto-detect'}",
                        size=12,
                        color=ft.Colors.with_opacity(0.7, ft.Colors.WHITE)),
            ], spacing=8),
            ft.Text(
                "Auto-selects backend: uses faster-whisper on GPU if available, "
                "otherwise falls back to OpenAI Whisper API (requires openai_api.txt). "
                "Processes all videos in manifest.json that lack a caption.",
                size=12,
                color=ft.Colors.with_opacity(0.6, ft.Colors.WHITE),
            ),
        ], spacing=8)),
        _card(ft.Column(controls=[
            single_path_f,
            ft.Container(height=4),
            _run_btn("Transcribe all pending", ft.Icons.PLAY_ARROW, _run),
        ], spacing=8)),
    ]
    return _page_layout(scroll_content)

# ── Page: Align ───────────────────────────────────────────────────────────────

def build_align(page: ft.Page, console: OutputConsole) -> ft.Column:
    course_val  = {"v": str(next(iter(COURSES), ""))}
    course_dd   = _course_dropdown(
        value=course_val["v"],
        # Bug fix: use e.data
        on_select=lambda e: course_val.update({"v": e.data}),
    )
    out_dir_f    = _text_field("Output directory",
                                hint="blank = [course]/alignment/")
    caption_f    = _text_field("Caption JSON path")
    slides_f     = _text_field("Slide file(s)",
                                hint="space-separated for multi-part")
    manual_out_f = _text_field("Output directory",
                                hint="blank = auto-inferred")

    def _run_course(_) -> None:
        cmd = [PYTHON, str(SCRIPTS["align"]), "--course", course_val["v"]]
        if out_dir_f.value.strip():
            cmd += ["--out", out_dir_f.value.strip()]
        console.run(cmd)

    def _run_manual(_) -> None:
        if not caption_f.value.strip() or not slides_f.value.strip():
            console.write("Caption and slide path(s) are required.", color=C_WARN)
            return
        cmd = ([PYTHON, str(SCRIPTS["align"]),
                "--caption", caption_f.value.strip(),
                "--slides"]  + slides_f.value.strip().split())
        if manual_out_f.value.strip():
            cmd += ["--out", manual_out_f.value.strip()]
        console.run(cmd)

    embed_model = _read_constant("align", "EMBED_MODEL")
    ctx_sec     = _read_constant("align", "CONTEXT_SEC")

    scroll_content = [
        _section_title("Align Transcripts to Slides", ft.Icons.LINK_OUTLINED),
        _card(ft.Row(controls=[
            ft.Icon(ft.Icons.INFO_OUTLINE, color=C_PRIMARY, size=15),
            ft.Text(
                f"Embed: {embed_model}   Context: ±{ctx_sec}s   "
                "Content-based fallback if names don't match.",
                size=12,
                color=ft.Colors.with_opacity(0.7, ft.Colors.WHITE),
            ),
        ], spacing=8)),

        _card(ft.Column(controls=[
            ft.Text("Auto-discover (whole course)", size=13,
                    weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
            ft.Text("Pairs all unaligned captions with matching slide files.",
                    size=12, color=ft.Colors.with_opacity(0.6, ft.Colors.WHITE)),
            ft.Container(height=6),
            ft.Row(controls=[
                ft.Column(controls=[_label("Course"), course_dd],
                          spacing=6, expand=True),
                ft.Column(controls=[_label("Output dir"), out_dir_f],
                          spacing=6, expand=True),
            ], spacing=16),
            ft.Container(height=4),
            _run_btn("Run auto-align", ft.Icons.AUTO_FIX_HIGH, _run_course),
        ], spacing=8)),

        _card(ft.Column(controls=[
            ft.Text("Manual (specific files)", size=13,
                    weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
            ft.Text("Align one caption to one or more slide files.",
                    size=12, color=ft.Colors.with_opacity(0.6, ft.Colors.WHITE)),
            ft.Container(height=6),
            caption_f,
            slides_f,
            ft.Row(controls=[
                ft.Column(controls=[manual_out_f], expand=True),
                _run_btn("Align", ft.Icons.LINK, _run_manual),
            ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.END),
        ], spacing=8)),
    ]
    return _page_layout(scroll_content)

# ── Page: Generate Notes ──────────────────────────────────────────────────────

def build_generate(page: ft.Page, console: OutputConsole) -> ft.Column:
    default_cid   = next(iter(COURSES), None)
    course_val    = {"v": str(default_cid) if default_cid else ""}

    course_name_f = _text_field("Course name",
                                 value=_course_name_from_notes(default_cid) if default_cid else "")

    def _on_course_select(e) -> None:
        # Bug fix: use e.data
        course_val.update({"v": e.data})
        try:
            course_name_f.value = _course_name_from_notes(int(e.data))
        except (ValueError, KeyError):
            pass
        page.update()

    course_dd   = _course_dropdown(
        value=course_val["v"],
        on_select=_on_course_select,
    )
    lec_filter_f = _text_field("Lecture filter",
                                hint="e.g. 1-5  or  1,3,5  (blank=all)")
    detail_label = ft.Text("7", size=22, weight=ft.FontWeight.BOLD, color=C_PRIMARY)
    detail_slider = ft.Slider(
        min=0, max=10, value=7, divisions=10,
        active_color=C_PRIMARY,
        on_change=lambda e: (
            setattr(detail_label, "value", str(int(e.control.value))),
            page.update(),
        ),
    )
    force_sw   = ft.Switch(label="Force regenerate all sections",
                            value=False, active_color=C_PRIMARY)
    merge_sw   = ft.Switch(label="Merge-only (skip generation)",
                            value=False, active_color=C_PRIMARY)
    iterate_sw = ft.Switch(label="Iterative mode (raise detail until quality target)",
                            value=False, active_color=C_PRIMARY)

    note_model   = _read_constant("generate", "NOTE_MODEL")
    verify_model = _read_constant("generate", "VERIFY_MODEL")
    quality      = _read_constant("generate", "QUALITY_TARGET")

    def _run(_) -> None:
        cid  = int(course_val["v"])
        name = course_name_f.value.strip() or _course_name_from_notes(cid)
        cmd  = [PYTHON, str(SCRIPTS["generate"]),
                "--course",      str(cid),
                "--course-name", name,
                "--detail",      str(int(detail_slider.value))]
        if lec_filter_f.value.strip():
            cmd += ["--lectures", lec_filter_f.value.strip()]
        if force_sw.value:
            cmd.append("--force")
        if merge_sw.value:
            cmd.append("--merge-only")
        if iterate_sw.value:
            cmd.append("--iterate")
        console.run(cmd)

    detail_styles = [
        ("0–2", "Outline"),
        ("3–5", "Hierarchical bullets"),
        ("6–8", "Full paragraphs"),
        ("9–10","Exhaustive"),
    ]

    scroll_content = [
        _section_title("Generate Study Notes", ft.Icons.ARTICLE_OUTLINED),
        _card(ft.Row(controls=[
            ft.Icon(ft.Icons.INFO_OUTLINE, color=C_PRIMARY, size=15),
            ft.Text(
                f"Generator: {note_model}   Verifier: {verify_model}   "
                f"Quality target: {quality}",
                size=12,
                color=ft.Colors.with_opacity(0.7, ft.Colors.WHITE),
            ),
        ], spacing=8)),
        ft.Row(controls=[
            _card(ft.Column(controls=[
                _label("Course"),
                course_dd,
                ft.Container(height=4),
                course_name_f,
                ft.Container(height=4),
                lec_filter_f,
            ], spacing=8), padding=16),
            _card(ft.Column(controls=[
                _label("Detail level"),
                ft.Row(controls=[
                    detail_label,
                    ft.Column(controls=[
                        detail_slider,
                        ft.Row(controls=[
                            ft.Text(f"  {rng} {desc}", size=10,
                                    color=ft.Colors.with_opacity(0.40, ft.Colors.WHITE))
                            for rng, desc in detail_styles
                        ], spacing=0, wrap=True),
                    ], expand=True),
                ], spacing=8,
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ft.Divider(height=10,
                           color=ft.Colors.with_opacity(0.08, ft.Colors.WHITE)),
                _label("Options"),
                force_sw,
                merge_sw,
                iterate_sw,
            ], spacing=6), padding=16),
        ], spacing=12, wrap=True),
        ft.Row(controls=[_run_btn("Generate Notes", ft.Icons.AUTO_AWESOME, _run)]),
    ]
    return _page_layout(scroll_content)

# ── Page: Settings ────────────────────────────────────────────────────────────

def build_settings(page: ft.Page,
                   on_courses_changed: callable | None = None) -> ft.Column:

    def _snack(msg: str, ok: bool = True) -> None:
        page.snack_bar = ft.SnackBar(
            content=ft.Text(msg, color=ft.Colors.BLACK),
            bgcolor=C_SUCCESS if ok else C_ERROR,
            duration=2500,
            open=True,
        )
        page.update()

    def _field_row(label: str, tf: ft.TextField) -> ft.Row:
        return ft.Row(controls=[
            ft.Text(label, size=12, color=ft.Colors.WHITE, width=140),
            tf,
        ], spacing=8)

    # ── Connection settings (config.json) ────────────────────────────────────

    config_file = DATA_DIR / "config.json"

    def _load_config() -> dict:
        return json.load(open(config_file)) if config_file.exists() else {}

    def _save_config_all(data: dict) -> None:
        cfg = _load_config()
        cfg.update(data)
        with open(config_file, "w") as f:
            json.dump(cfg, f, indent=2)

    # _v holds the live text for every field; on_change keeps it in sync.
    # Reading tf.value without on_change returns the *initial* value only.
    _cfg = _load_config()
    canvas_file    = DATA_DIR / "canvas_token.txt"
    openai_file    = DATA_DIR / "openai_api.txt"
    anthropic_file = DATA_DIR / "anthropic_key.txt"
    gemini_file    = DATA_DIR / "gemini_api.txt"

    _v = {
        "canvas_url":  _cfg.get("CANVAS_URL", ""),
        "panopto":     _cfg.get("PANOPTO_HOST", ""),
        "python_path": _cfg.get("PYTHON_PATH", ""),
        "canvas":      canvas_file.read_text().strip() if canvas_file.exists() else "",
        "openai":      openai_file.read_text().strip() if openai_file.exists() else "",
        "anthropic":   anthropic_file.read_text().strip() if anthropic_file.exists() else "",
        "gemini":      gemini_file.read_text().strip() if gemini_file.exists() else "",
    }

    def _mk_tf(key: str, **kwargs) -> ft.TextField:
        return ft.TextField(
            value=_v[key],
            on_change=lambda e, k=key: _v.update({k: e.control.value}),
            **kwargs,
        )

    tf_canvas_url = _mk_tf("canvas_url",
        hint_text="canvas.yourschool.edu  (https:// added automatically)",
        expand=True, dense=True, bgcolor=C_OUTPUT_BG, border_color=C_PRIMARY, text_size=12)
    tf_panopto = _mk_tf("panopto",
        hint_text="mediaweb.ap.panopto.com",
        expand=True, dense=True, bgcolor=C_OUTPUT_BG, border_color=C_PRIMARY, text_size=12)
    tf_python_path = _mk_tf("python_path",
        hint_text="/path/to/conda/envs/auto-note/bin/python  (leave blank for system python3)",
        expand=True, dense=True, bgcolor=C_OUTPUT_BG, border_color=C_PRIMARY, text_size=12)
    tf_canvas = _mk_tf("canvas",
        password=True, can_reveal_password=True,
        hint_text="Canvas API token",
        expand=True, dense=True, bgcolor=C_OUTPUT_BG, border_color=C_PRIMARY, text_size=12)
    tf_openai = _mk_tf("openai",
        password=True, can_reveal_password=True,
        hint_text="sk-…  (no default)",
        expand=True, dense=True, bgcolor=C_OUTPUT_BG, border_color=C_PRIMARY, text_size=12)
    tf_anthropic = _mk_tf("anthropic",
        password=True, can_reveal_password=True,
        hint_text="sk-ant-…  (no default)",
        expand=True, dense=True, bgcolor=C_OUTPUT_BG, border_color=C_PRIMARY, text_size=12)
    tf_gemini = _mk_tf("gemini",
        password=True, can_reveal_password=True,
        hint_text="AIza…  (Google AI Studio key, no default)",
        expand=True, dense=True, bgcolor=C_OUTPUT_BG, border_color=C_PRIMARY, text_size=12)

    conn_card = _card(ft.Column(controls=[
        ft.Text("Connection", size=13,
                weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
        ft.Text("Saved to config.json in the project directory.",
                size=11, color=ft.Colors.with_opacity(0.5, ft.Colors.WHITE)),
        ft.Container(height=4),
        _field_row("Canvas URL",   tf_canvas_url),
        _field_row("Panopto Host", tf_panopto),
        _field_row("Python Path",  tf_python_path),
    ], spacing=10))

    refresh_status = ft.Text("", size=11,
                             color=ft.Colors.with_opacity(0.6, ft.Colors.WHITE))

    def _do_refresh():
        refresh_status.value = "Refreshing…"
        refresh_status.color = ft.Colors.with_opacity(0.6, ft.Colors.WHITE)
        page.update()
        # Flush current credentials to disk before loading, so the user doesn't
        # need to click "Save All" first — whatever is typed right now is used.
        try:
            _save_config_all({
                "CANVAS_URL":   _v["canvas_url"].strip(),
                "PANOPTO_HOST": _v["panopto"].strip(),
                "PYTHON_PATH":  _v["python_path"].strip(),
            })
            if _v["canvas"].strip():
                canvas_file.write_text(_v["canvas"].strip())
        except Exception:
            pass
        err = _load_courses_from_canvas()
        n = len(COURSES)
        if n:
            refresh_status.value = f"✓ {n} course{'s' if n != 1 else ''} loaded."
            refresh_status.color = C_SUCCESS
        elif err:
            refresh_status.value = f"✗ {err}"
            refresh_status.color = C_ERROR
        else:
            refresh_status.value = "No courses found after filtering."
            refresh_status.color = C_WARN
        page.update()
        if on_courses_changed:
            on_courses_changed()

    keys_card = _card(ft.Column(controls=[
        ft.Text("API Keys & Credentials", size=13,
                weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
        ft.Text("Keys are stored in plaintext files in the project directory.",
                size=11, color=ft.Colors.with_opacity(0.5, ft.Colors.WHITE)),
        ft.Container(height=4),
        _field_row("Canvas Token",      tf_canvas),
        _field_row("OpenAI API Key",    tf_openai),
        _field_row("Anthropic API Key", tf_anthropic),
        _field_row("Gemini API Key",    tf_gemini),
        ft.Row(controls=[
            ft.Text("Project dir", size=11,
                    color=ft.Colors.with_opacity(0.5, ft.Colors.WHITE)),
            ft.Text(str(DATA_DIR), size=11, selectable=True,
                    color=ft.Colors.with_opacity(0.7, ft.Colors.WHITE)),
        ], spacing=8),
    ], spacing=10))

    # ── ML Environment ────────────────────────────────────────────────────────

    _venv_exists = Path(ML_VENV_PYTHON).exists()
    env_status = ft.Text(
        ("✓ Installed at " + str(ML_VENV_DIR)) if _venv_exists else "Not installed",
        size=11,
        color=C_SUCCESS if _venv_exists else ft.Colors.with_opacity(0.5, ft.Colors.WHITE),
    )
    env_log = ft.TextField(
        value="", multiline=True, read_only=True, min_lines=1, max_lines=12,
        text_size=11, bgcolor=C_OUTPUT_BG, border_color=ft.Colors.TRANSPARENT,
        color=ft.Colors.with_opacity(0.85, ft.Colors.WHITE),
        expand=True, visible=False,
    )
    env_setup_btn = ft.FilledButton(
        "Install ML Environment",
        icon=ft.Icons.DOWNLOAD_OUTLINED,
        style=ft.ButtonStyle(bgcolor=C_SECONDARY, color=ft.Colors.BLACK),
    )
    env_reinstall_btn = ft.OutlinedButton(
        "Reinstall",
        icon=ft.Icons.REFRESH,
        style=ft.ButtonStyle(side=ft.BorderSide(1, C_SECONDARY), color=C_SECONDARY),
        visible=_venv_exists,
    )
    # Manual Python path override for when auto-detection fails
    _base_py_v = {"v": ""}
    tf_base_py = ft.TextField(
        hint_text="Python path (auto-detected, override if needed — e.g. /usr/bin/python3.12)",
        expand=True, dense=True, bgcolor=C_OUTPUT_BG, border_color=C_PRIMARY,
        text_size=11, visible=False,
        on_change=lambda e: _base_py_v.update({"v": e.control.value}),
    )

    def _append_log(line: str) -> None:
        env_log.value = (env_log.value or "") + line + "\n"
        env_log.visible = True
        page.update()

    def _run_env_setup(_=None) -> None:
        global PYTHON
        env_setup_btn.disabled = True
        env_reinstall_btn.disabled = True
        env_log.value = ""
        env_log.visible = True
        env_status.value = "Setting up…"
        env_status.color = C_WARN
        page.update()

        def _worker():
            global PYTHON

            # ── Find a Python that has SSL (required for pip HTTPS) ──────────
            import shutil as _sh
            home = Path.home()

            # 1) Manual override from the text field
            base_py = _base_py_v["v"].strip()

            # 2) Login shell — respects .bashrc/.zshrc and conda init
            if not base_py:
                _append_log("► Probing login shell for Python …")
                for shell_cmd in [
                    ["bash", "-l", "-c",
                     "python3 -c 'import ssl,venv,sys; print(sys.executable)'"],
                    ["zsh",  "-l", "-c",
                     "python3 -c 'import ssl,venv,sys; print(sys.executable)'"],
                ]:
                    try:
                        r = subprocess.run(shell_cmd, capture_output=True,
                                           text=True, timeout=15)
                        if r.returncode == 0:
                            for line in reversed(r.stdout.strip().splitlines()):
                                line = line.strip()
                                if line and Path(line).exists():
                                    base_py = line
                                    break
                        if base_py:
                            break
                    except Exception:
                        pass

            # 3) Common install locations
            if not base_py:
                _candidates = [
                    str(home / "miniconda3/bin/python3"),
                    str(home / "miniconda3/bin/python"),
                    str(home / "anaconda3/bin/python3"),
                    str(home / "anaconda3/bin/python"),
                    str(home / "miniforge3/bin/python3"),
                    str(home / "miniforge3/bin/python"),
                    str(home / "mambaforge/bin/python3"),
                    str(home / "mambaforge/bin/python"),
                    str(home / ".local/share/mamba/bin/python3"),
                    "/opt/conda/bin/python3",
                    "/opt/miniconda3/bin/python3",
                    "/opt/anaconda3/bin/python3",
                    _sh.which("python3") or "",
                    _sh.which("python") or "",
                    "/usr/bin/python3",
                    "/usr/local/bin/python3",
                ]
                for cand in _candidates:
                    if not cand or not Path(cand).exists():
                        continue
                    r = subprocess.run([cand, "-c", "import ssl, venv"],
                                       capture_output=True, timeout=5)
                    if r.returncode == 0:
                        base_py = cand
                        break

            if not base_py:
                _append_log("ERROR: Could not find a Python 3 with SSL support.")
                _append_log("")
                _append_log("  Paste the path to your conda/system Python below")
                _append_log("  (e.g. /home/user/miniconda3/bin/python3)")
                _append_log("  then click Install again.")
                tf_base_py.visible = True
                env_status.value = "✗ Python not found — enter path below"
                env_status.color = C_ERROR
                env_setup_btn.disabled = False
                env_reinstall_btn.disabled = False
                page.update()
                return

            _append_log(f"► Using Python: {base_py}")

            # Step 1 — create venv (remove stale one first)
            import shutil as _sh2
            if ML_VENV_DIR.exists():
                _append_log(f"► Removing old venv …")
                _sh2.rmtree(str(ML_VENV_DIR))
            _append_log(f"► Creating venv at {ML_VENV_DIR} …")
            r = subprocess.run(
                [base_py, "-m", "venv", str(ML_VENV_DIR)],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                _append_log("STDERR: " + r.stderr.strip())
                _append_log("ERROR: venv creation failed.")
                env_status.value = "✗ Setup failed"
                env_status.color = C_ERROR
                env_setup_btn.disabled = False
                env_reinstall_btn.disabled = False
                page.update()
                return
            _append_log("  venv created.")

            pip = str(ML_VENV_DIR / (
                "Scripts/pip.exe" if sys.platform == "win32" else "bin/pip"
            ))

            # Step 2 — upgrade pip
            _append_log("► Upgrading pip …")
            subprocess.run([pip, "install", "--upgrade", "pip"],
                           capture_output=True, text=True)

            # Step 3 — detect CUDA and install torch
            cuda = _detect_cuda()
            idx  = _torch_index_url(cuda)
            if cuda:
                _append_log(f"► CUDA {cuda[0]}.{cuda[1]} detected — installing torch (GPU) …")
            else:
                _append_log("► No GPU detected — installing torch (CPU) …")
            torch_cmd = [pip, "install", "torch"]
            if idx:
                torch_cmd += ["--index-url", idx]
            proc = subprocess.Popen(
                torch_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            for line in proc.stdout:
                _append_log(line.rstrip())
            proc.wait()

            # Step 4 — install remaining ML packages
            _append_log("► Installing ML packages …")
            proc = subprocess.Popen(
                [pip, "install"] + _ML_PACKAGES,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            for line in proc.stdout:
                _append_log(line.rstrip())
            proc.wait()

            # Step 5 — playwright browsers
            _append_log("► Installing Playwright browsers …")
            venv_py = ML_VENV_PYTHON
            proc = subprocess.Popen(
                [venv_py, "-m", "playwright", "install", "chromium"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            for line in proc.stdout:
                _append_log(line.rstrip())
            proc.wait()

            # Done — activate venv python
            PYTHON = ML_VENV_PYTHON
            try:
                _save_config_all({"PYTHON_PATH": ""})   # clear manual override; venv is auto-detected
            except Exception:
                pass
            env_status.value = "✓ Installed at " + str(ML_VENV_DIR)
            env_status.color = C_SUCCESS
            env_setup_btn.disabled = False
            env_reinstall_btn.disabled = False
            env_reinstall_btn.visible = True
            _append_log("\n✓ ML environment ready. You can now run the pipeline.")
            page.update()

        threading.Thread(target=_worker, daemon=True).start()

    env_setup_btn.on_click = _run_env_setup
    env_reinstall_btn.on_click = _run_env_setup

    env_card = _card(ft.Column(controls=[
        ft.Text("ML Environment", size=13,
                weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
        ft.Text(
            "Creates a virtual environment at ~/.auto_note/venv/ and installs all "
            "pipeline dependencies (torch, faster-whisper, sentence-transformers, …) "
            "automatically. Only needed once.",
            size=11, color=ft.Colors.with_opacity(0.5, ft.Colors.WHITE),
        ),
        ft.Container(height=4),
        ft.Row(controls=[
            ft.Text("Status:", size=11,
                    color=ft.Colors.with_opacity(0.5, ft.Colors.WHITE), width=60),
            env_status,
        ], spacing=8),
        ft.Row(controls=[env_setup_btn, env_reinstall_btn], spacing=10),
        tf_base_py,
        env_log,
    ], spacing=10))

    # ── Tunable Constants ─────────────────────────────────────────────────────

    _SCRIPT_COLORS = {
        "transcribe": ft.Colors.CYAN_300,
        "align":      ft.Colors.AMBER_300,
        "generate":   ft.Colors.PURPLE_200,
    }

    # options: list of (display_label, stored_value) pairs, or None for free text
    CONSTANTS = [
        #  script       name                   desc                     default           options
        ("transcribe", "WHISPER_MODEL_SIZE", "Whisper model variant", "large-v3", [
            ("tiny",             "tiny"),
            ("base",             "base"),
            ("small",            "small"),
            ("medium",           "medium"),
            ("large",            "large"),
            ("large-v2",         "large-v2"),
            ("large-v3",         "large-v3"),
            ("large-v3-turbo",   "large-v3-turbo"),
            ("distil-large-v3",  "distil-large-v3"),
        ]),
        ("transcribe", "WHISPER_LANGUAGE", "Transcription language", "None", [
            ("Auto-detect",  "None"),
            ("English",      "en"),
            ("Chinese",      "zh"),
            ("Japanese",     "ja"),
            ("Korean",       "ko"),
            ("French",       "fr"),
            ("German",       "de"),
            ("Spanish",      "es"),
        ]),
        ("align", "EMBED_MODEL", "Sentence-transformer model", "all-mpnet-base-v2", [
            ("all-mpnet-base-v2 (best quality)",           "all-mpnet-base-v2"),
            ("all-MiniLM-L12-v2 (balanced)",               "all-MiniLM-L12-v2"),
            ("all-MiniLM-L6-v2 (fast)",                    "all-MiniLM-L6-v2"),
            ("paraphrase-multilingual-mpnet-base-v2",      "paraphrase-multilingual-mpnet-base-v2"),
        ]),
        ("align",      "CONTEXT_SEC",         "Context window (s)",      "30",   None),
        ("align",      "OFF_SLIDE_THRESHOLD",  "Off-slide cosine cutoff", "0.28", None),
        ("align",      "PRIOR_SIGMA",          "Temporal prior σ",        "5",    None),
        ("generate", "NOTE_LANGUAGE", "Note language", "en", [
            ("English",         "en"),
            ("Chinese (中文)",  "zh"),
        ]),
        ("generate", "NOTE_MODEL", "Note generation LLM", "gpt-5.1", [
            # OpenAI
            ("gpt-5.1",              "gpt-5.1"),
            ("gpt-5.2",              "gpt-5.2"),
            ("gpt-5.3",              "gpt-5.3"),
            ("gpt-5.4",              "gpt-5.4"),
            ("gpt-4.1",              "gpt-4.1"),
            ("gpt-4.1-mini",         "gpt-4.1-mini"),
            ("o3",                   "o3"),
            ("o1",                   "o1"),
            # Gemini
            ("Gemini 2.5 Pro",       "gemini-2.5-pro"),
            ("Gemini 2.5 Flash",     "gemini-2.5-flash"),
            ("Gemini 2.0 Flash",     "gemini-2.0-flash"),
            ("Gemini 1.5 Pro",       "gemini-1.5-pro"),
            # Anthropic
            ("Claude Opus 4.6",      "claude-opus-4-6"),
            ("Claude Sonnet 4.6",    "claude-sonnet-4-6"),
            ("Claude Sonnet 4.5",    "claude-sonnet-4-5"),
            ("Claude Sonnet 3.5",    "claude-3-5-sonnet-20241022"),
            ("Claude Haiku 4.5",     "claude-haiku-4-5-20251001"),
        ]),
        ("generate", "VERIFY_MODEL", "Verification LLM", "gpt-4.1-mini", [
            # OpenAI
            ("gpt-4.1-mini",         "gpt-4.1-mini"),
            ("gpt-4.1",              "gpt-4.1"),
            ("gpt-5.1",              "gpt-5.1"),
            # Gemini
            ("Gemini 2.5 Flash",     "gemini-2.5-flash"),
            ("Gemini 2.0 Flash",     "gemini-2.0-flash"),
            # Anthropic
            ("Claude Haiku 4.5",     "claude-haiku-4-5-20251001"),
            ("Claude Sonnet 3.5",    "claude-3-5-sonnet-20241022"),
            ("Claude Sonnet 4.5",    "claude-sonnet-4-5"),
        ]),
        ("generate",   "DETAIL_LEVEL",         "Default detail level",    "8",    None),
        ("generate",   "CHAPTER_SIZE",          "Slides per GPT call",     "15",   None),
        ("generate",   "QUALITY_TARGET",        "Self-score target",       "8.0",  None),
    ]

    # Collect (ctrl, script, name, default) for bulk save
    _const_ctrls: list[tuple] = []

    def _const_row(script: str, name: str, desc: str, default: str,
                   options: list[tuple[str, str]] | None = None) -> ft.Row:
        cur    = _read_constant(script, name)
        accent = _SCRIPT_COLORS.get(script, C_PRIMARY)

        if options:
            ctrl = ft.Dropdown(
                value=cur if any(v == cur for _, v in options) else default,
                options=[ft.dropdown.Option(key=val, text=label) for label, val in options],
                expand=True, dense=True,
                bgcolor=C_OUTPUT_BG, border_color=accent, text_size=12,
                content_padding=ft.Padding.symmetric(horizontal=8, vertical=4),
            )
        else:
            ctrl = ft.TextField(
                value=cur, expand=True, dense=True,
                bgcolor=C_OUTPUT_BG, border_color=accent,
                text_size=12, content_padding=ft.Padding.symmetric(horizontal=8, vertical=6),
                on_change=lambda e: None,  # ensures Flet syncs typed value to ctrl.value
            )

        _const_ctrls.append((ctrl, script, name, default))

        def _reset(_, c=ctrl, s=script, n=name, d=default):
            c.value = d
            c.update()
            if _write_constant(s, n, d):
                _snack(f"{n} reset to default.")
            else:
                _snack(f"Failed to reset {n}.", ok=False)

        return ft.Row(controls=[
            ft.Container(
                ft.Text(script, size=10, color=accent, weight=ft.FontWeight.BOLD),
                width=74, padding=ft.Padding.symmetric(horizontal=4, vertical=2),
                border_radius=4,
                bgcolor=ft.Colors.with_opacity(0.12, accent),
            ),
            ft.Text(desc, size=12, color=ft.Colors.WHITE, width=190),
            ft.Text(name, size=11, color=ft.Colors.with_opacity(0.5, ft.Colors.WHITE),
                    width=160, italic=True),
            ctrl,
            ft.OutlinedButton(
                "Default", on_click=_reset,
                style=ft.ButtonStyle(side=ft.BorderSide(1, C_SECONDARY),
                                     color=C_SECONDARY),
                tooltip=f"Default: {default}",
            ),
        ], spacing=8)

    const_rows = [_const_row(*c) for c in CONSTANTS]

    const_card = _card(ft.Column(controls=[
        ft.Text("Tunable Constants", size=13,
                weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
        ft.Text("Changes are written directly to the corresponding script file.",
                size=11, color=ft.Colors.with_opacity(0.5, ft.Colors.WHITE)),
        ft.Divider(height=12, color=ft.Colors.with_opacity(0.15, ft.Colors.WHITE)),
        *const_rows,
    ], spacing=10))

    # ── Single Save All button ────────────────────────────────────────────────

    def _save_all(_):
        global PYTHON
        errors: list[str] = []
        try:
            _save_config_all({
                "CANVAS_URL":   _v["canvas_url"].strip(),
                "PANOPTO_HOST": _v["panopto"].strip(),
                "PYTHON_PATH":  _v["python_path"].strip(),
            })
            p = _v["python_path"].strip()
            PYTHON = p if p else _DEFAULT_PYTHON
        except Exception as e:
            errors.append(f"Connection: {e}")
        for path, key in [
            (canvas_file,    "canvas"),
            (openai_file,    "openai"),
            (anthropic_file, "anthropic"),
            (gemini_file,    "gemini"),
        ]:
            try:
                val = _v[key].strip()
                if val:
                    path.write_text(val)
            except Exception as e:
                errors.append(str(e))
        for ctrl, script, name, default in _const_ctrls:
            # Dropdowns sync via on_select; TextFields sync via on_change
            val = (ctrl.value or default) if isinstance(ctrl, ft.Dropdown) \
                  else ctrl.value.strip()
            if not _write_constant(script, name, val):
                errors.append(f"Failed to write {name}")
        if errors:
            _snack("Errors: " + "; ".join(errors), ok=False)
        else:
            _snack("All settings saved successfully!")

    save_btn = ft.FilledButton(
        "Save All Settings",
        icon=ft.Icons.SAVE_OUTLINED,
        on_click=_save_all,
        style=ft.ButtonStyle(
            bgcolor=C_PRIMARY, color=ft.Colors.BLACK,
            padding=ft.Padding.symmetric(horizontal=28, vertical=14),
        ),
    )
    refresh_btn_settings = ft.OutlinedButton(
        "Refresh Courses",
        icon=ft.Icons.REFRESH,
        on_click=lambda _: _do_refresh(),
        style=ft.ButtonStyle(
            side=ft.BorderSide(1, C_SECONDARY), color=C_SECONDARY,
            padding=ft.Padding.symmetric(horizontal=20, vertical=14),
        ),
    )

    return ft.Column(
        controls=[
            ft.Column(controls=[
                _section_title("Settings", ft.Icons.SETTINGS_OUTLINED),
                conn_card,
                keys_card,
                env_card,
                const_card,
                ft.Container(height=8),
                ft.Row(controls=[
                    save_btn,
                    refresh_btn_settings,
                    refresh_status,
                ], spacing=12),
                ft.Container(height=16),
            ], spacing=12, scroll=ft.ScrollMode.AUTO, expand=True),
        ],
        expand=True,
    )

# ── Main app ──────────────────────────────────────────────────────────────────

def main(page: ft.Page) -> None:
    page.title      = "AUTO NOTE"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor    = C_SURFACE
    page.theme      = ft.Theme(color_scheme_seed=ft.Colors.CYAN)
    page.window.width      = 1080
    page.window.height     = 780
    page.window.min_width  = 720
    page.window.min_height = 520
    page.padding = 0

    # One shared console per session (preserves history across page switches)
    console = OutputConsole(page)

    _nav_target: list[callable] = [None]

    def navigate(idx: int) -> None:
        if _nav_target[0]:
            _nav_target[0](idx)

    # Mutable ref so _rebuild can be passed to build_settings before it's defined
    _rebuild_ref: list[callable | None] = [None]

    def _build_pages() -> list:
        return [
            build_dashboard(page, console, navigate=navigate,
                            on_refresh=lambda: _rebuild_ref[0] and _rebuild_ref[0]()),
            build_pipeline(page, console),
            build_download(page, console),
            build_transcribe(page, console),
            build_align(page, console),
            build_generate(page, console),
            build_settings(page,
                           on_courses_changed=lambda: _rebuild_ref[0] and _rebuild_ref[0]()),
        ]

    # Load user-configured Python interpreter (e.g. conda env) from config
    _load_python_from_config()

    # Try to populate courses immediately if credentials are already on disk
    _load_courses_from_canvas()

    pages = _build_pages()

    # page_content swaps between tab pages; console stays fixed at the bottom
    page_content = ft.Container(
        content=pages[0],
        expand=True,
        padding=ft.Padding.only(left=16, right=16, top=16, bottom=8),
    )

    # Right-side panel: scrollable page content + pinned console
    right_panel = ft.Column(
        controls=[
            page_content,
            ft.Container(
                content=console.container,
                padding=ft.Padding.only(left=16, right=16, bottom=12),
            ),
        ],
        spacing=0,
        expand=True,
    )

    def _on_nav(e: ft.ControlEvent) -> None:
        idx                  = e.control.selected_index
        page_content.content = pages[idx]
        page.update()

    def _navigate(idx: int) -> None:
        rail.selected_index  = idx
        page_content.content = pages[idx]
        page.update()

    _nav_target[0] = _navigate

    def _rebuild() -> None:
        """Reload courses from Canvas and rebuild all course-dependent pages."""
        _load_courses_from_canvas()  # return value intentionally ignored here
        new = _build_pages()
        pages.clear()
        pages.extend(new)
        if COURSES and rail.selected_index == 6:
            # Refreshed from Settings and courses are now available —
            # jump to Dashboard so the user immediately sees the result.
            rail.selected_index = 0
            page_content.content = pages[0]
        elif rail.selected_index != 6:
            # Normal navigation rebuild — update current page.
            page_content.content = pages[rail.selected_index]
        # If still on Settings with no courses, leave content alone (avoids blank render).
        page.update()

    _rebuild_ref[0] = _rebuild

    rail = ft.NavigationRail(
        selected_index=0,
        destinations=[
            ft.NavigationRailDestination(
                icon=ft.Icons.DASHBOARD_OUTLINED,
                selected_icon=ft.Icons.DASHBOARD, label="Dashboard"),
            ft.NavigationRailDestination(
                icon=ft.Icons.ACCOUNT_TREE_OUTLINED,
                selected_icon=ft.Icons.ACCOUNT_TREE, label="Pipeline"),
            ft.NavigationRailDestination(
                icon=ft.Icons.DOWNLOAD_OUTLINED,
                selected_icon=ft.Icons.DOWNLOAD, label="Download"),
            ft.NavigationRailDestination(
                icon=ft.Icons.MIC_NONE,
                selected_icon=ft.Icons.MIC, label="Transcribe"),
            ft.NavigationRailDestination(
                icon=ft.Icons.LINK_OUTLINED,
                selected_icon=ft.Icons.LINK, label="Align"),
            ft.NavigationRailDestination(
                icon=ft.Icons.ARTICLE_OUTLINED,
                selected_icon=ft.Icons.ARTICLE, label="Generate"),
            ft.NavigationRailDestination(
                icon=ft.Icons.SETTINGS_OUTLINED,
                selected_icon=ft.Icons.SETTINGS, label="Settings"),
        ],
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=80,
        bgcolor=C_RAIL,
        indicator_color=ft.Colors.with_opacity(0.15, C_PRIMARY),
        on_change=_on_nav,
        leading=ft.Container(
            content=ft.Column(controls=[
                ft.Container(height=8),
                ft.Icon(ft.Icons.AUTO_STORIES, color=C_PRIMARY, size=28),
                ft.Text("AUTO\nNOTE", size=9, color=C_PRIMARY,
                        weight=ft.FontWeight.BOLD,
                        text_align=ft.TextAlign.CENTER),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2),
            padding=ft.Padding.only(bottom=12),
        ),
    )

    page.add(
        ft.Row(
            controls=[
                rail,
                ft.VerticalDivider(
                    width=1,
                    color=ft.Colors.with_opacity(0.08, ft.Colors.WHITE),
                ),
                right_panel,
            ],
            expand=True,
            spacing=0,
        )
    )


if __name__ == "__main__":
    ft.run(main)
