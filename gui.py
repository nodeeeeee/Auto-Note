#!/usr/bin/env python3
"""
gui.py — Material Design desktop GUI for the auto_note pipeline.

Requires: pip install flet
Usage:    python gui.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import flet as ft

# ── Project layout ────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).parent

# The ML venv is always at ~/.auto_note/venv/ regardless of frozen/dev mode.
# (The installer creates it there; dev mode uses it directly.)
_VENV_SUBPATH = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
ML_VENV_DIR    = Path.home() / ".auto_note" / "venv"
ML_VENV_PYTHON = str(ML_VENV_DIR / _VENV_SUBPATH)

# Initial Python: prefer the managed venv if already installed; otherwise
# fall back to sys.executable (dev mode) or system python3 (frozen/AppImage).
if Path(ML_VENV_PYTHON).exists():
    PYTHON = ML_VENV_PYTHON
elif getattr(sys, "frozen", False):
    import shutil as _shutil
    if sys.platform == "win32":
        PYTHON = _shutil.which("python") or "python"
    else:
        PYTHON = _shutil.which("python3") or _shutil.which("python") or "python3"
else:
    PYTHON = sys.executable

# User data directory: persistent across app restarts.
# When running as a PyInstaller bundle (AppImage / .exe), __file__ resolves to
# a temporary extraction folder that is deleted on exit — any files written
# there are lost.  Use ~/.auto_note/ instead so credentials and config survive.
if getattr(sys, "frozen", False):
    DATA_DIR = Path.home() / ".auto_note"
else:
    DATA_DIR = PROJECT_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline scripts are installed to ~/.auto_note/scripts/ in AppImage mode.
# In dev mode the scripts live directly in PROJECT_DIR (no scripts/ subdir).
SCRIPTS_DIR = DATA_DIR / "scripts"


def _install_scripts() -> None:
    """Copy bundled pipeline scripts from AppImage to ~/.auto_note/scripts/."""
    if not getattr(sys, "frozen", False):
        return   # dev mode: use files in place
    import shutil as _sh
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    for fname in [
        "downloader.py", "extract_caption.py", "frame_extractor.py",
        "semantic_alignment.py", "alignment_parser.py", "note_generation.py",
    ]:
        src = PROJECT_DIR / fname
        dst = SCRIPTS_DIR / fname
        if src.exists():
            _sh.copy2(str(src), str(dst))


def _script(name: str) -> Path:
    """Return the path to a pipeline script, falling back to PROJECT_DIR in dev mode."""
    installed = SCRIPTS_DIR / name
    if installed.exists():
        return installed
    return PROJECT_DIR / name   # dev mode: scripts are in the project root


# Lazily-resolved script paths.  In AppImage mode, _install_scripts() copies
# scripts to SCRIPTS_DIR *after* module load, so we must re-resolve on first
# access rather than caching stale paths from module-init time.
class _ScriptDict(dict):
    """Dict that re-resolves script paths on every access."""
    _NAMES = {
        "downloader":      "downloader.py",
        "transcribe":      "extract_caption.py",
        "frame_extractor": "frame_extractor.py",
        "align":           "semantic_alignment.py",
        "generate":        "note_generation.py",
    }
    def __getitem__(self, key: str) -> Path:
        return _script(self._NAMES[key])
    def __contains__(self, key: object) -> bool:
        return key in self._NAMES

SCRIPTS = _ScriptDict()

_DEFAULT_PYTHON = PYTHON   # auto-detected fallback; may be overridden by user config

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
    # PanoptoDownloader is not on PyPI; install from GitHub.
    # Its declared version pins (requests~=2.27, tqdm~=4.62, yarl~=1.7) are
    # conservative — newer versions work fine.
    "ffmpeg-progress-yield",
    "pycryptodomex",
    "git+https://github.com/Panopto-Video-DL/Panopto-Video-DL-lib.git",
]


def _find_base_python(log_fn: callable) -> str:
    """Find a Python 3 with ssl+venv support.

    Tries (in order):
      1. Login-shell probe via bash/zsh (Unix only)
      2. Common conda/system install locations (platform-specific)
    Returns the executable path or '' if nothing is found.
    """
    import shutil as _sh
    home = Path.home()

    # 1) Login shell probe — bash/zsh don't exist on Windows
    if sys.platform != "win32":
        log_fn("► Probing login shell for Python …")
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
                            return line
            except Exception:
                pass

    # 2) Common install locations (platform-specific)
    if sys.platform == "win32":
        _candidates = [
            str(home / "miniconda3"  / "python.exe"),
            str(home / "Miniconda3"  / "python.exe"),
            str(home / "anaconda3"   / "python.exe"),
            str(home / "Anaconda3"   / "python.exe"),
            str(home / "miniforge3"  / "python.exe"),
            str(home / "mambaforge"  / "python.exe"),
            # Standard Windows Python installer locations
            str(home / "AppData" / "Local" / "Programs" / "Python" / "Python313" / "python.exe"),
            str(home / "AppData" / "Local" / "Programs" / "Python" / "Python312" / "python.exe"),
            str(home / "AppData" / "Local" / "Programs" / "Python" / "Python311" / "python.exe"),
            str(home / "AppData" / "Local" / "Programs" / "Python" / "Python310" / "python.exe"),
            str(Path("C:/miniconda3/python.exe")),
            str(Path("C:/anaconda3/python.exe")),
            str(Path("C:/ProgramData/miniconda3/python.exe")),
            str(Path("C:/ProgramData/anaconda3/python.exe")),
            _sh.which("python") or "",
        ]
    else:
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
            return cand

    return ""


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


OUTPUT_DIR: Path = Path.home() / "AutoNote"


def _load_output_dir_from_config() -> None:
    """Load user-configured output directory from config.json."""
    global OUTPUT_DIR
    config_file = DATA_DIR / "config.json"
    if config_file.exists():
        try:
            cfg = json.load(open(config_file))
            p = cfg.get("OUTPUT_DIR", "").strip()
            if p:
                OUTPUT_DIR = Path(p)
        except Exception:
            pass


def _get_output_dir() -> Path:
    """Return current output directory, creating it if needed."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


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
    d    = DATA_DIR / str(course_id) / "notes"
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

        def _format_val() -> str:
            # None keyword → bare None
            if new_display_val == "None":
                return "None"
            # Numeric → bare (int or float)
            try:
                float(new_display_val)
                return new_display_val
            except ValueError:
                pass
            # String → always quoted
            return f'"{new_display_val}"'

        formatted = _format_val()

        # Match: NAME = VALUE [# optional comment]
        # The value part may be a quoted string, bare identifier, or number.
        # Preserve any inline comment that follows.
        new_src, n = re.subn(
            rf'^({name}\s*=\s*)([^\n#]+)(#[^\n]*)?$',
            lambda m: m.group(1) + formatted + ("   " + m.group(3) if m.group(3) else ""),
            src, count=1, flags=re.MULTILINE,
        )
        if n == 0:
            # Fallback: value might be glued to comment (e.g. "None# comment")
            new_src, n = re.subn(
                rf'^({name}\s*=\s*)(\S+)(#[^\n]*)?$',
                lambda m: m.group(1) + formatted + ("   " + m.group(3) if m.group(3) else ""),
                src, count=1, flags=re.MULTILINE,
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

        _UPDATE_INTERVAL = 0.1   # seconds between UI refreshes while streaming
        _default_color   = ft.Colors.with_opacity(0.88, ft.Colors.WHITE)

        def _append_line(text: str, color: str | None = None) -> None:
            for line in text.splitlines():
                self._lines.controls.append(
                    ft.Text(
                        line, size=11, font_family=MONO,
                        color=color or _default_color,
                        no_wrap=False, selectable=True,
                    )
                )

        def _worker() -> None:
            rc = -1
            try:
                import io as _io
                _ANSI_RE  = re.compile(
                    r"\x1b\[[0-9;]*[mABCDEFGHJKST]|\x1b\][^\x07]*\x07"
                )
                _MAX_LINES   = 500
                _last_update = time.monotonic()
                buf          = ""   # current line being assembled

                def _cur() -> ft.Text:
                    """Return (or create) the last Text item — the live line."""
                    if not self._lines.controls:
                        t = ft.Text("", size=11, font_family=MONO,
                                    color=_default_color, no_wrap=False,
                                    selectable=True)
                        self._lines.controls.append(t)
                    return self._lines.controls[-1]

                def _new_line() -> None:
                    """Append a blank Text item; trim oldest if over cap."""
                    self._lines.controls.append(
                        ft.Text("", size=11, font_family=MONO,
                                color=_default_color, no_wrap=False,
                                selectable=True)
                    )
                    excess = len(self._lines.controls) - _MAX_LINES
                    if excess > 0:
                        del self._lines.controls[:excess]

                def _process_chunk(raw: bytes) -> None:
                    nonlocal buf
                    text = _ANSI_RE.sub(
                        "", raw.decode("utf-8", errors="replace")
                    )
                    for ch in text:
                        if ch == "\r":
                            buf = ""
                        elif ch == "\n":
                            if not buf.startswith("<frozen importlib"):
                                item = _cur()
                                item.value = buf
                                low = buf.lower().lstrip()
                                if low.startswith(
                                        ("error", "traceback", "exception")):
                                    item.color = C_ERROR
                                elif low.startswith("warning"):
                                    item.color = C_WARN
                                _new_line()
                            buf = ""
                        else:
                            buf += ch

                # ── Launch subprocess ─────────────────────────────────────
                # On Unix use a pty so tqdm/rich see a real terminal and use
                # \r for in-place refresh.  On Windows pty is unavailable so
                # fall back to a plain pipe (progress bars produce extra lines
                # but work correctly).
                env = {
                    **os.environ,
                    "PYTHONUNBUFFERED":  "1",
                    "PYTHONIOENCODING":  "utf-8",
                    "PYTHONUTF8":        "1",
                }
                _use_pty = False
                if sys.platform != "win32":
                    try:
                        import pty as _pty
                        master_fd, slave_fd = _pty.openpty()
                        env.update({"TERM": "xterm-256color", "COLUMNS": "100"})
                        state.proc = subprocess.Popen(
                            cmd,
                            stdout=slave_fd, stderr=slave_fd,
                            close_fds=True,
                            cwd=str(_get_output_dir()),
                            env=env,
                        )
                        os.close(slave_fd)
                        _use_pty = True
                    except Exception:
                        _use_pty = False

                if not _use_pty:
                    state.proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        cwd=str(_get_output_dir()),
                        env=env,
                    )

                # ── Stream output ─────────────────────────────────────────
                if _use_pty:
                    try:
                        with _io.open(master_fd, "rb", closefd=True) as master:
                            while True:
                                try:
                                    chunk = master.read(4096)
                                except OSError:
                                    break
                                if not chunk:
                                    break
                                _process_chunk(chunk)
                                now = time.monotonic()
                                if now - _last_update >= _UPDATE_INTERVAL:
                                    if buf:
                                        _cur().value = buf
                                    self.page.update()
                                    _last_update = now
                    except OSError:
                        pass
                else:
                    for line in state.proc.stdout:
                        _process_chunk(line)
                        now = time.monotonic()
                        if now - _last_update >= _UPDATE_INTERVAL:
                            if buf:
                                _cur().value = buf
                            self.page.update()
                            _last_update = now

                if buf:
                    _cur().value = buf
                self.page.update()

                state.proc.wait()
                rc = state.proc.returncode

                if rc == 0:
                    _append_line("\n✓  Completed successfully.", C_SUCCESS)
                    self.set_status("✓ done", C_SUCCESS)
                elif rc == -15:
                    pass  # user stopped it
                else:
                    _append_line(f"\n✗  Exited with code {rc}.", C_ERROR)
                    self.set_status(f"✗ code {rc}", C_ERROR)
            except Exception as exc:
                _append_line(f"\n✗  {exc}", C_ERROR)
                self.set_status("✗ error", C_ERROR)
            finally:
                state.running          = False
                state.proc             = None
                self._stop_btn.visible = False
                self.page.update()
                # Only advance the pipeline chain when the step succeeded.
                # On failure rc != 0 — stop the chain so the user sees the error.
                if on_done and rc == 0:
                    on_done()
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

    secretly_sw   = ft.Switch(label="Slack mode for downloads",
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
    # Per-video notes are the default (one note per video, named after the
    # video). Toggle off to merge everything into one course-wide file.
    per_video_sw = ft.Switch(label="Per-video notes", value=True, active_color=C_PRIMARY)
    # Image source: video screenshots (frames) by default, or PDF slide renders.
    # Falls back to the other source if the chosen one isn't available.
    image_src_val = {"v": "frames"}
    image_src_dd = ft.Dropdown(
        label="Image source",
        value="frames",
        options=[
            ft.dropdown.Option("frames", "Video screenshots (default)"),
            ft.dropdown.Option("slides", "Slide PDFs"),
        ],
        on_change=lambda e: image_src_val.update({"v": e.data or e.control.value}),
        width=260,
    )

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
                 "--course", str(cid), "--download-material-all",
                 "--path", str(_get_output_dir())]
            if secretly_sw.value:
                c.append("--secretly")
            cmds.append(("Download materials", c))

        if "dl_video" in steps:
            c = [PYTHON, str(SCRIPTS["downloader"]),
                 "--course", str(cid), "--download-video-all",
                 "--path", str(_get_output_dir())]
            if secretly_sw.value:
                c.append("--secretly")
            cmds.append(("Download videos", c))

        if "transcribe" in steps:
            cmds.append(("Transcribe", [PYTHON, str(SCRIPTS["transcribe"])]))

        # Frame extraction for screen share videos (runs between transcribe and align)
        if "align" in steps:
            cmds.append(("Extract frames (screen share)",
                         [PYTHON, str(SCRIPTS["frame_extractor"]),
                          "--course", str(cid),
                          "--path", str(_get_output_dir())]))
            # Use saved mapping if it exists
            mapping_file = _get_output_dir() / str(cid) / "alignment" / "video_slide_mapping.json"
            align_cmd = [PYTHON, str(SCRIPTS["align"]), "--course", str(cid)]
            if mapping_file.exists():
                align_cmd += ["--mapping", str(mapping_file)]
            cmds.append(("Align", align_cmd))

        if "generate" in steps:
            c = [PYTHON, str(SCRIPTS["generate"]),
                 "--course",      str(cid),
                 "--course-name", name,
                 "--detail",      str(int(detail_slider.value))]
            if lec_filter_f.value.strip():
                c += ["--lectures", lec_filter_f.value.strip()]
            if force_sw.value:
                c.append("--force")
            if not per_video_sw.value:
                # Per-video is the default; opt in to the merged layout.
                c.append("--merged")
            if image_src_val["v"] != "frames":
                c += ["--image-source", image_src_val["v"]]
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
                image_src_dd,
                force_sw,
                per_video_sw,
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
    secretly_sw  = ft.Switch(label="Slack mode", value=False, active_color=C_PRIMARY)
    video_nums_f = _text_field("Video numbers", hint="e.g. 1 3 5", expand=2)
    mat_names_f  = _text_field("Filename(s)", hint="partial names, space-separated",
                                expand=True)

    def _course_args() -> list[str]:
        cid = course_val["v"]
        return ["--course", cid] if cid != "0" else []

    def _secretly_args() -> list[str]:
        return ["--secretly"] if secretly_sw.value else []

    def _go(extra: list[str]) -> None:
        console.run([PYTHON, str(SCRIPTS["downloader"]),
                     "--path", str(_get_output_dir())]
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
        on_select=lambda e: course_val.update({"v": e.data}),
    )

    # ── Video ↔ Slide Matching ───────────────────────────────────────────────
    match_rows_col = ft.Column(controls=[], spacing=0)
    match_status   = ft.Text("", size=11,
                             color=ft.Colors.with_opacity(0.6, ft.Colors.WHITE))

    # State:  rows = [ {stem, title, aligned, container, slide_dropdowns: [dd,...]} ]
    _match_state: dict = {"rows": [], "base": None, "slide_opts": []}

    def _auto_suggest(cap_stem: str, slides: list, base: Path,
                      cap_dir: Path | None = None) -> list[str]:
        """Find auto-matched slides for a caption using multi-strategy scoring.

        Strategies (scored and summed):
          1. Lecture number match (Week3 → L3)      — +100
          2. Date match (06/03/2026 → ann060326)    — +80
          3. Filename token overlap                  — +0..30
          4. Transcript keyword overlap with slide   — +10 per hit
          5. Preference: "with notes" > "review"     — +5/+3/+2
        """
        import re as _re
        cl = cap_stem.lower().replace("-", " ").replace("_", " ")

        # Extract number from caption
        num_m = _re.search(r"week\s*(\d+)|lec(?:ture)?\s*(\d+)|\bl(\d+)\b", cl)
        cap_num = (num_m.group(1) or num_m.group(2) or num_m.group(3)) if num_m else None

        # Extract date from caption (e.g. "06_03_2026")
        date_m = _re.search(r"(\d{2})[/_](\d{2})[/_](\d{4})", cap_stem)
        cap_date = (date_m.group(1) + date_m.group(2) + date_m.group(3)[2:]) if date_m else None

        # Read transcript keywords
        cap_keywords: set[str] = set()
        if cap_dir:
            cap_file = cap_dir / f"{cap_stem}.json"
            if cap_file.exists():
                try:
                    import json as _json
                    segs = _json.loads(cap_file.read_text()).get("segments", [])
                    step = max(1, len(segs) // 20)
                    words = " ".join(s.get("text", "") for s in segs[::step][:20]).lower()
                    cap_keywords = {w for w in _re.findall(r"[a-z]{5,}", words)}
                except Exception:
                    pass

        best_score, best_rel = 0.0, ""
        for sp in slides:
            sl = sp.stem.lower().replace("-", " ").replace("_", " ")
            score = 0.0

            # Strategy 1: lecture number
            sl_num_m = _re.search(r"l(?:ecture)?\s*(\d+)", sl)
            if cap_num and sl_num_m and cap_num == sl_num_m.group(1):
                score += 100

            # Strategy 2: date match
            if cap_date and cap_date in sp.name:
                score += 80

            # Strategy 3: token overlap
            ct = set(cl.split())
            st = set(sl.split())
            if ct and st:
                score += (len(ct & st) / max(len(ct), len(st))) * 30

            # Strategy 4: transcript keyword overlap
            if cap_keywords:
                sl_words = set(_re.findall(r"[a-z]{5,}", sl))
                score += len(cap_keywords & sl_words) * 10

            # Strategy 5: prefer annotated versions
            if _re.search(r"with\s*notes", sp.name, _re.IGNORECASE):
                score += 5
            elif _re.search(r"review", sp.name, _re.IGNORECASE):
                score += 3
            elif _re.search(r"ann\d", sp.name, _re.IGNORECASE):
                score += 2

            if score > best_score:
                best_score = score
                best_rel = str(sp.relative_to(base))

        return [best_rel] if best_score > 5 and best_rel else []

    def _make_slide_dropdown(initial: str = "(none)") -> ft.Dropdown:
        """Create a single slide-file dropdown."""
        return ft.Dropdown(
            options=list(_match_state["slide_opts"]),
            value=initial,
            dense=True,
            bgcolor=C_OUTPUT_BG,
            border_color=(ft.Colors.GREEN_400 if initial != "(none)"
                          else ft.Colors.with_opacity(0.3, ft.Colors.WHITE)),
            text_size=11,
            expand=True,
            content_padding=ft.Padding.symmetric(horizontal=8, vertical=2),
        )

    def _build_video_row(row_data: dict) -> ft.Container:
        """Build the UI container for one video row."""
        stem     = row_data["stem"]
        title    = row_data["title"]
        aligned  = row_data["aligned"]
        dds      = row_data["slide_dropdowns"]

        # Status chip
        if aligned:
            status = ft.Tooltip(
                message="Already aligned",
                content=ft.Icon(ft.Icons.CHECK_CIRCLE, size=14,
                                color=ft.Colors.GREEN_400))
        else:
            status = ft.Tooltip(
                message="Not yet aligned",
                content=ft.Icon(ft.Icons.CIRCLE_OUTLINED, size=14,
                                color=ft.Colors.with_opacity(0.3, ft.Colors.WHITE)))

        # Delete (×) button — removes this video from the list
        def _remove(_evt, _stem=stem):
            _match_state["rows"] = [r for r in _match_state["rows"]
                                    if r["stem"] != _stem]
            _rebuild_rows()

        remove_btn = ft.IconButton(
            icon=ft.Icons.CLOSE, icon_size=14,
            icon_color=ft.Colors.RED_300,
            tooltip="Remove this video",
            on_click=_remove,
            style=ft.ButtonStyle(padding=0),
        )

        # Add (+) button — append another slide dropdown for multi-part lectures
        def _add_slide(_evt, _row=row_data):
            new_dd = _make_slide_dropdown()
            _row["slide_dropdowns"].append(new_dd)
            _rebuild_rows()

        add_btn = ft.IconButton(
            icon=ft.Icons.ADD, icon_size=14,
            icon_color=C_PRIMARY,
            tooltip="Add another slide file (multi-part lecture)",
            on_click=_add_slide,
            style=ft.ButtonStyle(padding=0),
        )

        # Remove (−) button for extra dropdowns (index > 0)
        def _make_remove_dd(_row, _idx):
            def _handler(_evt):
                if len(_row["slide_dropdowns"]) > 1:
                    _row["slide_dropdowns"].pop(_idx)
                    _rebuild_rows()
            return _handler

        # Build slide selector rows
        slide_controls: list[ft.Control] = []
        for i, dd in enumerate(dds):
            row_ctrls = [dd]
            if i == 0:
                row_ctrls.append(add_btn)
            else:
                rm = ft.IconButton(
                    icon=ft.Icons.REMOVE, icon_size=14,
                    icon_color=ft.Colors.RED_300,
                    tooltip="Remove this slide file",
                    on_click=_make_remove_dd(row_data, i),
                    style=ft.ButtonStyle(padding=0),
                )
                row_ctrls.append(rm)
            slide_controls.append(
                ft.Row(controls=row_ctrls, spacing=2,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER))

        # Assemble: [×] [status] [title]  →  [slide dropdown(s)] [+]
        return ft.Container(
            content=ft.Row(controls=[
                remove_btn,
                status,
                ft.Text(title, size=11, width=220,
                        color=ft.Colors.WHITE,
                        overflow=ft.TextOverflow.ELLIPSIS),
                ft.Icon(ft.Icons.ARROW_FORWARD, size=12,
                        color=ft.Colors.with_opacity(0.3, ft.Colors.WHITE)),
                ft.Column(controls=slide_controls, spacing=2, expand=True),
            ], spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            padding=ft.Padding(left=0, top=4, right=0, bottom=4),
            border=ft.Border(bottom=ft.BorderSide(
                1, ft.Colors.with_opacity(0.06, ft.Colors.WHITE))),
        )

    def _rebuild_rows() -> None:
        """Rebuild the visual row list from _match_state["rows"]."""
        match_rows_col.controls.clear()
        if not _match_state["rows"]:
            return
        # Header
        match_rows_col.controls.append(ft.Container(
            content=ft.Row(controls=[
                ft.Container(width=28),  # space for × button
                ft.Container(width=14),  # space for status icon
                ft.Text("Video", size=10, width=220,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.with_opacity(0.4, ft.Colors.WHITE)),
                ft.Container(width=12),
                ft.Text("Lecture slides", size=10,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.with_opacity(0.4, ft.Colors.WHITE),
                        expand=True),
            ], spacing=4),
            padding=ft.Padding(left=0, top=0, right=0, bottom=2),
        ))
        for row_data in _match_state["rows"]:
            row_data["container"] = _build_video_row(row_data)
            match_rows_col.controls.append(row_data["container"])

        n_matched = sum(1 for r in _match_state["rows"]
                        if any(d.value and d.value != "(none)"
                               for d in r["slide_dropdowns"]))
        n_total   = len(_match_state["rows"])
        match_status.value = (
            f"{n_total} video(s) shown, "
            f"{n_matched} matched to slides."
        )
        page.update()

    def _scan_matching(_) -> None:
        """Scan course folder for captions and slide files, build matching UI."""
        cid = course_val["v"]
        base = _get_output_dir() / cid
        cap_dir   = base / "captions"
        mat_dir   = base / "materials"
        align_dir = base / "alignment"

        captions = sorted(cap_dir.glob("*.json")) if cap_dir.exists() else []
        exts = {".pdf", ".pptx", ".ppt", ".docx", ".doc"}
        slides = sorted([
            p for p in mat_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
            and "image_cache" not in p.name
        ]) if mat_dir.exists() else []

        if not captions:
            console.write("No captions found. Transcribe videos first.", color=C_WARN)
            return
        if not slides:
            console.write("No slide files found. Download materials first.", color=C_WARN)
            return

        # Load manifest for video titles
        manifest: dict = {}
        mf = DATA_DIR / "manifest.json"
        if mf.exists():
            try:
                manifest = json.load(open(mf))
            except Exception:
                pass

        # Load existing mapping
        existing_mapping: dict = {}
        mapping_file = align_dir / "video_slide_mapping.json"
        if mapping_file.exists():
            try:
                existing_mapping = json.load(open(mapping_file))
            except Exception:
                pass

        # Build dropdown option list (shared across all dropdowns)
        slide_opts = [ft.dropdown.Option("(none)", "(none — auto-detect)")]
        for sp in slides:
            rel = str(sp.relative_to(base))
            slide_opts.append(ft.dropdown.Option(rel, rel))
        _match_state["slide_opts"] = slide_opts
        _match_state["base"] = base
        _match_state["rows"] = []

        valid_keys = {o.key for o in slide_opts}

        for cap in captions:
            # Video title from manifest
            title = cap.stem
            for entry in manifest.values():
                if (entry.get("status") == "done" and
                    entry.get("title", "").replace("/", "_").replace(":", "_") == cap.stem):
                    title = entry["title"]
                    break

            aligned = (align_dir / f"{cap.stem}.json").exists() if align_dir.exists() else False

            # Initial values: existing mapping > auto-suggest > (none)
            if cap.stem in existing_mapping and existing_mapping[cap.stem]:
                initials = [v for v in existing_mapping[cap.stem] if v in valid_keys]
                if not initials:
                    initials = ["(none)"]
            else:
                initials = _auto_suggest(cap.stem, slides, base, cap_dir)
                if not initials:
                    initials = ["(none)"]

            dds = [_make_slide_dropdown(v) for v in initials]

            _match_state["rows"].append({
                "stem": cap.stem,
                "title": title,
                "aligned": aligned,
                "slide_dropdowns": dds,
                "container": None,
            })

        _rebuild_rows()
        console.write(
            f"Found {len(captions)} video(s), {len(slides)} slide file(s). "
            "Adjust matches as needed, then click 'Align'.",
            color=C_SUCCESS)

    def _run_with_mapping(_) -> None:
        """Save the mapping and run alignment."""
        cid = course_val["v"]
        base = _match_state.get("base") or _get_output_dir() / cid

        if not _match_state["rows"]:
            console.write("Click 'Scan' first to load videos and slides.",
                          color=C_WARN)
            return

        mapping: dict[str, list[str]] = {}
        for row in _match_state["rows"]:
            vals = [d.value for d in row["slide_dropdowns"]
                    if d.value and d.value != "(none)"]
            if vals:
                mapping[row["stem"]] = vals

        # Save mapping
        mapping_file = base / "alignment" / "video_slide_mapping.json"
        mapping_file.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_file, "w") as f:
            json.dump(mapping, f, indent=2)

        cmd = [PYTHON, str(SCRIPTS["align"]),
               "--course", cid,
               "--mapping", str(mapping_file)]
        console.run(cmd)

    def _run_auto(_) -> None:
        """Run fully automatic alignment (no user mapping)."""
        cmd = [PYTHON, str(SCRIPTS["align"]), "--course", course_val["v"]]
        console.run(cmd)

    embed_model = _read_constant("align", "EMBED_MODEL")
    ctx_sec     = _read_constant("align", "CONTEXT_SEC")

    scroll_content = [
        _section_title("Align Transcripts to Slides", ft.Icons.LINK_OUTLINED),
        _card(ft.Row(controls=[
            ft.Icon(ft.Icons.INFO_OUTLINE, color=C_PRIMARY, size=15),
            ft.Text(
                f"Embed: {embed_model}   Context: ±{ctx_sec}s",
                size=12,
                color=ft.Colors.with_opacity(0.7, ft.Colors.WHITE),
            ),
        ], spacing=8)),

        _card(ft.Column(controls=[
            ft.Row(controls=[
                ft.Icon(ft.Icons.COMPARE_ARROWS, color=C_PRIMARY, size=18),
                ft.Text("Video ↔ Slide Matching", size=14,
                        weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
            ], spacing=8),
            ft.Text("Match each video to its lecture slides. "
                    "Auto-suggested matches are pre-filled. "
                    "Click + to add multiple slide files for one video, "
                    "or × to exclude a video.",
                    size=12, color=ft.Colors.with_opacity(0.6, ft.Colors.WHITE)),
            ft.Container(height=6),

            # Course selector + Scan
            ft.Row(controls=[
                ft.Column(controls=[_label("Course"), course_dd],
                          spacing=6, expand=True),
                _run_btn("Scan videos & slides", ft.Icons.SEARCH, _scan_matching),
            ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.END),
            ft.Container(height=2),
            match_status,
            ft.Container(height=4),

            # Video rows
            match_rows_col,

            # Action buttons
            ft.Container(height=8),
            ft.Row(controls=[
                ft.FilledButton(
                    "Align with mapping",
                    icon=ft.Icons.LINK,
                    on_click=_run_with_mapping,
                    style=ft.ButtonStyle(bgcolor=C_PRIMARY, color=ft.Colors.BLACK),
                ),
                ft.OutlinedButton(
                    "Auto-align (skip matching)",
                    icon=ft.Icons.AUTO_FIX_HIGH,
                    on_click=_run_auto,
                    style=ft.ButtonStyle(
                        side=ft.BorderSide(1, ft.Colors.with_opacity(0.3, ft.Colors.WHITE)),
                        color=ft.Colors.with_opacity(0.7, ft.Colors.WHITE),
                    ),
                ),
            ], spacing=12),
        ], spacing=6)),
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
    # Per-video output is the default; toggle off to get a single combined file.
    per_video_sw = ft.Switch(label="Per-video notes (one file per video)",
                              value=True, active_color=C_PRIMARY)
    # Image source: video screenshots (frames) vs PDF slide renders.
    image_src_val = {"v": "frames"}
    image_src_dd = ft.Dropdown(
        label="Image source",
        value="frames",
        options=[
            ft.dropdown.Option("frames", "Video screenshots (default)"),
            ft.dropdown.Option("slides", "Slide PDFs"),
        ],
        on_change=lambda e: image_src_val.update({"v": e.data or e.control.value}),
        width=260,
    )

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
        if not per_video_sw.value:
            cmd.append("--merged")
        if image_src_val["v"] != "frames":
            cmd += ["--image-source", image_src_val["v"]]
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
                per_video_sw,
                image_src_dd,
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
    jina_file      = DATA_DIR / "jina_api.txt"

    _v = {
        "canvas_url":  _cfg.get("CANVAS_URL", ""),
        "panopto":     _cfg.get("PANOPTO_HOST", ""),
        "python_path": _cfg.get("PYTHON_PATH", ""),
        "output_dir":  _cfg.get("OUTPUT_DIR",  str(Path.home() / "AutoNote")),
        "canvas":      canvas_file.read_text().strip() if canvas_file.exists() else "",
        "openai":      openai_file.read_text().strip() if openai_file.exists() else "",
        "anthropic":   anthropic_file.read_text().strip() if anthropic_file.exists() else "",
        "gemini":      gemini_file.read_text().strip() if gemini_file.exists() else "",
        "jina":        jina_file.read_text().strip() if jina_file.exists() else "",
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
    tf_output_dir = _mk_tf("output_dir",
        hint_text="Output directory for all pipeline files (default: ~/AutoNote)",
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
    tf_jina = _mk_tf("jina",
        password=True, can_reveal_password=True,
        hint_text="jina_…  (for multimodal alignment, optional)",
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
        _field_row("Output Dir",   tf_output_dir),
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
                "OUTPUT_DIR":   _v["output_dir"].strip(),
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
        _field_row("Jina API Key",      tf_jina),
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
            # 1) Manual override from the text field
            base_py = _base_py_v["v"].strip()

            # 2) Auto-detect via login shell + common install locations
            if not base_py:
                base_py = _find_base_python(_append_log)

            if not base_py:
                _append_log("ERROR: Could not find a Python 3 with SSL support.")
                _append_log("")
                if sys.platform == "win32":
                    _append_log("  Paste the path to your Python below")
                    _append_log(r"  (e.g. C:\Users\you\miniconda3\python.exe)")
                else:
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
            ("gpt-5.4",              "gpt-5.4"),
            ("gpt-5.3",              "gpt-5.3"),
            ("gpt-5.2",              "gpt-5.2"),
            ("gpt-5.1",              "gpt-5.1"),
            ("gpt-4.1",              "gpt-4.1"),
            ("gpt-4.1-mini",         "gpt-4.1-mini"),
            ("gpt-4.1-nano",         "gpt-4.1-nano"),
            ("o3",                   "o3"),
            ("o3-pro",               "o3-pro"),
            ("o1",                   "o1"),
            # Gemini
            ("Gemini 3.5 Pro",       "gemini-3.5-pro"),
            ("Gemini 3.5 Flash",     "gemini-3.5-flash"),
            ("Gemini 3.0 Pro",       "gemini-3.0-pro"),
            ("Gemini 3.0 Flash",     "gemini-3.0-flash"),
            ("Gemini 2.5 Pro",       "gemini-2.5-pro"),
            ("Gemini 2.5 Flash",     "gemini-2.5-flash"),
            ("Gemini 2.0 Flash",     "gemini-2.0-flash"),
            # Anthropic
            ("Claude Opus 4.6",      "claude-opus-4-6"),
            ("Claude Sonnet 4.6",    "claude-sonnet-4-6"),
            ("Claude Sonnet 4.5",    "claude-sonnet-4-5"),
            ("Claude Haiku 4.5",     "claude-haiku-4-5-20251001"),
        ]),
        ("generate", "VERIFY_MODEL", "Verification LLM", "gpt-4.1-mini", [
            # OpenAI
            ("gpt-4.1-mini",         "gpt-4.1-mini"),
            ("gpt-4.1-nano",         "gpt-4.1-nano"),
            ("gpt-4.1",              "gpt-4.1"),
            ("gpt-5.1",              "gpt-5.1"),
            # Gemini
            ("Gemini 3.5 Flash",     "gemini-3.5-flash"),
            ("Gemini 3.0 Flash",     "gemini-3.0-flash"),
            ("Gemini 2.5 Flash",     "gemini-2.5-flash"),
            ("Gemini 2.0 Flash",     "gemini-2.0-flash"),
            # Anthropic
            ("Claude Haiku 4.5",     "claude-haiku-4-5-20251001"),
            ("Claude Sonnet 4.5",    "claude-sonnet-4-5"),
            ("Claude Sonnet 4.6",    "claude-sonnet-4-6"),
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
                "OUTPUT_DIR":   _v["output_dir"].strip(),
            })
            p = _v["python_path"].strip()
            PYTHON = p if p else _DEFAULT_PYTHON
            global OUTPUT_DIR
            od = _v["output_dir"].strip()
            if od:
                OUTPUT_DIR = Path(od)
        except Exception as e:
            errors.append(f"Connection: {e}")
        for path, key in [
            (canvas_file,    "canvas"),
            (openai_file,    "openai"),
            (anthropic_file, "anthropic"),
            (gemini_file,    "gemini"),
            (jina_file,      "jina"),
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
            # Auto-refresh courses after saving so the user sees results immediately
            threading.Thread(target=_do_refresh, daemon=True).start()

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

def _show_installer(page: ft.Page) -> None:
    """Full-screen first-run setup wizard shown when venv is not yet installed."""

    def _on_install_complete() -> None:
        """Called by the installer after the venv is ready; launches main UI."""
        page.controls.clear()
        _install_scripts()
        _load_python_from_config()
        _load_output_dir_from_config()
        _load_courses_from_canvas()
        _show_main_app(page)

    # ── Log area ──────────────────────────────────────────────────────────
    inst_log = ft.TextField(
        value="", multiline=True, read_only=True, min_lines=4, max_lines=18,
        text_size=11, bgcolor=C_OUTPUT_BG,
        border_color=ft.Colors.TRANSPARENT,
        color=ft.Colors.with_opacity(0.85, ft.Colors.WHITE),
        expand=True,
    )
    inst_status = ft.Text("Ready to install.", size=12, color=C_PRIMARY)
    inst_btn = ft.FilledButton(
        "Install ML Environment",
        icon=ft.Icons.DOWNLOAD_OUTLINED,
        style=ft.ButtonStyle(bgcolor=C_SECONDARY, color=ft.Colors.BLACK),
    )

    _base_py_v2 = {"v": ""}
    tf_base_py2 = ft.TextField(
        hint_text="Python path (auto-detected — paste here only if auto-detect fails)",
        expand=True, dense=True, bgcolor=C_OUTPUT_BG, border_color=C_PRIMARY,
        text_size=11, visible=False,
        on_change=lambda e: _base_py_v2.update({"v": e.control.value}),
    )

    def _append(line: str) -> None:
        inst_log.value = (inst_log.value or "") + line + "\n"
        page.update()

    def _run_install(_=None) -> None:
        inst_btn.disabled = True
        inst_status.value = "Installing…"
        inst_status.color = C_WARN
        page.update()

        def _worker():
            base_py = _base_py_v2["v"].strip()

            if not base_py:
                base_py = _find_base_python(_append)

            if not base_py:
                _append("ERROR: No Python 3 with SSL support found.")
                if sys.platform == "win32":
                    _append(r"  Paste your Python path below (e.g. C:\Users\you\miniconda3\python.exe)")
                else:
                    _append("  Paste your Python path below (e.g. /home/user/miniconda3/bin/python3)")
                _append("  then click Install again.")
                tf_base_py2.visible = True
                inst_status.value = "✗ Python not found — enter path below"
                inst_status.color = C_ERROR
                inst_btn.disabled = False
                page.update()
                return

            _append(f"► Using Python: {base_py}")

            import shutil as _sh2
            if ML_VENV_DIR.exists():
                _append("► Removing old venv …")
                _sh2.rmtree(str(ML_VENV_DIR))
            _append(f"► Creating venv at {ML_VENV_DIR} …")
            r = subprocess.run([base_py, "-m", "venv", str(ML_VENV_DIR)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                _append("STDERR: " + r.stderr.strip())
                _append("ERROR: venv creation failed.")
                inst_status.value = "✗ Setup failed"
                inst_status.color = C_ERROR
                inst_btn.disabled = False
                page.update()
                return
            _append("  venv created.")

            pip = str(ML_VENV_DIR / (
                "Scripts/pip.exe" if sys.platform == "win32" else "bin/pip"
            ))
            _append("► Upgrading pip …")
            subprocess.run([pip, "install", "--upgrade", "pip"],
                           capture_output=True, text=True)

            cuda = _detect_cuda()
            idx = _torch_index_url(cuda)
            if cuda:
                _append(f"► CUDA {cuda[0]}.{cuda[1]} detected — installing torch (GPU) …")
            else:
                _append("► No GPU detected — installing torch (CPU) …")
            torch_cmd = [pip, "install", "torch"]
            if idx:
                torch_cmd += ["--index-url", idx]
            proc = subprocess.Popen(torch_cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                _append(line.rstrip())
            proc.wait()

            _append("► Installing ML packages …")
            proc = subprocess.Popen([pip, "install"] + _ML_PACKAGES,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                _append(line.rstrip())
            proc.wait()

            _append("► Installing Playwright browsers …")
            proc = subprocess.Popen(
                [ML_VENV_PYTHON, "-m", "playwright", "install", "chromium"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                _append(line.rstrip())
            proc.wait()

            global PYTHON
            PYTHON = ML_VENV_PYTHON
            _append("\n✓ Installation complete!")
            inst_status.value = "✓ Done — launching app…"
            inst_status.color = C_SUCCESS
            inst_btn.disabled = False
            page.update()
            import time as _t; _t.sleep(1)
            _on_install_complete()

        threading.Thread(target=_worker, daemon=True).start()

    inst_btn.on_click = _run_install

    page.add(
        ft.Container(
            content=ft.Column(
                controls=[
                    ft.Container(height=40),
                    ft.Row(
                        controls=[ft.Icon(ft.Icons.AUTO_AWESOME, color=C_PRIMARY, size=32),
                                  ft.Text("AUTO NOTE", size=28,
                                          weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE)],
                        alignment=ft.MainAxisAlignment.CENTER, spacing=12,
                    ),
                    ft.Container(height=8),
                    ft.Text(
                        "First-time setup: install the ML environment to get started.",
                        size=13, color=ft.Colors.with_opacity(0.6, ft.Colors.WHITE),
                        text_align=ft.TextAlign.CENTER,
                    ),
                    ft.Container(height=24),
                    _card(ft.Column(controls=[
                        ft.Text("ML Environment Setup", size=14,
                                weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                        ft.Text(
                            "This will create ~/.auto_note/venv/ and install torch, "
                            "faster-whisper, sentence-transformers and all pipeline "
                            "dependencies. Internet connection required. Takes ~5 min.",
                            size=11,
                            color=ft.Colors.with_opacity(0.55, ft.Colors.WHITE),
                        ),
                        ft.Container(height=6),
                        ft.Row(controls=[inst_status], spacing=8),
                        inst_btn,
                        tf_base_py2,
                        ft.Container(content=inst_log, expand=True),
                    ], spacing=10)),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                scroll=ft.ScrollMode.AUTO,
                expand=True,
            ),
            expand=True,
            padding=ft.Padding.symmetric(horizontal=80, vertical=0),
        )
    )
    page.update()


def _show_main_app(page: ft.Page) -> None:
    """Build and display the full navigation shell (called after install or on normal launch)."""

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
    page.update()


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

    # Install pipeline scripts to ~/.auto_note/scripts/ (no-op in dev mode)
    _install_scripts()

    # Load user-configured settings from config
    _load_python_from_config()
    _load_output_dir_from_config()

    # ── First-run installer ────────────────────────────────────────────────
    if not Path(ML_VENV_PYTHON).exists():
        _show_installer(page)
        return

    # Try to populate courses immediately if credentials are already on disk
    _load_courses_from_canvas()
    _show_main_app(page)


if __name__ == "__main__":
    ft.run(main)
