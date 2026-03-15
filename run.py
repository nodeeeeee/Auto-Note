#!/usr/bin/env python3
"""
run.py — Material-style workflow UI for the auto_note pipeline.

Presents a full-screen interactive menu that covers every option
exposed by downloader.py, extract_caption.py, semantic_alignment.py,
and note_generation.py.  Pipeline steps are executed via subprocess so
heavy GPU imports are never loaded into this process.

Usage:
    python run.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    from rich import box
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text
    from rich.theme import Theme
except ImportError:
    print("rich is required:  pip install rich")
    sys.exit(1)


# ── Project layout ────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).parent
PYTHON      = sys.executable

SCRIPTS = {
    "downloader":  PROJECT_DIR / "downloader.py",
    "transcribe":  PROJECT_DIR / "extract_caption.py",
    "align":       PROJECT_DIR / "semantic_alignment.py",
    "generate":    PROJECT_DIR / "note_generation.py",
}

# Known academic courses (kept in sync with downloader.py)
COURSES: dict[int, str] = {
    85367: "CS2101 Effective Communication",
    85377: "CS2103/T Software Engineering",
    85397: "CS2105 Computer Networks",
    85427: "CS3210 Parallel Computing",
}


# ── Theme & console ───────────────────────────────────────────────────────────

THEME = Theme({
    "primary":   "bold cyan",
    "secondary": "bold yellow",
    "success":   "bold green",
    "error":     "bold red",
    "warn":      "dark_orange",
    "muted":     "bright_black",
    "menu.key":  "bold cyan",
    "menu.text": "white",
    "chip.done":    "bold green",
    "chip.partial": "dark_orange",
    "chip.none":    "bright_black",
})

console = Console(theme=THEME)


# ── Pipeline-state inspection ─────────────────────────────────────────────────

def _manifest() -> dict:
    p = PROJECT_DIR / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _video_status(course_id: int) -> tuple[int, int]:
    """(done, total) videos for a course, from manifest."""
    m = _manifest()
    items = [v for v in m.values()
             if str(course_id) in v.get("path", "")]
    done = sum(1 for v in items if v.get("status") == "done")
    return done, len(items)


def _caption_count(course_id: int) -> int:
    d = PROJECT_DIR / str(course_id) / "captions"
    return len(list(d.glob("*.json"))) if d.exists() else 0


def _alignment_count(course_id: int) -> int:
    d = PROJECT_DIR / str(course_id) / "alignment"
    return len([f for f in d.glob("*.json")
                if "compact" not in f.name]) if d.exists() else 0


def _notes_path(course_id: int) -> Path | None:
    d = PROJECT_DIR / str(course_id) / "notes"
    if d.exists():
        mds = list(d.glob("*.md"))
        return mds[0] if mds else None
    return None


def _chip(done: int, total: int) -> Text:
    if total == 0:
        return Text("  ○ none  ", style="chip.none")
    if done >= total:
        return Text(f"  ✓ {done}/{total}  ", style="chip.done")
    return Text(f"  ◐ {done}/{total}  ", style="chip.partial")


def _bool_chip(ok: bool) -> Text:
    return Text("  ✓  ", style="chip.done") if ok else Text("  ○  ", style="chip.none")


# ── UI building blocks ────────────────────────────────────────────────────────

def _header() -> None:
    console.print()
    console.print(Panel(
        "[primary]◉  AUTO NOTE[/primary]\n"
        "[muted]NUS Canvas Lecture Note Pipeline[/muted]",
        border_style="cyan",
        padding=(0, 2),
        expand=False,
    ))


def _section(title: str) -> None:
    console.print()
    console.rule(f"[primary]{title}[/primary]", style="cyan")


def _ok(msg: str) -> None:
    console.print(f"[success]  ✓[/success]  {msg}")


def _err(msg: str) -> None:
    console.print(f"[error]  ✗[/error]  {msg}")


def _info(msg: str) -> None:
    console.print(f"[primary]  ·[/primary]  {msg}")


def _pause() -> None:
    console.print()
    Prompt.ask("[muted]Press Enter to return to the menu[/muted]",
               default="", show_default=False)


def _pick_course(allow_all: bool = True) -> int | None:
    """
    Prompt the user to choose one course.
    Returns course ID, or 0 meaning 'all courses', or None to cancel.
    """
    _section("Select Course")
    rows = list(COURSES.items())
    for i, (cid, name) in enumerate(rows, 1):
        console.print(f"  [menu.key]{i}[/menu.key]  {name}  [muted]({cid})[/muted]")
    if allow_all:
        console.print(f"  [menu.key]A[/menu.key]  All courses")
    console.print(f"  [menu.key]0[/menu.key]  Cancel")
    console.print()
    raw = Prompt.ask("  Choice", default="0")
    if raw.strip().upper() == "A" and allow_all:
        return 0
    try:
        idx = int(raw.strip())
    except ValueError:
        return None
    if idx == 0:
        return None
    if 1 <= idx <= len(rows):
        return rows[idx - 1][0]
    return None


def _course_name(course_id: int) -> str:
    base = COURSES.get(course_id, f"Course {course_id}")
    # Check if there's already a notes file to read the actual course name from
    d = PROJECT_DIR / str(course_id) / "notes"
    if d.exists():
        for md in d.glob("*.md"):
            stem = md.stem.replace("_notes", "").replace("_", " ")
            if stem:
                return stem
    return base


# ── Status dashboard ──────────────────────────────────────────────────────────

def show_status() -> None:
    console.clear()
    _header()
    _section("Course Pipeline Status")

    tbl = Table(box=box.ROUNDED, border_style="cyan", show_header=True,
                header_style="primary", expand=False)
    tbl.add_column("ID",      style="muted",    justify="right",  no_wrap=True)
    tbl.add_column("Course",  style="white",     min_width=28)
    tbl.add_column("Videos",  justify="center",  no_wrap=True)
    tbl.add_column("Captions",justify="center",  no_wrap=True)
    tbl.add_column("Aligned", justify="center",  no_wrap=True)
    tbl.add_column("Notes",   justify="center",  no_wrap=True)

    for cid, name in COURSES.items():
        vdone, vtotal = _video_status(cid)
        caps   = _caption_count(cid)
        aligns = _alignment_count(cid)
        notes  = _notes_path(cid) is not None

        tbl.add_row(
            str(cid),
            name,
            _chip(vdone, vtotal),
            _chip(caps, max(caps, vdone) or 0),
            _chip(aligns, max(aligns, caps) or 0),
            _bool_chip(notes),
        )

    console.print(tbl)

    # Extra detail for courses that have local data
    for cid, name in COURSES.items():
        d = PROJECT_DIR / str(cid)
        if not d.exists():
            continue
        notes_p = _notes_path(cid)
        if notes_p:
            size_kb = notes_p.stat().st_size // 1024
            console.print(f"  [muted]{name}[/muted] → [success]{notes_p.name}[/success] "
                          f"[muted]({size_kb} KB)[/muted]")


# ── 1. Download ───────────────────────────────────────────────────────────────

def menu_download() -> None:
    console.clear()
    _header()
    _section("Download")

    console.print(
        "  [menu.key]1[/menu.key]  List courses\n"
        "  [menu.key]2[/menu.key]  List videos\n"
        "  [menu.key]3[/menu.key]  Download specific video(s)\n"
        "  [menu.key]4[/menu.key]  Download all pending videos\n"
        "  [menu.key]5[/menu.key]  List materials\n"
        "  [menu.key]6[/menu.key]  Download specific material(s)\n"
        "  [menu.key]7[/menu.key]  Download all pending materials\n"
        "  [menu.key]0[/menu.key]  Back"
    )
    console.print()
    choice = Prompt.ask("  Choice", default="0")

    if choice == "0":
        return

    cmd = [PYTHON, str(SCRIPTS["downloader"])]

    # -- Course filter (shared by most actions) --------------------------------
    course_id: int | None = None
    if choice in ("2", "3", "4", "5", "6", "7"):
        raw = Prompt.ask(
            "  Course ID (leave blank for all courses)",
            default="",
        )
        if raw.strip():
            try:
                course_id = int(raw.strip())
                cmd += ["--course", str(course_id)]
            except ValueError:
                _err("Invalid course ID — using all courses.")

    # -- Stealth mode ----------------------------------------------------------
    secretly = False
    if choice in ("3", "4", "6", "7"):
        secretly = Confirm.ask(
            "  Enable stealth mode? (random delays between downloads)",
            default=False,
        )
        if secretly:
            cmd.append("--secretly")

    # -- Custom path -----------------------------------------------------------
    if choice in ("4", "7"):
        raw = Prompt.ask(
            "  Override download path (leave blank for default)",
            default="",
        )
        if raw.strip():
            cmd += ["--path", raw.strip()]

    # -- Build the specific sub-command ----------------------------------------
    if choice == "1":
        cmd.append("--course-list")

    elif choice == "2":
        cmd.append("--video-list")

    elif choice == "3":
        cmd.append("--video-list")
        console.print()
        _run(cmd)   # show list first

        cmd2 = [PYTHON, str(SCRIPTS["downloader"])]
        if course_id:
            cmd2 += ["--course", str(course_id)]
        if secretly:
            cmd2.append("--secretly")
        nums = Prompt.ask(
            "  Video numbers to download (space-separated, e.g. 1 3 5)",
        )
        cmd2 += ["--download-video"] + nums.strip().split()
        console.print()
        _run(cmd2)
        _pause()
        return

    elif choice == "4":
        cmd.append("--download-video-all")

    elif choice == "5":
        cmd.append("--material-list")

    elif choice == "6":
        cmd.append("--material-list")
        console.print()
        _run(cmd)   # show list first

        cmd2 = [PYTHON, str(SCRIPTS["downloader"])]
        if course_id:
            cmd2 += ["--course", str(course_id)]
        if secretly:
            cmd2.append("--secretly")
        names = Prompt.ask(
            "  Filename(s) to download (space-separated, partial names ok)",
        )
        cmd2 += ["--download-material"] + names.strip().split()
        console.print()
        _run(cmd2)
        _pause()
        return

    elif choice == "7":
        cmd.append("--download-material-all")

    console.print()
    _run(cmd)
    _pause()


# ── 2. Transcribe ─────────────────────────────────────────────────────────────

def menu_transcribe() -> None:
    console.clear()
    _header()
    _section("Transcribe Videos")

    console.print(
        "  Runs [primary]faster-whisper large-v3[/primary] on GPU.\n"
        "  Processes all videos in [primary]manifest.json[/primary] that don't "
        "have a caption yet.\n"
    )
    console.print(
        "  [menu.key]1[/menu.key]  Process all pending videos\n"
        "  [menu.key]2[/menu.key]  Process a single video file\n"
        "  [menu.key]0[/menu.key]  Back"
    )
    console.print()
    choice = Prompt.ask("  Choice", default="0")

    if choice == "0":
        return

    cmd = [PYTHON, str(SCRIPTS["transcribe"])]

    if choice == "2":
        path = Prompt.ask("  Path to video file")
        cmd += ["--video", path.strip()]

    console.print()
    _run(cmd)
    _pause()


# ── 3. Align ──────────────────────────────────────────────────────────────────

def menu_align() -> None:
    console.clear()
    _header()
    _section("Align Transcripts to Slides")

    console.print(
        "  Maps each transcript segment to a slide page using\n"
        "  [primary]all-mpnet-base-v2[/primary] embeddings + Viterbi smoothing.\n"
        "  Unmatched captions fall back to [primary]content-based[/primary] pairing.\n"
    )
    console.print(
        "  [menu.key]1[/menu.key]  Auto-process all unaligned pairs in a course\n"
        "  [menu.key]2[/menu.key]  Align a specific caption + slide file(s)\n"
        "  [menu.key]0[/menu.key]  Back"
    )
    console.print()
    choice = Prompt.ask("  Choice", default="0")

    if choice == "0":
        return

    cmd = [PYTHON, str(SCRIPTS["align"])]

    if choice == "1":
        cid = _pick_course(allow_all=False)
        if cid is None:
            return
        cmd += ["--course", str(cid)]

        raw = Prompt.ask(
            "  Output directory (leave blank for default: [course]/alignment/)",
            default="",
        )
        if raw.strip():
            cmd += ["--out", raw.strip()]

    elif choice == "2":
        cap  = Prompt.ask("  Caption JSON path")
        slds = Prompt.ask("  Slide file path(s) (space-separated for multi-part)")
        out  = Prompt.ask(
            "  Output directory (leave blank for default)",
            default="",
        )
        cmd += ["--caption", cap.strip()]
        cmd += ["--slides"] + slds.strip().split()
        if out.strip():
            cmd += ["--out", out.strip()]

    console.print()
    _run(cmd)
    _pause()


# ── 4. Generate notes ─────────────────────────────────────────────────────────

def menu_generate() -> None:
    console.clear()
    _header()
    _section("Generate Study Notes")

    cid = _pick_course(allow_all=False)
    if cid is None:
        return

    name = _course_name(cid)
    console.print()
    console.print(f"  Course: [primary]{name}[/primary]  [muted]({cid})[/muted]")

    # ── Options ───────────────────────────────────────────────────────────────
    _section("Options")

    # Course name override
    name_input = Prompt.ask(
        "  Course name for notes header",
        default=name,
    )

    # Detail level
    console.print()
    console.print(
        "  Detail level  [muted]0-2[/muted] outline  "
        "[muted]3-5[/muted] hierarchical bullets  "
        "[muted]6-8[/muted] paragraphs  "
        "[muted]9-10[/muted] exhaustive"
    )
    detail = Prompt.ask("  Detail level", default="7")

    # Lecture filter
    lec_filter = Prompt.ask(
        "  Lectures to process (e.g. 1-3  or  1,4,5 — blank for all)",
        default="",
    )

    # Advanced flags
    console.print()
    force      = Confirm.ask("  Force regenerate existing sections?", default=False)
    merge_only = Confirm.ask("  Merge-only? (skip generation, re-merge + re-filter images)", default=False)
    iterate    = Confirm.ask("  Iterative mode? (keep raising detail until quality target)", default=False)

    # ── Build command ─────────────────────────────────────────────────────────
    cmd = [
        PYTHON, str(SCRIPTS["generate"]),
        "--course",       str(cid),
        "--course-name",  name_input.strip(),
        "--detail",       detail.strip(),
    ]
    if lec_filter.strip():
        cmd += ["--lectures", lec_filter.strip()]
    if force:
        cmd.append("--force")
    if merge_only:
        cmd.append("--merge-only")
    if iterate:
        cmd.append("--iterate")

    console.print()
    _run(cmd)
    _pause()


# ── 5. Full pipeline wizard ───────────────────────────────────────────────────

def menu_full_pipeline() -> None:
    console.clear()
    _header()
    _section("Full Pipeline Wizard")

    # ── Course selection ──────────────────────────────────────────────────────
    cid = _pick_course(allow_all=False)
    if cid is None:
        return

    name = COURSES.get(cid, f"Course {cid}")
    console.print(f"\n  Course: [primary]{name}[/primary]  [muted]({cid})[/muted]")

    # ── Which steps to run ────────────────────────────────────────────────────
    _section("Steps")
    vdone, vtotal = _video_status(cid)
    caps   = _caption_count(cid)
    aligns = _alignment_count(cid)
    notes  = _notes_path(cid)

    steps = {
        "dl_material": ("Download materials",  True),
        "dl_video":    ("Download videos",      True),
        "transcribe":  ("Transcribe videos",    caps == 0),
        "align":       ("Align transcripts",    aligns == 0),
        "generate":    ("Generate notes",       notes is None),
    }

    enabled: dict[str, bool] = {}
    for key, (label, default) in steps.items():
        enabled[key] = Confirm.ask(f"  {label}?", default=default)

    # ── Step-specific options ─────────────────────────────────────────────────
    _section("Options")

    secretly   = False
    course_name = _course_name(cid)
    detail      = "7"
    force       = False
    lec_filter  = ""

    if enabled["dl_material"] or enabled["dl_video"]:
        secretly = Confirm.ask(
            "  Stealth mode for downloads? (random delays)",
            default=False,
        )

    if enabled["generate"]:
        console.print()
        course_name = Prompt.ask("  Course name for notes", default=course_name)
        detail      = Prompt.ask("  Detail level (0–10)", default="7")
        lec_filter  = Prompt.ask(
            "  Lecture filter (e.g. 1-5 — blank for all)",
            default="",
        )
        force = Confirm.ask("  Force regenerate existing sections?", default=False)

    # ── Confirm and run ───────────────────────────────────────────────────────
    _section("Summary")
    for key, (label, _) in steps.items():
        icon = "[success]✓[/success]" if enabled[key] else "[muted]–[/muted]"
        console.print(f"  {icon}  {label}")
    if secretly:
        console.print("  [muted]  stealth mode on[/muted]")
    if enabled["generate"]:
        console.print(f"  [muted]  detail={detail}  course='{course_name}'[/muted]")

    console.print()
    if not Confirm.ask("  Run pipeline now?", default=True):
        return

    # ── Execute steps ─────────────────────────────────────────────────────────
    console.print()

    if enabled["dl_material"]:
        _banner("Step 1/5 — Download materials")
        cmd = [PYTHON, str(SCRIPTS["downloader"]),
               "--course", str(cid), "--download-material-all"]
        if secretly:
            cmd.append("--secretly")
        _run(cmd)

    if enabled["dl_video"]:
        _banner("Step 2/5 — Download videos")
        cmd = [PYTHON, str(SCRIPTS["downloader"]),
               "--course", str(cid), "--download-video-all"]
        if secretly:
            cmd.append("--secretly")
        _run(cmd)

    if enabled["transcribe"]:
        _banner("Step 3/5 — Transcribe videos")
        _run([PYTHON, str(SCRIPTS["transcribe"])])

    if enabled["align"]:
        _banner("Step 4/5 — Align transcripts")
        _run([PYTHON, str(SCRIPTS["align"]), "--course", str(cid)])

    if enabled["generate"]:
        _banner("Step 5/5 — Generate notes")
        cmd = [PYTHON, str(SCRIPTS["generate"]),
               "--course",      str(cid),
               "--course-name", course_name,
               "--detail",      detail]
        if lec_filter:
            cmd += ["--lectures", lec_filter]
        if force:
            cmd.append("--force")
        _run(cmd)

    console.print()
    _ok("Pipeline complete.")
    _pause()


# ── 6. Settings (read-only view) ──────────────────────────────────────────────

def menu_settings() -> None:
    console.clear()
    _header()
    _section("Configuration")

    # Read canvas token from downloader.py
    dl_src = SCRIPTS["downloader"].read_text(errors="ignore")
    token_m = re.search(r'CANVAS_TOKEN\s*=\s*"([^"]+)"', dl_src)
    token_preview = (token_m.group(1)[:8] + "…") if token_m else "not found"

    # Check OpenAI key
    openai_file = PROJECT_DIR / "openai_api.txt"
    openai_env  = os.environ.get("OPENAI_API_KEY", "")
    if openai_file.exists():
        openai_src = "openai_api.txt ✓"
    elif openai_env:
        openai_src = "OPENAI_API_KEY env ✓"
    else:
        openai_src = "[error]not found[/error]"

    cfg = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    cfg.add_column("Key",   style="primary",  no_wrap=True)
    cfg.add_column("Value", style="white")

    cfg.add_row("Canvas token",   token_preview)
    cfg.add_row("OpenAI key",     openai_src)
    cfg.add_row("Project dir",    str(PROJECT_DIR))

    console.print(cfg)

    # Per-script tunable constants
    _section("Tunable Constants")

    const_tbl = Table(box=box.SIMPLE, show_header=True, header_style="primary",
                      padding=(0, 2))
    const_tbl.add_column("Script",   style="muted",   no_wrap=True)
    const_tbl.add_column("Constant", style="primary",  no_wrap=True)
    const_tbl.add_column("Value",    style="white")
    const_tbl.add_column("Description")

    def _const(script: str, name: str, desc: str) -> None:
        src = SCRIPTS[script].read_text(errors="ignore")
        m   = re.search(rf"^{name}\s*=\s*(.+)", src, re.MULTILINE)
        if m:
            # Strip inline comment and trailing whitespace
            val = re.sub(r"\s*#.*$", "", m.group(1)).strip()
        else:
            val = "?"
        const_tbl.add_row(script, name, val, desc)

    _const("transcribe", "WHISPER_MODEL_SIZE",  "Whisper model variant")
    _const("transcribe", "WHISPER_LANGUAGE",     "Forced language (None=auto)")
    _const("align",      "EMBED_MODEL",          "Sentence-transformer model")
    _const("align",      "CONTEXT_SEC",          "Transcript context window (s)")
    _const("align",      "OFF_SLIDE_THRESHOLD",  "Min cosine to stay on-slide")
    _const("align",      "PRIOR_SIGMA",          "Temporal prior width (slides)")
    _const("generate",   "NOTE_MODEL",           "LLM for note generation")
    _const("generate",   "VERIFY_MODEL",         "LLM for verification")
    _const("generate",   "DETAIL_LEVEL",         "Default detail level (0-10)")
    _const("generate",   "CHAPTER_SIZE",         "Slides per GPT call")
    _const("generate",   "QUALITY_TARGET",       "Self-score target for --iterate")

    console.print(const_tbl)

    _section("Modify")
    console.print(
        "  To change constants, edit the relevant script directly.\n"
        "  Canvas token → [primary]downloader.py[/primary]  CANVAS_TOKEN\n"
        "  OpenAI key   → [primary]openai_api.txt[/primary]  or  OPENAI_API_KEY env var"
    )
    _pause()


# ── Subprocess runner ─────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    console.print()
    console.print(Panel(
        f"[primary]{msg}[/primary]",
        border_style="cyan",
        expand=False,
        padding=(0, 2),
    ))
    console.print()


def _run(cmd: list[str]) -> int:
    """Run a pipeline command, streaming output directly to the terminal."""
    console.print(
        "[muted]$[/muted] " +
        " ".join(str(c) for c in cmd[2:]),   # skip python + script path
        style="muted",
    )
    console.print()
    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    console.print()
    if result.returncode == 0:
        _ok("Completed successfully.")
    else:
        _err(f"Exited with code {result.returncode}.")
    return result.returncode


# ── Main menu loop ────────────────────────────────────────────────────────────

_MENU = """\
  [menu.key]1[/menu.key]  [menu.text]🚀  Full Pipeline Wizard[/menu.text]
  [menu.key]2[/menu.key]  [menu.text]⬇   Download videos & materials[/menu.text]
  [menu.key]3[/menu.key]  [menu.text]🎙   Transcribe videos[/menu.text]
  [menu.key]4[/menu.key]  [menu.text]🔗   Align transcripts to slides[/menu.text]
  [menu.key]5[/menu.key]  [menu.text]📝   Generate study notes[/menu.text]
  [menu.key]6[/menu.key]  [menu.text]📊   Course status[/menu.text]
  [menu.key]7[/menu.key]  [menu.text]⚙    Settings & constants[/menu.text]
  [menu.key]Q[/menu.key]  [menu.text]Quit[/menu.text]\
"""


def main() -> None:
    while True:
        console.clear()
        _header()

        # ── Mini status strip ─────────────────────────────────────────────────
        console.print()
        row_parts: list[Text] = []
        for cid, name in COURSES.items():
            vd, vt = _video_status(cid)
            caps   = _caption_count(cid)
            has_n  = _notes_path(cid) is not None
            short  = name.split()[0]   # e.g. "CS3210"
            icon   = "✓" if has_n else ("◐" if caps > 0 else "○")
            color  = "chip.done" if has_n else ("chip.partial" if caps > 0 else "chip.none")
            row_parts.append(Text(f"  {icon} {short}", style=color))
        status_line = Text("  ")
        for t in row_parts:
            status_line.append_text(t)
        console.print(status_line)

        # ── Menu ──────────────────────────────────────────────────────────────
        console.print()
        console.print(_MENU)
        console.print()

        choice = Prompt.ask(
            "  [primary]>[/primary]",
            default="Q",
            show_default=False,
        )

        match choice.strip().upper():
            case "1":
                menu_full_pipeline()
            case "2":
                menu_download()
            case "3":
                menu_transcribe()
            case "4":
                menu_align()
            case "5":
                menu_generate()
            case "6":
                show_status()
                _pause()
            case "7":
                menu_settings()
            case "Q" | "":
                console.print("\n  [muted]Bye.[/muted]\n")
                break
            case _:
                _err("Invalid choice.")


if __name__ == "__main__":
    main()
