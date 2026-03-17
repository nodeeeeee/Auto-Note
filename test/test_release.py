"""
Release-readiness tests — verify the VENV (the Python used in the released
AppImage) has every package that every pipeline script imports.

Run with the project's conda env so pytest itself is available, but the
imports are tested against the *venv* Python (~/.auto_note/venv/bin/python)
which is what the GUI actually invokes.

  conda run -n auto-note python -m pytest test/test_release.py -v
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# ── Resolve paths ─────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).parent.parent
_AUTO_NOTE  = Path.home() / ".auto_note"
SCRIPTS_DIR = _AUTO_NOTE / "scripts"
VENV_PYTHON = str(_AUTO_NOTE / "venv" / "bin" / "python")

pytestmark = pytest.mark.skipif(
    not Path(VENV_PYTHON).exists(),
    reason=f"Venv not found at {VENV_PYTHON} — run the app installer first",
)


def _venv_import(module: str) -> tuple[bool, str]:
    """Return (success, error_msg) for importing module in the venv."""
    r = subprocess.run(
        [VENV_PYTHON, "-c", f"import {module}"],
        capture_output=True, text=True,
    )
    return r.returncode == 0, r.stderr.strip()


def _assert_import(module: str, pip_name: str | None = None) -> None:
    ok, err = _venv_import(module)
    install_hint = pip_name or module
    assert ok, (
        f"Venv is missing '{module}' (install: pip install {install_hint}).\n"
        f"Error: {err}"
    )


# ── Core packages (used at module-import time) ────────────────────────────────

class TestCorePackages:
    """Every package imported at the top of a pipeline script must be present."""

    def test_canvasapi(self):
        _assert_import("canvasapi")

    def test_tqdm(self):
        _assert_import("tqdm")

    def test_requests(self):
        _assert_import("requests")

    def test_faiss(self):
        _assert_import("faiss")

    def test_numpy(self):
        _assert_import("numpy")

    def test_torch(self):
        _assert_import("torch")


# ── Lazy / conditional packages ───────────────────────────────────────────────

class TestLazyPackages:
    """Packages imported only inside functions / on first use."""

    def test_faster_whisper(self):
        _assert_import("faster_whisper", "faster-whisper")

    def test_sentence_transformers(self):
        _assert_import("sentence_transformers", "sentence-transformers")

    def test_openai(self):
        _assert_import("openai")

    def test_anthropic(self):
        _assert_import("anthropic")

    def test_pymupdf(self):
        _assert_import("fitz", "pymupdf")

    def test_pptx(self):
        _assert_import("pptx", "python-pptx")

    def test_docx(self):
        _assert_import("docx", "python-docx")

    def test_pillow(self):
        _assert_import("PIL", "pillow")

    def test_playwright(self):
        _assert_import("playwright", "playwright")

    def test_httpx(self):
        _assert_import("httpx")

    def test_PanoptoDownloader(self):
        _assert_import(
            "PanoptoDownloader",
            "git+https://github.com/Panopto-Video-DL/Panopto-Video-DL-lib.git",
        )

    def test_ffmpeg_progress_yield(self):
        _assert_import("ffmpeg_progress_yield", "ffmpeg-progress-yield")

    def test_pycryptodomex(self):
        _assert_import("Cryptodome", "pycryptodomex")


# ── Script-level importability ────────────────────────────────────────────────

class TestScriptImports:
    """Each installed script must be importable from SCRIPTS_DIR."""

    @pytest.fixture(autouse=True)
    def _skip_if_not_installed(self):
        if not SCRIPTS_DIR.exists():
            pytest.skip("Scripts not installed yet")

    def _run_import(self, script_stem: str) -> tuple[bool, str]:
        r = subprocess.run(
            [VENV_PYTHON, "-c", f"import {script_stem}; print('OK')"],
            capture_output=True, text=True,
            cwd=str(SCRIPTS_DIR),
        )
        return r.returncode == 0, (r.stdout + r.stderr).strip()

    def test_downloader_importable(self):
        ok, out = self._run_import("downloader")
        assert ok, f"downloader import failed:\n{out}"

    def test_extract_caption_importable(self):
        ok, out = self._run_import("extract_caption")
        assert ok, f"extract_caption import failed:\n{out}"

    def test_semantic_alignment_importable(self):
        ok, out = self._run_import("semantic_alignment")
        assert ok, f"semantic_alignment import failed:\n{out}"

    def test_alignment_parser_importable(self):
        ok, out = self._run_import("alignment_parser")
        assert ok, f"alignment_parser import failed:\n{out}"

    def test_note_generation_importable(self):
        ok, out = self._run_import("note_generation")
        assert ok, (
            f"note_generation import failed (alignment_parser missing?):\n{out}"
        )

    def test_all_scripts_present(self):
        required = [
            "downloader.py",
            "extract_caption.py",
            "semantic_alignment.py",
            "alignment_parser.py",
            "note_generation.py",
        ]
        for name in required:
            assert (SCRIPTS_DIR / name).exists(), (
                f"{name} not in {SCRIPTS_DIR} — "
                "_install_scripts() may not have copied it"
            )


# ── CLI smoke tests (venv Python) ─────────────────────────────────────────────

class TestCLISmoke:
    """--help on every script should exit 0 with the venv Python."""

    @pytest.fixture(autouse=True)
    def _skip_if_not_installed(self):
        if not SCRIPTS_DIR.exists():
            pytest.skip("Scripts not installed yet")

    def _help(self, script: str, timeout: int = 15) -> subprocess.CompletedProcess:
        return subprocess.run(
            [VENV_PYTHON, str(SCRIPTS_DIR / script), "--help"],
            capture_output=True, text=True, timeout=timeout,
        )

    def test_downloader_help(self):
        r = self._help("downloader.py")
        assert r.returncode == 0, r.stderr[:500]
        assert "--material-list" in r.stdout

    def test_extract_caption_help(self):
        r = self._help("extract_caption.py")
        assert r.returncode == 0, r.stderr[:500]
        assert "--video" in r.stdout

    def test_note_generation_help(self):
        r = self._help("note_generation.py")
        assert r.returncode == 0, r.stderr[:500]
        assert "--course" in r.stdout

    def test_downloader_no_args_prints_help(self):
        r = subprocess.run(
            [VENV_PYTHON, str(SCRIPTS_DIR / "downloader.py")],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 0

    def test_note_generation_missing_materials_clean_exit(self):
        """note_generation with unknown course must print [error] not traceback."""
        r = subprocess.run(
            [VENV_PYTHON, str(SCRIPTS_DIR / "note_generation.py"),
             "--course", "99999"],
            capture_output=True, text=True, timeout=15,
            cwd=str(_AUTO_NOTE),
        )
        combined = r.stdout + r.stderr
        assert "Traceback" not in combined, (
            f"Unhandled traceback — should print [error] message:\n{combined[:1000]}"
        )
        assert r.returncode != 0  # should fail with sys.exit(1)


# ── ffmpeg availability (needed by PanoptoDownloader) ─────────────────────────

class TestSystemDeps:
    def test_ffmpeg_on_path(self):
        """PanoptoDownloader calls ffmpeg at runtime — it must be on PATH."""
        r = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True
        )
        assert r.returncode == 0, (
            "ffmpeg not found on PATH. Install with: sudo pacman -S ffmpeg"
        )

    def test_ffprobe_on_path(self):
        r = subprocess.run(
            ["ffprobe", "-version"], capture_output=True, text=True
        )
        assert r.returncode == 0, (
            "ffprobe not found on PATH. Install with: sudo pacman -S ffmpeg"
        )
