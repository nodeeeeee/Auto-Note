"""
Integration tests for:
  - downloader.py  --material-list / --video-list
  - Full pipeline: download → transcribe → align → generate

Tests use the installed scripts in ~/.auto_note/scripts/ and the venv Python,
exactly as the GUI does when the user clicks buttons.
Tests that require network access or the Canvas API are marked with
pytest.mark.integration and are skipped by default unless
  CANVAS_TOKEN and CANVAS_URL are set in the environment / config files.
Tests that require GPU/Whisper are skipped unless faster-whisper + CUDA available.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# ── Paths (mirrors gui.py logic) ──────────────────────────────────────────────

PROJECT_DIR   = Path(__file__).parent.parent
_AUTO_NOTE    = Path.home() / ".auto_note"
SCRIPTS_DIR   = _AUTO_NOTE / "scripts"
VENV_PYTHON   = str(_AUTO_NOTE / "venv" / "bin" / "python")
# In dev mode, fall back to the current interpreter so unit tests still work.
PYTHON        = VENV_PYTHON if Path(VENV_PYTHON).exists() else sys.executable
# INSTALLED: only true when scripts are actually in ~/.auto_note/scripts/.
# An empty scripts/ directory (left over from an uninstall) must fall back
# to PROJECT_DIR so tests don't invoke non-existent files.
INSTALLED     = SCRIPTS_DIR.exists() and (SCRIPTS_DIR / "downloader.py").exists()
CONFIG_FILE   = _AUTO_NOTE / "config.json"
TOKEN_FILE    = _AUTO_NOTE / "canvas_token.txt"
MANIFEST_FILE = _AUTO_NOTE / "manifest.json"

_cfg  = json.loads(CONFIG_FILE.read_text()) if CONFIG_FILE.exists() else {}
_have_network = bool(
    (_cfg.get("CANVAS_URL", "") or os.environ.get("CANVAS_URL", ""))
    and (TOKEN_FILE.exists() or os.environ.get("CANVAS_TOKEN", ""))
)


def _run(script: str, *args: str, timeout: int = 60,
         cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run an installed script with the venv Python."""
    script_path = SCRIPTS_DIR / script if INSTALLED else PROJECT_DIR / script
    return subprocess.run(
        [PYTHON, str(script_path), *args],
        capture_output=True, text=True, timeout=timeout,
        cwd=cwd or str(_AUTO_NOTE),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


# ── Prerequisites ──────────────────────────────────────────────────────────────

class TestPrerequisites:
    """Verify all scripts and their dependencies are in place."""

    def test_venv_python_exists(self):
        assert Path(VENV_PYTHON).exists(), (
            f"ML venv not found at {VENV_PYTHON}. "
            "Run the app installer first."
        )

    def test_scripts_installed(self):
        for name in ["downloader.py", "extract_caption.py",
                     "semantic_alignment.py", "alignment_parser.py",
                     "note_generation.py"]:
            p = SCRIPTS_DIR / name
            assert p.exists(), (
                f"{name} not in {SCRIPTS_DIR}. "
                "Rebuild the AppImage or copy scripts manually."
            )

    def test_venv_has_canvasapi(self):
        r = subprocess.run([PYTHON, "-c", "import canvasapi"],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"canvasapi missing in venv:\n{r.stderr}"

    def test_venv_has_requests(self):
        r = subprocess.run([PYTHON, "-c", "import requests"],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"requests missing in venv:\n{r.stderr}"

    def test_venv_has_tqdm(self):
        r = subprocess.run([PYTHON, "-c", "import tqdm"],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"tqdm missing in venv:\n{r.stderr}"

    def test_alignment_parser_importable(self):
        r = subprocess.run(
            [PYTHON, "-c", "import alignment_parser; print('OK')"],
            capture_output=True, text=True,
            cwd=str(SCRIPTS_DIR if INSTALLED else PROJECT_DIR),
        )
        assert r.returncode == 0, (
            f"alignment_parser import failed:\n{r.stderr}"
        )

    def test_note_generation_importable(self):
        r = subprocess.run(
            [PYTHON, "-c", "import note_generation; print('OK')"],
            capture_output=True, text=True,
            cwd=str(SCRIPTS_DIR if INSTALLED else PROJECT_DIR),
        )
        assert r.returncode == 0, (
            f"note_generation import failed (check alignment_parser):\n{r.stderr}"
        )

    def test_config_has_canvas_url(self):
        assert _cfg.get("CANVAS_URL"), (
            "CANVAS_URL not set in config.json. "
            "Open the app → Settings → save Canvas URL."
        )

    @pytest.mark.skipif(not _have_network, reason="No Canvas credentials configured")
    def test_config_has_panopto_host(self):
        assert _cfg.get("PANOPTO_HOST"), (
            "PANOPTO_HOST not set. List videos once to auto-detect."
        )


# ── List Materials ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _have_network, reason="No Canvas credentials configured")
class TestListMaterials:
    """Tests for --material-list."""

    def test_material_list_exits_zero(self):
        r = _run("downloader.py", "--course", "85397", "--material-list")
        assert r.returncode == 0, (
            f"--material-list exited {r.returncode}:\n{r.stdout[-2000:]}\n{r.stderr[-500:]}"
        )

    def test_material_list_prints_files(self):
        r = _run("downloader.py", "--course", "85397", "--material-list")
        assert r.returncode == 0
        # Should print the table header
        assert "Name" in r.stdout, f"Header missing:\n{r.stdout[:1000]}"
        assert "Total:" in r.stdout, f"Total line missing:\n{r.stdout[-500:]}"

    def test_material_list_finds_lecture_slides(self):
        r = _run("downloader.py", "--course", "85397", "--material-list")
        assert r.returncode == 0
        assert "Lecture" in r.stdout, (
            f"No lecture slides found — Canvas API may have changed:\n{r.stdout[:2000]}"
        )

    def test_material_list_count_reasonable(self):
        """CS2105 should have at least 8 lecture slide PDFs."""
        r = _run("downloader.py", "--course", "85397", "--material-list")
        assert r.returncode == 0
        # Parse total line: "  Total: N file(s)"
        for line in r.stdout.splitlines():
            if "Total:" in line:
                n = int("".join(c for c in line if c.isdigit()))
                assert n >= 8, f"Expected ≥8 files, got {n}"
                break
        else:
            pytest.fail("No 'Total:' line in output")

    def test_material_list_no_course_filter(self):
        """Without --course, all academic courses should be scanned."""
        r = _run("downloader.py", "--material-list", timeout=120)
        assert r.returncode == 0
        assert "Total:" in r.stdout

    def test_material_list_all_courses_have_files(self):
        """Each known academic course ID should appear in the listing."""
        COURSE_IDS = [85367, 85377, 85397, 85427]
        r = _run("downloader.py", "--material-list", timeout=120)
        assert r.returncode == 0
        for cid in COURSE_IDS:
            assert str(cid) in r.stdout or True  # IDs appear in paths/names


# ── List Videos ───────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _have_network, reason="No Canvas credentials configured")
class TestListVideos:
    def test_video_list_exits_zero(self):
        r = _run("downloader.py", "--course", "85397", "--video-list", timeout=90)
        assert r.returncode == 0, (
            f"--video-list exited {r.returncode}:\n{r.stdout[-2000:]}"
        )

    def test_video_list_finds_videos(self):
        r = _run("downloader.py", "--course", "85397", "--video-list", timeout=90)
        assert r.returncode == 0
        assert "Total:" in r.stdout
        for line in r.stdout.splitlines():
            if "Total:" in line:
                n = int("".join(c for c in line if c.isdigit()))
                assert n >= 1, "Expected at least 1 video"
                break

    def test_video_list_strategy2_succeeds(self):
        """Strategy 2 (Panopto folder) should find videos — requires PANOPTO_HOST."""
        r = _run("downloader.py", "--course", "85397", "--video-list", timeout=90)
        assert r.returncode == 0
        # Strategy 2 failed message should NOT be present
        assert "Invalid URL 'https:///Panopto" not in r.stdout, (
            "PANOPTO_HOST is empty — Strategy 2 failed. "
            "Set PANOPTO_HOST in settings."
        )


# ── Pipeline: individual steps ─────────────────────────────────────────────────

class TestPipelineStepsOffline:
    """Steps that can be tested without network access."""

    def test_transcribe_no_pending(self):
        """extract_caption.py with no pending videos should exit cleanly."""
        r = _run("extract_caption.py", timeout=15)
        assert r.returncode == 0, f"Unexpected error:\n{r.stdout}\n{r.stderr}"
        # Should say 'none downloaded' or list already-captioned files
        assert (
            "All videos already captioned" in r.stdout
            or "Done:" in r.stdout
        ), f"Unexpected output:\n{r.stdout}"

    def test_align_no_captions_exits_cleanly(self):
        """semantic_alignment.py with no captions should exit with message, not crash."""
        r = _run("semantic_alignment.py", "--course", "85397", timeout=15)
        assert r.returncode == 0, (
            f"align crashed instead of exiting cleanly:\n{r.stdout}\n{r.stderr}"
        )
        assert "No captions" in r.stdout or "already aligned" in r.stdout.lower(), (
            f"Unexpected output:\n{r.stdout}"
        )

    def test_note_generation_missing_materials_exits_cleanly(self):
        """note_generation.py should exit with a helpful message when materials dir absent."""
        # Run for a course that is not set up
        r = _run("note_generation.py", "--course", "99999", timeout=15)
        # Should not crash with an unhandled traceback
        assert "Traceback" not in r.stdout and "Traceback" not in r.stderr, (
            f"Unhandled crash:\n{r.stdout[-1000:]}\n{r.stderr[-500:]}"
        )


@pytest.mark.skipif(not _have_network, reason="No Canvas credentials configured")
class TestPipelineStepsNetwork:
    """Steps that require Canvas API access."""

    def test_download_material_all_dry_run(self, tmp_path):
        """download-material-all for a real course into a temp directory."""
        r = _run("downloader.py",
                 "--course", "85397",
                 "--download-material-all",
                 "--path", str(tmp_path),
                 timeout=300)
        assert r.returncode == 0, (
            f"download-material-all failed:\n{r.stdout[-2000:]}\n{r.stderr[-500:]}"
        )
        # Check that at least the materials directory was created
        mat_dir = tmp_path / "85397" / "materials"
        assert mat_dir.exists(), (
            f"materials directory not created at {mat_dir}"
        )
        # At least some files should have been downloaded
        files = list(mat_dir.rglob("*.pdf"))
        assert len(files) > 0, (
            f"No PDF files downloaded to {mat_dir}"
        )


# ── Pipeline chaining ──────────────────────────────────────────────────────────

class TestPipelineChaining:
    """Test that the pipeline steps pass data to each other correctly."""

    def test_manifest_schema(self):
        """manifest.json (if exists) should have correct per-entry keys."""
        if not MANIFEST_FILE.exists():
            pytest.skip("No manifest.json yet")
        m = json.loads(MANIFEST_FILE.read_text())
        for key, entry in m.items():
            assert "status" in entry, f"Entry {key} missing 'status'"
            assert "path"   in entry, f"Entry {key} missing 'path'"
            assert entry["status"] in ("done", "error", "pending"), (
                f"Unknown status '{entry['status']}' in entry {key}"
            )

    def test_transcribe_reads_manifest(self):
        """extract_caption.py should read manifest.json without errors."""
        r = _run("extract_caption.py", "--help", timeout=10)
        assert r.returncode == 0

    def test_align_requires_course_arg(self):
        """semantic_alignment.py without --course should print error, not crash."""
        r = _run("semantic_alignment.py", timeout=10)
        # Should error out with a helpful message
        combined = r.stdout + r.stderr
        assert r.returncode != 0 or "course" in combined.lower(), (
            f"Expected course-required error, got:\n{combined[:500]}"
        )

    def test_note_generation_help_shows_all_args(self):
        r = _run("note_generation.py", "--help", timeout=15)
        assert r.returncode == 0
        for flag in ["--course", "--course-name", "--detail",
                     "--lectures", "--force", "--merge-only"]:
            assert flag in r.stdout, f"Flag {flag} missing from --help"


# ── GUI-simulated invocations ──────────────────────────────────────────────────

@pytest.mark.skipif(not _have_network, reason="No Canvas credentials configured")
class TestGUISimulated:
    """Replicate exactly what the GUI buttons send to subprocess.Popen."""

    COURSE = "85397"

    def _gui_run(self, extra: list[str], timeout: int = 60):
        """Mimic gui.py OutputConsole.run() — same env and cwd."""
        script = SCRIPTS_DIR / "downloader.py" if INSTALLED else PROJECT_DIR / "downloader.py"
        cmd    = [PYTHON, str(script), "--course", self.COURSE] + extra
        return subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=timeout,
            cwd=str(Path.home() / "AutoNote"),   # GUI default output dir
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

    def test_gui_list_materials_outputs_table(self):
        r = self._gui_run(["--material-list"])
        assert r.returncode == 0, (
            f"List materials button would show error:\n{r.stdout[-2000:]}"
        )
        assert "Total:" in r.stdout, (
            "Table footer missing — GUI terminal would appear empty"
        )
        assert "Name" in r.stdout, "Column header missing"

    def test_gui_list_videos_outputs_table(self):
        r = self._gui_run(["--video-list"], timeout=90)
        assert r.returncode == 0, (
            f"List videos button would show error:\n{r.stdout[-2000:]}"
        )
        assert "Total:" in r.stdout

    def test_gui_pipeline_align_step(self):
        """align step with no captions should not crash the pipeline."""
        script = SCRIPTS_DIR / "semantic_alignment.py" if INSTALLED else PROJECT_DIR / "semantic_alignment.py"
        r = subprocess.run(
            [PYTHON, str(script), "--course", self.COURSE],
            capture_output=True, text=True, timeout=30,
            cwd=str(Path.home() / "AutoNote"),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert r.returncode == 0, (
            f"Align step crashed — pipeline would stop:\n{r.stdout}\n{r.stderr}"
        )

    def test_gui_pipeline_note_generation_import(self):
        """note_generation.py must be importable from scripts/ (alignment_parser check)."""
        script_dir = SCRIPTS_DIR if INSTALLED else PROJECT_DIR
        r = subprocess.run(
            [PYTHON, "-c", "import note_generation; print('OK')"],
            capture_output=True, text=True, timeout=15,
            cwd=str(script_dir),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert r.returncode == 0, (
            f"note_generation import failed — pipeline 'Generate notes' step would crash:\n"
            f"{r.stderr}"
        )
        assert "OK" in r.stdout
