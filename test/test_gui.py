"""
Tests for gui.py helper functions.

The Flet GUI itself cannot be unit-tested without a running Flutter process,
so we test all the pure Python helpers directly.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# gui.py imports flet at module level; guard so import failure gives a clear skip
try:
    import gui
    FLET_AVAILABLE = True
except ImportError:
    FLET_AVAILABLE = False

pytestmark = pytest.mark.skipif(not FLET_AVAILABLE, reason="flet not installed")


# ── _sanitize (shared with downloader) ───────────────────────────────────────
# gui.py doesn't have its own _sanitize, but the COURSES dict and SCRIPTS
# paths should be consistent.

class TestConstants:
    def test_courses_dict_is_a_dict(self):
        # COURSES is populated dynamically from Canvas API at runtime; in tests it's empty
        assert isinstance(gui.COURSES, dict)

    def test_scripts_paths_exist(self):
        # SCRIPTS points to ~/.auto_note/scripts/ in frozen/installed mode.
        # In a dev environment the scripts live in PROJECT_DIR, so skip when
        # the installed scripts directory doesn't exist yet.
        if not gui.SCRIPTS_DIR.exists():
            pytest.skip("SCRIPTS_DIR not installed; run from a built AppImage to test")
        for key, path in gui.SCRIPTS.items():
            assert path.exists(), f"Script not found: {key} -> {path}"

    def test_project_dir_is_correct(self):
        assert gui.PROJECT_DIR == Path(__file__).parent.parent


# ── _manifest ─────────────────────────────────────────────────────────────────

class TestManifest:
    def test_returns_dict(self, tmp_path, monkeypatch):
        mf = tmp_path / "manifest.json"
        mf.write_text('{"key": "value"}')
        monkeypatch.setattr(gui, "DATA_DIR", tmp_path)
        result = gui._manifest()
        assert result == {"key": "value"}

    def test_returns_empty_dict_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gui, "DATA_DIR", tmp_path)
        result = gui._manifest()
        assert result == {}


# ── _video_status ─────────────────────────────────────────────────────────────

class TestVideoStatus:
    def test_counts_done_for_course(self, tmp_path, monkeypatch):
        mf = tmp_path / "manifest.json"
        mf.write_text(json.dumps({
            "1": {"status": "done",  "path": "/home/user/85427/videos/L01.mp4"},
            "2": {"status": "done",  "path": "/home/user/85427/videos/L02.mp4"},
            "3": {"status": "error", "path": "/home/user/85427/videos/L03.mp4"},
        }))
        monkeypatch.setattr(gui, "DATA_DIR", tmp_path)
        done, total = gui._video_status(85427)
        assert done  == 2
        assert total == 3

    def test_ignores_other_courses(self, tmp_path, monkeypatch):
        mf = tmp_path / "manifest.json"
        mf.write_text(json.dumps({
            "1": {"status": "done", "path": "/home/user/85427/videos/L01.mp4"},
            "2": {"status": "done", "path": "/home/user/85377/videos/L01.mp4"},
        }))
        monkeypatch.setattr(gui, "DATA_DIR", tmp_path)
        done, total = gui._video_status(85427)
        assert total == 1

    def test_zero_when_no_manifest(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gui, "DATA_DIR", tmp_path)
        done, total = gui._video_status(85427)
        assert done == 0
        assert total == 0


# ── _caption_count ────────────────────────────────────────────────────────────

class TestCaptionCount:
    def test_counts_json_files(self, tmp_path, monkeypatch):
        cap_dir = tmp_path / "85427" / "captions"
        cap_dir.mkdir(parents=True)
        for i in range(3):
            (cap_dir / f"L{i:02d}.json").write_text("{}")
        monkeypatch.setattr(gui, "DATA_DIR", tmp_path)
        assert gui._caption_count(85427) == 3

    def test_zero_when_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gui, "DATA_DIR", tmp_path)
        assert gui._caption_count(85427) == 0

    def test_counts_only_json(self, tmp_path, monkeypatch):
        cap_dir = tmp_path / "85427" / "captions"
        cap_dir.mkdir(parents=True)
        (cap_dir / "L01.json").write_text("{}")
        (cap_dir / "L01.mp3").write_bytes(b"audio")
        monkeypatch.setattr(gui, "DATA_DIR", tmp_path)
        assert gui._caption_count(85427) == 1


# ── _alignment_count ──────────────────────────────────────────────────────────

class TestAlignmentCount:
    def test_excludes_compact_files(self, tmp_path, monkeypatch):
        aln_dir = tmp_path / "85427" / "alignment"
        aln_dir.mkdir(parents=True)
        (aln_dir / "L01.json").write_text("{}")
        (aln_dir / "L01.compact.json").write_text("{}")
        monkeypatch.setattr(gui, "DATA_DIR", tmp_path)
        assert gui._alignment_count(85427) == 1

    def test_zero_when_no_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gui, "DATA_DIR", tmp_path)
        assert gui._alignment_count(85427) == 0

    def test_counts_multiple(self, tmp_path, monkeypatch):
        aln_dir = tmp_path / "85427" / "alignment"
        aln_dir.mkdir(parents=True)
        for i in range(4):
            (aln_dir / f"L{i:02d}.json").write_text("{}")
        monkeypatch.setattr(gui, "DATA_DIR", tmp_path)
        assert gui._alignment_count(85427) == 4


# ── _notes_path ───────────────────────────────────────────────────────────────

class TestNotesPath:
    def test_returns_md_path(self, tmp_path, monkeypatch):
        notes_dir = tmp_path / "85427" / "notes"
        notes_dir.mkdir(parents=True)
        md = notes_dir / "CS3210_notes.md"
        md.write_text("# Notes")
        monkeypatch.setattr(gui, "DATA_DIR", tmp_path)
        result = gui._notes_path(85427)
        assert result == md

    def test_returns_none_when_no_md(self, tmp_path, monkeypatch):
        notes_dir = tmp_path / "85427" / "notes"
        notes_dir.mkdir(parents=True)
        monkeypatch.setattr(gui, "DATA_DIR", tmp_path)
        assert gui._notes_path(85427) is None

    def test_returns_none_when_no_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gui, "DATA_DIR", tmp_path)
        assert gui._notes_path(85427) is None


# ── _course_name_from_notes ───────────────────────────────────────────────────

class TestCourseNameFromNotes:
    def test_extracts_from_md_filename(self, tmp_path, monkeypatch):
        notes_dir = tmp_path / "85427" / "notes"
        notes_dir.mkdir(parents=True)
        (notes_dir / "CS3210 Parallel Computing_notes.md").write_text("# Notes")
        monkeypatch.setattr(gui, "DATA_DIR", tmp_path)
        name = gui._course_name_from_notes(85427)
        assert "CS3210" in name or "Parallel" in name

    def test_falls_back_to_courses_dict(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gui, "DATA_DIR", tmp_path)
        monkeypatch.setattr(gui, "COURSES", {85427: "CS3210 Parallel Computing"})
        name = gui._course_name_from_notes(85427)
        assert "CS3210" in name


# ── _read_constant ────────────────────────────────────────────────────────────

# Evaluated only when flet is importable; without the guard, pytest would
# NameError at collection time because `gui` is unbound.
_DEV_SCRIPTS = {
    "downloader": gui.PROJECT_DIR / "downloader.py",
    "transcribe":  gui.PROJECT_DIR / "extract_caption.py",
    "align":       gui.PROJECT_DIR / "semantic_alignment.py",
    "generate":    gui.PROJECT_DIR / "note_generation.py",
} if FLET_AVAILABLE else {}


class TestReadConstant:
    def test_reads_note_model(self, monkeypatch):
        monkeypatch.setattr(gui, "SCRIPTS", _DEV_SCRIPTS)
        model = gui._read_constant("generate", "NOTE_MODEL")
        assert model  # should be a non-empty string like "gpt-5.1"
        assert "?" not in model

    def test_reads_detail_level(self, monkeypatch):
        monkeypatch.setattr(gui, "SCRIPTS", _DEV_SCRIPTS)
        detail = gui._read_constant("generate", "DETAIL_LEVEL")
        assert detail.isdigit() or detail == "?"

    def test_missing_constant_returns_question_mark(self, monkeypatch):
        monkeypatch.setattr(gui, "SCRIPTS", _DEV_SCRIPTS)
        result = gui._read_constant("generate", "NONEXISTENT_CONSTANT_XYZ")
        assert result == "?"


# ── AppState ──────────────────────────────────────────────────────────────────

class TestAppState:
    def test_initial_state(self):
        s = gui.AppState()
        assert s.running is False
        assert s.proc is None

    def test_stop_when_not_running(self):
        s = gui.AppState()
        s.stop()  # should not raise

    def test_stop_terminates_process(self):
        from unittest.mock import MagicMock
        s = gui.AppState()
        proc = MagicMock()
        proc.poll.return_value = None   # process is alive
        s.proc    = proc
        s.running = True
        s.stop()
        proc.terminate.assert_called_once()
        assert s.running is False
