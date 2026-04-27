"""
Regression tests for the v1.0.1 lecture-filter propagation fix.

Before this fix, picking a lecture filter (e.g. "1-3") in the Pipeline page
only narrowed the note-generation step. Download / transcribe / align all
silently processed every video in the course. After the fix:

  • pipeline_worker.py exposes --lectures and parses '1-5' / '1,3,5' /
    '1-2,7' formats consistently with note_generation.
  • get_videos() returns videos sorted by stem so the 1-based position
    matches the alphabetical-caption numbering used downstream.
  • semantic_alignment.py accepts --lectures and filters captions to
    those positions before running BGE-M3.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))


class TestParseLectureFilter:
    def test_range(self):
        from pipeline_worker import _parse_lecture_filter
        assert _parse_lecture_filter("1-5") == {1, 2, 3, 4, 5}

    def test_list(self):
        from pipeline_worker import _parse_lecture_filter
        assert _parse_lecture_filter("1,3,5") == {1, 3, 5}

    def test_mixed(self):
        from pipeline_worker import _parse_lecture_filter
        assert _parse_lecture_filter("1-3,7") == {1, 2, 3, 7}

    def test_empty(self):
        from pipeline_worker import _parse_lecture_filter
        assert _parse_lecture_filter("") == set()

    def test_garbage_tokens_ignored(self):
        from pipeline_worker import _parse_lecture_filter
        # 'foo' and 'bar' should be silently dropped — treat the parser as
        # tolerant rather than crashing the pipeline mid-run.
        assert _parse_lecture_filter("foo,1-2,bar") == {1, 2}

    def test_whitespace_tolerant(self):
        from pipeline_worker import _parse_lecture_filter
        assert _parse_lecture_filter(" 1 - 3 , 7 ") == {1, 2, 3, 7}


class TestGetVideosOrdering:
    """Lecture-N must mean the same video at every stage. That requires
    get_videos() to sort by stem (matching note_generation's caption sort).
    """

    def test_videos_sorted_alphabetically_by_stem(self, tmp_path, monkeypatch):
        # Build a fake manifest where insertion order is REVERSED relative
        # to alphabetical order. If get_videos() returned manifest order,
        # lecture 1 would be the wrong file.
        course_dir = tmp_path / "85397"
        videos_dir = course_dir / "videos"
        videos_dir.mkdir(parents=True)
        for stem in ("CS2105_Lecture_3", "CS2105_Lecture_1", "CS2105_Lecture_2"):
            (videos_dir / f"{stem}.mp4").touch()

        manifest = {
            "30": {"status": "done", "path": str(videos_dir / "CS2105_Lecture_3.mp4")},
            "10": {"status": "done", "path": str(videos_dir / "CS2105_Lecture_1.mp4")},
            "20": {"status": "done", "path": str(videos_dir / "CS2105_Lecture_2.mp4")},
        }
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest))

        import pipeline_worker as pw
        monkeypatch.setattr(pw, "MANIFEST_FILE", manifest_file)

        videos = pw.get_videos("85397", tmp_path)
        stems = [v["stem"] for v in videos]
        assert stems == [
            "CS2105_Lecture_1",
            "CS2105_Lecture_2",
            "CS2105_Lecture_3",
        ]


class TestSemanticAlignmentLectureFilter:
    """semantic_alignment.process_course must accept and apply a lectures
    filter. Numbers are 1-based positions over alphabetically sorted
    captions — matching pipeline_worker.get_videos and
    note_generation._discover_video_lectures.
    """

    def test_filter_keeps_only_selected_captions(self, tmp_path, monkeypatch):
        course_dir = tmp_path / "85397"
        captions_dir = course_dir / "captions"
        captions_dir.mkdir(parents=True)

        # Create 5 caption files; lecture 2 + 4 should survive '2,4'
        names = [
            "Lecture_1.json",
            "Lecture_2.json",
            "Lecture_3.json",
            "Lecture_4.json",
            "Lecture_5.json",
        ]
        for n in names:
            (captions_dir / n).write_text("{}")

        # Prevent process_course from doing any real work — we only need
        # the filter step to execute. Stub everything that runs after.
        import semantic_alignment as sa
        monkeypatch.setattr(sa, "COURSE_DATA_DIR", tmp_path)
        monkeypatch.setattr(sa, "_candidate_slides", lambda _d: [])

        # Should return early with the no-slides warning, but only AFTER
        # filtering captions. Capture stdout to verify the count printed.
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sa.process_course("85397", lectures={2, 4})
        out = buf.getvalue()
        assert "Lecture filter: 2/5 caption(s) selected" in out
