"""
Tests for note_generation.py

API-dependent functions (generate_section, etc.) are mocked.
"""
import json
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import note_generation as ng


# ── _detail_instr ─────────────────────────────────────────────────────────────

class TestDetailInstr:
    def test_level_0_is_bullet(self):
        result = ng._detail_instr(0)
        assert "要点" in result or "bullet" in result.lower() or len(result) > 0

    def test_level_2_still_bullet_range(self):
        assert ng._detail_instr(2) == ng._detail_instr(0)

    def test_level_5_has_hierarchy(self):
        result = ng._detail_instr(5)
        assert "层次" in result or "bullet" in result.lower()

    def test_level_8_is_detailed(self):
        result = ng._detail_instr(8)
        assert "详细" in result or "detail" in result.lower()

    def test_level_10_is_max(self):
        result = ng._detail_instr(10)
        assert len(result) > 0

    def test_level_11_does_not_crash(self):
        # Out of range should return the level-6-8 default
        assert len(ng._detail_instr(11)) > 0


# ── _key_terms ────────────────────────────────────────────────────────────────

class TestKeyTerms:
    def test_extracts_uppercase_terms(self):
        text = "The TCP protocol uses ACK messages to confirm delivery"
        terms = ng._key_terms(text)
        assert any("tcp" in t for t in terms)

    def test_extracts_cs_keywords(self):
        terms = ng._key_terms("A mutex guards the critical section from race conditions")
        lowered = [t.lower() for t in terms]
        assert "mutex" in lowered or "critical" in lowered or "race" in lowered

    def test_deduplicates(self):
        text = "mutex mutex mutex"
        terms = ng._key_terms(text)
        assert len(terms) == len(set(terms))

    def test_returns_at_most_n(self):
        text = " ".join(["Thread", "Process", "Kernel", "Mutex", "Semaphore",
                         "Deadlock", "Fork", "Scheduler", "Socket", "Packet",
                         "Protocol", "Checksum", "TCP", "UDP", "ACK"])
        terms = ng._key_terms(text, n=5)
        assert len(terms) <= 5

    def test_empty_text(self):
        assert ng._key_terms("") == []


# ── SlideInfo ─────────────────────────────────────────────────────────────────

class TestSlideInfo:
    def test_basic_construction(self):
        s = ng.SlideInfo(0, "Intro", "Introduction to Parallel Computing")
        assert s.index == 0
        assert s.label == "Intro"
        assert "Parallel" in s.text

    def test_has_code_detects_c_include(self):
        s = ng.SlideInfo(0, "Code", "#include <stdio.h>\nint main() {}")
        assert s.has_code is True

    def test_has_code_detects_pthread(self):
        s = ng.SlideInfo(0, "Threads", "pthread_create(&tid, NULL, func, NULL);")
        assert s.has_code is True

    def test_no_code_for_plain_text(self):
        s = ng.SlideInfo(0, "Text", "This is just plain text about concepts")
        assert s.has_code is False

    def test_word_count(self):
        s = ng.SlideInfo(0, "L", "one two three four five")
        assert s.word_count == 5

    def test_empty_slide(self):
        s = ng.SlideInfo(0, "Empty", "")
        assert s.word_count == 0
        assert s.has_code is False


# ── _TITLE_PATTERN ────────────────────────────────────────────────────────────

class TestTitlePattern:
    def test_matches_outline(self):
        assert ng._TITLE_PATTERN.search("Outline")

    def test_matches_agenda(self):
        assert ng._TITLE_PATTERN.search("Agenda")

    def test_matches_lecture_number(self):
        assert ng._TITLE_PATTERN.search("Lecture 3")

    def test_matches_cs_code(self):
        assert ng._TITLE_PATTERN.search("CS3210")

    def test_no_match_on_content_slide(self):
        text = "Process Scheduling: Round-Robin and Priority Algorithms"
        # Should not match because it's content
        assert not ng._TITLE_PATTERN.search(text)

    def test_matches_questions(self):
        assert ng._TITLE_PATTERN.search("Questions?")


# ── _img_ref_pattern ──────────────────────────────────────────────────────────

class TestImgRefPattern:
    def setup_method(self):
        self.pat = ng._img_ref_pattern()

    def test_standard_path(self):
        m = self.pat.search("![Slide 5](images/L02/slide_005.png)")
        assert m is not None
        assert m.group(1) == "images/L02/slide_005.png"

    def test_multi_file_path(self):
        m = self.pat.search("![Slide 5](images/L02_F02/slide_005.png)")
        assert m is not None
        assert m.group(1) == "images/L02_F02/slide_005.png"

    def test_no_match_on_invalid_path(self):
        assert self.pat.search("![image](http://example.com/img.png)") is None

    def test_no_match_on_wrong_prefix(self):
        assert self.pat.search("![Slide 1](files/L01/slide_001.png)") is None


# ── _desc_has_visual ─────────────────────────────────────────────────────────

class TestDescHasVisual:
    def test_detects_diagram(self):
        assert ng._desc_has_visual("This slide contains a diagram of the pipeline")

    def test_detects_flowchart(self):
        assert ng._desc_has_visual("A flowchart showing the scheduling algorithm")

    def test_detects_table(self):
        assert ng._desc_has_visual("Comparison table of algorithms")

    def test_false_for_pure_text(self):
        assert not ng._desc_has_visual("Text describing the process lifecycle")

    def test_case_insensitive(self):
        assert ng._desc_has_visual("System Architecture overview")


# ── filter_images_pass ────────────────────────────────────────────────────────

class TestFilterImagesPass:
    def _make_lecture_data(self, slides_with_text: list[tuple[int, str, str]]):
        """slides_with_text: list of (index, label, text)"""
        ld = MagicMock()
        ld.num = 1
        ld.file_idx = 1
        ld.img_cache = {}
        ld.slides = [
            ng.SlideInfo(idx, label, text)
            for idx, label, text in slides_with_text
        ]
        return ld

    def test_keeps_visual_slide_via_cache(self, tmp_path):
        ld = self._make_lecture_data([
            (0, "Pipeline", "This is a diagram of the pipeline architecture")
        ])
        ld.img_cache = {"page_0": "A flowchart showing the system architecture diagram"}

        notes = "Some text\n![Slide 1](images/L01/slide_001.png)\nMore text"

        img_dir = tmp_path / "images" / "L01"
        img_dir.mkdir(parents=True)
        (img_dir / "slide_001.png").write_bytes(b"PNG")

        with patch("note_generation._vision_keep") as mock_vk:
            cleaned, kept, removed = ng.filter_images_pass(notes, tmp_path, [ld])
            assert kept == 1
            assert removed == 0
            assert "![Slide 1]" in cleaned
            mock_vk.assert_not_called()  # cache hit → no vision API call

    def test_removes_title_slide(self, tmp_path):
        ld = self._make_lecture_data([
            (0, "Outline", "Outline")  # title pattern
        ])
        notes = "Intro\n![Slide 1](images/L01/slide_001.png)\nBody"

        with patch("note_generation._vision_keep") as mock_vk:
            cleaned, kept, removed = ng.filter_images_pass(notes, tmp_path, [ld])
            assert removed == 1
            assert "![Slide 1]" not in cleaned
            mock_vk.assert_not_called()  # title pattern → no vision API call

    def test_calls_vision_api_for_ambiguous(self, tmp_path):
        ld = self._make_lecture_data([
            (0, "Content", "Some content that is not obviously title or visual")
        ])
        img_dir = tmp_path / "images" / "L01"
        img_dir.mkdir(parents=True)
        from PIL import Image
        Image.new("RGB", (10, 10), "white").save(img_dir / "slide_001.png")

        notes = "Text\n![Slide 1](images/L01/slide_001.png)\nText"

        with patch("note_generation._vision_keep", return_value=True):
            cleaned, kept, removed = ng.filter_images_pass(notes, tmp_path, [ld])
            assert kept == 1
            assert removed == 0

    def test_collapses_triple_blank_lines(self, tmp_path):
        ld = self._make_lecture_data([(0, "Outline", "Outline")])
        notes = "A\n\n\n\n![Slide 1](images/L01/slide_001.png)\n\n\n\nB"

        with patch("note_generation._vision_keep", return_value=False):
            cleaned, _, _ = ng.filter_images_pass(notes, tmp_path, [ld])
            assert "\n\n\n" not in cleaned

    def test_deduplicates_decisions(self, tmp_path):
        """Same image referenced twice → _vision_keep called only once."""
        ld = self._make_lecture_data([
            (0, "Content", "Some content without visual keywords")
        ])
        img_dir = tmp_path / "images" / "L01"
        img_dir.mkdir(parents=True)
        from PIL import Image
        Image.new("RGB", (10, 10)).save(img_dir / "slide_001.png")

        notes = ("![Slide 1](images/L01/slide_001.png)\n"
                 "text\n"
                 "![Slide 1](images/L01/slide_001.png)")

        with patch("note_generation._vision_keep", return_value=True) as mock_vk:
            ng.filter_images_pass(notes, tmp_path, [ld])
            # _vision_keep called once even though image appears twice
            assert mock_vk.call_count == 1


# ── self_score ────────────────────────────────────────────────────────────────

class TestSelfScore:
    def _make_slides(self, texts: list[str]) -> list[ng.SlideInfo]:
        return [ng.SlideInfo(i, f"Slide {i}", t) for i, t in enumerate(texts)]

    def test_perfect_coverage(self):
        slides = self._make_slides(["word"] * 5)
        notes  = " word" * (5 * ng.MIN_NOTE_WORDS_PER_SLIDE)
        scores = ng.self_score(slides, notes, [])
        assert scores["coverage"] == 10.0

    def test_zero_coverage(self):
        slides = self._make_slides(["important content here"] * 5)
        scores = ng.self_score(slides, "x", [])
        assert scores["coverage"] < 2.0

    def test_terminology_all_present(self):
        slides = self._make_slides(["Process Thread Mutex Semaphore Deadlock"] * 3)
        notes  = "process thread mutex semaphore deadlock " * 50
        scores = ng.self_score(slides, notes, [])
        assert scores["terminology"] > 8.0

    def test_callouts_score_no_callouts_needed(self):
        slides = self._make_slides(["Just some text without exam cues"] * 3)
        compact = [{"slide": i+1, "transcript": "no cues here"} for i in range(3)]
        notes = "text " * 200
        scores = ng.self_score(slides, notes, compact)
        assert scores["callouts"] == 10.0  # no callouts needed → perfect

    def test_callouts_score_with_callouts_written(self):
        slides = self._make_slides(["exam content"] * 2)
        compact = [
            {"slide": 1, "transcript": "this is important remember for exam"},
            {"slide": 2, "transcript": "key point here"},
        ]
        notes = "text " * 200 + "\n> [!IMPORTANT]\n> point 1\n> [!IMPORTANT]\n> point 2"
        scores = ng.self_score(slides, notes, compact)
        assert scores["callouts"] > 0

    def test_code_blocks_counted(self):
        slides = self._make_slides(["#include <pthread.h>\nint main() {return 0;}"] * 3)
        notes = "```c\nint main(){}\n```\n" * 3 + "text " * 100
        scores = ng.self_score(slides, notes, [])
        assert scores["code_blocks"] > 0

    def test_overall_is_weighted_average(self):
        slides = self._make_slides(["word"] * 3)
        notes  = " word" * (3 * ng.MIN_NOTE_WORDS_PER_SLIDE * 2)  # over-fill
        scores = ng.self_score(slides, notes, [])
        expected = (
            ng.SCORE_WEIGHTS["coverage"]    * scores["coverage"] +
            ng.SCORE_WEIGHTS["terminology"] * scores["terminology"] +
            ng.SCORE_WEIGHTS["callouts"]    * scores["callouts"] +
            ng.SCORE_WEIGHTS["code_blocks"] * scores["code_blocks"]
        )
        assert abs(scores["overall"] - round(expected, 2)) < 0.01

    def test_stats_keys_present(self):
        slides = self._make_slides(["text"] * 2)
        scores = ng.self_score(slides, "note text " * 20, [])
        st = scores["stats"]
        assert "note_words"     in st
        assert "expected_words" in st
        assert "term_hits"      in st
        assert "term_total"     in st


# ── _discover_lectures ────────────────────────────────────────────────────────

class TestDiscoverLectures:
    def _touch(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4")  # minimal PDF magic bytes
        return path

    def test_finds_slides_in_lecture_notes_subfolder(self, tmp_path):
        ln = tmp_path / "materials" / "LectureNotes"
        self._touch(ln / "L01-Intro.pdf")
        self._touch(ln / "L02-Process.pdf")
        lectures = ng._discover_lectures(tmp_path)
        nums = {l.num for l in lectures}
        assert 1 in nums
        assert 2 in nums

    def test_fallback_to_direct_children_when_no_lecture_notes(self, tmp_path):
        mat = tmp_path / "materials"
        self._touch(mat / "L01-slides.pdf")
        self._touch(mat / "L02-slides.pdf")
        # No LectureNotes subdirectory
        lectures = ng._discover_lectures(tmp_path)
        assert len(lectures) == 2

    def test_does_not_recurse_into_subdirs_without_lecture_notes(self, tmp_path):
        mat = tmp_path / "materials"
        self._touch(mat / "L01-slides.pdf")
        # Create Tutorials and Assignments subdirs with PDFs
        self._touch(mat / "Tutorials" / "T01.pdf")
        self._touch(mat / "Assignments" / "A01.pdf")
        # Falls back to direct children only
        lectures = ng._discover_lectures(tmp_path)
        assert len(lectures) == 1  # only L01-slides.pdf

    def test_extracts_lecture_number_from_filename(self, tmp_path):
        ln = tmp_path / "materials" / "LectureNotes"
        self._touch(ln / "Lecture-03-Threads.pdf")
        lectures = ng._discover_lectures(tmp_path)
        assert lectures[0].num == 3

    def test_extracts_lecture_number_with_space_separator(self, tmp_path):
        """'Lecture 3 - Topic.pdf' style (space before number)."""
        mat = tmp_path / "materials" / "Lecture Slides"
        self._touch(mat / "Lecture 1 - Introduction.pdf")
        self._touch(mat / "Lecture 2 - Application Layer.pdf")
        lectures = ng._discover_lectures(tmp_path)
        nums = {l.num for l in lectures}
        assert 1 in nums
        assert 2 in nums

    def test_finds_lecture_slides_subfolder(self, tmp_path):
        """'Lecture Slides' folder should be recognised before falling back."""
        mat = tmp_path / "materials" / "Lecture Slides"
        self._touch(mat / "Lecture 1.pdf")
        self._touch(mat / "Lecture 2.pdf")
        # Also create Tutorials — should NOT be picked up
        self._touch(tmp_path / "materials" / "Tutorials" / "T01.pdf")
        lectures = ng._discover_lectures(tmp_path)
        assert len(lectures) == 2

    def test_ignores_image_cache_files(self, tmp_path):
        mat = tmp_path / "materials"
        self._touch(mat / "L01.pdf")
        self._touch(mat / "L01.image_cache.json")
        lectures = ng._discover_lectures(tmp_path)
        # image_cache.json should be ignored (wrong ext anyway), just no crash
        assert len(lectures) == 1

    def test_no_slides_returns_empty(self, tmp_path):
        (tmp_path / "materials").mkdir(parents=True)
        lectures = ng._discover_lectures(tmp_path)
        assert lectures == []

    def test_pptx_supported(self, tmp_path):
        mat = tmp_path / "materials"
        p = mat / "L04-Lecture.pptx"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"PK")   # ZIP/PPTX magic
        lectures = ng._discover_lectures(tmp_path)
        assert any(l.num == 4 for l in lectures)


# ── _load_slides ──────────────────────────────────────────────────────────────

class TestLoadSlides:
    def test_unsupported_format_raises(self, tmp_path):
        bad = tmp_path / "file.xyz"
        bad.write_text("content")
        with pytest.raises(ValueError, match="Unsupported"):
            ng._load_slides(bad)

    def test_load_real_pdf(self):
        """Load a real PDF from the project's alignment test data if available."""
        import fitz
        pdf_path = (Path(__file__).parent.parent /
                    "85427_medium" / "materials" / "LectureNotes" / "L02-Processes-Threads.pdf")
        if not pdf_path.exists():
            pytest.skip("Real PDF not present")
        slides = ng._load_slides(pdf_path)
        assert len(slides) > 0
        assert all(isinstance(s, ng.SlideInfo) for s in slides)
        assert all(s.index >= 0 for s in slides)

    def test_load_small_pdf(self, tmp_path):
        """Create a minimal valid PDF and verify _load_slides doesn't crash."""
        try:
            import fitz
        except ImportError:
            pytest.skip("fitz not available")

        pdf_path = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Hello Parallel Computing\nProcess and Thread")
        doc.save(str(pdf_path))
        doc.close()

        slides = ng._load_slides(pdf_path)
        assert len(slides) == 1
        assert "Hello" in slides[0].text


# ── System prompt quality ──────────────────────────────────────────────────────

def _all_system_prompts() -> str:
    """Concatenate all system prompts from _PROMPTS for inspection."""
    return "\n".join(
        v.get("system", "") if isinstance(v, dict) else ""
        for v in ng._PROMPTS.values()
    )


class TestSystemPrompt:
    def test_no_professor_centric_narration_rule(self):
        # System prompts are now authored in English (Chinese output is
        # produced by a separate post-generation _translate() pass).
        prompts = _all_system_prompts()
        assert "third-person" in prompts.lower() or "third person" in prompts.lower()
        assert "professor" in prompts.lower() or "lecturer" in prompts.lower()

    def test_latex_math_rule(self):
        prompts = _all_system_prompts()
        assert "LaTeX" in prompts or "latex" in prompts.lower()

    def test_callout_format_specified(self):
        prompts = _all_system_prompts()
        assert "[!IMPORTANT]" in prompts

    def test_image_path_rule(self):
        prompts = _all_system_prompts()
        assert "images/" in prompts


# ── CLI smoke test ────────────────────────────────────────────────────────────

def test_cli_help():
    import subprocess
    result = subprocess.run(
        [sys.executable,
         str(Path(__file__).parent.parent / "note_generation.py"),
         "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--course" in result.stdout
    assert "--detail" in result.stdout
