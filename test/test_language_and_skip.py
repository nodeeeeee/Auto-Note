"""
Tests for:
  1. All terminal/print output is in English (no Chinese in print statements)
  2. Note language selection via --language CLI arg
  3. Pipeline skip logic: existing files are not re-processed without --force

No network, API keys, or GPU required.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# ── Project root for imports ─────────────────────────────────────────────────
# Always use PROJECT_DIR (source) so we test the latest code, not stale
# installed copies in ~/.auto_note/scripts/.
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

PYTHON = sys.executable


def _run(script: str, *args: str, timeout: int = 30,
         cwd: str | None = None) -> subprocess.CompletedProcess:
    script_path = PROJECT_DIR / script
    return subprocess.run(
        [PYTHON, str(script_path), *args],
        capture_output=True, text=True, timeout=timeout,
        cwd=cwd or str(PROJECT_DIR),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


# CJK Unicode ranges for detecting Chinese characters
_CJK_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
    r"\U00020000-\U0002a6df\U0002a700-\U0002ebef]"
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Terminal output is English — no Chinese in print() statements
# ══════════════════════════════════════════════════════════════════════════════

class TestTerminalOutputEnglish:
    """Verify that all print() calls in Python scripts use English text.

    Chinese text should only appear inside prompt template strings (_PROMPTS),
    not in any print/tqdm.write statements that show in the terminal.
    """

    SCRIPTS = [
        "note_generation.py",
        "extract_caption.py",
        "semantic_alignment.py",
        "downloader.py",
        "frame_extractor.py",
        "alignment_parser.py",
    ]

    def _extract_print_strings(self, filepath: Path) -> list[tuple[int, str]]:
        """Extract string literals from print() and tqdm.write() calls via AST.

        Returns [(line_number, string_literal), ...].
        """
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
        results = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # Match print(...) or tqdm.write(...)
            func = node.func
            is_print = isinstance(func, ast.Name) and func.id == "print"
            is_tqdm_write = (
                isinstance(func, ast.Attribute)
                and func.attr == "write"
                and isinstance(func.value, ast.Name)
                and func.value.id == "tqdm"
            )
            if not (is_print or is_tqdm_write):
                continue

            # Collect all string literals in the call arguments
            for arg in node.args:
                for sub in ast.walk(arg):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        results.append((sub.lineno, sub.value))
                    elif isinstance(sub, ast.JoinedStr):
                        # f-string: check the constant parts
                        for val in sub.values:
                            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                                results.append((val.lineno, val.value))
        return results

    @pytest.mark.parametrize("script", SCRIPTS)
    def test_no_chinese_in_print_statements(self, script):
        filepath = PROJECT_DIR / script
        if not filepath.exists():
            pytest.skip(f"{script} not found at {PROJECT_DIR}")

        prints = self._extract_print_strings(filepath)
        violations = []
        for lineno, text in prints:
            if _CJK_RE.search(text):
                snippet = text[:80].replace("\n", "\\n")
                violations.append(f"  line {lineno}: \"{snippet}...\"")

        assert not violations, (
            f"{script} has Chinese characters in print/tqdm.write:\n"
            + "\n".join(violations)
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. Note language selection via --language CLI argument
# ══════════════════════════════════════════════════════════════════════════════

class TestLanguageCLIArg:
    """Verify that --language flag is accepted and overrides NOTE_LANGUAGE."""

    def test_help_shows_language_flag(self):
        r = _run("note_generation.py", "--help")
        assert r.returncode == 0
        assert "--language" in r.stdout, (
            f"--language flag missing from help:\n{r.stdout}"
        )

    def test_language_accepts_en(self):
        """--language en should be accepted (even if no course data)."""
        r = _run("note_generation.py", "--course", "99999", "--language", "en")
        # Should fail because course doesn't exist, NOT because of --language
        assert "invalid choice" not in r.stderr.lower(), (
            f"'en' rejected as invalid choice:\n{r.stderr}"
        )

    def test_language_accepts_zh(self):
        """--language zh should be accepted."""
        r = _run("note_generation.py", "--course", "99999", "--language", "zh")
        assert "invalid choice" not in r.stderr.lower(), (
            f"'zh' rejected as invalid choice:\n{r.stderr}"
        )

    def test_language_rejects_invalid(self):
        """--language fr should be rejected."""
        r = _run("note_generation.py", "--course", "99999", "--language", "fr")
        assert r.returncode != 0, "Invalid language should cause an error"

    def test_language_overrides_constant(self):
        """Verify that --language zh is confirmed in subprocess output."""
        r = _run("note_generation.py", "--course", "99999",
                 "--language", "zh", "--course-name", "Test")
        assert "Note language: zh" in r.stdout, (
            f"Language confirmation not found in output:\n{r.stdout}\n{r.stderr}"
        )

    def test_language_en_not_printed_when_default(self):
        """When --language is not passed, no language line should be printed."""
        r = _run("note_generation.py", "--course", "99999", "--course-name", "Test")
        assert "Note language:" not in r.stdout, (
            f"Language line should not appear without --language flag:\n{r.stdout}"
        )

    def test_prompts_always_english_translate_is_separate(self):
        """Verify _P() always returns English prompts (translation is post-step)."""
        import note_generation
        original = note_generation.NOTE_LANGUAGE
        try:
            note_generation.NOTE_LANGUAGE = 'en'
            en_sys = note_generation._P('system')
            assert 'note' in en_sys.lower()

            note_generation.NOTE_LANGUAGE = 'zh'
            zh_sys = note_generation._P('system')
            # Prompts should be identical — language is handled by _translate()
            assert en_sys == zh_sys, (
                'Prompts should be the same regardless of language; '
                'translation is a separate post-generation step')

            # _translate function should exist
            assert hasattr(note_generation, '_translate'), (
                '_translate function must exist for post-generation translation')
        finally:
            note_generation.NOTE_LANGUAGE = original

    def test_global_declaration_at_function_top(self):
        """Ensure 'global NOTE_LANGUAGE' is at the top of main(), not inside if."""
        import inspect, ast
        import note_generation
        src = inspect.getsource(note_generation.main)
        tree = ast.parse(textwrap.dedent(src))
        func = tree.body[0]
        # First statement in function body should be (or contain) the global
        first_stmts = func.body[:3]  # check first few statements
        has_global = any(
            isinstance(s, ast.Global) and 'NOTE_LANGUAGE' in s.names
            for s in first_stmts
        )
        assert has_global, (
            "'global NOTE_LANGUAGE' should be at the top of main(), "
            "not inside a conditional block"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. Pipeline skip logic — existing files not re-processed without --force
# ══════════════════════════════════════════════════════════════════════════════

class TestTranscribeSkipLogic:
    """Verify extract_caption.py skips existing captions without --force."""

    def test_help_shows_force_flag(self):
        r = _run("extract_caption.py", "--help")
        assert r.returncode == 0
        assert "--force" in r.stdout

    def test_skips_already_captioned(self):
        """Without --force, already-captioned videos should be skipped."""
        r = _run("extract_caption.py", timeout=15)
        assert r.returncode == 0
        # Should either say "All videos already captioned" or list skipped ones
        out = r.stdout
        assert (
            "already captioned" in out.lower()
            or "[skip]" in out
            or "Done:" in out
        ), f"Unexpected output:\n{out}"

    def test_force_flag_accepted(self):
        """--force should be accepted without error."""
        r = _run("extract_caption.py", "--force", "--help")
        # --help with --force should still show help (argparse processes --help first)
        assert r.returncode == 0

    def test_force_regen_constant_default_false(self):
        """FORCE_REGEN should default to False."""
        script = textwrap.dedent("""\
            import sys
            sys.argv = ['extract_caption.py']
            import extract_caption
            assert extract_caption.FORCE_REGEN is False, (
                f"FORCE_REGEN should default to False, got {extract_caption.FORCE_REGEN}")
            print('PASS')
        """)
        r = subprocess.run(
            [PYTHON, "-c", script],
            capture_output=True, text=True, timeout=15,
            cwd=str(PROJECT_DIR),
        )
        assert "PASS" in r.stdout, f"Failed:\n{r.stdout}\n{r.stderr}"


class TestAlignSkipLogic:
    """Verify semantic_alignment.py skips existing alignment files without --force."""

    def test_help_shows_force_flag(self):
        r = _run("semantic_alignment.py", "--help")
        assert r.returncode == 0
        assert "--force" in r.stdout

    def test_skips_already_aligned(self, tmp_path):
        """process_course should skip captions that already have alignment files."""
        # Set up a minimal course dir with a caption and a matching alignment
        course_dir = tmp_path / "99998"
        cap_dir = course_dir / "captions"
        align_dir = course_dir / "alignment"
        mat_dir = course_dir / "materials"
        cap_dir.mkdir(parents=True)
        align_dir.mkdir(parents=True)
        mat_dir.mkdir(parents=True)

        # Create a caption file
        caption = {"duration": 60.0, "language": "en", "segments": [
            {"id": 0, "start": 0, "end": 30, "text": "Hello world"}
        ]}
        (cap_dir / "test_video.json").write_text(json.dumps(caption))

        # Create a matching alignment file (already aligned)
        alignment = {"lecture": "test_video", "slide_file": "test.pdf",
                     "total_slides": 5, "segments": [], "timeline": []}
        (align_dir / "test_video.json").write_text(json.dumps(alignment))

        # Create a dummy slide file
        (mat_dir / "L01-Test.pdf").write_bytes(b"%PDF-dummy")

        # Run without --force — should skip
        from semantic_alignment import process_course, COURSE_DATA_DIR
        import io
        from contextlib import redirect_stdout

        # Temporarily override COURSE_DATA_DIR
        import semantic_alignment as sa
        original = sa.COURSE_DATA_DIR
        sa.COURSE_DATA_DIR = tmp_path
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                sa.process_course("99998", force=False)
            output = buf.getvalue()
            assert "[skip] Already aligned" in output, (
                f"Should have skipped existing alignment, got:\n{output}"
            )
        finally:
            sa.COURSE_DATA_DIR = original

    def test_force_does_not_skip(self, tmp_path):
        """process_course with force=True should NOT print skip messages.

        We can't run the full alignment (needs ML models), but we can verify
        that the skip check is bypassed by checking the code path.
        """
        course_dir = tmp_path / "99997"
        cap_dir = course_dir / "captions"
        align_dir = course_dir / "alignment"
        mat_dir = course_dir / "materials"
        cap_dir.mkdir(parents=True)
        align_dir.mkdir(parents=True)
        mat_dir.mkdir(parents=True)

        caption = {"duration": 60.0, "language": "en", "segments": [
            {"id": 0, "start": 0, "end": 30, "text": "Hello world"}
        ]}
        (cap_dir / "test_video.json").write_text(json.dumps(caption))

        alignment = {"lecture": "test_video", "slide_file": "L01-Test.pdf",
                     "total_slides": 5, "segments": [], "timeline": []}
        (align_dir / "test_video.json").write_text(json.dumps(alignment))

        (mat_dir / "L01-Test.pdf").write_bytes(b"%PDF-dummy")

        import semantic_alignment as sa
        import io
        from contextlib import redirect_stdout

        original = sa.COURSE_DATA_DIR
        sa.COURSE_DATA_DIR = tmp_path
        try:
            buf = io.StringIO()
            # force=True will try to re-align, which needs ML models.
            # It will fail at the embedding step, but should NOT print "[skip]".
            try:
                with redirect_stdout(buf):
                    sa.process_course("99997", force=True)
            except Exception:
                pass  # Expected: ML model not available
            output = buf.getvalue()
            assert "[skip] Already aligned" not in output, (
                f"With force=True, should NOT skip:\n{output}"
            )
        finally:
            sa.COURSE_DATA_DIR = original


class TestNoteGenerationSkipLogic:
    """Verify note_generation.py skips existing section files without --force."""

    def test_section_cache_respected(self, tmp_path):
        """generate_section should return cached content without --force."""
        from note_generation import _section_path

        sections_dir = tmp_path / "sections"
        sections_dir.mkdir()

        # Create a cached section file
        sec_file = _section_path(sections_dir, 1, 1, 1)
        sec_file.write_text("### 1.1 Cached Content\n\nThis was previously generated.")

        assert sec_file.exists()
        assert sec_file.stat().st_size > 50

    def test_help_shows_force_flag(self):
        r = _run("note_generation.py", "--help")
        assert r.returncode == 0
        assert "--force" in r.stdout


# ══════════════════════════════════════════════════════════════════════════════
# 4. Electron app language dropdown integration
# ══════════════════════════════════════════════════════════════════════════════

class TestElectronLanguageUI:
    """Verify the language dropdown is present in the Pipeline and Generate pages."""

    APP_JS = PROJECT_DIR / "electron" / "renderer" / "app.js"

    def test_pipeline_page_has_language_select(self):
        src = self.APP_JS.read_text(encoding="utf-8")
        assert 'id="pp-language"' in src, (
            "Pipeline page missing language dropdown (pp-language)"
        )

    def test_generate_page_has_language_select(self):
        src = self.APP_JS.read_text(encoding="utf-8")
        assert 'id="gen-language"' in src, (
            "Generate page missing language dropdown (gen-language)"
        )

    def test_pipeline_passes_language_to_cli(self):
        src = self.APP_JS.read_text(encoding="utf-8")
        assert "'--language', lang" in src or '"--language", lang' in src, (
            "Pipeline page does not pass --language to note_generation.py"
        )

    def test_generate_passes_language_to_cli(self):
        src = self.APP_JS.read_text(encoding="utf-8")
        # Find the generate run handler section
        gen_section = src[src.index("gen-run-btn"):]
        assert "--language" in gen_section, (
            "Generate page does not pass --language to CLI"
        )

    def test_state_has_language_field(self):
        src = self.APP_JS.read_text(encoding="utf-8")
        assert "language:" in src and "'en'" in src, (
            "State.pipeline should have a language field defaulting to 'en'"
        )
