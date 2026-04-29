"""
Regression tests for the v1.0.4 truncation fixes.

Two bugs caused DeepSeek-generated notes to be silently truncated:

  1. The verifier saw only ``draft[:2500]`` but its prompt asked it to
     return the "full note excerpt." For long drafts, the verifier's
     ~2.5K-char revision was accepted as the new draft, losing every-
     thing past the verify window. Result: 9K-char drafts collapsed to
     ~2K-char sections.

  2. ``_translate`` never propagated the truncation flag, so when
     gpt-4o's 16K output cap was hit during Chinese translation, the
     truncated text was silently cached. Tell-tale: sections ending mid
     image link (``![Frame 67](images``) or with the model's own
     ``...`` ellipsis.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))


class TestTranslatorChunking:
    def test_chunked_translate_preserves_paragraphs(self):
        from note_generation import _translate_chunked

        # Three paragraphs; mock _translate to echo "[T]" + content so we
        # can verify both the chunk boundaries and the rejoin order.
        text = "para1\n\npara2 with stuff\n\npara3 final"
        with patch("note_generation._translate", side_effect=lambda t, l: f"[T]{t}"):
            out = _translate_chunked(text, "Chinese", max_chunk_chars=15)
        # Each paragraph went through the mocked translator
        assert "[T]para1" in out
        assert "[T]para2 with stuff" in out
        assert "[T]para3 final" in out
        # Paragraph boundary preserved
        assert "\n\n" in out

    def test_chunked_translate_falls_back_to_english_on_chunk_failure(self):
        from note_generation import _translate_chunked

        def fake_translate(t, l):
            if "fail" in t:
                raise RuntimeError("simulated 429")
            return f"[T]{t}"

        text = "good para\n\nfail para\n\ngood again"
        with patch("note_generation._translate", side_effect=fake_translate):
            out = _translate_chunked(text, "Chinese", max_chunk_chars=15)
        # Successful chunks translated, failed chunk kept in English so
        # content isn't dropped silently.
        assert "[T]good para" in out
        assert "fail para" in out          # English kept
        assert "[T]good again" in out


class TestPickTranslateModel:
    def test_uses_note_model_when_translate_cap_is_smaller(self, monkeypatch):
        # When NOTE_MODEL is deepseek-v4-pro (384K cap) and TRANSLATE_MODEL
        # is gpt-4o (16K cap), translation should pick deepseek so long
        # Chinese outputs don't hit the gpt-4o ceiling.
        import note_generation as ng
        monkeypatch.setattr(ng, "NOTE_MODEL", "deepseek-v4-pro")
        monkeypatch.setattr(ng, "TRANSLATE_MODEL", "gpt-4o")
        assert ng._pick_translate_model() == "deepseek-v4-pro"

    def test_keeps_translate_model_when_caps_are_close(self, monkeypatch):
        import note_generation as ng
        monkeypatch.setattr(ng, "NOTE_MODEL", "gpt-4.1")      # 32K
        monkeypatch.setattr(ng, "TRANSLATE_MODEL", "gpt-4o")  # 16K
        # 32K vs 16K — only 2x, NOT > 2x. Keep TRANSLATE_MODEL since the
        # gap is small enough that switching wouldn't materially help.
        assert ng._pick_translate_model() == "gpt-4o"

    def test_passes_through_cli_models(self, monkeypatch):
        import note_generation as ng
        monkeypatch.setattr(ng, "NOTE_MODEL", "claude-cli")
        assert ng._pick_translate_model() == "claude-cli"
        monkeypatch.setattr(ng, "NOTE_MODEL", "codex-cli")
        assert ng._pick_translate_model() == "codex-cli"


class TestLooksTruncated:
    """Heuristic detector for the case where the provider returned
    finish_reason='stop' but the content was actually cut mid-stream.
    DeepSeek V4 chat completions hit this against real workloads — see
    /tmp/cs2105_L01_test.log for the in-the-wild example.
    """

    def test_clean_chinese_ending(self):
        from note_generation import _looks_truncated
        # Ends with `。` — proper Chinese sentence end
        assert _looks_truncated("内容齐全。") is False
        # Ends with markdown italic close after `。`
        assert _looks_truncated("内容齐全。*") is False
        # Western sentence end
        assert _looks_truncated("All good.") is False

    def test_broken_utf8_at_end(self):
        from note_generation import _looks_truncated
        assert _looks_truncated("现代商业产品通常会�") is True

    def test_mid_image_link_cut(self):
        from note_generation import _looks_truncated
        assert _looks_truncated("![Frame 67](images") is True
        assert _looks_truncated("foo bar\n\n![Slide 12](path/to/slide_") is True

    def test_mid_word_cut_ascii(self):
        from note_generation import _looks_truncated
        # The exact pathology from CS2105 L01 S04 — ends mid-word "pri"
        assert _looks_truncated("由 access、regional 和 global ISP 以及 pri") is True

    def test_complete_image_link_with_caption(self):
        from note_generation import _looks_truncated
        assert _looks_truncated(
            "![Slide 47](images/L01/slide_047.png) *(底线：something。)*"
        ) is False

    def test_empty_string(self):
        from note_generation import _looks_truncated
        assert _looks_truncated("") is True
        assert _looks_truncated("   \n  ") is True


class TestVerifierRemoved:
    """The verifier/revision pass was removed in v1.0.5 — modern flagship
    note models rarely make terminology mistakes worth a separate review
    call, and the limited verify window had been silently truncating
    long drafts. Make sure no constant or prompt template is left over.
    """

    def test_constants_gone(self):
        import note_generation as ng
        assert not hasattr(ng, "VERIFY_NOTES")
        assert not hasattr(ng, "VERIFY_MODEL")

    def test_verify_prompt_not_in_template(self):
        import note_generation as ng
        # Both language tables should have lost the "verify" key
        assert "verify" not in ng._PROMPTS["en"]
        if "zh" in ng._PROMPTS:
            assert "verify" not in ng._PROMPTS["zh"]

