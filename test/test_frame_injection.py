"""
Regression tests for the v1.0.7 frame-injection safety net.

DeepSeek V4 Pro (and some other models) is inconsistent at embedding
`![Frame N](path)` markdown even when the prompt explicitly asks for
it — empirically observed dropping 23-of-23 frames on one EE2022
lecture and 0/3 on another. `_ensure_frames_embedded` post-processes
the draft and appends any frame in `img_map` that the LLM didn't cite.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def _make_slide(idx, label="Slide"):
    return SimpleNamespace(index=idx, label=label, text="", word_count=0,
                           has_code=False)


class TestEnsureFramesEmbedded:
    def test_appends_missing_frames_with_caption(self, tmp_path):
        from note_generation import _ensure_frames_embedded
        out_dir = tmp_path
        # Create dummy frame paths under out_dir
        frames_dir = out_dir / "images" / "L01_Foo"
        frames_dir.mkdir(parents=True)
        f1 = frames_dir / "frame_001.png"; f1.touch()
        f2 = frames_dir / "frame_002.png"; f2.touch()

        slides = [_make_slide(0), _make_slide(1)]
        img_map = {0: f1, 1: f2}
        img_cache = {
            "page_0": "Renewable Energy Integration – Power Electronic Convertors. Diagram.",
            "page_1": "Synchronous reactance equivalent circuit.",
        }
        # Draft with NO frame markdown — the bug case.
        draft = "Some Chinese 内容 about the topic."

        out = _ensure_frames_embedded(
            draft, slides, img_map, img_cache, out_dir, source="screenshare",
        )
        assert "![Frame 1](" in out
        assert "![Frame 2](" in out
        assert "Renewable Energy Integration" in out  # caption preserved
        assert "Synchronous reactance" in out

    def test_keeps_already_referenced_frames(self, tmp_path):
        from note_generation import _ensure_frames_embedded
        out_dir = tmp_path
        frames_dir = out_dir / "images" / "L01_Foo"
        frames_dir.mkdir(parents=True)
        f1 = frames_dir / "frame_001.png"; f1.touch()
        f2 = frames_dir / "frame_002.png"; f2.touch()

        slides = [_make_slide(0), _make_slide(1)]
        img_map = {0: f1, 1: f2}
        img_cache = {"page_0": "first", "page_1": "second"}
        # Draft already references frame 1 — don't duplicate it.
        draft = "Foo. ![Frame 1](images/L01_Foo/frame_001.png) *(first)* Bar."

        out = _ensure_frames_embedded(
            draft, slides, img_map, img_cache, out_dir, source="screenshare",
        )
        # Frame 1 not duplicated
        assert out.count("![Frame 1](") == 1
        # Frame 2 appended
        assert "![Frame 2](" in out

    def test_no_op_when_img_map_empty(self, tmp_path):
        from note_generation import _ensure_frames_embedded
        slides = [_make_slide(0), _make_slide(1)]
        draft = "Some prose."
        out = _ensure_frames_embedded(
            draft, slides, {}, {}, tmp_path, source="screenshare",
        )
        assert out == draft

    def test_uses_slide_prefix_for_slide_pdf_source(self, tmp_path):
        from note_generation import _ensure_frames_embedded
        out_dir = tmp_path
        slides_dir = out_dir / "images" / "L01_Foo"
        slides_dir.mkdir(parents=True)
        s1 = slides_dir / "slide_001.png"; s1.touch()

        slides = [_make_slide(0)]
        img_map = {0: s1}
        img_cache = {"page_0": "Architecture diagram."}
        draft = "Prose without images."

        out = _ensure_frames_embedded(
            draft, slides, img_map, img_cache, out_dir, source="slides",
        )
        # Slide-PDF source uses "Slide N" prefix, not "Frame N"
        assert "![Slide 1](" in out
        assert "![Frame 1](" not in out

    def test_caption_falls_back_to_index_when_no_description(self, tmp_path):
        from note_generation import _ensure_frames_embedded
        out_dir = tmp_path
        frames_dir = out_dir / "images" / "L01_Foo"
        frames_dir.mkdir(parents=True)
        f1 = frames_dir / "frame_007.png"; f1.touch()

        slides = [_make_slide(6)]   # 0-based index 6 → frame_007
        img_map = {6: f1}
        # Empty cache — no description available.
        out = _ensure_frames_embedded(
            "draft", slides, img_map, {}, out_dir, source="screenshare",
        )
        assert "![Frame 7](" in out
        assert "*(Frame 7)*" in out

    def test_long_description_truncated_at_first_sentence(self, tmp_path):
        from note_generation import _ensure_frames_embedded
        out_dir = tmp_path
        frames_dir = out_dir / "images" / "L01_Foo"
        frames_dir.mkdir(parents=True)
        f1 = frames_dir / "frame_001.png"; f1.touch()

        slides = [_make_slide(0)]
        img_map = {0: f1}
        # Two sentences — only the first should appear in the caption.
        img_cache = {"page_0": "First sentence about reactance. Second sentence about flux."}

        out = _ensure_frames_embedded(
            "draft", slides, img_map, img_cache, out_dir, source="screenshare",
        )
        # Caption text appears as `*(...)*` — find it
        import re
        m = re.search(r"\*\(([^)]+)\)\*", out)
        assert m is not None
        caption = m.group(1)
        assert "First sentence" in caption
        assert "Second sentence" not in caption
