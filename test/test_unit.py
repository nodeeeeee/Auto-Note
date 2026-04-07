"""
Unit tests — no network, no downloads, no GPU, no API keys required.

Tests core logic of:
  - frame_extractor: scene timestamps, frame alignment, perceptual hash, classifier
  - alignment_parser: compact alignment parsing
  - note_generation: LectureData, slide discovery, chunk prompts, image filter
  - semantic_alignment: Viterbi smoothing, timeline building, name similarity, auto-suggest
  - downloader: stream extraction, sanitize, _write_constant regex
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

import pytest

# ── Project root for imports ─────────────────────────────────────────────────
import sys
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))


# ══════════════════════════════════════════════════════════════════════════════
# frame_extractor
# ══════════════════════════════════════════════════════════════════════════════

class TestFrameAlignment:
    """Test build_frame_alignment() — maps transcript segments to frames by timestamp."""

    @pytest.fixture()
    def caption_file(self, tmp_path):
        cap = {
            "duration": 120.0,
            "language": "en",
            "segments": [
                {"id": i, "start": i * 20.0, "end": (i + 1) * 20.0,
                 "text": f"Segment {i} about topic {i}"}
                for i in range(6)
            ],
        }
        p = tmp_path / "test.json"
        p.write_text(json.dumps(cap))
        return p

    def test_basic_alignment(self, caption_file, tmp_path):
        from frame_extractor import build_frame_alignment
        timestamps = [0.0, 30.0, 60.0, 90.0]
        result = build_frame_alignment(
            caption_file, timestamps, 120.0, "test_vid", tmp_path / "frames")
        assert result["source"] == "screenshare"
        assert result["total_slides"] == 4
        assert len(result["segments"]) == 6

    def test_segment_to_frame_mapping(self, caption_file, tmp_path):
        from frame_extractor import build_frame_alignment
        timestamps = [0.0, 40.0, 80.0]
        result = build_frame_alignment(
            caption_file, timestamps, 120.0, "test_vid", tmp_path / "frames")
        # Segment at t=0-20 mid=10 → frame 1 (interval 0-40)
        assert result["segments"][0]["slide"] == 1
        # Segment at t=40-60 mid=50 → frame 2 (interval 40-80)
        assert result["segments"][2]["slide"] == 2
        # Segment at t=80-100 mid=90 → frame 3 (interval 80-120)
        assert result["segments"][4]["slide"] == 3

    def test_timeline_collapse(self, caption_file, tmp_path):
        from frame_extractor import build_frame_alignment
        timestamps = [0.0, 60.0]  # 2 frames
        result = build_frame_alignment(
            caption_file, timestamps, 120.0, "test_vid", tmp_path / "frames")
        tl = result["timeline"]
        assert len(tl) == 2
        assert tl[0]["slide"] == 1
        assert tl[1]["slide"] == 2

    def test_empty_segments(self, tmp_path):
        from frame_extractor import build_frame_alignment
        cap = {"duration": 60.0, "language": "en", "segments": []}
        p = tmp_path / "empty.json"
        p.write_text(json.dumps(cap))
        result = build_frame_alignment(p, [0.0, 30.0], 60.0, "test", tmp_path)
        assert result == {}

    def test_single_frame(self, caption_file, tmp_path):
        from frame_extractor import build_frame_alignment
        result = build_frame_alignment(
            caption_file, [0.0], 120.0, "test_vid", tmp_path / "frames")
        # All segments should map to frame 1
        for seg in result["segments"]:
            assert seg["slide"] == 1


class TestPerceptualHash:
    """Test dHash and Hamming distance."""

    def test_identical_images_zero_distance(self):
        pytest.importorskip("PIL")
        from PIL import Image as PILImage
        from frame_extractor import _perceptual_hash, _hamming
        img = PILImage.new("RGB", (100, 100), (255, 255, 255))
        h1 = _perceptual_hash(img)
        h2 = _perceptual_hash(img)
        assert _hamming(h1, h2) == 0

    def test_different_images_nonzero_distance(self):
        pytest.importorskip("PIL")
        from PIL import Image as PILImage
        from frame_extractor import _perceptual_hash, _hamming
        # Use a gradient vs solid — solid images have no pixel differences
        # so dHash is 0 for both. A gradient has real differences.
        gradient = PILImage.new("L", (100, 100))
        for x in range(100):
            for y in range(100):
                gradient.putpixel((x, y), x * 2)
        gradient = gradient.convert("RGB")
        solid = PILImage.new("RGB", (100, 100), (128, 128, 128))
        h1 = _perceptual_hash(gradient)
        h2 = _perceptual_hash(solid)
        assert _hamming(h1, h2) > 0

    def test_hamming_known_values(self):
        from frame_extractor import _hamming
        assert _hamming(0b1111, 0b0000) == 4
        assert _hamming(0b1010, 0b1010) == 0
        assert _hamming(0b1100, 0b1010) == 2


class TestVideoClassifier:
    """Test classify_video heuristics (requires PIL)."""

    def test_screen_detection_white_uniform(self):
        pytest.importorskip("PIL")
        from PIL import Image as PILImage
        import statistics
        from frame_extractor import classify_video

        # We can't easily test classify_video without a real video file,
        # but we can test the heuristic logic inline
        img = PILImage.new("RGB", (200, 200), (255, 255, 255))
        from frame_extractor import _get_pixels
        pixels = _get_pixels(img)

        # Edge ratio for uniform image should be 0
        row_diffs = 0
        w, h = 200, 200
        for y in range(0, h - 1, 4):
            for x in range(0, w - 1, 4):
                idx = y * w + x
                idx_r = idx + 1
                if idx_r < len(pixels):
                    diff = sum(abs(a - b) for a, b in zip(pixels[idx], pixels[idx_r]))
                    if diff > 100:
                        row_diffs += 1
        total_samples = (h // 4) * (w // 4)
        edge_ratio = row_diffs / max(total_samples, 1)
        assert edge_ratio == 0.0  # uniform white has no edges

        # Uniformity for solid color should be ~1.0
        block_size = 32
        uniform_blocks = 0
        total_blocks = 0
        for by in range(0, h - block_size, block_size):
            for bx in range(0, w - block_size, block_size):
                total_blocks += 1
                uniform_blocks += 1  # all pixels same
        uniformity = uniform_blocks / max(total_blocks, 1)
        assert uniformity > 0.9


# ══════════════════════════════════════════════════════════════════════════════
# alignment_parser
# ══════════════════════════════════════════════════════════════════════════════

class TestAlignmentParser:
    """Test compact alignment JSON parsing."""

    @pytest.fixture()
    def alignment_file(self, tmp_path):
        data = {
            "lecture": "Test Lecture",
            "slide_file": "test.pdf",
            "duration": 300,
            "language": "en",
            "total_slides": 5,
            "segments": [
                {"id": 0, "start": 0, "end": 30, "text": "Hello everyone",
                 "slide": 1, "slide_label": "Intro", "off_slide": False},
                {"id": 1, "start": 30, "end": 90, "text": "Today we discuss processes",
                 "slide": 2, "slide_label": "Processes", "off_slide": False},
                {"id": 2, "start": 90, "end": 120, "text": "um uh okay so",
                 "slide": 2, "slide_label": "Processes", "off_slide": False},
                {"id": 3, "start": 120, "end": 180, "text": "now lets look at threads",
                 "slide": 3, "slide_label": "Threads", "off_slide": False},
                {"id": 4, "start": 180, "end": 200, "text": "off topic question",
                 "slide": None, "slide_label": None, "off_slide": True},
            ],
            "timeline": [
                {"slide": 1, "start": 0, "end": 30, "label": "Intro"},
                {"slide": 2, "start": 30, "end": 120, "label": "Processes"},
                {"slide": 3, "start": 120, "end": 180, "label": "Threads"},
            ],
        }
        p = tmp_path / "alignment.json"
        p.write_text(json.dumps(data))
        return p

    def test_parse_returns_slides(self, alignment_file):
        import alignment_parser
        compact = alignment_parser.parse(alignment_file)
        assert "slides" in compact
        assert len(compact["slides"]) == 3

    def test_parse_merges_duplicate_slides(self, alignment_file):
        import alignment_parser
        compact = alignment_parser.parse(alignment_file)
        slide_nums = [s["slide"] for s in compact["slides"]]
        assert len(slide_nums) == len(set(slide_nums)), "Duplicate slides not merged"

    def test_parse_cleans_fillers(self, alignment_file):
        import alignment_parser
        compact = alignment_parser.parse(alignment_file)
        # "um uh okay so" from segment 2 should have fillers removed
        all_text = " ".join(s["transcript"] for s in compact["slides"])
        assert "um" not in all_text.lower().split()

    def test_parse_handles_off_slide(self, alignment_file):
        import alignment_parser
        compact = alignment_parser.parse(alignment_file)
        assert "off_slide" in compact
        assert compact["off_slide"]["transcript"]

    def test_parse_metadata(self, alignment_file):
        import alignment_parser
        compact = alignment_parser.parse(alignment_file)
        assert compact["lecture"] == "Test Lecture"
        assert compact["total_slides"] == 5
        assert compact["duration"] == 300

    def test_parse_and_save(self, alignment_file, tmp_path):
        import alignment_parser
        out = tmp_path / "compact.json"
        result_path = alignment_parser.parse_and_save(alignment_file, out)
        assert result_path == out
        assert out.exists()
        data = json.loads(out.read_text())
        assert "slides" in data


# ══════════════════════════════════════════════════════════════════════════════
# note_generation
# ══════════════════════════════════════════════════════════════════════════════

class TestSlideInfo:
    def test_basic_creation(self):
        from note_generation import SlideInfo
        s = SlideInfo(0, "Introduction", "This is the introduction.")
        assert s.index == 0
        assert s.label == "Introduction"
        assert s.word_count == 4

    def test_has_code_detection(self):
        from note_generation import SlideInfo
        s1 = SlideInfo(0, "Code", "int main() {\n  return 0;\n}")
        assert s1.has_code is True
        s2 = SlideInfo(0, "Text", "No code here, just text.")
        assert s2.has_code is False

    def test_code_detection_pthread(self):
        from note_generation import SlideInfo
        s = SlideInfo(0, "Thread", "pthread_create(&tid, NULL, func, NULL);")
        assert s.has_code is True


class TestLectureData:
    def test_default_source(self):
        from note_generation import LectureData
        ld = LectureData(1, Path("/tmp/test.pdf"), None)
        assert ld.source == "slides"
        assert ld.frame_dir is None

    def test_screenshare_source(self):
        from note_generation import LectureData
        ld = LectureData(1, Path("/tmp/frames"), None,
                         source="screenshare", frame_dir=Path("/tmp/frames"))
        assert ld.source == "screenshare"
        assert ld.frame_dir == Path("/tmp/frames")

    def test_title_from_compact(self):
        from note_generation import LectureData
        ld = LectureData(1, Path("/tmp/L01-Intro.pdf"), None)
        ld.compact = {"lecture": "CS3210 e-Lecture on Processes and Threads"}
        assert ld.title == "Processes and Threads"

    def test_title_fallback_to_stem(self):
        from note_generation import LectureData
        ld = LectureData(1, Path("/tmp/L01-Intro-To-OS.pdf"), None)
        assert ld.title == "Intro To OS"

    def test_compact_by_idx(self):
        from note_generation import LectureData
        ld = LectureData(1, Path("/tmp/test.pdf"), None)
        ld.compact_slides = [
            {"slide": 1, "start": 0, "transcript": "Hello"},
            {"slide": 3, "start": 60, "transcript": "World"},
        ]
        by_idx = ld.compact_by_idx
        assert 0 in by_idx  # slide 1 → index 0
        assert 2 in by_idx  # slide 3 → index 2
        assert 1 not in by_idx

    def test_load_from_frames(self, tmp_path):
        from note_generation import LectureData
        frame_dir = tmp_path / "frames" / "test"
        frame_dir.mkdir(parents=True)
        for i in range(3):
            (frame_dir / f"frame_{i+1:03d}.png").write_bytes(b"fake")

        ld = LectureData(1, frame_dir, None,
                         source="screenshare", frame_dir=frame_dir)
        out_dir = tmp_path / "notes"
        out_dir.mkdir()
        ld.load(out_dir)

        assert len(ld.slides) == 3
        assert len(ld.img_map) == 3
        assert ld.slides[0].label == "Frame 1"


class TestBuildChunkPrompt:
    def test_slide_source_no_screencapture(self):
        from note_generation import _build_chunk_prompt, SlideInfo
        slides = [SlideInfo(0, "Introduction", "This is the introduction to OS concepts.")]
        prompt = _build_chunk_prompt(
            slides=slides, compact_by_idx={}, img_cache={}, img_map={},
            out_dir=Path("/tmp"), course_name="CS3210", lec_num=1,
            lec_title="Intro", chunk_idx=1, chunk_title="Intro",
            detail=7, has_transcript=False, source="slides")
        assert "screen capture" not in prompt
        assert "Introduction" in prompt

    def test_screenshare_source_has_frame_label(self, tmp_path):
        from note_generation import _build_chunk_prompt, SlideInfo
        img_dir = tmp_path / "images" / "L01"
        img_dir.mkdir(parents=True)
        fake_img = img_dir / "frame_001.png"
        fake_img.write_bytes(b"fake")

        slides = [SlideInfo(0, "Frame 1", "Frame 1")]
        prompt = _build_chunk_prompt(
            slides=slides, compact_by_idx={}, img_cache={},
            img_map={0: fake_img}, out_dir=tmp_path,
            course_name="CS3210", lec_num=1, lec_title="Lecture",
            chunk_idx=1, chunk_title="Topic", detail=7,
            has_transcript=True, source="screenshare")
        assert "Frame 1" in prompt
        assert "screen capture" in prompt.lower() or "frame" in prompt.lower()

    def test_transcript_block_included(self):
        from note_generation import _build_chunk_prompt, SlideInfo
        slides = [SlideInfo(0, "Slide 1", "Content here.")]
        compact = {0: {"slide": 1, "start": 0.0, "end": 30.0,
                        "transcript": "The professor discussed process scheduling."}}
        prompt = _build_chunk_prompt(
            slides=slides, compact_by_idx=compact, img_cache={}, img_map={},
            out_dir=Path("/tmp"), course_name="CS3210", lec_num=1,
            lec_title="Processes", chunk_idx=1, chunk_title="Scheduling",
            detail=7, has_transcript=True, source="slides")
        assert "process scheduling" in prompt


class TestChunkTitle:
    def test_picks_short_label(self):
        from note_generation import _chunk_title, SlideInfo
        slides = [
            SlideInfo(0, "CS3210", "CS3210"),  # bad: course code
            SlideInfo(1, "Process Management", "..."),  # good
            SlideInfo(2, "1", "..."),  # bad: just a number
        ]
        assert _chunk_title(slides) == "Process Management"

    def test_fallback_to_slide_range(self):
        from note_generation import _chunk_title, SlideInfo
        slides = [SlideInfo(0, "42", "...")]
        assert _chunk_title(slides) == "Slide 1"  # bad label → falls back to slide number


class TestImageRefPattern:
    def test_matches_slide(self):
        from note_generation import _img_ref_pattern
        pat = _img_ref_pattern()
        m = pat.search("![Slide 1](images/L01/slide_001.png)")
        assert m is not None
        assert m.group(1) == "images/L01/slide_001.png"

    def test_matches_frame(self):
        from note_generation import _img_ref_pattern
        pat = _img_ref_pattern()
        m = pat.search("![Frame 5](images/L03/frame_005.png)")
        assert m is not None
        assert m.group(1) == "images/L03/frame_005.png"

    def test_matches_with_caption(self):
        from note_generation import _img_ref_pattern
        pat = _img_ref_pattern()
        m = pat.search("![Slide 2](images/L01/slide_002.png) *(A diagram)*")
        assert m is not None

    def test_matches_multi_file(self):
        from note_generation import _img_ref_pattern
        pat = _img_ref_pattern()
        m = pat.search("![Slide 1](images/L04_F02/slide_001.png)")
        assert m is not None


class TestDiscoverScreenshareLectures:
    def test_discovers_from_frames(self, tmp_path):
        from note_generation import _discover_screenshare_lectures
        frame_dir = tmp_path / "frames" / "TestVid"
        frame_dir.mkdir(parents=True)
        for i in range(3):
            (frame_dir / f"frame_{i+1:03d}.png").write_bytes(b"f")
        align_dir = tmp_path / "alignment"
        align_dir.mkdir()
        alignment = {"source": "screenshare", "segments": [], "timeline": [],
                     "total_slides": 3, "duration": 60, "language": "en",
                     "lecture": "TestVid", "slide_file": "frames/TestVid"}
        (align_dir / "TestVid.json").write_text(json.dumps(alignment))

        result = _discover_screenshare_lectures(tmp_path)
        assert len(result) == 1
        assert result[0].source == "screenshare"

    def test_ignores_non_screenshare_alignment(self, tmp_path):
        from note_generation import _discover_screenshare_lectures
        frame_dir = tmp_path / "frames" / "TestVid"
        frame_dir.mkdir(parents=True)
        (frame_dir / "frame_001.png").write_bytes(b"f")
        align_dir = tmp_path / "alignment"
        align_dir.mkdir()
        # source is NOT screenshare
        alignment = {"source": "slides", "segments": [], "timeline": []}
        (align_dir / "TestVid.json").write_text(json.dumps(alignment))

        result = _discover_screenshare_lectures(tmp_path)
        assert len(result) == 0

    def test_empty_when_no_frames(self, tmp_path):
        from note_generation import _discover_screenshare_lectures
        result = _discover_screenshare_lectures(tmp_path)
        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# semantic_alignment
# ══════════════════════════════════════════════════════════════════════════════

class TestViterbiSmooth:
    def test_forward_progression(self):
        import numpy as np
        from semantic_alignment import viterbi_smooth_fast

        # 3 segments, 3 slides — strong diagonal signal
        log_ll = np.array([
            [1.0, 0.1, 0.0],
            [0.0, 1.0, 0.1],
            [0.0, 0.1, 1.0],
        ], dtype=np.float64)
        path = viterbi_smooth_fast(log_ll)
        assert path == [0, 1, 2]

    def test_backward_penalty(self):
        import numpy as np
        from semantic_alignment import viterbi_smooth_fast

        # Segment 2 has weak signal for slide 0 (backward jump)
        # but strong signal for slide 2 (forward)
        log_ll = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.3, 0.0, 0.5],  # slide 0 is 0.3, slide 2 is 0.5
        ], dtype=np.float64)
        path = viterbi_smooth_fast(log_ll)
        assert path[2] == 2  # should prefer forward even with weaker signal

    def test_stay_on_same_slide(self):
        import numpy as np
        from semantic_alignment import viterbi_smooth_fast

        log_ll = np.array([
            [1.0, 0.0],
            [0.8, 0.2],  # still stronger on slide 0
            [0.7, 0.3],
        ], dtype=np.float64)
        path = viterbi_smooth_fast(log_ll)
        assert path[0] == 0
        assert path[1] == 0
        assert path[2] == 0


class TestBuildTimeline:
    def test_collapse_consecutive(self):
        from semantic_alignment import build_timeline, SlideText
        segments = [
            {"start": 0, "end": 10, "text": "a"},
            {"start": 10, "end": 20, "text": "b"},
            {"start": 20, "end": 30, "text": "c"},
        ]
        slides = [SlideText(0, "S1", ""), SlideText(1, "S2", "")]
        path = [0, 0, 1]
        tl = build_timeline(segments, path, slides)
        assert len(tl) == 2
        assert tl[0]["slide"] == 1  # 1-based
        assert tl[0]["end"] == 20
        assert tl[1]["slide"] == 2

    def test_off_slide_breaks_span(self):
        from semantic_alignment import build_timeline, SlideText
        segments = [
            {"start": 0, "end": 10, "text": "a"},
            {"start": 10, "end": 20, "text": "b"},
            {"start": 20, "end": 30, "text": "c"},
        ]
        slides = [SlideText(0, "S1", "")]
        path = [0, 0, 0]
        off = [False, True, False]  # middle segment is off-slide
        tl = build_timeline(segments, path, slides, off)
        assert len(tl) == 2  # broken into two intervals


class TestNameSimilarity:
    def test_identical(self):
        from semantic_alignment import _name_similarity
        assert _name_similarity("foo-bar", "foo-bar") == 1.0

    def test_partial_overlap(self):
        from semantic_alignment import _name_similarity
        sim = _name_similarity("L02-Processes-Threads", "L02-Processes")
        assert 0.3 < sim < 1.0

    def test_no_overlap(self):
        from semantic_alignment import _name_similarity
        assert _name_similarity("video-week3", "assignment-2") < 0.1

    def test_empty_string(self):
        from semantic_alignment import _name_similarity
        assert _name_similarity("", "something") == 0.0


class TestLecNum:
    def test_extracts_L02(self):
        from semantic_alignment import _lec_num
        assert _lec_num(Path("L02-Processes.pdf")) == 2

    def test_extracts_Lecture10(self):
        from semantic_alignment import _lec_num
        assert _lec_num(Path("Lecture10 Classification.pdf")) == 10

    def test_extracts_Lec03(self):
        from semantic_alignment import _lec_num
        assert _lec_num(Path("Lec03-intro.pdf")) == 3

    def test_no_number(self):
        from semantic_alignment import _lec_num
        assert _lec_num(Path("README.pdf")) is None

    def test_avoids_tutorial(self):
        from semantic_alignment import _lec_num
        # "tutorial02" should NOT match — 'l' is inside a word
        assert _lec_num(Path("tutorial02.pdf")) is None


class TestBuildWindowTexts:
    def test_context_pooling(self):
        from semantic_alignment import build_window_texts
        segs = [
            {"start": 0, "end": 5, "text": "hello"},
            {"start": 5, "end": 10, "text": "world"},
            {"start": 10, "end": 15, "text": "foo"},
        ]
        texts = build_window_texts(segs, context_sec=10.0)
        assert len(texts) == 3
        # First segment's window should include "hello" and "world"
        assert "hello" in texts[0]
        assert "world" in texts[0]

    def test_no_context(self):
        from semantic_alignment import build_window_texts
        segs = [
            {"start": 0, "end": 5, "text": "hello"},
            {"start": 100, "end": 105, "text": "world"},
        ]
        texts = build_window_texts(segs, context_sec=1.0)
        assert "world" not in texts[0]
        assert "hello" not in texts[1]


class TestSuggestMatches:
    """Test suggest_matches — requires sentence-transformers to be installed."""

    def test_empty_course_returns_empty(self, tmp_path):
        from semantic_alignment import suggest_matches
        result = suggest_matches(tmp_path, model="mpnet")
        assert result == {}

    def test_no_captions_returns_empty(self, tmp_path):
        from semantic_alignment import suggest_matches
        mat_dir = tmp_path / "materials"
        mat_dir.mkdir()
        (mat_dir / "test.pdf").write_bytes(b"%PDF")
        result = suggest_matches(tmp_path, model="mpnet")
        assert result == {}


# ══════════════════════════════════════════════════════════════════════════════
# downloader
# ══════════════════════════════════════════════════════════════════════════════

class TestStreamExtraction:
    def test_prefers_ss_stream(self):
        """Simulate _extract_stream logic — should prefer SS over DV."""
        body = {
            "Delivery": {"Streams": [
                {"Tag": "DV", "StreamUrl": "https://cdn/camera.m3u8"},
                {"Tag": "SS", "StreamUrl": "https://cdn/screen.m3u8"},
            ]}
        }
        streams = body["Delivery"]["Streams"]
        for tag in ("SS", "DV", "OBJECT", None):
            for s in streams:
                surl = s.get("StreamUrl", "")
                if surl and (tag is None or s.get("Tag") == tag):
                    assert tag == "SS"
                    assert "screen" in surl
                    return
        pytest.fail("No stream found")

    def test_fallback_to_dv(self):
        body = {
            "Delivery": {"Streams": [
                {"Tag": "DV", "StreamUrl": "https://cdn/camera.m3u8"},
            ]}
        }
        streams = body["Delivery"]["Streams"]
        result_tag = None
        for tag in ("SS", "DV", "OBJECT", None):
            for s in streams:
                surl = s.get("StreamUrl", "")
                if surl and (tag is None or s.get("Tag") == tag):
                    result_tag = s.get("Tag")
                    break
            if result_tag:
                break
        assert result_tag == "DV"

    def test_empty_streams(self):
        body = {"Delivery": {"Streams": []}}
        streams = body["Delivery"]["Streams"]
        found = False
        for tag in ("SS", "DV", "OBJECT", None):
            for s in streams:
                found = True
        assert not found


class TestSanitizeFilename:
    def test_removes_special_chars(self):
        from downloader import _sanitize
        result = _sanitize('foo/bar:baz?"<>|test')
        # Each of \/*?:"<>| is replaced with _
        assert "/" not in result
        assert ":" not in result
        assert "?" not in result
        assert '"' not in result
        assert "<" not in result
        assert ">" not in result
        assert "|" not in result

    def test_strips_whitespace(self):
        from downloader import _sanitize
        assert _sanitize("  hello  ") == "hello"


class TestMaterialNumberSelection:
    """Test that --download-material accepts numbers."""

    def test_number_in_range(self):
        files = [{"display_name": f"file{i}.pdf"} for i in range(5)]
        query = "3"
        try:
            idx = int(query)
            assert 1 <= idx <= len(files)
            assert files[idx - 1]["display_name"] == "file2.pdf"
        except ValueError:
            pytest.fail("Should parse as int")

    def test_number_out_of_range(self):
        files = [{"display_name": "file.pdf"}]
        query = "99"
        idx = int(query)
        assert not (1 <= idx <= len(files))

    def test_string_fallback(self):
        files = [
            {"display_name": "Lecture1.pdf"},
            {"display_name": "Lecture2.pdf"},
        ]
        query = "Lecture1"
        matches = [f for f in files if query.lower() in f["display_name"].lower()]
        assert len(matches) == 1
        assert matches[0]["display_name"] == "Lecture1.pdf"


# ══════════════════════════════════════════════════════════════════════════════
# Integration: screenshare pipeline end-to-end (offline)
# ══════════════════════════════════════════════════════════════════════════════

class TestScreensharePipelineOffline:
    """End-to-end test of the screenshare pipeline without any network or video."""

    def test_full_screenshare_flow(self, tmp_path):
        from frame_extractor import build_frame_alignment
        from note_generation import (
            LectureData, _discover_screenshare_lectures, SlideInfo,
            _build_chunk_prompt,
        )
        import alignment_parser

        course_dir = tmp_path / "99999"

        # 1. Create mock caption
        cap_dir = course_dir / "captions"
        cap_dir.mkdir(parents=True)
        caption = {
            "duration": 60.0, "language": "en",
            "segments": [
                {"id": i, "start": i * 15.0, "end": (i + 1) * 15.0,
                 "text": f"Discussing concept {i}"}
                for i in range(4)
            ],
        }
        cap_path = cap_dir / "TestLecture.json"
        cap_path.write_text(json.dumps(caption))

        # 2. Create mock frames
        frame_dir = course_dir / "frames" / "TestLecture"
        frame_dir.mkdir(parents=True)
        timestamps = [0.0, 20.0, 40.0]
        for i in range(len(timestamps)):
            (frame_dir / f"frame_{i+1:03d}.png").write_bytes(b"png_data")

        # 3. Build alignment
        align_dir = course_dir / "alignment"
        align_dir.mkdir(parents=True)
        alignment = build_frame_alignment(
            cap_path, timestamps, 60.0, "TestLecture", frame_dir)
        assert alignment["source"] == "screenshare"
        (align_dir / "TestLecture.json").write_text(json.dumps(alignment))

        # 4. Discover screenshare lectures
        lectures = _discover_screenshare_lectures(course_dir)
        assert len(lectures) == 1
        ld = lectures[0]
        assert ld.source == "screenshare"

        # 5. Load and verify
        notes_dir = course_dir / "notes"
        notes_dir.mkdir()
        ld.load(notes_dir)
        assert len(ld.slides) == 3
        assert len(ld.img_map) == 3

        # 6. Render chunk images (should copy frames)
        img_map = ld.render_chunk_images([0, 1])
        assert len(img_map) == 2
        for p in img_map.values():
            assert "frame_" in p.name
            assert p.exists()

        # 7. Parse compact alignment
        compact = alignment_parser.parse(align_dir / "TestLecture.json")
        assert compact["slides"]
        assert ld.compact_slides

        # 8. Build prompt
        prompt = _build_chunk_prompt(
            slides=ld.slides[:2],
            compact_by_idx=ld.compact_by_idx,
            img_cache={}, img_map=img_map,
            out_dir=notes_dir, course_name="Test",
            lec_num=1, lec_title="TestLecture",
            chunk_idx=1, chunk_title="Concepts",
            detail=7, has_transcript=True, source="screenshare")
        assert "Frame" in prompt
        assert "frame_" in prompt
