"""
Regression tests for the v0.13.0 Panopto split-stream download path.

When a Panopto recording stores screen video in OBJECT/SS (no audio) and the
microphone audio in a separate DV stream, downloader.py must:
  • Prefer the screen-recording stream as the VIDEO source
  • Pick the first audio-bearing stream as the AUDIO source
  • Use ffmpeg to merge them into one MP4 (so the frame extractor sees slides
    AND the transcript sees lecture audio)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def _select_split(candidates, hls_audio_lookup):
    """Reimplement downloader.py's video/audio classification.

    Mirrors the logic in download_video so tests don't need to monkeypatch
    Panopto auth, ffprobe, or the progress bar. ``candidates`` is a list of
    (url, headers, tag) and ``hls_audio_lookup`` maps url → bool (has audio).
    Returns (video_cand, audio_cand, merge_needed).
    """
    video_cand = audio_cand = None
    for cu, ch, ct in candidates:
        is_m3u8 = "master.m3u8" in cu
        has_audio = (not is_m3u8) or hls_audio_lookup.get(cu, True)
        if video_cand is None and ct in ("SS", "OBJECT"):
            video_cand = (cu, ch, ct, is_m3u8, has_audio)
        if audio_cand is None and has_audio:
            audio_cand = (cu, ch, ct, is_m3u8, has_audio)
    if video_cand is None and audio_cand is not None:
        video_cand = audio_cand
    merge = (
        video_cand is not None and audio_cand is not None
        and video_cand[0] != audio_cand[0]
        and video_cand[3] and audio_cand[3]
    )
    return video_cand, audio_cand, merge


class TestSplitStreamClassification:
    """Verify the OBJECT-video + DV-audio split-stream selector."""

    def test_object_video_dv_audio_merges(self):
        cands = [
            ("https://cdn/obj.master.m3u8", None, "OBJECT"),
            ("https://cdn/dv.master.m3u8",  None, "DV"),
        ]
        audio = {
            "https://cdn/obj.master.m3u8": False,   # screen has no audio
            "https://cdn/dv.master.m3u8":  True,    # camera carries audio
        }
        v, a, merge = _select_split(cands, audio)
        assert v[2] == "OBJECT"
        assert a[2] == "DV"
        assert merge is True

    def test_ss_with_audio_no_merge(self):
        """If the screen stream already has audio, no merge is needed."""
        cands = [
            ("https://cdn/ss.master.m3u8", None, "SS"),
            ("https://cdn/dv.master.m3u8", None, "DV"),
        ]
        audio = {
            "https://cdn/ss.master.m3u8": True,
            "https://cdn/dv.master.m3u8": True,
        }
        v, a, merge = _select_split(cands, audio)
        assert v[2] == "SS"
        assert a[2] == "SS"   # same stream → audio source is video source
        assert merge is False

    def test_dv_only_camera_recording(self):
        """No screen stream — fall back to DV as both video and audio."""
        cands = [("https://cdn/dv.master.m3u8", None, "DV")]
        audio = {"https://cdn/dv.master.m3u8": True}
        v, a, merge = _select_split(cands, audio)
        assert v[2] == "DV"
        assert a[2] == "DV"
        assert merge is False

    def test_screen_without_audio_anywhere_returns_no_audio(self):
        """OBJECT exists but no candidate has audio — caller should error."""
        cands = [
            ("https://cdn/obj.master.m3u8", None, "OBJECT"),
            ("https://cdn/dv.master.m3u8",  None, "DV"),
        ]
        audio = {
            "https://cdn/obj.master.m3u8": False,
            "https://cdn/dv.master.m3u8":  False,
        }
        v, a, merge = _select_split(cands, audio)
        assert v is not None
        assert a is None
        assert merge is False

    def test_direct_url_treated_as_audio_bearing(self):
        """Non-HLS authenticated URLs skip the audio probe (always assumed audio)."""
        cands = [("https://cdn/dv.mp4?token=abc", {"Auth": "x"}, "DV")]
        v, a, merge = _select_split(cands, {})
        assert v[2] == "DV"
        assert merge is False
        assert v[4] is True   # has_audio


class TestMergeFunctionExists:
    """The new merge helper must be importable with the right signature."""

    def test_run_ffmpeg_hls_merge_signature(self):
        from downloader import _run_ffmpeg_hls_merge
        import inspect

        sig = inspect.signature(_run_ffmpeg_hls_merge)
        params = list(sig.parameters.keys())
        assert params == ["video_url", "audio_url", "out_path", "progress_cb"]

    def test_merge_raises_when_ffmpeg_missing(self, tmp_path):
        from downloader import _run_ffmpeg_hls_merge

        with patch("downloader._resolve_ffmpeg", return_value=None):
            with pytest.raises(RuntimeError, match="ffmpeg not found"):
                _run_ffmpeg_hls_merge(
                    "https://cdn/v.m3u8",
                    "https://cdn/a.m3u8",
                    tmp_path / "out.mp4",
                    lambda pct: None,
                )
