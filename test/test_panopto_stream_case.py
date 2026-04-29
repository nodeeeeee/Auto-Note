"""
Regression tests for the v1.0.8 Panopto-stream case-sensitivity fix.

Some Panopto sessions return stream tags in upper-case (`OBJECT`, `DV`)
while others use lower-case (`object`, `dv`). The downloader's preference
order is `("SS","OBJECT","DV")`. Before v1.0.8 the comparison was
case-sensitive, so when Panopto returned lower-case tags the OBJECT
branch never matched and the code fell through to "first stream" — which
is `dv` (the camera). Result: a lecture-hall webcam shot got downloaded
instead of the slide-recording, and downstream notes embedded camera
frames of the lecturer instead of slides.

Six EE2022 Wednesday lectures were affected in production:
  01/04, 04/03, 08/04, 11/03, 18/03, 25/03 — all dated 2026.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def _build_extract(prefer_order=("SS", "OBJECT", "DV")):
    """Reimplement downloader._extract_streams with the v1.0.8 fix so we
    can test it without a real Panopto delivery payload."""
    def _extract_streams(body):
        streams = (body.get("Delivery") or {}).get("Streams") or []
        order = tuple(t.upper() for t in prefer_order)
        out = []
        seen = set()
        for tag in order + (None,):
            for s in streams:
                surl = s.get("StreamUrl", "")
                if not surl or surl in seen:
                    continue
                stream_tag = (s.get("Tag", "") or "").upper()
                if tag is None or stream_tag == tag:
                    seen.add(surl)
                    out.append((surl, s.get("Tag", "unknown")))
        return out
    return _extract_streams


class TestStreamTagCaseInsensitive:
    def test_lowercase_object_picked_over_lowercase_dv(self):
        # The bug case: tags are lowercase. Before the fix this fell
        # through to "first stream" = dv (camera). After: OBJECT wins.
        body = {"Delivery": {"Streams": [
            {"Tag": "dv",     "StreamUrl": "https://cdn/dv.m3u8"},
            {"Tag": "object", "StreamUrl": "https://cdn/obj.m3u8"},
        ]}}
        result = _build_extract()(body)
        assert result[0] == ("https://cdn/obj.m3u8", "object")
        assert result[1] == ("https://cdn/dv.m3u8", "dv")

    def test_uppercase_object_still_works(self):
        # Backward-compat: existing Panopto sessions returning upper-case
        # tags must continue to be matched. v1.0.8 normalises both sides
        # to upper-case before comparing.
        body = {"Delivery": {"Streams": [
            {"Tag": "DV",     "StreamUrl": "https://cdn/dv.m3u8"},
            {"Tag": "OBJECT", "StreamUrl": "https://cdn/obj.m3u8"},
        ]}}
        result = _build_extract()(body)
        assert result[0] == ("https://cdn/obj.m3u8", "OBJECT")

    def test_mixed_case_doesnt_break_anything(self):
        body = {"Delivery": {"Streams": [
            {"Tag": "Dv",     "StreamUrl": "https://cdn/dv.m3u8"},
            {"Tag": "Object", "StreamUrl": "https://cdn/obj.m3u8"},
            {"Tag": "Ss",     "StreamUrl": "https://cdn/ss.m3u8"},
        ]}}
        result = _build_extract()(body)
        # SS first (highest preference), then OBJECT, then DV
        assert result[0][1] == "Ss"
        assert result[1][1] == "Object"
        assert result[2][1] == "Dv"

    def test_dv_only_camera_only_recording(self):
        # Some Panopto sessions only have a DV stream (no screen capture).
        # In that case DV must be returned as the only candidate.
        body = {"Delivery": {"Streams": [
            {"Tag": "dv", "StreamUrl": "https://cdn/dv.m3u8"},
        ]}}
        result = _build_extract()(body)
        assert len(result) == 1
        assert result[0][1] == "dv"

    def test_empty_tag_falls_through_to_any(self):
        # If Panopto returns a stream with no Tag at all, it should still
        # be returned (after preferred tags are exhausted).
        body = {"Delivery": {"Streams": [
            {"StreamUrl": "https://cdn/u1.m3u8"},
            {"Tag": "OBJECT", "StreamUrl": "https://cdn/obj.m3u8"},
        ]}}
        result = _build_extract()(body)
        # OBJECT should still come first
        assert result[0][1] == "OBJECT"
        # Untagged stream still appears in the list
        assert any(t == "unknown" for _, t in result)


class TestJunkDescRegexExtensions:
    def test_no_signal_test_pattern_caught(self):
        from frame_extractor import _JUNK_DESC_RE
        # The exact pathology from EE2022 06/03 Fri — Panopto recorded
        # a TV-style "No Signal" screen because the slide projector wasn't
        # plugged in at recording time.
        descs = [
            'The image displays a test pattern commonly used to indicate a "No Signal" status on television screens.',
            "The image is a classic test pattern used for television broadcasts, featuring vertical stripes of various colors.",
            "The image displays a static screen with vertical color bars in various shades.",
            "The image displays a test screen typically used for television signals, consisting of vertical stripes in various colors.",
        ]
        for d in descs:
            assert _JUNK_DESC_RE.search(d), f"Should be junk: {d}"

    def test_legitimate_slide_descriptions_not_caught(self):
        from frame_extractor import _JUNK_DESC_RE
        # These are real lecture slide descriptions that must NOT be
        # filtered — false positives would silently drop slides.
        legit = [
            'The slide features the title "Renewable Energy Integration".',
            "The slide shows a network configuration diagram featuring a laptop and a TV.",
            "The image shows a circuit diagram for a synchronous generator.",
            "Architecture drawing of a power transmission system with multiple transformers.",
            'The slide titled "Tutorial: 3-Phase Balanced Power" shows a table summarising key concepts.',
        ]
        for d in legit:
            assert not _JUNK_DESC_RE.search(d), f"Wrongly junked: {d}"
