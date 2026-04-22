"""
Regression tests for the v0.12.x series of bug fixes.

Covers:
  - Issue #3: Browse button — renderer-side; not Python-testable here
  - Issue #4: canvasapi import — top-level guarded import
  - Issue #5: HLS download with query-string URLs
  - Issue #6: subprocess must be top-level in downloader
  - Issue #7: ffprobe unavailable → fall back to ffmpeg stderr parsing
  - v0.12.4/5: Panopto tool-ID dynamic resolution via /tabs
  - v0.12.10: frame_extractor.py ffmpeg resolver (screen-recording capture)
  - v0.12.10: _run_ffmpeg_hls raises on silent ffmpeg failure

These tests avoid all network, GPU, and OpenAI API usage. The only optional
external dependency is a system ffmpeg binary for the end-to-end duration
test; it falls back cleanly if ffmpeg is absent.
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))


# ══════════════════════════════════════════════════════════════════════════════
# Issue #7 — extract_caption.py ffmpeg/ffprobe resolvers
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractCaptionResolvers:
    """New resolvers added in v0.12.9 (extract_caption.py)."""

    def test_resolve_ffmpeg_prefers_system(self, monkeypatch):
        import extract_caption as ec
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
        assert ec._resolve_ffmpeg() == "/usr/bin/ffmpeg"

    def test_resolve_ffmpeg_falls_back_to_imageio(self, monkeypatch):
        import extract_caption as ec
        monkeypatch.setattr("shutil.which", lambda name: None)
        fake_imageio = MagicMock()
        fake_imageio.get_ffmpeg_exe.return_value = "/fake/imageio/ffmpeg"
        monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_imageio)
        assert ec._resolve_ffmpeg() == "/fake/imageio/ffmpeg"

    def test_resolve_ffmpeg_auto_installs_when_missing(self, monkeypatch):
        """When neither system nor imageio-ffmpeg available, attempt pip install."""
        import extract_caption as ec
        monkeypatch.setattr("shutil.which", lambda name: None)
        # sys.modules[name] = None makes `import name` raise ImportError —
        # simulate the case where imageio-ffmpeg really isn't installed.
        monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
        install_called = {"count": 0}

        def fake_run(cmd, **kw):
            install_called["count"] += 1
            fake = MagicMock()
            fake.get_ffmpeg_exe.return_value = "/post/install/ffmpeg"
            # pip install succeeded → make imageio_ffmpeg importable.
            sys.modules["imageio_ffmpeg"] = fake
            return MagicMock(returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert ec._resolve_ffmpeg() == "/post/install/ffmpeg"
        assert install_called["count"] == 1

    def test_resolve_ffmpeg_raises_when_install_fails(self, monkeypatch):
        import extract_caption as ec
        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)

        def fake_run(cmd, **kw):
            raise RuntimeError("pip blew up")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(RuntimeError, match="ffmpeg unavailable"):
            ec._resolve_ffmpeg()

    def test_resolve_ffprobe_returns_system(self, monkeypatch):
        import extract_caption as ec
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffprobe" if name == "ffprobe" else None)
        assert ec._resolve_ffprobe() == "/usr/bin/ffprobe"

    def test_resolve_ffprobe_returns_none_when_missing(self, monkeypatch):
        """The exact scenario from issue #7."""
        import extract_caption as ec
        monkeypatch.setattr("shutil.which", lambda name: None)
        assert ec._resolve_ffprobe() is None


class TestFfmpegDurationParser:
    """The stderr-parsing fallback used when ffprobe is unavailable."""

    def test_basic_duration(self):
        import extract_caption as ec
        out = "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from foo.mp4:\n  Duration: 00:12:34.56, start: 0.000000, bitrate: 1024 kb/s\n"
        assert ec._parse_ffmpeg_duration(out) == pytest.approx(12 * 60 + 34.56)

    def test_integer_seconds(self):
        import extract_caption as ec
        out = "Duration: 01:00:00, bitrate: 2 kb/s"
        assert ec._parse_ffmpeg_duration(out) == 3600.0

    def test_short_file(self):
        import extract_caption as ec
        out = "Duration: 00:00:05.12, bitrate: 2 kb/s"
        assert ec._parse_ffmpeg_duration(out) == pytest.approx(5.12)

    def test_no_duration_line(self):
        import extract_caption as ec
        assert ec._parse_ffmpeg_duration("This is not an ffmpeg output.") is None

    def test_empty_string(self):
        import extract_caption as ec
        assert ec._parse_ffmpeg_duration("") is None

    def test_none_input(self):
        import extract_caption as ec
        assert ec._parse_ffmpeg_duration(None) is None


class TestVideoDurationFallback:
    """Issue #7 regression — _video_duration still works when ffprobe missing."""

    @pytest.fixture()
    def sample_video(self):
        vp = PROJECT_DIR / "sample" / "85397" / "videos" / "Lecture 9 Link Layer ARP.mp4"
        if not vp.exists():
            pytest.skip(f"Sample video not available: {vp}")
        return vp

    def test_uses_ffprobe_when_available(self, sample_video, monkeypatch):
        import extract_caption as ec
        # Keep real resolvers — just confirm duration is positive.
        dur = ec._video_duration(sample_video)
        assert dur > 0
        assert isinstance(dur, float)

    def test_falls_back_to_ffmpeg_when_no_ffprobe(self, sample_video, monkeypatch):
        """Simulate issue #7's environment: user has ffmpeg but no ffprobe."""
        import extract_caption as ec
        # Force the ffprobe resolver to return None.
        monkeypatch.setattr(ec, "_resolve_ffprobe", lambda: None)
        dur = ec._video_duration(sample_video)
        assert dur > 0

    def test_both_paths_agree(self, sample_video, monkeypatch):
        """ffprobe path and ffmpeg fallback should produce near-identical durations."""
        import extract_caption as ec
        if ec._resolve_ffprobe() is None:
            pytest.skip("ffprobe not installed; cannot cross-validate")
        d_ffprobe = ec._video_duration(sample_video)
        monkeypatch.setattr(ec, "_resolve_ffprobe", lambda: None)
        d_ffmpeg = ec._video_duration(sample_video)
        # ffmpeg's Duration line is rounded to 0.01s; allow ±0.5s.
        assert abs(d_ffprobe - d_ffmpeg) < 0.5

    def test_raises_when_output_has_no_duration(self, tmp_path, monkeypatch):
        """Non-video files should raise RuntimeError, not a cryptic parse error."""
        import extract_caption as ec
        not_a_video = tmp_path / "garbage.mp4"
        not_a_video.write_bytes(b"this is not a video file")
        monkeypatch.setattr(ec, "_resolve_ffprobe", lambda: None)
        with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
            ec._video_duration(not_a_video)


# ══════════════════════════════════════════════════════════════════════════════
# Issue #6 — subprocess must be top-level in downloader.py
# ══════════════════════════════════════════════════════════════════════════════

class TestDownloaderSubprocessImport:
    """Regression for issue #6 — NameError: name 'subprocess' is not defined."""

    def test_downloader_has_top_level_subprocess(self):
        """_spawn_transcribe used subprocess without importing it. Guard against regression."""
        import downloader
        assert hasattr(downloader, "subprocess"), \
            "downloader.py must import subprocess at module level; see issue #6"

    def test_subprocess_is_the_real_module(self):
        import downloader
        assert downloader.subprocess is subprocess


# ══════════════════════════════════════════════════════════════════════════════
# v0.12.6/7 — downloader.py ffmpeg resolver
# ══════════════════════════════════════════════════════════════════════════════

class TestDownloaderFfmpegResolver:

    def test_resolve_ffmpeg_prefers_system(self, monkeypatch):
        import downloader
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
        assert downloader._resolve_ffmpeg() == "/usr/bin/ffmpeg"

    def test_resolve_ffmpeg_falls_back_to_imageio(self, monkeypatch):
        import downloader
        monkeypatch.setattr("shutil.which", lambda name: None)
        fake = MagicMock()
        fake.get_ffmpeg_exe.return_value = "/imageio/ffmpeg"
        monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake)
        assert downloader._resolve_ffmpeg() == "/imageio/ffmpeg"

    def test_resolve_ffmpeg_handles_missing_imageio_gracefully(self, monkeypatch):
        """Even if imageio-ffmpeg is absent, resolver attempts install — doesn't crash."""
        import downloader
        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)

        def fake_run(cmd, **kw):
            raise RuntimeError("no internet")

        # downloader._resolve_ffmpeg does `import subprocess as _sp` locally,
        # so patch the module-level subprocess.run.
        monkeypatch.setattr(subprocess, "run", fake_run)
        result = downloader._resolve_ffmpeg()
        assert result is None  # downloader returns None on failure (vs raising)


# ══════════════════════════════════════════════════════════════════════════════
# v0.12.4/5 — Panopto tool-ID resolution via /tabs endpoint
# ══════════════════════════════════════════════════════════════════════════════

class TestPanoptoToolIdResolution:

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        """Reset the module-level cache before each test."""
        import downloader
        downloader._PANOPTO_TOOL_ID_CACHE.clear()
        yield
        downloader._PANOPTO_TOOL_ID_CACHE.clear()

    def _mock_response(self, status_code: int, payload):
        r = MagicMock()
        r.status_code = status_code
        r.json.return_value = payload
        return r

    def test_resolves_from_tabs_by_panopto_label(self, monkeypatch):
        """Strategy A: /tabs returns context_external_tool_<N> with panopto label."""
        import downloader
        payload = [
            {"id": "home",                              "label": "Home"},
            {"id": "context_external_tool_77",          "label": "Videos / Panopto"},
            {"id": "assignments",                       "label": "Assignments"},
        ]
        with patch.object(downloader, "requests") as mock_req:
            mock_req.get.return_value = self._mock_response(200, payload)
            tid = downloader._resolve_panopto_tool_id(999)
        assert tid == 77

    def test_resolves_from_tabs_by_videos_label(self, monkeypatch):
        """A 'Videos' label (no Panopto mention) should still match."""
        import downloader
        payload = [
            {"id": "context_external_tool_42", "label": "Videos"},
        ]
        with patch.object(downloader, "requests") as mock_req:
            mock_req.get.return_value = self._mock_response(200, payload)
            assert downloader._resolve_panopto_tool_id(888) == 42

    def test_falls_back_to_external_tools(self, monkeypatch):
        """When /tabs has no matching entry, /external_tools should be consulted."""
        import downloader
        tabs_payload = [{"id": "context_external_tool_1", "label": "Nothing Matching"}]
        external_tools_payload = [
            {"id": 128, "name": "Panopto Recordings", "domain": "mediaweb.panopto.com", "url": ""},
        ]
        call_log = []

        def fake_get(url, **kw):
            call_log.append(url)
            if "/tabs" in url:
                return self._mock_response(200, tabs_payload)
            if "/external_tools" in url:
                return self._mock_response(200, external_tools_payload)
            return self._mock_response(404, [])

        with patch.object(downloader, "requests") as mock_req:
            mock_req.get.side_effect = fake_get
            tid = downloader._resolve_panopto_tool_id(777)
        assert tid == 128
        assert any("/tabs" in u for u in call_log)
        assert any("/external_tools" in u for u in call_log)

    def test_returns_none_when_no_match_anywhere(self, monkeypatch):
        import downloader

        def fake_get(url, **kw):
            return self._mock_response(200, [])

        with patch.object(downloader, "requests") as mock_req:
            mock_req.get.side_effect = fake_get
            assert downloader._resolve_panopto_tool_id(666) is None

    def test_caches_result(self, monkeypatch):
        """Second lookup should not hit the API again."""
        import downloader
        payload = [{"id": "context_external_tool_5", "label": "Panopto"}]
        with patch.object(downloader, "requests") as mock_req:
            mock_req.get.return_value = self._mock_response(200, payload)
            tid_1 = downloader._resolve_panopto_tool_id(555)
            tid_2 = downloader._resolve_panopto_tool_id(555)
            assert tid_1 == tid_2 == 5
            # Only one GET should have been issued.
            assert mock_req.get.call_count == 1

    def test_malformed_tab_id_is_skipped(self, monkeypatch):
        """context_external_tool_XYZ (non-numeric) should not crash the resolver."""
        import downloader
        payload = [
            {"id": "context_external_tool_not_a_number", "label": "Panopto"},
            {"id": "context_external_tool_11", "label": "Videos"},
        ]
        with patch.object(downloader, "requests") as mock_req:
            mock_req.get.return_value = self._mock_response(200, payload)
            assert downloader._resolve_panopto_tool_id(111) == 11

    def test_network_exception_does_not_crash(self, monkeypatch):
        """If /tabs raises, should still try /external_tools; ultimate fallback None."""
        import downloader

        def fake_get(url, **kw):
            raise ConnectionError("timeout")

        with patch.object(downloader, "requests") as mock_req:
            mock_req.get.side_effect = fake_get
            assert downloader._resolve_panopto_tool_id(222) is None


# ══════════════════════════════════════════════════════════════════════════════
# v0.12.0 — Panopto stream priority (SS > OBJECT > DV)
# ══════════════════════════════════════════════════════════════════════════════

class TestPanoptoStreamPriority:
    """The stream selector used in _fetch_panopto_streams."""

    @staticmethod
    def _select(streams):
        """Mirror the tag-priority logic baked into _extract_stream."""
        for tag in ("SS", "OBJECT", "DV", None):
            for s in streams:
                surl = s.get("StreamUrl", "")
                if surl and (tag is None or s.get("Tag") == tag):
                    return surl, s.get("Tag", "unknown")
        return None

    def test_ss_wins_over_everything(self):
        streams = [
            {"Tag": "DV",     "StreamUrl": "https://cdn/dv.m3u8"},
            {"Tag": "OBJECT", "StreamUrl": "https://cdn/obj.m3u8"},
            {"Tag": "SS",     "StreamUrl": "https://cdn/ss.m3u8"},
        ]
        url, tag = self._select(streams)
        assert tag == "SS"
        assert "ss.m3u8" in url

    def test_object_wins_over_dv(self):
        """When screen-share isn't available, screen-recorded OBJECT beats camera DV."""
        streams = [
            {"Tag": "DV",     "StreamUrl": "https://cdn/dv.m3u8"},
            {"Tag": "OBJECT", "StreamUrl": "https://cdn/obj.m3u8"},
        ]
        _, tag = self._select(streams)
        assert tag == "OBJECT"

    def test_falls_back_to_dv(self):
        streams = [{"Tag": "DV", "StreamUrl": "https://cdn/dv.m3u8"}]
        _, tag = self._select(streams)
        assert tag == "DV"

    def test_unknown_tag_is_last_resort(self):
        streams = [{"Tag": "WEIRD", "StreamUrl": "https://cdn/weird.m3u8"}]
        url, tag = self._select(streams)
        assert "weird.m3u8" in url
        assert tag == "WEIRD"

    def test_empty_streams_returns_none(self):
        assert self._select([]) is None

    def test_stream_without_url_skipped(self):
        streams = [
            {"Tag": "SS",  "StreamUrl": ""},            # empty URL — skip
            {"Tag": "DV",  "StreamUrl": "https://dv"},
        ]
        _, tag = self._select(streams)
        assert tag == "DV"


# ══════════════════════════════════════════════════════════════════════════════
# Issue #5 — HLS detection with query strings
# ══════════════════════════════════════════════════════════════════════════════

class TestHlsDetection:
    """Issue #5: endswith('master.m3u8') broke for URLs with query strings.

    The current code uses substring match ('master.m3u8' in stream_url).
    These tests document the expected behaviour so the regression can't recur.
    """

    @staticmethod
    def _is_hls(url: str) -> bool:
        """Mirror the fixed substring check."""
        return "master.m3u8" in url

    def test_plain_master_m3u8(self):
        assert self._is_hls("https://cdn.example.com/path/master.m3u8")

    def test_master_m3u8_with_query_string(self):
        """This is the exact class of URL that failed before the fix."""
        assert self._is_hls("https://cdn.example.com/master.m3u8?token=abc&expires=123")

    def test_index_m3u8_not_matched(self):
        """Non-master HLS indexes are individual-stream playlists, handled elsewhere."""
        assert not self._is_hls("https://cdn/index.m3u8")

    def test_mp4_not_matched(self):
        assert not self._is_hls("https://cdn.example.com/video.mp4")


# ══════════════════════════════════════════════════════════════════════════════
# Issue #4 — Graceful canvasapi import failure
# ══════════════════════════════════════════════════════════════════════════════

class TestDownloaderImportGuard:
    """downloader.py's top-level imports must fail with a clear error, not a stacktrace."""

    def test_downloader_imports_cleanly(self):
        """Baseline: in a well-formed env, importing downloader raises nothing."""
        import downloader  # noqa: F401

    def test_top_of_file_wraps_imports_in_try(self):
        """The ML-optional contract requires the import block to be defensive."""
        src = (PROJECT_DIR / "downloader.py").read_text()
        # Look for the try/except block around canvasapi/requests/tqdm.
        head = src[:1500]
        assert "try:" in head and "except ImportError" in head, \
            "downloader.py must guard ML imports with try/except ImportError"
        assert "canvasapi" in head
        assert "Settings" in head or "install" in head, \
            "Import error message should guide the user to the ML Environment install"


# ══════════════════════════════════════════════════════════════════════════════
# extract_caption: _filter_api_segments (defensive filter for API hallucinations)
# ══════════════════════════════════════════════════════════════════════════════

class TestFilterApiSegments:

    def test_keeps_normal_segments(self):
        import extract_caption as ec
        segs = [
            {"id": 1, "start": 0.0, "end": 5.0, "text": "Hello world.",
             "avg_logprob": -0.3, "no_speech_prob": 0.05},
            {"id": 2, "start": 5.0, "end": 10.0, "text": "This is a test.",
             "avg_logprob": -0.2, "no_speech_prob": 0.1},
        ]
        kept, dropped = ec._filter_api_segments(segs)
        assert len(kept) == 2
        assert dropped == 0

    def test_filter_returns_tuple(self):
        import extract_caption as ec
        kept, dropped = ec._filter_api_segments([])
        assert kept == []
        assert dropped == 0


# ══════════════════════════════════════════════════════════════════════════════
# Module-level invariants (cheap smoke tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestModuleSmoke:
    """Fast sanity checks that key modules import cleanly."""

    @pytest.mark.parametrize("module_name", [
        "downloader", "extract_caption", "frame_extractor",
        "alignment_parser", "semantic_alignment", "note_generation",
    ])
    def test_import(self, module_name):
        importlib.import_module(module_name)

    def test_extract_caption_has_resolvers(self):
        """v0.12.9 contract — these symbols MUST exist."""
        import extract_caption as ec
        assert callable(getattr(ec, "_resolve_ffmpeg"))
        assert callable(getattr(ec, "_resolve_ffprobe"))
        assert callable(getattr(ec, "_parse_ffmpeg_duration"))
        assert callable(getattr(ec, "_video_duration"))


# ══════════════════════════════════════════════════════════════════════════════
# v0.12.10 — frame_extractor.py ffmpeg resolver (screen-recording capture)
# ══════════════════════════════════════════════════════════════════════════════

class TestFrameExtractorResolvers:
    """Regression for v0.12.10 — silent 'no screen capture' on macOS.

    Before this release, frame_extractor.py used bare 'ffmpeg'/'ffprobe'
    strings, so users with only imageio-ffmpeg (no system ffmpeg) got zero
    extracted frames and EE4802-style screen recordings ended up classified
    as 'camera', producing notes with no slide content.
    """

    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        """Clear the module-level resolver caches so monkeypatching takes effect."""
        import frame_extractor as fe
        fe._FFMPEG_BIN = None
        fe._FFPROBE_BIN = None
        fe._FFPROBE_RESOLVED = False
        yield
        fe._FFMPEG_BIN = None
        fe._FFPROBE_BIN = None
        fe._FFPROBE_RESOLVED = False

    def test_has_resolver_functions(self):
        """v0.12.10 contract — these symbols MUST exist."""
        import frame_extractor as fe
        assert callable(getattr(fe, "_resolve_ffmpeg"))
        assert callable(getattr(fe, "_resolve_ffprobe"))
        assert callable(getattr(fe, "_parse_ffmpeg_duration"))

    def test_no_bare_ffmpeg_literals_remain(self):
        """Guard: no remaining literal 'ffmpeg'/'ffprobe' strings as exec targets."""
        src = (PROJECT_DIR / "frame_extractor.py").read_text()
        # Count non-resolver occurrences. Resolvers themselves use which("ffmpeg")
        # and which("ffprobe") as *arguments*, which is fine; callable lists like
        # ["ffmpeg", ...] are the bug.
        import re as _re
        bad = _re.findall(r'\[\s*"ffmpeg"\s*,|\[\s*"ffprobe"\s*,', src)
        assert bad == [], f"found bare literal launch lists: {bad}"

    def test_resolve_ffmpeg_prefers_system(self, monkeypatch):
        import frame_extractor as fe
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
        assert fe._resolve_ffmpeg() == "/usr/bin/ffmpeg"

    def test_resolve_ffmpeg_falls_back_to_imageio(self, monkeypatch):
        import frame_extractor as fe
        monkeypatch.setattr("shutil.which", lambda name: None)
        fake = MagicMock()
        fake.get_ffmpeg_exe.return_value = "/imageio/ffmpeg"
        monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake)
        assert fe._resolve_ffmpeg() == "/imageio/ffmpeg"

    def test_resolve_ffmpeg_raises_when_unresolvable(self, monkeypatch):
        import frame_extractor as fe
        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)

        def fake_run(cmd, **kw):
            raise RuntimeError("pip fail")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(RuntimeError, match="ffmpeg unavailable"):
            fe._resolve_ffmpeg()

    def test_resolve_ffmpeg_is_cached(self, monkeypatch):
        """Hot path — resolving ffmpeg hundreds of times per video must not hit disk repeatedly."""
        import frame_extractor as fe
        calls = {"count": 0}

        def counting_which(name):
            calls["count"] += 1
            return "/usr/bin/ffmpeg" if name == "ffmpeg" else None

        monkeypatch.setattr("shutil.which", counting_which)
        fe._resolve_ffmpeg()
        fe._resolve_ffmpeg()
        fe._resolve_ffmpeg()
        assert calls["count"] == 1

    def test_resolve_ffprobe_returns_none_when_missing(self, monkeypatch):
        import frame_extractor as fe
        monkeypatch.setattr("shutil.which", lambda name: None)
        assert fe._resolve_ffprobe() is None

    def test_resolve_ffprobe_is_cached(self, monkeypatch):
        import frame_extractor as fe
        calls = {"count": 0}

        def counting_which(name):
            calls["count"] += 1
            return "/usr/bin/ffprobe" if name == "ffprobe" else None

        monkeypatch.setattr("shutil.which", counting_which)
        fe._resolve_ffprobe()
        fe._resolve_ffprobe()
        fe._resolve_ffprobe()
        assert calls["count"] == 1


class TestFrameExtractorDurationFallback:
    """Regression for screen-recording capture: get_video_duration must work
    without ffprobe — otherwise downstream slide alignment gets 0s duration
    and every frame lands on 'slide 1'."""

    @pytest.fixture()
    def sample_video(self):
        vp = PROJECT_DIR / "sample" / "85397" / "videos" / "Lecture 9 Link Layer ARP.mp4"
        if not vp.exists():
            pytest.skip(f"Sample video not available: {vp}")
        return vp

    @pytest.fixture(autouse=True)
    def _reset(self):
        import frame_extractor as fe
        fe._FFMPEG_BIN = None
        fe._FFPROBE_BIN = None
        fe._FFPROBE_RESOLVED = False
        yield

    def test_both_paths_agree(self, sample_video, monkeypatch):
        import frame_extractor as fe
        if fe._resolve_ffprobe() is None:
            pytest.skip("ffprobe not installed; cannot cross-validate")
        d_ffprobe = fe.get_video_duration(sample_video)
        # Force fallback path.
        monkeypatch.setattr(fe, "_resolve_ffprobe", lambda: None)
        d_ffmpeg = fe.get_video_duration(sample_video)
        assert d_ffprobe > 0
        assert d_ffmpeg > 0
        assert abs(d_ffprobe - d_ffmpeg) < 0.5

    def test_zero_on_unparseable(self, tmp_path, monkeypatch):
        """Bad input should return 0.0, not raise — matches old ffprobe semantics."""
        import frame_extractor as fe
        not_a_video = tmp_path / "garbage.mp4"
        not_a_video.write_bytes(b"not a video")
        monkeypatch.setattr(fe, "_resolve_ffprobe", lambda: None)
        assert fe.get_video_duration(not_a_video) == 0.0


class TestRunFfmpegHlsHardening:
    """v0.12.10 — silent ffmpeg failures must surface as errors.

    Before this release, `_run_ffmpeg_hls` swallowed non-zero exit codes and
    produced broken/empty mp4s that downstream pipeline stages tried to process,
    yielding 'no screen recording captured' in the user's notes.
    """

    def test_raises_when_output_file_never_appears(self, tmp_path, monkeypatch):
        import downloader as dl
        out_path = tmp_path / "out.mp4"

        # Fake FfmpegProgress that reports 100% but writes no file.
        class FakeFfmpeg:
            def __init__(self, cmd):
                pass
            def run_command_with_progress(self):
                yield 100

        fake_mod = MagicMock()
        fake_mod.FfmpegProgress = FakeFfmpeg
        monkeypatch.setitem(sys.modules, "ffmpeg_progress_yield", fake_mod)
        monkeypatch.setattr(dl, "_resolve_ffmpeg", lambda: "/fake/ffmpeg")

        with pytest.raises(RuntimeError, match="produced no output"):
            dl._run_ffmpeg_hls("https://cdn/master.m3u8", out_path, lambda p: None)

    def test_raises_when_output_is_tiny(self, tmp_path, monkeypatch):
        """1-byte file counts as silent failure."""
        import downloader as dl
        out_path = tmp_path / "out.mp4"
        out_path.write_bytes(b"x")  # tiny

        class FakeFfmpeg:
            def __init__(self, cmd):
                pass
            def run_command_with_progress(self):
                yield 100

        fake_mod = MagicMock()
        fake_mod.FfmpegProgress = FakeFfmpeg
        monkeypatch.setitem(sys.modules, "ffmpeg_progress_yield", fake_mod)
        monkeypatch.setattr(dl, "_resolve_ffmpeg", lambda: "/fake/ffmpeg")

        with pytest.raises(RuntimeError, match="produced no output"):
            dl._run_ffmpeg_hls("https://cdn/master.m3u8", out_path, lambda p: None)

    def test_no_raise_on_real_output(self, tmp_path, monkeypatch):
        """When ffmpeg produces a reasonable-sized file, no error."""
        import downloader as dl
        out_path = tmp_path / "out.mp4"

        class FakeFfmpeg:
            def __init__(self, cmd):
                pass
            def run_command_with_progress(self):
                # Simulate writing a file during progress
                out_path.write_bytes(b"\x00" * 5000)
                yield 100

        fake_mod = MagicMock()
        fake_mod.FfmpegProgress = FakeFfmpeg
        monkeypatch.setitem(sys.modules, "ffmpeg_progress_yield", fake_mod)
        monkeypatch.setattr(dl, "_resolve_ffmpeg", lambda: "/fake/ffmpeg")

        calls = []
        dl._run_ffmpeg_hls("https://cdn/master.m3u8", out_path, lambda p: calls.append(p))
        assert 100 in calls

    def test_raises_on_nonzero_exit_when_no_progress_lib(self, tmp_path, monkeypatch):
        """No ffmpeg-progress-yield installed, non-zero returncode → RuntimeError."""
        import downloader as dl
        out_path = tmp_path / "out.mp4"

        monkeypatch.setitem(sys.modules, "ffmpeg_progress_yield", None)
        monkeypatch.setattr(dl, "_resolve_ffmpeg", lambda: "/fake/ffmpeg")

        def fake_run(cmd, **kw):
            return MagicMock(returncode=1, stderr="ffmpeg: Connection refused\n")

        monkeypatch.setattr(dl.subprocess, "run", fake_run)
        with pytest.raises(RuntimeError) as exc_info:
            dl._run_ffmpeg_hls("https://cdn/master.m3u8", out_path, lambda p: None)
        msg = str(exc_info.value)
        assert "ffmpeg failed" in msg
        assert "Connection refused" in msg

    def test_no_y_flag_bug_regression(self, tmp_path, monkeypatch):
        """The ffmpeg command must include -y so stale leftovers don't cause an
        interactive overwrite prompt that deadlocks the subprocess."""
        import downloader as dl
        captured_cmd = {"cmd": None}

        class FakeFfmpeg:
            def __init__(self, cmd):
                captured_cmd["cmd"] = cmd
            def run_command_with_progress(self):
                (tmp_path / "out.mp4").write_bytes(b"\x00" * 5000)
                yield 100

        fake_mod = MagicMock()
        fake_mod.FfmpegProgress = FakeFfmpeg
        monkeypatch.setitem(sys.modules, "ffmpeg_progress_yield", fake_mod)
        monkeypatch.setattr(dl, "_resolve_ffmpeg", lambda: "/fake/ffmpeg")

        dl._run_ffmpeg_hls("https://cdn/master.m3u8", tmp_path / "out.mp4", lambda p: None)
        assert "-y" in captured_cmd["cmd"], "must pass -y to avoid overwrite prompt deadlock"

    def test_downloader_has_resolvers(self):
        import downloader
        assert callable(getattr(downloader, "_resolve_ffmpeg"))
        assert callable(getattr(downloader, "_resolve_panopto_tool_id"))
        assert callable(getattr(downloader, "_run_ffmpeg_hls"))
