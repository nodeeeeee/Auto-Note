# AutoNote Developer Guide

This document covers everything that doesn't belong in the user-facing
[README.md](./README.md): repository layout, the CLI surface of each
script, the pipeline data flow, how the multi-provider LLM dispatch
works, and the release procedure.

If you just want to use AutoNote, start with the README.

---

## Repository layout

```
Auto-Note/
├── README.md                  User guide (this is the user-facing one)
├── DeveloperGuide.md          You are here
├── requirements.txt           Python deps for the venv
├── build.spec                 PyInstaller spec for the standalone Flet GUI
├── downloader.py              Canvas + Panopto download
├── extract_caption.py         Whisper transcription wrapper
├── frame_extractor.py         Scene detection + frame selection + classifier
├── semantic_alignment.py      Caption ↔ slide / frame alignment (BGE-M3 + Viterbi)
├── alignment_parser.py        Compact alignment JSON helper
├── note_generation.py         LLM-driven Markdown note generation
├── pipeline_worker.py         Threaded transcribe + frames + align orchestrator
├── gui.py                     Flet GUI (legacy, used by the AppImage standalone)
├── electron/                  Modern Electron + Node frontend
│   ├── package.json           electron-builder config + version pin
│   ├── main.js                IPC + venv installer
│   ├── preload.js             contextBridge surface
│   └── renderer/              UI (app.js, index.html, style.css)
└── test/                      pytest suite (no network, no GPU, no API keys)
    ├── test_unit.py           Pure-function regressions
    ├── test_note_generation.py
    ├── test_language_and_skip.py
    ├── test_pipeline.py       Requires a real ML venv + Canvas config
    ├── test_release.py        AppImage venv presence checks
    ├── test_v0_12_fixes.py    Per-issue regression tests
    └── test_v0_13_split_stream.py
```

---

## Pipeline data flow

```
┌──────────────┐    ┌──────────────┐    ┌────────────────┐    ┌────────────────────┐    ┌──────────────────┐
│ downloader   │ →  │ extract_     │ →  │ frame_         │ →  │ semantic_alignment │ →  │ note_generation  │
│ .py          │    │ caption.py   │    │ extractor.py   │    │ .py                │    │ .py              │
└──────────────┘    └──────────────┘    └────────────────┘    └────────────────────┘    └──────────────────┘
       │                   │                    │                       │                          │
       ▼                   ▼                    ▼                       ▼                          ▼
   videos/            captions/             frames/                 alignment/                  notes/
   materials/         (whisper json)        + alignment/            (caption ↔ slide /          + sections/
                                            (source=screenshare)    frame mapping)              + images/
```

`pipeline_worker.py` is the threaded orchestrator: while video N+1 is being
transcribed, video N's frames are extracted concurrently. Alignment is
batched at the end so the BGE-M3 model loads once.

---

## Per-script CLI

### `downloader.py`

```
python downloader.py --course-list
python downloader.py --video-list --course 85427
python downloader.py --download-video 2 4 --course 85427
python downloader.py --download-video-all --course 85427 --secretly
python downloader.py --download-material-all --course 85427
```

Reads `~/.auto_note/canvas_token.txt` (one token per file) and
`~/.auto_note/config.json` for `CANVAS_URL` / `PANOPTO_HOST`. Set
`AUTONOTE_DATA_DIR` to point elsewhere (the Electron main process does this
for the bundled venv). `--secretly` adds 5–15 min jitter between videos.

**Panopto split-stream merge** (v0.13.0). Some Panopto recordings put the
screen video in `OBJECT/SS` (no audio) and the microphone in a separate
`DV` stream. The downloader classifies each candidate as video-bearing or
audio-bearing, picks the screen stream as the video source and the
audio-bearing stream as the audio source, and merges them with
`_run_ffmpeg_hls_merge()` — `ffmpeg -map 0:v:0 -map 1:a:0 -c copy -shortest`.
The merge is suppressed when video and audio are the same URL.

### `extract_caption.py`

```
python extract_caption.py --video <path>           # single video
python extract_caption.py --course 85427           # batch
```

Uses `faster-whisper large-v3` if the venv has GPU torch, falls back to
OpenAI's `whisper-1` API when the venv is CPU-only and `OPENAI_API_KEY`
is set. Output: `captions/<stem>.json` with `segments[]` and `duration`.

### `frame_extractor.py`

```
python frame_extractor.py --video <path> --caption <captions/...json> \
                          --course-dir <course> [--force-screen]
python frame_extractor.py --course 85427 [--force-screen]
```

Three logical phases:

1. **Classify** the video as `screen` or `camera` from 6 evenly-spaced
   sample frames using three pixel heuristics (edge density, color
   uniformity, brightness). The decision rule lives in `classify_video()`.
2. **Detect scenes** — ffmpeg `select='gt(scene,T)'` filter (T=0.15 for
   `OBJECT/SS` streams, 0.3 otherwise) plus periodic samples every 10–30 s.
3. **Dedup + extract** — perceptual dHash clustering groups consecutive
   near-duplicate frames; the highest `_information_score` wins per cluster.
   The frame extractor also runs `_describe_frames()` which posts each
   surviving frame to GPT-4o-mini for a one-paragraph description (cached
   in `frames/<stem>/image_cache.json`).

Output: `frames/<stem>/frame_NNN.png` + `alignment/<stem>.json` with
`source: "screenshare"` and a `slide` field mapping each transcript
segment to a frame index.

#### Classifier decision rule (v0.13.6)

```
screen_ratio = screen_votes / n_sampled_frames
if screen_ratio >= 1/3        → "screen"
elif screen_votes == 0        → "camera"
else (1/n votes, borderline)  → GPT-4o-mini vision tiebreaker, fall back to "screen"
```

The bias is intentional: a wrongly-detected "screen" video produces a few
camera-style frames the vision pass describes faithfully (minor noise);
a wrongly-detected "camera" video drops every slide image (catastrophic).

`--force-screen` skips classification entirely; the Electron Pipeline page
sets it whenever the user picks "Video screenshots" in the UI.

### `semantic_alignment.py`

```
python semantic_alignment.py --course 85427                          # batch
python semantic_alignment.py --caption <cap.json> --slides <pdf>     # one pair
python semantic_alignment.py --suggest-matches --match-model bge-m3
```

Loads slide text + speaker notes from PDF/PPTX/DOCX, embeds slides and
transcript-context windows with the chosen embedder (`bge-m3` /
`mpnet` / `jina` / `google`), runs FAISS cosine search, then a Viterbi
smoother over a Gaussian time-position prior. Output: `alignment/<stem>.json`.

Per-day annotation snapshots like `Lecture5.ann070426.pdf` are
auto-promoted to the master deck `Lecture5.pdf` when both exist (v0.12.24).
Weak filename-token matches are overridden by BGE-M3 results when the name
match isn't lecture-number-based (v0.12.23).

### `note_generation.py`

```
python note_generation.py --course 85427 --course-name CS3210 \
                          --image-source frames --language zh \
                          --model gpt-5.1 --detail 8
```

Discovery order (`_discover_video_lectures`):

1. Iterate `captions/*.json` (the canonical list of lectures)
2. For each caption, prefer the screenshare alignment (frames) when present
3. Fall back to the slide alignment when frames not extracted
4. Emit a transcript-only `LectureData` when neither is available

`--image-source slides` flips that preference. Each lecture's slides are
chunked (`CHAPTER_SIZE = 15` slides) and rendered as a section via
`generate_section()`; sections cache to `notes/sections/<dir_key>_S{NN}.md`
where `dir_key = L{num:02d}_{slug}[_F{file_idx:02d}]`. The section cache
survives caption-list reshuffles because `slug` is derived from the slide
file stem or frame_dir name, not the volatile sequential `num`.

`filter_images_pass()` runs after merge to drop title-slide/divider images
the vision API thinks aren't worth keeping.

#### Soft-fail on missing materials (v0.13.3)

`_discover_lectures()` (the slide-centric helper) returns `[]` when
`materials/` doesn't exist, instead of `sys.exit(1)`. `main()` then prints
a tailored hint based on which directories actually exist:

| State | Hint |
|-------|------|
| Neither `captions/` nor `materials/` | Download videos and transcribe, OR download slides |
| `captions/` missing | Transcribe first, OR rerun with `--image-source slides` |
| `materials/` missing | Rerun with `--image-source frames` (default) |
| Both exist, no pairs | Filename / alignment mismatch |

### `pipeline_worker.py`

```
python pipeline_worker.py --course 85427 [--force] [--sequential] \
                          [--skip-frames] [--force-screen]
```

The Electron Pipeline page wires `--skip-frames` when the user picked
slide PDFs (frame extraction would be wasted), and `--force-screen` when
the user picked video screenshots (so picture-in-picture lectures still
extract frames).

---

## Multi-provider LLM dispatch

`note_generation.py` routes every prompt through `_call(model, system,
user, max_tokens)`. Provider is decided by `_provider(model)`:

| Prefix / value | Provider | Implementation |
|----------------|----------|----------------|
| `claude-cli` | Claude Code CLI subprocess | `claude -p --output-format text` |
| `codex-cli` | OpenAI Codex CLI subprocess | `codex exec -m gpt-5.2 --skip-git-repo-check -s read-only -o <file>` |
| `gemini*` | Google Gemini | OpenAI-compatible endpoint at `generativelanguage.googleapis.com/v1beta/openai/` |
| `claude*` | Anthropic API | `anthropic.Anthropic.messages.create` |
| `deepseek*` | DeepSeek | OpenAI-compatible endpoint |
| `grok*` | xAI Grok | OpenAI-compatible endpoint |
| `mistral*` / `codestral*` / `pixtral*` / `magistral*` | Mistral | OpenAI-compatible endpoint |
| `gpt*` / `o3` / `o4-mini` | OpenAI | `openai.OpenAI` |

API keys come from `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`
env vars first, then `~/.auto_note/{provider}_api.txt`. `_cap_tokens()`
clamps `max_tokens` per-model to avoid 400s from gpt-4o-style limits.

`AUTONOTE_CODEX_MODEL` overrides the codex model (default `gpt-5.2`).
`gpt-5.1` requires an API-key codex login; `gpt-5.4`/`5.5`/`5.4-mini` work
on ChatGPT-only accounts.

### Resilience

The verify and translate passes are wrapped in try/except so that an
upstream 429 / 5xx doesn't discard a successfully-generated draft —
the original draft is kept and logged with a `[warn]` line (v0.12.22).

---

## Two data dirs

AutoNote reads / writes two directories:

| Directory | Purpose |
|-----------|---------|
| `~/AutoNote/` (configurable, "Output Dir") | Course content: videos, captions, frames, alignment, notes |
| `~/.auto_note/` (fixed) | App state: API keys, Canvas token, config.json, manifest.json, ML venv, scripts/ |

`AUTONOTE_DATA_DIR` env var changes the second dir — the Electron main
process always sets it before spawning Python so the bundled scripts find
the bundled credentials. The downloader writes videos under
`AUTONOTE_DATA_DIR/<course>/videos/` while the rest of the pipeline
reads `OUTPUT_DIR/<course>/videos/`. The Electron app symlinks the two,
so both locations resolve to the same files.

---

## File caches

### Section cache: `notes/sections/<dir_key>_S{NN}.md`

`dir_key = L{num:02d}_{slug}[_F{file_idx:02d}]`, where `slug` comes from
the slide file stem or the frame_dir name. Stable across runs even when
the sorted caption order changes.

### Image cache: `frames/<stem>/image_cache.json` and `<slide_pdf>.image_cache.json`

Maps `page_<index>` → vision-API description string. Used both during
alignment (to enrich text-poor slides) and during note generation (to
decide whether an image is worth embedding via `_desc_has_visual()`).

### Language marker: `notes/sections/.language`

A two-char language code (`en` / `zh`). When the user changes language
between runs, the marker mismatches and `--force` is implicitly added so
the wrong-language cached sections are re-run.

---

## Electron app

```
electron/
├── package.json           appId, productName, electron-builder targets, version pin
├── main.js                Window, IPC, install:components, install:start
├── preload.js             contextBridge: window.api.*
└── renderer/
    ├── app.js             All page builders + state machine
    ├── index.html         Single shell, all pages mount under #page-content
    └── style.css
```

The renderer is a single-page app with seven page builders (`buildDashboard`,
`buildPipeline`, …) and a shared `attachPageHandlers()` that binds the
right click handlers based on `State.currentPage`. State persists across
page navigation only for the Pipeline form (so it survives accidental
nav-rail clicks).

After install or skip, `enterMainApp()` always lands on Dashboard
(v0.13.2) — empty-state nudges users to Settings via a clear button rather
than auto-jumping there.

### Adding a new pipeline option to the UI

1. Add the form control to the relevant page builder.
2. Persist it in `State.pipeline` if it's on the Pipeline page (so the
   value survives nav).
3. Read it in the click handler and append the corresponding CLI flag to
   the `cmd` array passed to `runChain()`.
4. Make sure the underlying Python script accepts the flag — add to
   `argparse` if not.
5. Update the Flet GUI (`gui.py`) the same way if the option should be
   reachable from the standalone build.

---

## Tests

```
# Pure unit + offline regressions (default for CI)
python -m pytest test/ \
  --ignore=test/test_pipeline.py \
  --ignore=test/test_release.py \
  --ignore=test/test_gui.py

# Full suite (requires AppImage venv installed + Canvas configured)
python -m pytest test/
```

`test_pipeline.py` calls real Python subprocesses against the bundled
`scripts/` dir and a real Canvas token — skip it on CI.
`test_release.py` checks the AppImage venv has every package each script
imports.

When you change a public API (function signature, CLI flag, JSON schema),
add a regression test in `test_v0_13_split_stream.py` (or a new
`test_v0_<minor>_<feature>.py` for the next minor bump).

---

## Release procedure

Patch releases are cheap — every code change ends with a tag + push.

```bash
# 1. Verify
python -m pytest test/ \
  --ignore=test/test_pipeline.py \
  --ignore=test/test_release.py \
  --ignore=test/test_gui.py
node --check electron/renderer/app.js

# 2. Bump electron/package.json "version" (single-line edit, don't reformat
#    the file — JSON.stringify(p, null, 2) expands inline arrays).

# 3. Commit
git commit -am "Brief subject line of the change"

# 4. Tag + push
git tag vX.Y.Z
git push origin main
git push origin vX.Y.Z
```

Minor bumps (e.g. v0.13.0 → v0.14.0) signal a user-visible feature or a
breaking change in the JSON cache layout. Patch bumps cover bug fixes, UI
plumbing, and new tests.

---

## Common pitfalls

- **`AUTONOTE_DATA_DIR` is unset in dev.** The downloader reads
  `config.json` from `DATA_DIR`, which falls back to `PROJECT_DIR` in dev
  mode — meaning `Auto-Note/config.json`, not `~/.auto_note/config.json`.
  Always export `AUTONOTE_DATA_DIR=~/.auto_note` when running scripts
  directly.
- **Stray `*_notes.md` at the repo root.** When `AUTONOTE_DATA_DIR` isn't
  set, generated notes can land in the cwd. `.gitignore` excludes
  `*_notes.md` so these don't get committed.
- **Python's `sorted()` puts `EE4802IE4213*` before `EE4802_IE4213*`.**
  ASCII `_` (0x5F) > letters (0x41–0x7A). The shell `ls` uses locale-aware
  sort, so `--lectures N` from CLI may map to a different lecture than
  what `ls` shows. Use `python -c "from pathlib import Path; ..."` to
  preview the sort order.
- **Codex on a ChatGPT-only account doesn't support `gpt-5.1`.** The
  default was changed to `gpt-5.2` in v0.12.21. Override with
  `AUTONOTE_CODEX_MODEL`.

---

## Contributing

1. Fork + branch.
2. Run the offline test suite.
3. If you change UI: also start the Electron app (`cd electron && npm
   start`) and verify the form renders + the run still works.
4. Tag the bump in `electron/package.json` AND in git after merge.
