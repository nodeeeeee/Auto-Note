# AutoNote Architecture Design

## System Overview

AutoNote is a desktop application that generates comprehensive study notes from Canvas LMS lecture materials and videos. It uses an Electron frontend, a set of Python pipeline scripts executed as subprocesses, and an isolated ML virtual environment for GPU-accelerated processing.

```
User Interface (Electron)
    │
    ├── main.js          IPC bridge, subprocess management, file operations
    ├── renderer/app.js  Single-page app (7 pages), terminal, state management
    └── preload.js       Secure IPC bridge between renderer and main process
    │
    ▼
Pipeline Scripts (Python, executed as subprocesses)
    │
    ├── downloader.py            Canvas + Panopto download
    ├── extract_caption.py       Whisper transcription
    ├── frame_extractor.py       Screen-share frame extraction + dedup + junk filter
    ├── semantic_alignment.py    Transcript ↔ slide alignment (FAISS + Viterbi)
    ├── alignment_parser.py      Compact alignment JSON for LLM prompts
    ├── pipeline_worker.py       Orchestrates transcribe + frame-extract + align
    ├── note_generation.py       LLM-based note writing + image filtering
    └── benchmark.py             Quality evaluation (coverage / image / coherency)
    │
    ▼
ML Environment (~/.auto_note/venv/)
    PyTorch, faster-whisper, sentence-transformers, FAISS, BGE-M3
```

## Data Flow

```
Canvas API                 Panopto API
    │                          │
    ▼                          ▼
materials/  (PDF, PPTX)    videos/  (MP4)
    │                          │
    │                    ┌─────┴──────┐
    │                    │            │
    │               Camera?      Screen share?
    │                    │            │
    │                    ▼            ▼
    │              captions/    frames/<stem>/
    │               (Whisper)    (ffmpeg scene detect
    │                            + periodic sampling
    │                            + perceptual hash dedup
    │                            + junk filter via vision API)
    │                            + image_cache.json (descriptions)
    │                    │            │
    └────────┬───────────┘            │
             │                        │
             ▼                        ▼
        alignment/               alignment/
    (BGE-M3 matching →       (timestamp-based
     FAISS + Viterbi)         frame assignment)
             │                        │
             └────────┬───────────────┘
                      │
                      ▼
              alignment/*.compact.json
              (10x smaller for LLM prompts)
                      │
                      ▼
                notes/sections/L{N}_S{ci}.md
              (per-chunk LLM generation, cached)
                      │
                      ▼
              notes/CourseName_notes.md
              (merged + image-filtered final output)
```

## Component Details

### Electron Frontend

**Architecture**: Single-page application with 7 pages and a persistent terminal panel.

**State management**: Global `State` object in `app.js` persists form values across page navigation. Key fields include `pipeline.courseId`, `pipeline.language`, `pipeline.detail`, `pipeline.force`, and step checkboxes.

**Subprocess execution**: `main.js` spawns Python scripts via `child_process.spawn()` (or `node-pty` when available for tty support). Stdout/stderr are streamed to the renderer via IPC events (`process:data`, `process:done`).

**Dashboard detail modal**: Clicking a course card opens a modal overlay that lists all transcribed videos with their processing status (caption/alignment/notes). Each video has a delete button that removes the transcript, alignment, note sections, and per-video note files via the `course:deleteVideo` IPC handler.

**Packaging** (`electron/package.json`):
- **Windows**: NSIS installer only (`target: ["nsis"]`). The `appx` target was removed — it requires Microsoft Store signing credentials that aren't available in CI.
- **macOS**: Universal DMG (x64 + arm64) with signing explicitly disabled (`identity: null`, `hardenedRuntime: false`, `gatekeeperAssess: false`, `dmg.sign: false`). Without these, users see a "AutoNote.app is damaged and can't be opened" quarantine error. Users still need to right-click → Open on first launch (standard for unsigned apps).
- **Linux**: AppImage + deb.
- **`extraResources`** must list every Python script referenced by `main.js:SCRIPTS` or the renderer. Current list: `downloader.py`, `extract_caption.py`, `frame_extractor.py`, `semantic_alignment.py`, `alignment_parser.py`, `note_generation.py`, `pipeline_worker.py`. Missing a script from this list leaves it out of `resources/scripts/` in the packaged app and causes "No such file or directory" errors on users' machines.

### Pipeline Scripts

#### downloader.py
- Downloads videos from Panopto and materials from Canvas
- **Stream priority**: OBJECT > SS > DV > untagged. OBJECT streams are screen recordings and preferred over camera (DV) when both are available for a lecture.
- Tracks download state in `manifest.json` and `download_log.json`. Each entry records `stream_tag` (used downstream by `frame_extractor`).
- Slack mode adds random delays to avoid rate-limiting
- Smart size filter uses LLM to select relevant files when > 1 GB

#### extract_caption.py
- Selects backend automatically: faster-whisper (GPU) or OpenAI Whisper API
- Produces timestamped segment-level JSON
- `--force` flag re-transcribes even if captions already exist
- Language detection probes from mid-audio for accuracy
- Whisper hallucination filter: segments whose text is only `. . . . .` (common during silent audio from OBJECT-stream intro/loading screens) are dropped downstream in `alignment_parser._clean_transcript`.

#### frame_extractor.py
- Classifies videos as screen-share or camera using edge/uniformity heuristics. Screen recordings have sharp edges, high brightness, and large uniform regions.
- **Per-stream extraction tuning** (`detect_scenes(stream_tag=...)`):
  - For `OBJECT`/`SS` streams (stable slide recordings with subtle text-only changes): scene threshold lowered to 0.15 and periodic samples taken every 30s unconditionally. Without periodic sampling, 2-hour OBJECT streams yield only ~10 frames.
  - Other streams: default 0.3 threshold, periodic sampling only as fallback when < 5 scene changes detected.
- **Same-page deduplication**: Groups consecutive frames by perceptual hash similarity (dHash, 16×16 = 256-bit hash, Hamming distance < `PAGE_SIMILARITY_THRESHOLD = 35` bits). Merges incremental bullet reveals while keeping genuinely different pages. From each group, selects the frame with the highest visual information score (edge density on 160x120 grayscale).
- **Junk-frame filter** (`_JUNK_DESC_RE` + `_is_blank_frame`):
  - Pre-vision: pure-black or pure-white frames (>95% of samples <15 or >240 in grayscale) are skipped before calling the vision API.
  - Post-vision: frames whose vision description matches `_JUNK_DESC_RE` (desktop wallpaper, taskbar, Windows 11, loading screens, vision-API refusals, memes, XKCD, four-panel comics) are deleted. Remaining frames are contiguously renumbered on disk and in the alignment so they stay in sync.
- **Vision API descriptions**: After extraction, `_describe_frames` calls GPT-4o-mini (via `semantic_alignment.ImageDescriber`) to describe each frame. Descriptions are cached in `frames/<stem>/image_cache.json` keyed by `page_N` (0-indexed matching frame_{N+1:03d}.png).
- Builds timestamp-based alignment JSON compatible with the rest of the pipeline. The `source` field is `"screenshare"` for frame-based alignments and `"slides"` for PDF-based ones.

#### pipeline_worker.py
- Single subprocess that orchestrates `extract_caption.py` + `frame_extractor.py` + `semantic_alignment.py` for every video in a course.
- Invoked by the Electron app (`main.js:SCRIPTS.pipeline_worker`) for the "Transcribe + Align" button, letting one progress bar cover the whole pipeline.
- `_script(name)` resolves script paths by looking in `~/.auto_note/scripts/` first (production install), then the script's own directory (development). This matches how the Electron app syncs packaged `resources/scripts/*.py` into `~/.auto_note/scripts/` on first launch.
- Must be listed in `electron/package.json:build.extraResources` or the packaged Windows/macOS installer will be missing it.

#### semantic_alignment.py
- Extracts text from slides (PDF/PPTX/DOCX) with image enrichment for sparse slides
- **Matching priority** (in `process_course`):
  1. User-supplied mapping JSON
  2. Automatic name/number matching
  3. BGE-M3 embedding match (pre-computed via `suggest_matches`)
  4. mpnet content embedding fallback
- Embeds slides and transcript windows with sentence-transformers
- FAISS IndexFlatIP for fast cosine-similarity K-NN lookup
- Viterbi temporal smoothing with forward bias and temporal position prior
- Off-slide detection for Q&A/demo segments (cosine < threshold)
- `--force` flag re-aligns even if alignment files already exist
- `ImageDescriber` is reused by `frame_extractor` to describe screen-capture frames.

#### alignment_parser.py
- Compresses full alignment JSON (300 KB) into compact per-slide format (30 KB)
- `_clean_transcript` strips Whisper's dot-hallucination segments (`re.fullmatch(r"[\s.]+", text)`) and common ASR filler words.
- Used by note_generation to build token-efficient LLM prompts

#### note_generation.py
- Multi-provider LLM support: OpenAI, Anthropic, Google Gemini, DeepSeek, xAI, Mistral, Claude CLI (`claude -p`)
- **Language system**: `--language en|zh` CLI flag overrides the `NOTE_LANGUAGE` constant. `_P(key)` always returns English prompts; Chinese output is produced by a separate post-generation `_translate()` call per section.
- **Translation prompt preserves technical terms in English** when target is Chinese. The prompt lists protected term categories (protocols, crypto, algorithms, proper nouns, code identifiers) and instructs the LLM to translate only connecting prose. Example output: `symmetric key cryptography 方案在 encryption 和 decryption 时使用同一个 key。` This matches how students study in English-taught courses — the Chinese provides narrative, the English terms remain exam-ready.
- **Per-lecture chunking**: CHAPTER_SIZE slides per LLM call. `MAX_NOTE_CHARS = 120000` is the total prompt-char cap (transcript + slide outlines + image hints) — replaces the older per-slide `MAX_TRANSCRIPT_CHARS` limit that was truncating content too aggressively.
- **Parallel section generation**: `ThreadPoolExecutor(max_workers=PARALLEL_SECTIONS = 6)` fans out chunk generation + translation concurrently. Each chunk returns `(ci, (content, fresh))` — make sure to unpack as `ci_ret, (content, fresh) = fut.result()`.
- **Section caching**: each chunk saved as `L{N}_S{ci}.md`. `notes/sections/.language` marker triggers auto-force-regen when the language changes.
- **Image hints for screen-share lectures** (`make_chunk_prompt`, line 819): now include the cached vision description (`img_cache["page_N"]`) instead of only the transcript context. Without this the LLM invents captions for frames based on what it's writing about, causing captions that don't match the displayed image.
- **Image rendering** (`LectureData.render_chunk_images`, line 1222): copies source frames from `frames/<stem>/frame_NNN.png` into `notes/images/L{NN}/`. **Refreshes stale copies** when source mtime or size differs from the destination — important after re-extraction, or the displayed image will be an outdated frame with different content than the caption describes.
- `--force` flag re-generates all sections from scratch
- Image filtering: multi-step decision pipeline (cache description keywords → title pattern → vision API). Includes all slides with visual elements; only excludes administrative/non-course elements.
- Self-scoring: coverage, terminology, callouts, code blocks (weighted average)
- Per-video mode (`--per-video`): one note file per lecture instead of merged
- Iterative mode: raises detail level until quality target is reached
- All terminal output (print/tqdm.write) is in English regardless of note language

#### benchmark.py
- Stand-alone quality evaluator for generated notes. Usage: `python benchmark.py --course ID [--verbose]` or `--note PATH --transcript PATH --image-cache PATH`.
- **Three metrics, each scored 0-10**:
  - **Content coverage** (`content_coverage`): extracts key terms (capitalized phrases, acronyms, hyphenated terms appearing ≥2 times) from the transcript; scores the % of those terms that appear in the note.
  - **Image density** (`image_density`): ratio of images inserted in the note vs. number of content-rich images available in `image_cache.json` (filtered via `_CONTENT_KEYWORDS` to exclude loading-screen/desktop/blank descriptions).
  - **Logic coherency** (`logic_coherency`): structural checks — sections headings, long image clusters (≥3 consecutive images without intervening prose; pairs are allowed), orphan images (no preceding paragraph within 6 lines), mid-sentence truncation, leaked artifacts (APPROVED, © CS, LLM refusals).
- **Overall** = `0.4 × coverage + 0.3 × image_density + 0.3 × coherency`.
- Limitations: coverage extractor matches English terms only, so Chinese notes score lower on coverage even when content is complete. Image density returns 10.0 when no cache exists (division-by-zero fallback for slide-based notes).

### Force Regenerate Behavior

The "Force regenerate" toggle applies to whichever pipeline steps are selected:

| Step selected | Without force | With force |
|---|---|---|
| Transcribe | Skips videos with existing captions | Re-transcribes all videos |
| Align | Skips captions with existing alignment | Re-aligns all captions |
| Generate notes | Uses cached section .md files | Re-calls LLM for all sections |

Without force, the pipeline is incremental: only missing files are processed.

### Image Inclusion Pipeline

Images pass through multiple filtering layers before appearing in the final notes:

1. **Frame extraction** (`frame_extractor.py`):
   - `_is_blank_frame` pre-vision skip for all-black / all-white frames
   - `_JUNK_DESC_RE` post-vision filter for desktop wallpaper, loading screens, memes, XKCD, vision-API refusals
2. **Image hints generation**: Slides with word_count < 80, cached descriptions, or code are offered to the LLM as available images. Screen-share frames always pass through their `image_cache` description so the LLM caption matches the actual frame.
3. **LLM prompt instructions**: System prompt instructs to insert all slides with visual elements (diagrams, charts, code, math, etc.) and skip pure text or administrative slides
4. **Post-generation filter** (`filter_images_pass`):
   - Screen-share frames: always kept
   - Cache-verified visual description: kept
   - Title/divider pattern: removed
   - Vision API (GPT-4o-mini): decides uncertain cases; defaults to keep
5. **Image rendering** (`render_chunk_images`): always refreshes stale copies in `notes/images/L{NN}/` on mtime/size mismatch so the final `.md` never points at stale content.

### NSIS Uninstaller (Windows)

`electron/build/uninstaller.nsh` lets users choose what to keep on uninstall via three sequential `MessageBox` prompts, all inside the `customUnInstall` macro:

1. Keep generated notes and downloaded course files in `%USERPROFILE%\AutoNote`? (default: No)
2. Keep ML environment (~2 GB) in `%USERPROFILE%\.auto_note\venv`? (default: No)
3. Keep settings and API keys in `%USERPROFILE%\.auto_note`? (default: No)

Each MessageBox uses the NSIS label-jump idiom: `IDYES keep_label` → skip the following delete commands and jump past them. This keeps the implementation free of `Var` declarations, `${If}/${EndIf}` macros, and `LogicLib` dependencies — all of which turned out to break electron-builder's multi-pass NSIS compile.

Three pragmas suppress warnings that fire harmlessly in one of the two compile passes but would be escalated to errors by CI:

- `!pragma warning disable 6010` — "un.* function not referenced"
- `!pragma warning disable 6020` — "uninstaller script code but no WriteUninstaller"
- `!pragma warning disable 8000` — "Uninstall page instfiles not used"

**Things that did not work** during iteration (kept here to save future attempts):

- **Custom `UninstPage` via `!macro customUnInstallPage`**: that macro name is not a real electron-builder hook, so the `un.` function is never referenced and NSIS zeros it out with warning 6010.
- **`UninstPage custom ...` inside `!macro customHeader`**: valid NSIS location but triggers warning 8000 because it overrides MUI2's default `MUI_UNPAGE_INSTFILES` without a replacement.
- **`customUnInit` + `Var` + `${If}`**: compiled but failed in one of electron-builder's passes (exact error unavailable without admin log access). Moving the prompts into `customUnInstall` directly succeeded.

## Data Storage

**No database** — all data is file-based JSON/Markdown.

| File | Location | Purpose |
|------|----------|---------|
| `config.json` | `~/.auto_note/` | Canvas URL, Panopto host, output dir |
| `*_api.txt` / `*_token.txt` | `~/.auto_note/` | API keys and tokens |
| `manifest.json` | `~/.auto_note/` | Video download state + `stream_tag` |
| `download_log.json` | Per-course | Material download tracking |
| `captions/*.json` | Per-course | Whisper transcript (timestamped segments) |
| `frames/<stem>/frame_NNN.png` | Per-course | Extracted screen-share frames |
| `frames/<stem>/image_cache.json` | Per-course | Vision-API descriptions per frame |
| `alignment/*.json` | Per-course | Full segment-level alignment |
| `alignment/*.compact.json` | Per-course | Token-efficient alignment for LLM |
| `notes/sections/L*_S*.md` | Per-course | Cached per-chunk note sections |
| `notes/sections/.language` | Per-course | Current language marker (forces regen on change) |
| `notes/*_notes.md` | Per-course | Final merged/per-video notes (English default) |
| `notes/*_notes.zh.md` | Per-course | Chinese version (user renames after `--language zh`) |
| `notes/*_notes.en.md` | Per-course | Optional English backup before running ZH regen |
| `notes/*.score.json` | Per-course | Self-score breakdown |
| `notes/images/L*/` | Per-course | Rendered slide PNGs / copied frames |

## Testing

Tests are in `test/` and organized by scope:

| File | Scope |
|------|-------|
| `test_unit.py` | Offline unit tests: Viterbi, timeline, hashing, alignment parsing, slide discovery |
| `test_pipeline.py` | Integration tests: script execution, manifest schema, CLI flags (some require network) |
| `test_language_and_skip.py` | Language selection, terminal output (no CJK in prints), skip logic with/without --force |
| `test_note_generation.py` | Note generation specific tests |
| `test_gui.py` | GUI-specific tests |
| `electron/test/main.test.js` | Electron main process tests (26 cases) |

Run all offline tests: `python -m pytest test/ -v -k "not integration"`
Run Electron tests: `cd electron && npx jest test/`

Ad-hoc quality check: `python benchmark.py --course <ID> --verbose` runs the three quality metrics against every note in a course.
