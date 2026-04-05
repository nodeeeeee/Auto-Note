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
    ├── frame_extractor.py       Screen-share frame extraction + dedup
    ├── semantic_alignment.py    Transcript ↔ slide alignment (FAISS + Viterbi)
    ├── alignment_parser.py      Compact alignment JSON for LLM prompts
    └── note_generation.py       LLM-based note writing + image filtering
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
    │              captions/    frames/ + captions/
    │               (Whisper)    (ffmpeg scene detect
    │                            + perceptual hash dedup
    │                            + info-score selection)
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

### Pipeline Scripts

#### downloader.py
- Downloads videos from Panopto and materials from Canvas
- Tracks download state in `manifest.json` and `download_log.json`
- Slack mode adds random delays to avoid rate-limiting
- Smart size filter uses LLM to select relevant files when > 1 GB

#### extract_caption.py
- Selects backend automatically: faster-whisper (GPU) or OpenAI Whisper API
- Produces timestamped segment-level JSON
- `--force` flag re-transcribes even if captions already exist
- Language detection probes from mid-audio for accuracy

#### frame_extractor.py
- Classifies videos as screen-share or camera using edge/uniformity heuristics
- Scene detection via ffmpeg scene filter + periodic sampling fallback
- **Same-page deduplication**: Groups consecutive frames by perceptual hash similarity (dHash, Hamming distance < 45 bits). From each group, selects the frame with the highest visual information score (edge density on 160x120 grayscale). This ensures incremental bullet reveals keep only the most complete version.
- Builds timestamp-based alignment JSON compatible with the rest of the pipeline

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

#### alignment_parser.py
- Compresses full alignment JSON (300 KB) into compact per-slide format (30 KB)
- Cleans filler words from transcripts
- Used by note_generation to build token-efficient LLM prompts

#### note_generation.py
- Multi-provider LLM support: OpenAI, Anthropic, Google Gemini, DeepSeek, xAI, Mistral
- **Language system**: `--language en|zh` CLI flag overrides the `NOTE_LANGUAGE` constant. The `_P(key)` function selects from `_PROMPTS["en"]` or `_PROMPTS["zh"]` dictionaries containing complete prompt sets (system, chunk, slide_only, verify, exam, detail_instructions). Language is selectable per-run from the Pipeline and Generate page dropdowns.
- Per-lecture chunking: CHAPTER_SIZE slides per LLM call
- Section caching: each chunk saved as `L{N}_S{ci}.md` for resume support
- `--force` flag re-generates all sections from scratch
- Image filtering: multi-step decision pipeline (cache description keywords → title pattern → vision API). Includes all slides with visual elements; only excludes administrative/non-course elements.
- Self-scoring: coverage, terminology, callouts, code blocks (weighted average)
- Per-video mode (`--per-video`): one note file per lecture instead of merged
- Iterative mode: raises detail level until quality target is reached
- All terminal output (print/tqdm.write) is in English regardless of note language

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

1. **Image hints generation**: Slides with word_count < 80, cached descriptions, or code are offered to the LLM as available images
2. **LLM prompt instructions**: System prompt instructs to insert all slides with visual elements (diagrams, charts, code, math, etc.) and skip pure text or administrative slides
3. **Post-generation filter** (`filter_images_pass`):
   - Screen-share frames: always kept
   - Cache-verified visual description: kept
   - Title/divider pattern: removed
   - Vision API (GPT-4o-mini): decides uncertain cases; defaults to keep

## Data Storage

**No database** — all data is file-based JSON/Markdown.

| File | Location | Purpose |
|------|----------|---------|
| `config.json` | `~/.auto_note/` | Canvas URL, Panopto host, output dir |
| `*_api.txt` / `*_token.txt` | `~/.auto_note/` | API keys and tokens |
| `manifest.json` | Output dir root | Video download state tracking |
| `download_log.json` | Per-course | Material download tracking |
| `captions/*.json` | Per-course | Whisper transcript (timestamped segments) |
| `alignment/*.json` | Per-course | Full segment-level alignment |
| `alignment/*.compact.json` | Per-course | Token-efficient alignment for LLM |
| `notes/sections/L*_S*.md` | Per-course | Cached per-chunk note sections |
| `notes/*_notes.md` | Per-course | Final merged/per-video notes |
| `notes/*.score.json` | Per-course | Self-score breakdown |
| `notes/images/L*/` | Per-course | Rendered slide PNGs |

## Testing

Tests are in `test/` and organized by scope:

| File | Scope |
|------|-------|
| `test_unit.py` | Offline unit tests: Viterbi, timeline, hashing, alignment parsing, slide discovery |
| `test_pipeline.py` | Integration tests: script execution, manifest schema, CLI flags (some require network) |
| `test_language_and_skip.py` | Language selection, terminal output (no CJK in prints), skip logic with/without --force |
| `test_note_generation.py` | Note generation specific tests |
| `test_gui.py` | GUI-specific tests |
| `electron/test/main.test.js` | Electron main process tests |

Run all offline tests: `python -m pytest test/ -v -k "not integration"`
