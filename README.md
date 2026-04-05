# AutoNote

AutoNote is a desktop application that automatically generates comprehensive study notes from Canvas LMS lecture materials and videos. It downloads slides and videos, transcribes audio, aligns the transcript to slide pages, and produces a single Markdown note file per course using an LLM.

---

## Quick start

1. **Launch** the AutoNote AppImage (or run `python gui.py`)
2. **Settings** → enter Canvas URL and Canvas token → **Save All** → **Refresh Courses**
3. **Settings** → enter at least one LLM API key (OpenAI / Anthropic / Gemini) → **Save All**
4. **Settings → ML Environment** → click **Install ML Environment** (first time only, takes ~10 min)
5. **Pipeline** → select course → click **Run Pipeline**

---

## Interface overview

The app has six pages, accessible from the left navigation rail:

| Page | Purpose |
|------|---------|
| **Dashboard** | Course overview: video / caption / alignment counts per course; click a course to view per-video status and delete generated files; quick-action shortcuts |
| **Pipeline** | One-click full pipeline wizard: runs all five steps in sequence |
| **Download** | Fine-grained control over video and material downloads |
| **Transcribe** | Transcribe downloaded videos to timestamped captions |
| **Align** | Map caption segments to specific slide pages |
| **Generate Notes** | Generate the final Markdown study notes |
| **Settings** | API keys, ML environment, model selection, tunable constants |

Every page has a **terminal panel** pinned to the bottom that streams live output. The **Stop** button (×) cancels the running process at any time.

---

## First-time setup

### 1. Connection settings

Open **Settings** and fill in the **Connection** card:

| Field | Value |
|-------|-------|
| **Canvas URL** | Your institution's Canvas domain (e.g. `canvas.nus.edu.sg`). The `https://` prefix is added automatically. |
| **Panopto Host** | Panopto video host (e.g. `mediaweb.ap.panopto.com`). Leave blank — it is auto-detected the first time you list videos. |
| **Output Dir** | Directory where all pipeline files are stored (default: `~/AutoNote`). |

### 2. API keys

Fill in the **API Keys & Credentials** card:

| Field | Required for |
|-------|-------------|
| **Canvas Token** | Downloading materials and listing videos. Get it from Canvas → Account → Settings → New Access Token. |
| **OpenAI API Key** | Note generation when using an OpenAI model (gpt-5.1, gpt-4.1, o3, …). |
| **Anthropic API Key** | Note generation when using a Claude model. |
| **Gemini API Key** | Note generation when using a Gemini model (get from Google AI Studio). |

Click **Save All**, then **Refresh Courses**. Your enrolled courses appear in the Dashboard and all course dropdowns.

### 3. ML environment

The pipeline uses GPU-accelerated ML libraries (Whisper, sentence-transformers, FAISS) that are kept in a dedicated virtual environment at `~/.auto_note/venv/`. This environment is separate from the app itself so that heavy ML dependencies do not ship with the AppImage.

Go to **Settings → ML Environment** and click **Install ML Environment**. The installer:

1. Detects Python from your login shell or common conda/system locations
2. Creates a fresh venv
3. Detects CUDA and installs the matching version of PyTorch
4. Installs all pipeline packages (whisper, sentence-transformers, canvasapi, PanoptoDownloader, …)
5. Installs Playwright's Chromium browser (used to authenticate Panopto video downloads)

This only needs to be done once. Use **Reinstall** if packages become broken.

> **GPU note:** Whisper large-v3 and the sentence-transformer embedding model both run on GPU. A CUDA-capable GPU with ≥ 8 GB VRAM is recommended. The app still works on CPU but transcription will be much slower.

---

## Running the full pipeline

Go to **Pipeline**, select a course from the dropdown, and click **Run Pipeline**.

### Pipeline steps (executed in order)

| # | Step | What it does |
|---|------|-------------|
| 1 | **Download materials** | Downloads all lecture slides, PDFs, and other files from Canvas |
| 2 | **Download videos** | Downloads all Panopto lecture recordings (MP4) |
| 3 | **Transcribe videos** | Runs Whisper on each video to produce timestamped caption JSON |
| 4 | **Align transcripts** | Maps each caption segment to the slide page being shown at that moment |
| 5 | **Generate study notes** | Sends slides + aligned transcripts to an LLM to generate a Markdown note file |

Each step only runs if the previous step succeeded. You can uncheck any step to skip it.

### Pipeline options

| Option | Description |
|--------|-------------|
| **Slack mode** | Adds random delays between downloads (5–15 min between videos, 2–5 min between material folders) to avoid rate-limiting. Enable when downloading in bulk. |
| **Course name** | Name that appears in the final notes file (auto-filled when you select a course). |
| **Language** | Language for generated notes: English or Chinese. Selectable per-run without changing Settings. |
| **Detail level** | Controls note verbosity (see [Detail levels](#detail-levels) below). |
| **Lecture filter** | Process only specific lectures, e.g. `1-5` or `1,3,5`. Leave blank for all. |
| **Force regenerate** | Re-run the selected pipeline steps even if their output files already exist. Applies to whichever steps are checked (transcribe, align, and/or generate). Without this, only missing files are processed. |

---

## Download page

For fine-grained control without running the full pipeline.

### Videos

| Button | Action |
|--------|--------|
| **List videos** | Show all available Panopto videos with their numbers |
| **Download all pending** | Download every video not yet in the manifest |
| **Download selected** | Enter video number(s) in the text field (e.g. `1 3 5`), then click |

### Course materials

| Button | Action |
|--------|--------|
| **List materials** | Show all downloadable files on Canvas with sizes |
| **Download all pending** | Download all files not yet in `download_log.json` |
| **Download selected** | Enter partial filename(s), space-separated, then click |

**Smart size filter:** If a course has more than 1 GB of files, an LLM automatically identifies only lecture notes and tutorials for download, skipping large media assets.

All files are saved under the configured output directory:
- Videos → `<course_id>/videos/<title>.mp4`
- Materials → `<course_id>/materials/<canvas_subfolder>/<filename>`

---

## Transcribe page

Transcribes MP4 videos to word-level timestamped JSON using Whisper.

- **Transcribe all pending** — processes every video in the download manifest that does not yet have a caption file
- **Single video path** — enter a specific `.mp4` path to transcribe just that file

**Automatic backend selection:** If a CUDA GPU is available, faster-whisper runs locally. Otherwise, the OpenAI Whisper API is used (requires an OpenAI key in Settings).

Caption files are saved to `<course_id>/captions/<title>.json`.

---

## Align page

Maps each Whisper caption segment to the slide page being presented at that moment.

### Auto-discover (recommended)

Select a course and click **Run auto-align**. The aligner finds all unaligned caption/slide pairs automatically using a three-strategy matching system:

1. **Lecture number** — matches `L02-foo.json` to `L02-Processes.pdf`
2. **BGE-M3 embedding** — pre-matches all captions to slides using the BGE-M3 model for high-quality multilingual matching (default first option)
3. **Token overlap** — Jaccard similarity on filename words
4. **Content embedding fallback** — embeds transcript samples and slide text with mpnet; pairs by highest cosine similarity (fires when all above methods fail)

### Manual alignment

Enter a specific caption JSON path and one or more slide file paths (space-separated for multi-part lectures), then click **Align**.

Alignment results are saved to `<course_id>/alignment/<slide_stem>.json`.

---

## Generate Notes page

Generates a single comprehensive Markdown study note file covering all lectures.

### Options

| Option | Description |
|--------|-------------|
| **Course name** | Used as the note file name and title. Auto-filled when you select a course. |
| **Language** | English or Chinese — selectable per-run. |
| **Lecture filter** | Generate notes for specific lectures only (`1-5` or `1,3,5`). |
| **Detail level** | 0–10 slider (see below). Default: 7. |
| **Force regenerate** | Re-generate all section files. Without this, existing sections are reused from cache. |
| **Merge-only** | Re-run the merge and image-filter pass without re-generating any sections. Useful when you change only the merge prompt or image filter. |
| **Iterative mode** | Automatically increases the detail level until the self-score target is reached. |

### Detail levels

| Range | Style |
|-------|-------|
| 0–2 | Bullet-point outline only |
| 3–5 | Hierarchical bullets — main concepts with indented sub-details |
| 6–8 | Full paragraphs with examples and analogies |
| 9–10 | Exhaustive — includes edge cases and cross-lecture connections |

### Output structure

```
<Output Dir>/<course_id>/notes/
├── CourseName_notes.md          Final merged note
├── CourseName_notes.score.json  Self-score breakdown
├── sections/
│   ├── L01_S01.md               Cached section files (resumable)
│   └── exam_notes.md
└── images/
    ├── L01/                     Rendered slide images
    └── L02/
```

Notes are **resumable**: if generation stops midway, re-running skips sections already saved on disk. Use **Force regenerate** to redo them.

---

## Settings — advanced options

### Tunable constants

Settings exposes all configurable parameters directly in the UI. Changes are written back into the pipeline scripts.

**Transcription**

| Setting | Description | Default |
|---------|-------------|---------|
| Whisper model variant | Size vs. accuracy trade-off | `large-v3` |
| Transcription language | Language hint for Whisper; auto-detect works for most lectures | auto-detect |

Available Whisper models: `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`, `large-v3-turbo`, `distil-large-v3`.

**Alignment**

| Setting | Description | Default |
|---------|-------------|---------|
| Sentence-transformer model | Embedding model for semantic matching | `all-mpnet-base-v2` |
| Context window (s) | Seconds of speech pooled around each segment for richer queries | `30` |
| Off-slide cosine cutoff | Segments below this similarity are marked `off_slide` | `0.28` |
| Temporal prior σ | Strength of the forward-time prior in Viterbi smoothing | `5` |

Available embedding models: `all-mpnet-base-v2` (best quality), `all-MiniLM-L12-v2` (balanced), `all-MiniLM-L6-v2` (fast), `paraphrase-multilingual-mpnet-base-v2`.

**Note generation**

| Setting | Description | Default |
|---------|-------------|---------|
| Note language | Default language for generated notes (can be overridden per-run from Pipeline/Generate pages) | English |
| Note generation LLM | Primary model used to write sections | `gpt-5.1` |
| Verification LLM | Lighter model used to check each section for accuracy | `gpt-4.1-mini` |
| Default detail level | Starting detail level when none is specified | `8` |
| Slides per GPT call | Slides sent in one LLM call (one section) | `15` |
| Self-score target | Minimum score to accept in iterative mode | `8.0` |

### Supported LLM models

| Provider | Note generation | Verification |
|----------|----------------|-------------|
| **OpenAI** | gpt-5.1, gpt-5.2, gpt-4.1, gpt-4.1-mini, o3, o1 | gpt-4.1-mini, gpt-4.1, gpt-5.1 |
| **Anthropic** | Claude Opus 4.6, Sonnet 4.6, Sonnet 4.5, Sonnet 3.5, Haiku 4.5 | Claude Haiku 4.5, Sonnet 3.5, Sonnet 4.5 |
| **Google** | Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 2.0 Flash, Gemini 1.5 Pro | Gemini 2.5 Flash, Gemini 2.0 Flash |

The API key for the selected provider must be saved in Settings.

---

## Incremental updates

| Scenario | What to do |
|----------|-----------|
| New lecture added to Canvas | Re-run the full pipeline — existing sections are cached, only new ones are generated |
| Slides updated for an existing lecture | **Generate Notes → Lecture filter + Force regenerate** for that lecture |
| Want to change note style or language | Select the new language in the Pipeline/Generate page dropdown, then re-run with **Force regenerate** |
| Want to delete bad notes for a video | Dashboard → click the course card → click **Delete** next to the video |
| Pipeline stopped partway | Just run again — already-completed sections are skipped automatically |

---

## File layout

All pipeline output lives under the configured **Output Dir** (`~/AutoNote` by default):

```
~/AutoNote/
├── manifest.json                    Video download manifest (tracks download state)
├── <course_id>/
│   ├── videos/                      Downloaded MP4 lecture recordings
│   ├── materials/                   Downloaded slides, PDFs, etc.
│   │   └── <canvas_folder>/
│   ├── captions/                    Whisper transcript JSON per video
│   ├── alignment/                   Alignment JSON per slide file
│   └── notes/
│       ├── <CourseName>_notes.md    Final generated note
│       ├── sections/                Per-section cache (enables resume)
│       └── images/                  Rendered slide images embedded in notes
└── download_log.json                Per-course material download log
```

App configuration and the ML environment are stored in `~/.auto_note/`:

```
~/.auto_note/
├── config.json          Canvas URL, Panopto host, output dir
├── canvas_token.txt     Canvas API token
├── openai_api.txt       OpenAI API key
├── anthropic_key.txt    Anthropic API key
├── gemini_api.txt       Gemini API key
├── scripts/             Pipeline scripts installed from the AppImage
└── venv/                ML virtual environment (installed via Settings)
```

---

## Troubleshooting

**"No courses loaded" on the Dashboard**
→ Go to Settings, enter Canvas URL and Canvas token, click Save All, then Refresh Courses.

**Video list shows 0 videos for a course**
→ The Panopto host has not been detected yet. Click **List videos** once; it is auto-detected from browser network requests. If it still fails, enter the Panopto domain manually in Settings → Panopto Host (e.g. `mediaweb.ap.panopto.com`).

**Transcription is very slow**
→ The app is running on CPU. Ensure a CUDA GPU is present and PyTorch was installed with GPU support. Use Settings → ML Environment → **Reinstall** if needed.

**ModuleNotFoundError when running the pipeline**
→ The ML environment is missing packages. Go to Settings → ML Environment → **Reinstall**.

**Note generation produces empty or very short sections**
→ If using a reasoning model (o3, gpt-5.1), ensure the model is receiving enough output tokens. The app sets `max_completion_tokens=10000` automatically for reasoning models.

**Pipeline jumps ahead to a later step after an error**
→ Update to the latest version (v0.6.35+). Earlier builds had a bug where the pipeline chain advanced even when a step failed.
