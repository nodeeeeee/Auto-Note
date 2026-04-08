# AutoNote

AutoNote is a desktop application that automatically generates comprehensive study notes from Canvas LMS lecture materials and videos. It downloads slides and videos, transcribes audio, aligns the transcript to slide pages, and produces Markdown note files per lecture using an LLM.

---

## Quick start

1. **Launch** the AutoNote AppImage / installer. On first launch, a **setup wizard** guides you through installing the ML environment.
2. **Settings** → enter Canvas URL and Canvas token → **Save All** → **Refresh Courses**
3. **Settings** → enter at least one LLM API key (OpenAI / Anthropic / Gemini), or use **Claude CLI** if you have Claude Code installed (no API key needed)
4. **Pipeline** → select course → click **Run Pipeline**

---

## Interface overview

The app has seven pages, accessible from the left navigation rail:

| Page | Purpose |
|------|---------|
| **Dashboard** | Course overview: video / caption / alignment counts per course; click a course to view per-video status and delete generated files |
| **Pipeline** | One-click full pipeline wizard with pipelined execution |
| **Download** | Fine-grained control over video and material downloads |
| **Transcribe** | Transcribe downloaded videos to timestamped captions |
| **Align** | Map caption segments to specific slide pages |
| **Generate Notes** | Generate the final Markdown study notes |
| **Settings** | API keys, ML environment, model selection, tunable constants |

Every page has a **terminal panel** pinned to the bottom that streams live output. The **Stop** button cancels the running process at any time.

---

## First-time setup

### Setup wizard (automatic)

On first launch, AutoNote checks if the required ML environment is installed. If not, a **setup wizard** appears with:

- **Required components** (pre-checked, cannot be skipped): Core Python packages for the pipeline
- **Optional components** (pre-checked, can be unchecked):
  - Local transcription (faster-whisper + GPU)
  - Local embeddings (sentence-transformers + FAISS)
  - BGE-M3 model for high-quality video-slide matching
  - Panopto video download (Playwright browser)

Click **Install and Continue** to install everything, or **Skip for now** to configure later in Settings.

### Connection settings

Open **Settings** and fill in the **Connection** card:

| Field | Value |
|-------|-------|
| **Canvas URL** | Your institution's Canvas domain (e.g. `canvas.nus.edu.sg`). |
| **Panopto Host** | Panopto video host (e.g. `mediaweb.ap.panopto.com`). Leave blank — it is auto-detected. |
| **Output Dir** | Directory where all pipeline files are stored (default: `~/AutoNote`). |

### API keys

Fill in the **API Keys & Credentials** card:

| Field | Required for |
|-------|-------------|
| **Canvas Token** | Downloading materials and listing videos. Get it from Canvas → Account → Settings → New Access Token. |
| **OpenAI API Key** | Note generation with OpenAI models (gpt-5.1, gpt-4.1, o3, ...). |
| **Anthropic API Key** | Note generation with Claude models via API. |
| **Gemini API Key** | Note generation with Gemini models. |

Alternatively, if you have **Claude Code** installed on your computer, you can select **Claude CLI (local)** as the note generation model — no API key needed.

Click **Save All**, then **Refresh Courses**.

> **GPU note:** Whisper large-v3 and sentence-transformer embeddings run on GPU. A CUDA-capable GPU with >= 8 GB VRAM is recommended. The app works on CPU but transcription will be much slower.

---

## Running the full pipeline

Go to **Pipeline**, select a course from the dropdown, and click **Run Pipeline**.

### Pipeline steps

| # | Step | What it does |
|---|------|-------------|
| 1 | **Download materials** | Downloads all lecture slides, PDFs, and other files from Canvas |
| 2 | **Download videos** | Downloads all Panopto lecture recordings (MP4) |
| 3 | **Transcribe + Align** | Transcribes videos, extracts frames from screen recordings, and aligns transcripts to slides. Uses **pipelined threading**: while video N+1 is transcribing, video N's frames are extracted concurrently. |
| 4 | **Generate study notes** | Sends slides + aligned transcripts to an LLM to generate one Markdown note file per lecture |

Each step only runs if the previous step succeeded. You can uncheck any step to skip it.

### Pipeline options

| Option | Description |
|--------|-------------|
| **Slack mode** | Adds random delays between downloads to avoid rate-limiting. |
| **Course name** | Name that appears in the final notes file (auto-filled when you select a course). |
| **Language** | Language for generated notes: English or Chinese. Selectable per-run — notes are generated in English first, then translated. Changing language auto-regenerates cached sections. |
| **Detail level** | Controls note verbosity (0-2 outline, 3-5 bullets, 6-8 paragraphs, 9-10 exhaustive). |
| **Lecture filter** | Process only specific lectures, e.g. `1-5` or `1,3,5`. Leave blank for all. |
| **Force regenerate** | Re-run the selected pipeline steps even if output files already exist. Without this, only missing files are processed. |

### Screen recording detection

For screen-share videos (slide recordings), the pipeline automatically:
1. Classifies the video as screen vs camera using heuristics
2. Extracts unique frames via scene detection + perceptual hash deduplication
3. Picks the most informative frame per slide page (handles incremental reveals)
4. Builds timestamp-based transcript-to-frame alignment

Camera recordings use traditional slide-based alignment with PDF/PPTX files.

---

## Dashboard

Click any course card to open a **detail modal** showing all transcribed videos with their processing status:

- **Aligned** / **No align**: whether the video has been aligned to slides
- **Notes** / **No notes**: whether note sections exist for this video
- **Delete** button: removes the transcript, alignment, note sections, per-video notes, and rendered images for that video

---

## Generate Notes page

Generates one Markdown note file per lecture (multi-part lectures are merged into a single file).

### Options

| Option | Description |
|--------|-------------|
| **Course name** | Used as the note file name and title. Auto-filled when you select a course. |
| **Language** | English or Chinese — selectable per-run. |
| **Lecture filter** | Generate notes for specific lectures only. |
| **Detail level** | 0-10 slider. Default: 7. |
| **Force regenerate** | Re-generate all sections. Without this, cached sections are reused. |
| **Merge-only** | Re-run the merge pass without re-generating any sections. |
| **Iterative mode** | Automatically raises detail level until quality target is met. |

### Output structure

```
<Output Dir>/<course_id>/notes/
├── CS2105_L01_notes.md       Per-lecture note files
├── CS2105_L02_notes.md
├── sections/
│   ├── L01_S01.md            Cached per-chunk sections (resumable)
│   ├── L01_S02.md
│   └── .language             Language marker for cache invalidation
└── images/
    ├── L01/                  Rendered slide images
    └── L02/
```

Each note file includes **source metadata** at the top:
```
- **Video**: CS2105 Lecture on 1_23_2026 (Fri)
- **Slides**: Lecture 1 - Introduction.pdf
```

Notes are **resumable**: re-running skips sections already cached on disk. Only missing sections are generated. Truncated responses (LLM token limit) are detected and not cached, so they auto-regenerate on the next run.

---

## Supported LLM models

| Provider | Models |
|----------|--------|
| **OpenAI** | gpt-5.1, gpt-5.2, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, o3, o4-mini |
| **Anthropic** | Claude Opus 4.6, Sonnet 4.6, Sonnet 4.5, Haiku 4.5 |
| **Google** | Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 2.0 Flash |
| **DeepSeek** | DeepSeek V3, DeepSeek R1 |
| **xAI** | Grok 3, Grok 3 mini |
| **Mistral** | Mistral Large, Medium, Small, Codestral |
| **Claude CLI** | Uses local `claude -p` (no API key needed, requires Claude Code installed) |

Use `--model` on the CLI to override: `python note_generation.py --model claude-cli --language zh`

---

## Incremental updates

| Scenario | What to do |
|----------|-----------|
| New lecture added to Canvas | Re-run the pipeline — only new sections are generated |
| Slides updated for a lecture | **Generate Notes → Lecture filter + Force regenerate** for that lecture |
| Want to change note language | Select the new language — cached sections auto-regenerate when the language differs |
| Want to delete bad notes | Dashboard → click course card → click **Delete** next to the video |
| Pipeline stopped partway | Just run again — completed steps are skipped |

---

## File layout

All pipeline output lives under the configured **Output Dir** (`~/AutoNote` by default):

```
~/AutoNote/
├── manifest.json                    Video download state
├── <course_id>/
│   ├── videos/                      Downloaded MP4 recordings
│   ├── materials/                   Downloaded slides, PDFs
│   ├── captions/                    Whisper transcripts (JSON)
│   ├── frames/                      Extracted screen recording frames
│   ├── alignment/                   Alignment JSON per slide file
│   └── notes/                       Generated notes + images
└── download_log.json                Material download log
```

App configuration and ML environment:

```
~/.auto_note/
├── config.json          Canvas URL, Panopto host, output dir
├── canvas_token.txt     Canvas API token
├── openai_api.txt       OpenAI API key
├── anthropic_key.txt    Anthropic API key
├── gemini_api.txt       Gemini API key
├── scripts/             Pipeline scripts (installed from AppImage)
└── venv/                ML virtual environment
```

---

## Windows installation

Two installer formats are available:
- **NSIS (.exe)** — traditional installer
- **MSIX (.appx)** — modern Windows package (less SmartScreen friction)

When uninstalling from Windows Settings, a dialog asks what to keep:
- Generated notes and downloads (kept by default)
- Settings and API keys (kept by default)
- ML environment ~2 GB (deleted by default)

---

## Troubleshooting

**"No courses loaded" on the Dashboard**
→ Go to Settings, enter Canvas URL and Canvas token, click Save All, then Refresh Courses.

**Video list shows 0 videos for a course**
→ The Panopto host has not been detected yet. Click **List videos** once; it is auto-detected. If it still fails, enter the Panopto domain manually in Settings.

**Transcription is very slow**
→ Running on CPU. Ensure CUDA GPU is present and PyTorch was installed with GPU support. Use Settings → ML Environment → **Reinstall** if needed.

**ModuleNotFoundError when running the pipeline**
→ The ML environment is missing packages. Go to Settings → ML Environment → **Reinstall**.

**Notes are in the wrong language**
→ Select the correct language from the Pipeline or Generate page dropdown. If cached sections exist in the old language, they are automatically regenerated.

**Notes have truncated sections**
→ The LLM hit its token limit. Truncated sections are not cached, so re-running will regenerate them. If it persists, try a model with a larger output window.
