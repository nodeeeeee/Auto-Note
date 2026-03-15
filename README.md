# auto_note

Automated lecture note generator for NUS Canvas courses.
Downloads lecture videos and slides, transcribes audio, aligns transcripts to slides, and produces comprehensive Markdown study notes using GPT.

---

## Overview

```
Canvas LMS
   │
   ├─ downloader.py            Unified downloader (videos + materials)
   │     ├─ [course_id]/videos/      .mp4 files
   │     └─ [course_id]/materials/   PDFs, PPTXs, etc.
   │
   ▼
   extract_caption.py          Transcribe video → timestamped JSON
         │
         ▼
   semantic_alignment.py       Align transcript segments → slide pages
         │
         ▼
   note_generation.py          Generate Markdown study notes (GPT)
         │
         ▼
   [course_id]/notes/
         ├─ CourseName_notes.md
         └─ sections/          Per-chunk cached section files
```

---

## Setup

### Conda environment

```bash
conda activate auto-note
```

### API keys

| Key | Location |
|-----|----------|
| Canvas token | `canvas_token.txt` or `CANVAS_TOKEN` constant in `downloader.py` |
| OpenAI key | `openai_api.txt` or `OPENAI_API_KEY` env var |

### Course IDs

Known NUS courses:

| ID | Course |
|----|--------|
| 85367 | CS2101 |
| 85377 | CS2103 |
| 85397 | CS2105 |
| 85427 | CS3210 |

---

## Pipeline — Step by Step

### 1. Download videos and materials — `downloader.py`

Unified Canvas downloader combining video (Panopto) and material (Files API) downloads into a single CLI.

#### Course discovery

```bash
python downloader.py --course-list          # list all academic courses with IDs
```

#### Video operations

```bash
python downloader.py --video-list                       # list all available videos (global numbers)
python downloader.py --video-list --course 85427        # list videos for one course (course-local numbers)

python downloader.py --download-video 1 3 5             # download by global number
python downloader.py --download-video 2 4 --course 85427  # download by course-local number
python downloader.py --download-video-all               # download all pending videos
python downloader.py --download-video-all --course 85427  # all pending for one course
```

**Output:** `[course_id]/videos/[title].mp4`

A `manifest.json` in the project root tracks downloaded videos and prevents re-downloads.

#### Material operations

```bash
python downloader.py --material-list                    # list all downloadable files
python downloader.py --material-list --course 85427     # files for one course

python downloader.py --download-material "L02"          # download by partial filename match
python downloader.py --download-material-all            # download all pending materials
python downloader.py --download-material-all --course 85427
```

**Smart size filter:** If a course's total files exceed 1 GB, GPT identifies only lecture notes and tutorials for download, skipping large multimedia assets.

**Output:** `[course_id]/materials/[canvas_subfolder]/[filename]`
A `download_log.json` per course prevents re-downloading already-present files.

#### Common flags

| Flag | Description |
|------|-------------|
| `--course ID` | Restrict all operations to a single course |
| `--secretly` | Random delays between downloads to avoid rate-limiting (5–15 min between videos, 2–5 min between folders) with tqdm countdown bars |
| `--path PATH` | Override base directory (default: directory of `downloader.py`) |

#### Stealth mode example

```bash
python downloader.py --download-video-all --course 85427 --secretly
python downloader.py --download-material-all --secretly --path /data/courses
```

---

### 2. Transcribe videos — `extract_caption.py`

Transcribes `.mp4` lecture videos to word-level timestamped JSON using **faster-whisper large-v3** on GPU. The model reads video files directly (no separate audio extraction step).

```bash
python extract_caption.py              # process all pending videos
python extract_caption.py --video PATH # single video file
```

**Requires:** CUDA GPU (enforced at startup — will not silently fall back to CPU).

**Output:** `[course_id]/captions/[title].json`
Each JSON contains word-level timestamps and confidence scores.

---

### 3. Align transcript to slides — `semantic_alignment.py`

Maps Whisper transcript segments to specific slide pages using dense semantic search.

```bash
# Align one caption to one slide file
python semantic_alignment.py \
    --caption  85427/captions/lecture.json \
    --slides   85427/materials/LectureNotes/L02.pdf

# Align one caption to multiple slide files (multi-part lecture)
python semantic_alignment.py \
    --caption  85427/captions/L04.json \
    --slides   85427/materials/L04-Part1.pdf 85427/materials/L04-Part2.pdf

# Auto-discover all unaligned pairs in a course
python semantic_alignment.py --course 85427
```

**How it works:**

1. Extracts text from each slide (PDF / PPTX / DOCX), including speaker notes
2. For slides with fewer than 20 words, uses **GPT-4o-mini vision** to generate a text description of the visual content (result cached in `[slide].image_cache.json`)
3. Embeds all slides with **all-mpnet-base-v2** (sentence-transformers, GPU) into a FAISS cosine-similarity index
4. Queries the index for each transcript segment, using a ±30 s context window to pool nearby speech into richer queries
5. Applies **Viterbi temporal smoothing** with a forward bias so slide assignments only move forward in time (with configurable backward-revisit cost)
6. Flags segments that match no slide well as `off_slide`
7. Collapses consecutive same-slide assignments into a compact timeline

**Caption↔slide file matching (auto-discovery):**
When running `--course`, the aligner must decide which slide file(s) belong to each caption. It tries three strategies in order:

| Priority | Strategy | Example |
|----------|-----------|---------|
| 1 | **Lecture number** — regex extracts the number from both filenames and pairs equal numbers | `L02-foo.json` → `L02-Processes-Threads.pdf` |
| 2 | **Token overlap** — Jaccard similarity on filename words (threshold > 0.05) | `Processes-Threads.json` → `L02-Processes-Threads.pdf` |
| 3 | **Content embedding** — samples transcript text and slide text, embeds both with `all-mpnet-base-v2`, pairs the slide file with the highest cosine similarity (threshold ≥ 0.20) | `alpha_recording.json` → `L02-Processes-Threads.pdf` (sim=0.79) |

The content-based fallback means **files can be named anything** — if name matching fails, the aligner reads the actual content to find the right pair. A `[content-match]` line in the log indicates the fallback fired.

**Multi-file support:** When multiple slide files share the same lecture number (e.g. `L04-Part1.pdf` and `L04-Part2.pdf`), the aligner builds a single combined FAISS index, runs one Viterbi pass, then splits results back per file with local slide numbering. One JSON is saved per slide file (named `{slide_stem}.json`) so `note_generation.py` can locate them automatically.

**Output:** `[course_id]/alignment/[slide_stem].json`

---

### 4. Generate study notes — `note_generation.py`

Produces a single comprehensive Markdown note file covering all lectures in a course.

```bash
python note_generation.py --course 85427 --course-name "CS3210 Parallel Computing"
```

#### Key options

| Flag | Description |
|------|-------------|
| `--detail 0–10` | Note verbosity. `5` = medium (bullet points), `8` = detailed paragraphs (default `7`) |
| `--lectures N-N` | Process a subset of lectures, e.g. `--lectures 1-3` or `--lectures 1,4,5` |
| `--force` | Regenerate all sections even if cached |
| `--merge-only` | Skip generation; re-run only the merge + image filter pass |
| `--iterate` | Auto-increase detail level until self-score target is reached |

#### Architecture

**Section-by-section generation:**
Each lecture is split into chunks of ~15 slides. Each chunk is sent to GPT in a separate API call and saved as an individual section file (`sections/L04_S02.md`). On re-runs, existing section files are reused unless `--force` is set.

**Multi-file lecture support:**
Multiple slide files for the same lecture number (e.g. `L04-Part1.pdf` and `L04-Part2.pdf`) are grouped automatically. Their sections are written to separate files (`L04_S01.md`, `L04_F02_S01.md`) and merged under a single `## Lecture 4` heading with `### Part 1:` / `### Part 2:` sub-headings.

**Detail levels:**

| Range | Style |
|-------|-------|
| 0–2 | Bullet-point outline only |
| 3–5 | Hierarchical bullets — main concept + indented sub-details |
| 6–8 | Full paragraphs with examples and analogies |
| 9–10 | Maximum detail including edge cases and cross-lecture links |

**Image injection:**
Slide images are rendered to `notes/images/L{N}/` and injected into notes at relevant positions. For multi-file lectures, additional files render to `notes/images/L{N}_F{idx}/`.

**Image filter agent:**
After all sections are merged, a vision-based agent (GPT-4o-mini) reviews every embedded image and removes slides that do not contain meaningful visual elements. The decision priority is:

1. Slide has a cached visual description from semantic alignment → **KEEP**
2. Slide text matches a title/divider pattern → **REMOVE**
3. All other slides → **vision API** with a structured prompt that keeps diagrams, flowcharts, architecture illustrations, graphs, and tables, and removes text-only or code-only slides

**Generator–verifier loop:**
Each section draft is verified by `gpt-4.1-mini`, which checks technical term accuracy against the slide content. Suspicious verifier responses are discarded to prevent overwriting valid drafts.

**Exam notes:**
A final `## Exam Notes` section is auto-generated summarising up to 30 key exam points across all lectures.

**Self-scoring:**
After generation, the pipeline prints a heuristic self-score (word coverage, terminology hit-rate, callout density, code block count).

#### Output structure

```
[course_id]/notes/
├─ CourseName_notes.md          Final merged note
├─ CourseName_notes.score.json  Self-score breakdown
├─ sections/
│   ├─ L01_S01.md               Cached section files (single-file lectures)
│   ├─ L04_F02_S01.md           Cached section files (multi-file lecture, part 2)
│   └─ exam_notes.md
└─ images/
    ├─ L01/                     Rendered slide PNGs (single-file lectures)
    └─ L04_F02/                 Rendered slide PNGs (multi-file lecture, part 2)
```

---

## Incremental updates

| Scenario | Command |
|----------|---------|
| New lecture added | Normal run — new sections generated, existing ones cached |
| New slides added to existing lecture | `--lectures N --force` to regenerate that lecture only |
| New video/alignment added to existing lecture | `--lectures N --force` |
| Only prompts or image filter changed | `--merge-only` to re-merge without regenerating sections |

---

## Utilities

### `alignment_parser.py`

Converts the full alignment JSON (~300 KB) into a compact per-slide representation (~30 KB) for token-efficient LLM prompting. Used internally by `note_generation.py`.

```bash
python alignment_parser.py 85427/alignment/lecture.json
python alignment_parser.py 85427/alignment/lecture.json --out compact.json
```

---

## Hardware requirements

| Component | Requirement |
|-----------|-------------|
| GPU | CUDA-capable (RTX series recommended; tested on RTX 5070 Ti 16 GB) |
| VRAM | ≥ 8 GB for Whisper large-v3 + sentence-transformers |
| PyTorch | 2.10+ with CUDA 12.8 for Blackwell (sm_120) support |

---

## Notes on writing style

Notes are written in **Chinese** by default (as configured in `_SYSTEM` prompt), with English technical terms preserved inline (e.g. 进程 (Process)). The prompt enforces:

- No professor-centric narration ("老师说…", "教授指出…") — only first-person knowledge statements
- LaTeX for formulas (`$...$` inline, `$$...$$` block)
- Complete, compilable code examples with correct language tags
- `> [!IMPORTANT]` callout blocks for exam-critical content
