# AutoNote

AutoNote turns Canvas / Panopto lecture recordings into Markdown study notes.
It downloads videos and slides, transcribes the audio, lines up each transcript
with the slide that was on screen, and asks an LLM to write up the lecture as
clean notes — one Markdown file per video, with images embedded.

If you have lecture videos and slides somewhere on Canvas, AutoNote will turn
them into notes you can read.

---

## Quick start

1. **Install** the AutoNote app for your platform (Windows, macOS, Linux).
2. On first launch the **setup wizard** installs the Python ML environment
   (~2 GB). Click **Install and Continue** and wait ~10 min.
3. Open **Settings** and fill in:
   - **Canvas URL** (e.g. `canvas.nus.edu.sg`)
   - **Canvas Token** (Canvas → Account → Settings → New Access Token)
   - one **LLM API key** — OpenAI, Anthropic, or Gemini.
     *(Or skip API keys: pick **Codex CLI** or **Claude CLI** as the backend
     if you have either installed and logged in.)*
4. Click **Save All** → **Refresh Courses**.
5. Go to **Pipeline**, pick a course, click **Run Pipeline**.

When it's done, your notes are in `~/AutoNote/<course>/notes/` (or your
chosen Output Dir).

---

## The seven pages

| Page | What it's for |
|------|---------------|
| **Dashboard** | Overview of every course you have access to. Click a course card to see per-video status and delete bad notes. |
| **Pipeline** | One-click full run: download → transcribe → align → generate. The fastest path to "I want notes for this course." |
| **Download** | Just downloading: pick specific videos or materials. |
| **Transcribe** | Just Whisper transcription, on already-downloaded videos. |
| **Align** | Just match transcripts to slides / video frames. |
| **Generate Notes** | Just the LLM step. Use this to re-generate or tweak settings without redoing the heavy work. |
| **Settings** | API keys, Canvas / Panopto URLs, ML environment management. |

A terminal panel at the bottom of every page streams live output. Click **Stop**
to cancel a run.

---

## Pipeline page

Tick which steps to run, set the options below, click **Run Pipeline**.

### Steps

You can independently enable or disable each step. Run the whole thing on a
new course, or just the last step when you're tweaking notes.

- Download materials
- Download videos
- Transcribe videos
- Align transcripts
- Generate study notes

### Note generation options

| Option | What it does |
|--------|--------------|
| **Course name** | Title that appears on the generated notes. Auto-filled when you pick a course. |
| **Language** | English or 中文. Switching languages re-runs the LLM step. |
| **Image source** | **Video screenshots** (default) — extract frames straight from the recording. **Slide PDFs** — render the downloaded slides instead. Pick screenshots when the lecturer used materials they didn't upload to Canvas. |
| **Backend** | **Default** uses your configured API key (OpenAI / Anthropic / Gemini). **Codex CLI** uses your `codex login` (no API key). **Claude CLI** uses your Claude Code login (no API key). |
| **Detail level** | 0–10 slider. 0–2 outline, 3–5 bullets, 6–8 paragraphs (default 7), 9–10 exhaustive. |
| **Lecture filter** | Only process specific lectures, e.g. `1-5` or `1,3,5`. Blank = all. |
| **Force regenerate** | Ignore cached sections and start over. |
| **Slack mode** | Adds a 5–15 min random pause between video downloads to avoid rate-limiting. |

### Image source: which one to pick?

| You have… | Pick |
|-----------|------|
| The lecture's slide PDFs in Canvas Files | Either works. **Video screenshots** is slightly more aligned with what was actually shown. |
| Only the videos (no slide PDFs) | **Video screenshots** — it doesn't need the slides to exist. |
| Slides only, no videos | **Slide PDFs** (and skip the transcribe / align steps). |

If you pick **Video screenshots**, you can leave **Download materials**
unchecked — AutoNote won't ask for slides it doesn't need.

> **Tip.** For picture-in-picture lectures (slides + a webcam overlay)
> AutoNote's screen-vs-camera classifier sometimes used to bail out and skip
> frame extraction. Picking **Video screenshots** explicitly now forces frame
> extraction regardless of how the classifier votes.

### Backend: which one to pick?

| Backend | When to use it |
|---------|----------------|
| **Default (NOTE_MODEL)** | You have an API key. Set the model in Settings → Tunable Constants. Best quality and parallelism. |
| **Codex CLI** | You have an OpenAI ChatGPT subscription and `codex login` was successful. No API key needed. |
| **Claude CLI** | You have Claude Code installed locally. No API key needed. |

The first time you switch to Codex or Claude CLI on the Settings page,
AutoNote checks that the binary is on your `PATH`.

---

## Generate Notes page

Same options as the Pipeline page's note-generation block, but without the
download / transcribe / align steps. Useful when you want to:

- Re-run with a different language
- Re-run with a different backend
- Try a different image source
- Re-generate a single lecture

The note generator caches every chunk it produces under `notes/sections/`,
so re-running picks up where it left off — only the chunks that need
regenerating are re-asked. Switching language or hitting **Force regenerate**
invalidates the cache automatically.

---

## Where your files end up

By default everything lives in `~/AutoNote/` (configurable in Settings).
For each course:

```
~/AutoNote/<course_id>/
├── videos/        Downloaded MP4 recordings
├── materials/     Slides, PDFs (only if you downloaded them)
├── captions/      Whisper transcripts (JSON)
├── frames/        Video screenshots (only if image source = frames)
├── alignment/     Caption ↔ slide / frame mapping (JSON)
└── notes/
    ├── <Course> on 1_23_2026 (Fri)_notes.md     ← your notes
    ├── sections/                                 ← cached chunks
    └── images/                                   ← embedded pictures
```

The notes file is a self-contained Markdown document — open it in any
viewer, push it to Obsidian, paste it into Notion, share it with classmates.

---

## Dashboard

Click any course card. The detail modal shows every video and per-video
status badges:

- **Aligned** / **No align** — has the transcript been matched to slides
  or frames yet?
- **Notes** / **No notes** — do generated note sections exist?
- **Delete** — wipes captions, alignment, note sections, and images for
  that one video, so you can re-run cleanly.

---

## Output language

AutoNote can write notes in **English** or **中文**. The LLM drafts in
English first, then a separate translation pass produces the Chinese version
when you pick `zh`. Switching language regenerates only what needs to change.

---

## Troubleshooting

**"No courses loaded" on the Dashboard.**
Go to **Settings**, fill in Canvas URL + Canvas token, click **Save All**,
then **Refresh Courses**.

**Generation finished but the notes have no images.**
The screen-vs-camera classifier can mis-classify lectures with a webcam
overlay. Re-run the Pipeline with **Image source = Video screenshots** —
that bypasses the classifier and forces frames to be extracted.

**Video list shows 0 videos for a course.**
The Panopto host hasn't been auto-detected yet. Click **List videos** once.
If it still fails, fill in the Panopto domain manually in Settings.

**Transcription is very slow.**
You're running on CPU. Make sure CUDA is installed and use **Settings →
ML Environment → Reinstall**.

**`ModuleNotFoundError` mid-pipeline.**
The ML venv is missing a package. **Settings → ML Environment → Reinstall**.

**A note section looks cut off.**
The LLM hit its output token limit. Truncated sections aren't cached, so
re-running just regenerates them. If it keeps happening, switch to a model
with a larger output window in Settings.

**The downloaded video has no audio.**
Some Panopto recordings split the screen video and microphone audio into
separate streams. AutoNote merges them automatically with ffmpeg, but the
merge needs `ffmpeg` installed (the installer ships it). If you see this,
**Settings → ML Environment → Reinstall**.

---

## Need to script it?

AutoNote is a wrapper over a small set of Python scripts (`downloader.py`,
`extract_caption.py`, `frame_extractor.py`, `semantic_alignment.py`,
`note_generation.py`). If you want CLI-level control, batch automation, or
to contribute, see **DeveloperGuide.md**.
