# 📓 AutoNote

> 🎬 Canvas / Panopto lecture videos → 📝 clean Markdown study notes — automatically.

AutoNote downloads videos and slides, transcribes the audio, lines up each
transcript with the slide that was on screen, and asks an LLM to write up
the lecture as clean notes — one Markdown file per video, with images
embedded.

**🎯 If you have lecture videos and slides somewhere on Canvas, AutoNote will
turn them into notes you can read!**

🏫 Currently Auto-Note only works for the NUS system. Feel free to
[migrate this project to your school's system](#need-to-script-it) if needed.

✨ **Highlights**

- 🎥 Pulls Panopto recordings directly via your Canvas LTI session
- 🗣️ Whisper-API transcription with optional GPU acceleration
- 🖼️ Auto-extracts slide screenshots from video frames
- 🤖 Multi-LLM backend — OpenAI · Anthropic · Gemini · DeepSeek · xAI · Mistral · Codex CLI · Claude CLI
- 🌏 English or 中文 output
- 🔁 Resumable cache — re-runs only re-do what changed

**Who this app is targeted at?**

1. University students who are cramming for midterms/finals;
2. Lecturers who want to make lecture notes for his own course;
3. Students who want to keep the notes for future revision.

Please advertise it to your friends if this app helps you survive the exams!
\
Your support and acknowledgement is very important to me.

---

## 🚀 Quick start

1. 💻 **Install** the AutoNote app for your platform (Windows, macOS, Linux).
2. ⚙️ On first launch the **setup wizard** installs the Python ML environment
   (~2 GB). Click **Install and Continue** and wait ~10 min.
3. 🔑 Open **Settings** and fill in:
   - 🎓 **Canvas URL** (e.g. `canvas.nus.edu.sg`)
   - 🪪 **Canvas Token** (Canvas → Account → Settings → New Access Token)
   - 🧠 one **LLM API key** — OpenAI, Anthropic, Gemini, or Deepseek.
     *(Or skip API keys: pick **Codex CLI** or **Claude CLI** as the backend
     if you have either installed and logged in.)*
4. 💾 Click **Save All** → 🔄 **Refresh Courses**.
5. ▶️ Go to **Pipeline**, pick a course, click **Run Pipeline**.

📂 When it's done, your notes are in `~/AutoNote/<course>/notes/` (or your
chosen Output Dir).


⭐ **It is recommended to use Deepseek-v4-pro as your note-generation agent.**

> ⚠️ *Warning:* If you are using Gemini / Claude / OpenAI, your account should
> have **~$1 per lecture** in credit so it doesn't burn out all your tokens
> before the run finishes.


---

## 🧭 The seven pages

| Page | What it's for |
|------|---------------|
| 📊 **Dashboard** | Overview of every course you have access to. Click a course card to see per-video status and delete bad notes. |
| ▶️ **Pipeline** | One-click full run: download → transcribe → align → generate. The fastest path to "I want notes for this course." |
| ⬇️ **Download** | Just downloading: pick specific videos or materials. |
| 🗣️ **Transcribe** | Just Whisper transcription, on already-downloaded videos. |
| 🧩 **Align** | Just match transcripts to slides / video frames. |
| ✍️ **Generate Notes** | Just the LLM step. Use this to re-generate or tweak settings without redoing the heavy work. |
| ⚙️ **Settings** | API keys, Canvas / Panopto URLs, ML environment management. |

💻 A terminal panel at the bottom of every page streams live output. Click ⏹️ **Stop**
to cancel a run.

---

## ▶️ Pipeline page

Tick which steps to run, set the options below, click **Run Pipeline**.

### 🔧 Steps

You can independently enable or disable each step. Run the whole thing on a
new course, or just the last step when you're tweaking notes.

- 📚 Download materials
- 🎥 Download videos
- 🗣️ Transcribe videos
- 🧩 Align transcripts
- ✍️ Generate study notes

### 🛠️ Note generation options

| Option | What it does |
|--------|--------------|
| 🏷️ **Course name** | Title that appears on the generated notes. Auto-filled when you pick a course. |
| 🌏 **Language** | English or 中文. Switching languages re-runs the LLM step. |
| 🖼️ **Image source** | **Video screenshots** (default) — extract frames straight from the recording. **Slide PDFs** — render the downloaded slides instead. Pick screenshots when the lecturer used materials they didn't upload to Canvas. |
| 🤖 **Backend** | **Default** uses your configured API key (OpenAI / Anthropic / Gemini). **Codex CLI** uses your `codex login` (no API key). **Claude CLI** uses your Claude Code login (no API key). |
| 🎚️ **Detail level** | 0–10 slider. 0–2 outline, 3–5 bullets, 6–8 paragraphs (default 7), 9–10 exhaustive. |
| 🔍 **Lecture filter** | Only process specific lectures, e.g. `1-5` or `1,3,5`. Blank = all. |
| ♻️ **Force regenerate** | Ignore cached sections and start over. |
| 🐢 **Slack mode** | Adds a 5–15 min random pause between video downloads to avoid rate-limiting. (Though this is never reached when doing intensive testing) |

### 🖼️ Image source: which one to pick?

| You have… | Pick |
|-----------|------|
| 📄 The lecture's slide PDFs in Canvas Files | Either works. **Video screenshots** is slightly more aligned with what was actually shown. |
| 🎥 Only the videos (no slide PDFs) | **Video screenshots** — it doesn't need the slides to exist. |
| 🗂️ Slides only, no videos | **Slide PDFs** (and skip the transcribe / align steps). |

If you pick **Video screenshots**, you can leave **Download materials**
unchecked — AutoNote won't ask for slides it doesn't need.

> 💡 **Tip.** For picture-in-picture lectures (slides + a webcam overlay)
> AutoNote's screen-vs-camera classifier sometimes used to bail out and skip
> frame extraction. Picking **Video screenshots** explicitly now forces frame
> extraction regardless of how the classifier votes.

### 🤖 Backend: which one to pick?

| Backend | When to use it |
|---------|----------------|
| 🔑 **Default (NOTE_MODEL)** | You have an API key. Set the model in Settings → Tunable Constants. Best quality and parallelism. |
| 🦊 **Codex CLI** | You have an OpenAI ChatGPT subscription and `codex login` was successful. No API key needed. |
| 🟣 **Claude CLI** | You have Claude Code installed locally. No API key needed. |

The first time you switch to Codex or Claude CLI on the Settings page,
AutoNote checks that the binary is on your `PATH`.

---

## ✍️ Generate Notes page

Same options as the Pipeline page's note-generation block, but without the
download / transcribe / align steps. Useful when you want to:

- 🌏 Re-run with a different language
- 🔁 Re-run with a different backend
- 🖼️ Try a different image source
- 🎯 Re-generate a single lecture

💾 The note generator caches every chunk it produces under `notes/sections/`,
so re-running picks up where it left off — only the chunks that need
regenerating are re-asked. Switching language or hitting **Force regenerate**
invalidates the cache automatically.

---

## 📁 Where your files end up

By default everything lives in `~/AutoNote/` (configurable in Settings).
For each course:

```
~/AutoNote/<course_id>/
├── 🎥 videos/        Downloaded MP4 recordings
├── 📚 materials/     Slides, PDFs (only if you downloaded them)
├── 🗣️ captions/      Whisper transcripts (JSON)
├── 🖼️ frames/        Video screenshots (only if image source = frames)
├── 🧩 alignment/     Caption ↔ slide / frame mapping (JSON)
└── 📝 notes/
    ├── <Course> on 1_23_2026 (Fri)_notes.md     ← your notes
    ├── sections/                                 ← cached chunks
    └── images/                                   ← embedded pictures
```

📄 The notes file is a self-contained Markdown document — open it in any
viewer, push it to Obsidian, paste it into Notion, share it with classmates.

---

## 📊 Dashboard

Click any course card. The detail modal shows every video and per-video
status badges:

- 🧩 **Aligned** / **No align** — has the transcript been matched to slides
  or frames yet?
- 📝 **Notes** / **No notes** — do generated note sections exist?
- 🗑️ **Delete** — wipes captions, alignment, note sections, and images for
  that one video, so you can re-run cleanly.

---

## 🌏 Output language

AutoNote can write notes in 🇬🇧 **English** or 🇨🇳 **中文**. The LLM drafts in
English first, then a separate translation pass produces the Chinese version
when you pick `zh`. Switching language regenerates only what needs to change.

---

## 🛟 Troubleshooting

⚠️ **"No courses loaded" on the Dashboard.**
Go to **Settings**, fill in Canvas URL + Canvas token, click **Save All**,
then **Refresh Courses**.

⚠️ **Generation finished but the notes have no images.**
The screen-vs-camera classifier can mis-classify lectures with a webcam
overlay. Re-run the Pipeline with **Image source = Video screenshots** —
that bypasses the classifier and forces frames to be extracted.

⚠️ **Video list shows 0 videos for a course.**
The Panopto host hasn't been auto-detected yet. Click **List videos** once.
If it still fails, fill in the Panopto domain manually in Settings.

⚠️ **Transcription is very slow.**
You're running on CPU. Make sure CUDA is installed and use **Settings →
ML Environment → Reinstall**.

⚠️ **`ModuleNotFoundError` mid-pipeline.**
The ML venv is missing a package. **Settings → ML Environment → Reinstall**.

⚠️ **A note section looks cut off.**
The LLM hit its output token limit. Truncated sections aren't cached, so
re-running just regenerates them. If it keeps happening, switch to a model
with a larger output window in Settings.

⚠️ **The downloaded video has no audio.**
Some Panopto recordings split the screen video and microphone audio into
separate streams. AutoNote merges them automatically with ffmpeg, but the
merge needs `ffmpeg` installed (the installer ships it). If you see this,
**Settings → ML Environment → Reinstall**.

---

<a id="need-to-script-it"></a>

## 🛠️ Need to script it?

AutoNote is a wrapper over a small set of Python scripts (`downloader.py`,
`extract_caption.py`, `frame_extractor.py`, `semantic_alignment.py`,
`note_generation.py`). If you want CLI-level control, batch automation, or
to contribute, see 📖 **DeveloperGuide.md**.

The note generation part is the most fine-tuned part of this app. To adapt it to other universities, you only need to modify the material and video downloading part.

---

## 📜 Disclaimer

AutoNote is a personal study aid. Use it only on lecture material you
already have legitimate access to — recordings on your own course's Canvas,
slides your university distributes to enrolled students, or your own teaching
content. Do **not** use it to redistribute or republish copyrighted material.
You are responsible for complying with your institution's Acceptable Use
Policy and any applicable copyright law.

**🤖 AI output is not authoritative.** Generated notes are produced by large
language models and may contain errors, hallucinations, or omissions. They
are not a substitute for the lecture itself. Always verify against the
original recording / slides before relying on them for assessments.

**🔒 Privacy.** AutoNote runs entirely on your own machine and keeps all
data under `~/AutoNote/` and `~/.auto_note/`. There are no centralized
servers, no telemetry, and no analytics. The only outbound network traffic
goes to (a) Canvas + Panopto using your own token and (b) whichever LLM
provider you configured (OpenAI, Anthropic, Gemini, DeepSeek, …).
Those providers log requests under their own terms — do not feed the app
content you wouldn't be comfortable sending to them.

**⚖️ No warranty.** This software is provided "as is" without warranty of
any kind. The author is not liable for any damages arising from its use,
including but not limited to lost data, exceeded API quotas, or
academic-integrity decisions made by your institution.

**™️ Trademarks.** Canvas®, Panopto®, OpenAI®, Anthropic®, Google Gemini™,
DeepSeek™, and other marks belong to their respective owners. AutoNote is
not affiliated with, endorsed by, or sponsored by any of them.