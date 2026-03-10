Role: You are a Senior Full-Stack Engineer and AI Automation Expert. Your task is to design and implement a Python-based automation pipeline (or a browser extension + backend) that transforms a Canvas course page into structured, high-fidelity study notes.

The Workflow you must implement:
Step 1: Data Acquisition (Canvas & Panopto)

    Authentication: Implement a session-handling mechanism to log into Canvas (using API Tokens or Playwright-based SSO).

    Discovery: Recursively scan the Modules, Files, and Pages sections of a specific Course ID to find:

        Lecture Videos (primarily hosted on Panopto).

        Lecture Materials (PDF, PPTX).

    Action: Trigger Panopto-Video-DL via CLI to download high-resolution video streams. Save them to a structured directory: /course_id/module_name/video/.

Step 2: Multi-Modal Pre-processing

    Voice Isolation: Call the audio-separator library/CLI to isolate the professor's vocals from background noise or music.

    Transcription: Process the isolated audio using OpenAI Whisper (specifically faster-whisper with large-v3).

        Output Requirement: Generate a JSON file containing words and phrases with precise start/end timestamps.

    Visual Extraction: Convert downloaded PPTX/PDF slides into a series of high-resolution .png files. Use OCR (like Tesseract or a Vision model) to index the text content of each slide for searchability.

Step 3: Semantic Alignment (The "Agent" Logic)

    Contextual Mapping: Build a mapping engine that uses the Whisper timestamps and Slide OCR data.

    Heuristic: If the professor says "As you can see on this graph of the sigmoid function," the Agent must locate the slide containing "sigmoid" and "graph" and mark that timestamp as the "active period" for that slide.

Step 4: Intelligent Synthesis & Formatting

    The Detail Slider: Implement a variable DETAIL_LEVEL (0.0 to 1.0).

    Drafting: Generate a Markdown document where:

        Terminology Grounding: The Slides are the "Source of Truth." Correct any ASR (Speech-to-Text) errors based on slide text.

        Image Injection: Insert ![Slide X](path) at the exact semantic transition points.

        Formatting: Use professional LaTeX for math, syntax-highlighted blocks for code, and callouts (e.g., > [!IMPORTANT]) for exam-relevant tips mentioned by the professor.

Technical Constraints for Implementation:

    Modular Code: Write the code in a way that each step (Download, Transcribe, Align, Summarize) can be run independently or as a full pipeline.

    Error Handling: Include retry logic for failed Panopto downloads or API timeouts.

    State Management: Use a local database (e.g., SQLite) or a manifest.json to track which videos have already been processed to avoid redundant computation.

    Environment: Assume the environment has ffmpeg, nvidia-cuda-toolkit (for Whisper/Audio-separator), and python-canvasapi installed.

Output Request: Please provide the system architecture diagram, the database schema for tracking course progress, and a Python boilerplate that orchestrates these four steps.

