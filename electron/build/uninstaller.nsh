; AutoNote NSIS uninstaller customization.
;
; During uninstall, ask the user whether to keep each of three data areas.
; Implemented entirely inside customUnInstall — confirmed working in CI —
; without pages, vars, or LogicLib.

!pragma warning disable 6010
!pragma warning disable 6020
!pragma warning disable 8000

!macro customUnInstall
  ; ── Generated notes + downloaded course files ─────────────────────────────
  MessageBox MB_YESNO|MB_ICONQUESTION "Keep generated notes and downloaded course files in %USERPROFILE%\AutoNote?$\n$\nYes = keep them, No = delete" /SD IDNO IDYES autonote_keep_notes
  RMDir /r "$PROFILE\AutoNote"
autonote_keep_notes:

  ; ── ML environment (~2 GB) ────────────────────────────────────────────────
  MessageBox MB_YESNO|MB_ICONQUESTION "Keep ML environment (~2 GB) in %USERPROFILE%\.auto_note\venv?$\n$\nYes = keep it, No = delete" /SD IDNO IDYES autonote_keep_venv
  RMDir /r "$PROFILE\.auto_note\venv"
autonote_keep_venv:

  ; ── Settings, API keys, cached scripts ────────────────────────────────────
  MessageBox MB_YESNO|MB_ICONQUESTION "Keep settings and API keys in %USERPROFILE%\.auto_note?$\n$\nYes = keep them, No = delete" /SD IDNO IDYES autonote_keep_settings
  Delete "$PROFILE\.auto_note\config.json"
  Delete "$PROFILE\.auto_note\canvas_token.txt"
  Delete "$PROFILE\.auto_note\openai_api.txt"
  Delete "$PROFILE\.auto_note\anthropic_key.txt"
  Delete "$PROFILE\.auto_note\gemini_api.txt"
  Delete "$PROFILE\.auto_note\deepseek_api.txt"
  Delete "$PROFILE\.auto_note\grok_api.txt"
  Delete "$PROFILE\.auto_note\mistral_api.txt"
  Delete "$PROFILE\.auto_note\manifest.json"
  RMDir /r "$PROFILE\.auto_note\scripts"
  RMDir "$PROFILE\.auto_note"
autonote_keep_settings:
!macroend
