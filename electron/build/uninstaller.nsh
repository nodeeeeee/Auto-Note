; AutoNote NSIS uninstaller customization
; Shows a dialog asking the user what to retain when uninstalling.

!include "MUI2.nsh"

Var KeepNotes
Var KeepSettings
Var KeepVenv

; ── Custom uninstall page ──────────────────────────────────────────────────────
Function un.customUninstallPage
  nsDialogs::Create 1018
  Pop $0

  ${NSD_CreateLabel} 0 0 100% 24u "Choose what to keep after uninstalling AutoNote:"
  Pop $0

  ${NSD_CreateCheckBox} 16u 32u 100% 14u "Keep generated notes and downloaded files (~AutoNote folder)"
  Pop $KeepNotes
  ${NSD_Check} $KeepNotes

  ${NSD_CreateCheckBox} 16u 52u 100% 14u "Keep settings and API keys (~/.auto_note/config)"
  Pop $KeepSettings
  ${NSD_Check} $KeepSettings

  ${NSD_CreateCheckBox} 16u 72u 100% 14u "Keep ML environment (~/.auto_note/venv, ~2 GB)"
  Pop $KeepVenv

  ${NSD_CreateLabel} 16u 100u 100% 24u "Unchecked items will be permanently deleted."
  Pop $0

  nsDialogs::Show
FunctionEnd

Function un.customUninstallPageLeave
  ${NSD_GetState} $KeepNotes $KeepNotes
  ${NSD_GetState} $KeepSettings $KeepSettings
  ${NSD_GetState} $KeepVenv $KeepVenv
FunctionEnd

; ── Register the custom page ──────────────────────────────────────────────────
!macro customUnInstallPage
  UninstPage custom un.customUninstallPage un.customUninstallPageLeave
!macroend

; ── Cleanup after standard uninstall ──────────────────────────────────────────
!macro customUnInstall
  ; Delete ML venv if user chose not to keep it
  ${If} $KeepVenv != ${BST_CHECKED}
    RMDir /r "$PROFILE\.auto_note\venv"
  ${EndIf}

  ; Delete settings if user chose not to keep them
  ${If} $KeepSettings != ${BST_CHECKED}
    ; Remove config, credentials, scripts — but keep venv if retained above
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
    ; Only remove the dir if it's empty (venv might still be there)
    RMDir "$PROFILE\.auto_note"
  ${EndIf}

  ; Delete generated content if user chose not to keep it
  ${If} $KeepNotes != ${BST_CHECKED}
    RMDir /r "$PROFILE\AutoNote"
  ${EndIf}
!macroend
