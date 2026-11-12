Unicode true
!include "MUI2.nsh"
!include "LogicLib.nsh"

Name "Voice Changer"
OutFile "${OUTPUT}"
InstallDir "$LOCALAPPDATA\Programs\Voice Changer"
InstallDirRegKey HKCU "Software\VoiceChanger" "InstallDir"
RequestExecutionLevel user
SetCompressor /SOLID lzma
!ifdef ICON
  !define MUI_ICON "${ICON}"
  !define MUI_UNICON "${ICON}"
!endif
VIProductVersion "${VERSION}.0"
VIAddVersionKey /LANG=1033 "ProductName" "Voice Changer"
VIAddVersionKey /LANG=1033 "FileDescription" "Voice Changer installer"
VIAddVersionKey /LANG=1033 "FileVersion" "${VERSION}"
VIAddVersionKey /LANG=1033 "LegalCopyright" "Voice Changer contributors"

!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Section "Voice Changer"
  SetShellVarContext current
  SetOutPath "$INSTDIR"
  File /r "${PAYLOAD}\*.*"
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  CreateDirectory "$SMPROGRAMS\Voice Changer"
  CreateShortcut "$SMPROGRAMS\Voice Changer\Voice Changer.lnk" "$INSTDIR\voice-changer-native.exe"
  CreateShortcut "$SMPROGRAMS\Voice Changer\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
  WriteRegStr HKCU "Software\VoiceChanger" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\VoiceChanger" "DisplayName" "Voice Changer"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\VoiceChanger" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\VoiceChanger" "DisplayIcon" "$INSTDIR\voice-changer-native.exe"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\VoiceChanger" "UninstallString" '$\"$INSTDIR\Uninstall.exe$\"'
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\VoiceChanger" "QuietUninstallString" '$\"$INSTDIR\Uninstall.exe$\" /S'
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\VoiceChanger" "InstallLocation" "$INSTDIR"
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\VoiceChanger" "NoModify" 1
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\VoiceChanger" "NoRepair" 1
SectionEnd

Section "Uninstall"
  SetShellVarContext current
  Delete "$SMPROGRAMS\Voice Changer\Voice Changer.lnk"
  Delete "$SMPROGRAMS\Voice Changer\Uninstall.lnk"
  RMDir "$SMPROGRAMS\Voice Changer"
  Delete "$INSTDIR\voice-changer-native.exe"
  Delete "$INSTDIR\DirectML.dll"
  Delete "$INSTDIR\lib\onnxruntime.dll"
  Delete "$INSTDIR\lib\onnxruntime_providers_shared.dll"
  Delete "$INSTDIR\lib\onnxruntime_providers_cuda.dll"
  RMDir "$INSTDIR\lib"
  ; Voice Changer Mic: removing the installed driver needs administrator rights, so
  ; the app's Settings has "Remove Voice Changer Mic"; this only deletes the package.
  Delete "$INSTDIR\driver\VoiceChangerMic.inf"
  Delete "$INSTDIR\driver\VoiceChangerMic.sys"
  Delete "$INSTDIR\driver\voicechangermic.cat"
  Delete "$INSTDIR\driver\LICENSE"
  Delete "$INSTDIR\driver\THIRD_PARTY_NOTICES.md"
  RMDir "$INSTDIR\driver"
  Delete "$INSTDIR\README.md"
  Delete "$INSTDIR\LICENSE"
  Delete "$INSTDIR\models\example-model.json.txt"
  ; The shared models the app downloaded on first start.
  Delete "$INSTDIR\models\shared\contentvec.onnx"
  Delete "$INSTDIR\models\shared\rmvpe.onnx"
  RMDir "$INSTDIR\models\shared"
  RMDir "$INSTDIR\models"
  Delete "$INSTDIR\Uninstall.exe"
  ; Leave imported models and any other user files intact.
  RMDir "$INSTDIR"
  DeleteRegKey HKCU "Software\VoiceChanger"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\VoiceChanger"
SectionEnd
