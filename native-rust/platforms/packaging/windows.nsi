Unicode true
!include "MUI2.nsh"
!include "LogicLib.nsh"

Name "Voice Changer"
OutFile "${OUTPUT}"
InstallDir "$LOCALAPPDATA\Programs\Voice Changer"
InstallDirRegKey HKCU "Software\VoiceChanger" "InstallDir"
RequestExecutionLevel user
SetCompressor /SOLID lzma
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
  Delete "$INSTDIR\lib\onnxruntime.dll"
  Delete "$INSTDIR\lib\onnxruntime_providers_shared.dll"
  RMDir "$INSTDIR\lib"
  Delete "$INSTDIR\README.md"
  Delete "$INSTDIR\LICENSE"
  Delete "$INSTDIR\models\example-model.json.txt"
  RMDir "$INSTDIR\models"
  Delete "$INSTDIR\Uninstall.exe"
  ; Leave imported models and any other user files intact.
  RMDir "$INSTDIR"
  DeleteRegKey HKCU "Software\VoiceChanger"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\VoiceChanger"
SectionEnd
