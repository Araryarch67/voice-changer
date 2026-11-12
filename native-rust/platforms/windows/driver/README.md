# Voice Changer Mic (Windows driver)

A kernel audio driver that gives Windows a virtual microphone without VB-Cable. It adds two endpoints:

- **Voice Changer Input**, a speaker. The app plays the converted voice here.
- **Voice Changer Mic**, a microphone. Whatever was played to the speaker comes out here, about 20 ms later. Discord, OBS and games record from it.

Both endpoints use a single format (48 kHz, stereo, 32-bit PCM), so the driver copies bytes from one to the other through a one-second ring buffer ([Source/Main/loopback.cpp](Source/Main/loopback.cpp)). When nothing is playing, the microphone is silent. If the microphone falls more than 200 ms behind the speaker, or gets ahead of it, the driver moves its read position back to 20 ms behind.

The driver is based on [VirtualDrivers/Virtual-Audio-Driver](https://github.com/VirtualDrivers/Virtual-Audio-Driver) at commit `bb34fba` (MIT, see [LICENSE](LICENSE)), which in turn is based on Microsoft's SYSVAD sample (MS-PL, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). In that driver the microphone only produced silence and the speaker wrote to a file. Here the speaker feeds the microphone. The device also has its own hardware ID (`ROOT\VoiceChangerMic`), service and file names, so it can be installed next to the original driver.

## Building

You need Visual Studio 2022 or its Build Tools, with the *Windows Driver Kit* component and the *MSVC Spectre-mitigated libraries*, plus WDK 10.0.26100.

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1
```

`dist/` then holds `VoiceChangerMic.inf`, `.sys`, `.cat` and `VoiceChangerMic-x64.cab`. The script runs `infverif /w` (the rules for a Windows Driver) on the INF. The [driver workflow](../../../../.github/workflows/driver.yml) does the same on every change and keeps `dist/` as an artifact.

## Signing for release

Windows 10 and 11 load a kernel driver only if Microsoft has signed it. That signature comes from **attestation signing**:

1. Buy an **EV code signing certificate**.
2. Register the company or person in the **Windows Hardware Dev Center** (Partner Center) using that certificate.
3. Sign the CAB with the EV certificate:
   ```powershell
   signtool sign /fd sha256 /tr http://timestamp.digicert.com /td sha256 /a dist\VoiceChangerMic-x64.cab
   ```
4. In Partner Center, open **Hardware > Submit new hardware**, upload the signed CAB, and request the Windows 10/11 x64 signatures.
5. When it is done, download the signed package. Put its `VoiceChangerMic.inf`, `VoiceChangerMic.sys` and `voicechangermic.cat` in `signed/` in this folder, commit them, and tag a release.

The [release workflow](../../../../.github/workflows/release.yml) copies `signed/` into the Windows downloads as `driver/`. From then on, **Settings > Audio routing > Install Voice Changer Mic** installs it after one UAC prompt (`voice-changer-native --install-mic-driver driver\VoiceChangerMic.inf`), and **Remove Voice Changer Mic** uninstalls it (`--uninstall-mic-driver`). Downloads made without `signed/` say that the driver is not included yet.

## Testing before it is signed

An unsigned driver loads only with test signing on, and test signing requires Secure Boot off. Do this in a virtual machine, never on an everyday PC:

1. In the VM, run `bcdedit /set testsigning on` and restart.
2. Create a test certificate, sign `voicechangermic.cat` and `VoiceChangerMic.sys` with it, and trust the certificate in the VM (`Trusted Root` and `Trusted Publishers`).
3. Install with `voice-changer-native --install-mic-driver path\to\VoiceChangerMic.inf`.
4. Check that Sound settings lists *Voice Changer Input* and *Voice Changer Mic*. Play a sound to the first and record from the second.
