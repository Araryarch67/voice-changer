//! The virtual microphone on Windows is VB-Cable by VB-Audio: the voice changer plays
//! to its "CABLE Input" speaker, whose sound comes out of its "CABLE Output"
//! microphone, which Discord, OBS or a game record from. Its licence does not allow
//! installing it from another program, so the app opens its page instead.
use anyhow::{bail, Result};
use std::{ffi::OsStr, os::windows::ffi::OsStrExt, path::PathBuf, ptr::null_mut};
use windows_sys::Win32::{
    Foundation::{GetLastError, ERROR_CANCELLED},
    System::Registry::{RegGetValueW, HKEY_LOCAL_MACHINE, RRF_RT_REG_DWORD},
    UI::{
        Shell::{ShellExecuteExW, SHELLEXECUTEINFOW},
        WindowsAndMessaging::SW_SHOWNORMAL,
    },
};

/// VB-Cable's download page.
pub const VB_CABLE_PAGE: &str = "https://vb-audio.com/Cable/";
/// The speaker the voice is played to.
pub const SPEAKER: &str = "CABLE Input";
/// The microphone to pick in Discord, OBS or a game.
pub const MICROPHONE: &str = "CABLE Output";

/// Whether the output device called `name` is VB-Cable's speaker. Its 16-channel
/// "CABLE In 16ch" sibling does not match.
pub fn is_cable(name: &str) -> bool {
    name.contains(SPEAKER)
}

/// VB-Cable's speaker, which the voice is played to.
#[cfg(feature = "audio")]
pub fn output_device(host: &cpal::Host) -> Result<cpal::Device> {
    use anyhow::Context;
    use cpal::traits::HostTrait;
    host.output_devices()?
        .find(|device| is_cable(&device.to_string()))
        .context("VB-Cable is not installed; get it in Settings under Audio routing")
}

fn wide(text: impl AsRef<OsStr>) -> Vec<u16> {
    text.as_ref().encode_wide().chain(Some(0)).collect()
}

/// VB-Cable's buffer ("Max Latency" in its control panel) in samples, and its
/// sample rate, read from its settings.
pub fn vb_cable_buffer() -> Option<(u32, u32)> {
    let read = |name: &str| -> Option<u32> {
        let key = wide("SOFTWARE\\VB-Audio\\Cable");
        let name = wide(name);
        let mut value = 0u32;
        let mut size = std::mem::size_of::<u32>() as u32;
        let status = unsafe {
            RegGetValueW(
                HKEY_LOCAL_MACHINE,
                key.as_ptr(),
                name.as_ptr(),
                RRF_RT_REG_DWORD,
                null_mut(),
                (&mut value as *mut u32).cast(),
                &mut size,
            )
        };
        (status == 0).then_some(value)
    };
    let samples = read("VBAudioCableWDM_Latency")?;
    let rate = read("VBAudioCableWDM_SR")
        .filter(|&rate| rate > 0)
        .unwrap_or(48000);
    Some((samples, rate))
}

/// Opens VB-Cable's control panel, where its buffer is set; it needs administrator
/// rights, so Windows asks first.
pub fn open_vb_cable_settings() -> Result<()> {
    let program = std::env::var_os("ProgramFiles")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("C:\\Program Files"))
        .join("VB\\CABLE\\VBCABLE_ControlPanel.exe");
    anyhow::ensure!(
        program.is_file(),
        "VB-Cable's control panel was not found at {}",
        program.display()
    );
    let file = wide(&program);
    let verb = wide("runas");
    let mut info: SHELLEXECUTEINFOW = unsafe { std::mem::zeroed() };
    info.cbSize = std::mem::size_of::<SHELLEXECUTEINFOW>() as u32;
    info.lpVerb = verb.as_ptr();
    info.lpFile = file.as_ptr();
    info.nShow = SW_SHOWNORMAL;
    if unsafe { ShellExecuteExW(&mut info) } == 0 {
        let code = unsafe { GetLastError() };
        if code == ERROR_CANCELLED {
            bail!("Administrator permission was not given");
        }
        bail!("Opening VB-Cable's control panel failed (error 0x{code:08X})");
    }
    Ok(())
}

/// Opens `url` in the default browser.
pub fn open_page(url: &str) {
    if let Err(error) = std::process::Command::new("explorer").arg(url).spawn() {
        log::warn!("Could not open {url}: {error}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognises_only_the_stereo_cable() {
        assert!(is_cable("CABLE Input (VB-Audio Virtual Cable)"));
        assert!(!is_cable("CABLE In 16ch (VB-Audio Virtual Cable)"));
        assert!(!is_cable("Speakers (Realtek(R) Audio)"));
    }
}
