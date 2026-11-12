//! Voice Changer Mic on Windows: the app's own virtual microphone driver
//! (`platforms/windows/driver`). Sound played to its "Voice Changer Input" speaker comes
//! out of its "Voice Changer Mic" microphone, so the voice changer plays there and
//! Discord, OBS or a game record from the microphone.
//!
//! Installing a driver needs administrator rights: the app starts itself elevated
//! (one UAC prompt) with `--install-mic-driver`, which creates the root-enumerated
//! device and installs the driver on it, as `devcon install` does.
use anyhow::{bail, Context, Result};
use std::{
    ffi::OsStr,
    os::windows::ffi::OsStrExt,
    path::{Path, PathBuf},
    ptr::{null, null_mut},
};
use windows_sys::{
    core::GUID,
    Win32::{
        Devices::DeviceAndDriverInstallation::{
            DiUninstallDevice, SetupDiCallClassInstaller, SetupDiCreateDeviceInfoList,
            SetupDiCreateDeviceInfoW, SetupDiDestroyDeviceInfoList, SetupDiEnumDeviceInfo,
            SetupDiGetClassDevsW, SetupDiGetDeviceRegistryPropertyW, SetupDiGetINFClassW,
            SetupDiOpenDevRegKey, SetupDiSetDeviceRegistryPropertyW, SetupUninstallOEMInfW,
            UpdateDriverForPlugAndPlayDevicesW, DICD_GENERATE_ID, DICS_FLAG_GLOBAL,
            DIF_REGISTERDEVICE, DIGCF_ALLCLASSES, DIREG_DRV, HDEVINFO, INSTALLFLAG_FORCE,
            SPDRP_HARDWAREID, SP_DEVINFO_DATA, SUOI_FORCEDELETE,
        },
        Foundation::{CloseHandle, GetLastError, ERROR_CANCELLED, INVALID_HANDLE_VALUE},
        System::{
            Registry::{RegCloseKey, RegQueryValueExW, KEY_READ},
            Threading::{GetExitCodeProcess, WaitForSingleObject, INFINITE},
        },
        UI::{
            Shell::{ShellExecuteExW, SEE_MASK_NOCLOSEPROCESS, SHELLEXECUTEINFOW},
            WindowsAndMessaging::SW_HIDE,
        },
    },
};

/// Hardware ID in `VoiceChangerMic.inf`.
pub const HARDWARE_ID: &str = "ROOT\\VoiceChangerMic";
/// Both endpoints carry this in their names: "Voice Changer Input" (speaker) and
/// "Voice Changer Mic" (microphone).
pub const ENDPOINT_NAME: &str = "Voice Changer";
const INF_NAME: &str = "VoiceChangerMic.inf";

/// The signed driver package shipped beside the executable, when this build has one.
pub fn packaged_inf() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let inf = exe.parent()?.join("driver").join(INF_NAME);
    inf.is_file().then_some(inf)
}

/// Whether an audio device name belongs to the driver.
pub fn is_endpoint(name: &str) -> bool {
    name.contains(ENDPOINT_NAME)
}

/// The "Voice Changer Input" speaker the voice is played to.
#[cfg(feature = "audio")]
pub fn output_device(host: &cpal::Host) -> Result<cpal::Device> {
    use cpal::traits::HostTrait;
    host.output_devices()?
        .find(|device| is_endpoint(&device.to_string()))
        .context("Voice Changer Mic is not installed; install it in Settings under Audio routing")
}

fn wide(text: impl AsRef<OsStr>) -> Vec<u16> {
    text.as_ref().encode_wide().chain(Some(0)).collect()
}

/// REG_MULTI_SZ holding the hardware ID.
fn hardware_id_list() -> Vec<u16> {
    OsStr::new(HARDWARE_ID)
        .encode_wide()
        .chain([0, 0])
        .collect()
}

fn last_error(what: &str) -> anyhow::Error {
    let code = unsafe { GetLastError() };
    anyhow::anyhow!("{what} failed (error 0x{code:08X})")
}

struct DeviceSet(HDEVINFO);
impl Drop for DeviceSet {
    fn drop(&mut self) {
        unsafe { SetupDiDestroyDeviceInfoList(self.0) };
    }
}

fn devinfo() -> SP_DEVINFO_DATA {
    SP_DEVINFO_DATA {
        cbSize: std::mem::size_of::<SP_DEVINFO_DATA>() as u32,
        ..unsafe { std::mem::zeroed() }
    }
}

/// Every device node (present or not) whose first hardware ID is ours.
fn our_devices(mut each: impl FnMut(HDEVINFO, &SP_DEVINFO_DATA)) -> Result<()> {
    let set = unsafe { SetupDiGetClassDevsW(null(), null(), null_mut(), DIGCF_ALLCLASSES) };
    if set == INVALID_HANDLE_VALUE as HDEVINFO {
        return Err(last_error("Listing devices"));
    }
    let set = DeviceSet(set);
    let mut index = 0;
    loop {
        let mut info = devinfo();
        if unsafe { SetupDiEnumDeviceInfo(set.0, index, &mut info) } == 0 {
            break;
        }
        index += 1;
        let mut buffer = [0u16; 256];
        let mut size = 0;
        let read = unsafe {
            SetupDiGetDeviceRegistryPropertyW(
                set.0,
                &info,
                SPDRP_HARDWAREID,
                null_mut(),
                buffer.as_mut_ptr().cast(),
                std::mem::size_of_val(&buffer) as u32,
                &mut size,
            )
        };
        if read == 0 {
            continue;
        }
        let end = buffer.iter().position(|&c| c == 0).unwrap_or(buffer.len());
        if String::from_utf16_lossy(&buffer[..end]).eq_ignore_ascii_case(HARDWARE_ID) {
            each(set.0, &info);
        }
    }
    Ok(())
}

fn device_exists() -> Result<bool> {
    let mut found = false;
    our_devices(|_, _| found = true)?;
    Ok(found)
}

/// Creates the device node and installs `inf` on it. Needs administrator rights.
pub fn install(inf: &Path) -> Result<()> {
    let inf = inf
        .canonicalize()
        .with_context(|| format!("Driver package not found: {}", inf.display()))?;
    let inf_w = wide(&inf);
    let mut class: GUID = unsafe { std::mem::zeroed() };
    let mut class_name = [0u16; 64];
    if unsafe {
        SetupDiGetINFClassW(
            inf_w.as_ptr(),
            &mut class,
            class_name.as_mut_ptr(),
            class_name.len() as u32,
            null_mut(),
        )
    } == 0
    {
        return Err(last_error("Reading the driver package"));
    }
    let created = !device_exists()?;
    if created {
        let set = unsafe { SetupDiCreateDeviceInfoList(&class, null_mut()) };
        if set == INVALID_HANDLE_VALUE as HDEVINFO {
            return Err(last_error("Creating the device"));
        }
        let set = DeviceSet(set);
        let mut info = devinfo();
        if unsafe {
            SetupDiCreateDeviceInfoW(
                set.0,
                class_name.as_ptr(),
                &class,
                null(),
                null_mut(),
                DICD_GENERATE_ID,
                &mut info,
            )
        } == 0
        {
            return Err(last_error("Creating the device"));
        }
        let ids = hardware_id_list();
        if unsafe {
            SetupDiSetDeviceRegistryPropertyW(
                set.0,
                &mut info,
                SPDRP_HARDWAREID,
                ids.as_ptr().cast(),
                (ids.len() * 2) as u32,
            )
        } == 0
        {
            return Err(last_error("Naming the device"));
        }
        if unsafe { SetupDiCallClassInstaller(DIF_REGISTERDEVICE, set.0, &info) } == 0 {
            return Err(last_error("Registering the device"));
        }
    }
    let id = wide(HARDWARE_ID);
    let mut reboot = 0;
    let installed = unsafe {
        UpdateDriverForPlugAndPlayDevicesW(
            null_mut(),
            id.as_ptr(),
            inf_w.as_ptr(),
            INSTALLFLAG_FORCE,
            &mut reboot,
        )
    };
    if installed == 0 {
        let error = last_error("Installing the driver");
        if created {
            let _ = uninstall();
        }
        return Err(error.context(
            "Windows refused the driver; a driver must be signed by Microsoft to load with Secure Boot",
        ));
    }
    log::info!("Voice Changer Mic installed from {}", inf.display());
    Ok(())
}

/// Removes the device and its driver package. Needs administrator rights.
pub fn uninstall() -> Result<()> {
    let mut packages: Vec<Vec<u16>> = Vec::new();
    let mut failed = None;
    our_devices(|set, info| unsafe {
        let key = SetupDiOpenDevRegKey(set, info, DICS_FLAG_GLOBAL, 0, DIREG_DRV, KEY_READ);
        if !std::ptr::eq(key, INVALID_HANDLE_VALUE) {
            let mut buffer = [0u16; 260];
            let mut size = std::mem::size_of_val(&buffer) as u32;
            let name = wide("InfPath");
            if RegQueryValueExW(
                key,
                name.as_ptr(),
                null(),
                null_mut(),
                buffer.as_mut_ptr().cast(),
                &mut size,
            ) == 0
            {
                let end = buffer.iter().position(|&c| c == 0).unwrap_or(buffer.len());
                let package: Vec<u16> = buffer[..end].iter().copied().chain(Some(0)).collect();
                if !packages.contains(&package) {
                    packages.push(package);
                }
            }
            RegCloseKey(key);
        }
        if DiUninstallDevice(null_mut(), set, info, 0, null_mut()) == 0 {
            failed = Some(last_error("Removing the device"));
        }
    })?;
    for package in packages {
        unsafe { SetupUninstallOEMInfW(package.as_ptr(), SUOI_FORCEDELETE, null()) };
    }
    match failed {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Runs this executable elevated with `arguments` and waits. The elevated process
/// writes its error, if any, to `report`.
fn run_elevated(arguments: &str, report: &Path) -> Result<()> {
    let _ = std::fs::remove_file(report);
    let exe = wide(std::env::current_exe()?);
    let verb = wide("runas");
    let parameters = wide(arguments);
    let mut info: SHELLEXECUTEINFOW = unsafe { std::mem::zeroed() };
    info.cbSize = std::mem::size_of::<SHELLEXECUTEINFOW>() as u32;
    info.fMask = SEE_MASK_NOCLOSEPROCESS;
    info.lpVerb = verb.as_ptr();
    info.lpFile = exe.as_ptr();
    info.lpParameters = parameters.as_ptr();
    info.nShow = SW_HIDE;
    if unsafe { ShellExecuteExW(&mut info) } == 0 {
        if unsafe { GetLastError() } == ERROR_CANCELLED {
            bail!("Administrator permission was not given");
        }
        return Err(last_error("Starting the installer"));
    }
    let process = info.hProcess;
    let mut code = 1;
    unsafe {
        WaitForSingleObject(process, INFINITE);
        GetExitCodeProcess(process, &mut code);
        CloseHandle(process);
    }
    if code != 0 {
        let reason = std::fs::read_to_string(report).unwrap_or_default();
        let _ = std::fs::remove_file(report);
        bail!(
            "{}",
            if reason.trim().is_empty() {
                format!("The driver installer stopped (exit code {code})")
            } else {
                reason
            }
        );
    }
    Ok(())
}

fn report_file() -> PathBuf {
    std::env::temp_dir().join("voice-changer-mic-install.txt")
}

fn quoted(path: &Path) -> String {
    format!("\"{}\"", path.display())
}

/// Installs the packaged driver after one UAC prompt.
pub fn install_with_prompt() -> Result<()> {
    let inf = packaged_inf().context("This build does not include the Voice Changer Mic driver")?;
    let report = report_file();
    run_elevated(
        &format!("--install-mic-driver {} {}", quoted(&inf), quoted(&report)),
        &report,
    )
}

/// Removes the driver after one UAC prompt.
pub fn uninstall_with_prompt() -> Result<()> {
    let report = report_file();
    run_elevated(
        &format!("--uninstall-mic-driver {}", quoted(&report)),
        &report,
    )
}

/// Entry point of the elevated process: runs `action` and leaves its error in
/// `report` for the app that started it.
pub fn elevated(action: impl FnOnce() -> Result<()>, report: Option<&Path>) -> Result<()> {
    let result = action();
    if let (Err(error), Some(report)) = (&result, report) {
        let _ = std::fs::write(report, format!("{error:#}"));
    }
    result
}
