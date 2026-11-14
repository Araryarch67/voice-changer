//! Local voice training: the same RVC v2 training the Colab notebook
//! (`colab/train_voice.ipynb`) runs, on this computer's NVIDIA GPU.
//!
//! The trainer is [Applio](https://github.com/IAHispano/Applio) 3.7.1, as in the
//! notebook. It is Python and PyTorch, which this app does not otherwise use, so it
//! lives on its own in `trainer/` beside the models and datasets folders: uv
//! (Astral's Python manager, from `PATH` or downloaded), a private Python 3.12 and
//! virtual environment, Applio's engine requirements with PyTorch for CUDA 12.8
//! (which knows RTX 50 GPUs), and its pretrained models. Nothing is installed
//! system-wide, and removing `trainer/` removes all of it. The downloads that are
//! files (uv and Applio's source) are checked against their size and SHA-256;
//! Python packages come through uv from PyPI and PyTorch's own index.
//!
//! Training runs Applio's `core.py` steps with the notebook's settings: preprocess
//! the takes into 3 s slices, extract ContentVec features and RMVPE pitch, train
//! from the pretrained HiFi-GAN generator saving every 10 epochs, build the index,
//! then import the voice. A stopped training continues from its last save.
use crate::catalog::agent;
use anyhow::{bail, ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::{
    collections::VecDeque,
    io::{BufRead, Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

pub const APPLIO_VERSION: &str = "3.7.1";
/// Roughly what the first training downloads: Python, PyTorch with its CUDA
/// libraries, Applio and its pretrained models.
pub const INSTALL_GB: u32 = 8;
const APPLIO_URL: &str = "https://github.com/IAHispano/Applio/archive/refs/tags/3.7.1.zip";
const APPLIO_BYTES: u64 = 18_652_781;
const APPLIO_SHA256: &str = "2a341a81283826957a08f6eb2a2739e4d92dd505d38e8baa792df10c3626f58e";
const TERMS_URL: &str = "https://github.com/IAHispano/Applio/blob/main/TERMS_OF_USE.md";
const TORCH_INDEX: &str = "https://download.pytorch.org/whl/cu128";

/// A pinned uv release for this system.
struct Uv {
    url: &'static str,
    bytes: u64,
    sha256: &'static str,
}

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
const UV: Option<Uv> = Some(Uv {
    url: "https://github.com/astral-sh/uv/releases/download/0.13.0/uv-x86_64-pc-windows-msvc.zip",
    bytes: 15_722_003,
    sha256: "088962f9e7b7bd9ea740c04c650b2a21c8928c345bd99ac24350dc924dba656c",
});
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const UV: Option<Uv> = Some(Uv {
    url: "https://github.com/astral-sh/uv/releases/download/0.13.0/uv-x86_64-unknown-linux-gnu.tar.gz",
    bytes: 17_380_567,
    sha256: "1468ebd5a5541121837c5a2817b9972ba6090fa6caa3d142620850a47fb75154",
});
#[cfg(not(any(
    all(target_os = "windows", target_arch = "x86_64"),
    all(target_os = "linux", target_arch = "x86_64")
)))]
const UV: Option<Uv> = None;

/// Applio's terms of use, which training accepts.
pub fn terms_url() -> &'static str {
    TERMS_URL
}

/// What to train.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    /// The voice's name; also Applio's model name, as [`crate::dataset::safe_name`].
    pub name: String,
    /// The folder of WAV takes (`datasets/<name>/`).
    pub dataset: PathBuf,
    pub epochs: u32,
    pub batch_size: u32,
    /// 40 000 or 48 000.
    pub sample_rate: u32,
}

impl Options {
    pub fn new(name: &str, dataset: PathBuf) -> Self {
        Self {
            name: name.to_owned(),
            dataset,
            epochs: 300,
            batch_size: 8,
            sample_rate: 40_000,
        }
    }

    fn check(&self) -> Result<()> {
        ensure!(
            !crate::dataset::safe_name(&self.name).is_empty(),
            "Give the voice a name"
        );
        ensure!(
            (1..=10_000).contains(&self.epochs),
            "Epochs must be 1 to 10000"
        );
        ensure!(
            (1..=50).contains(&self.batch_size),
            "The batch size must be 1 to 50"
        );
        ensure!(
            matches!(self.sample_rate, 40_000 | 48_000),
            "The sample rate must be 40000 or 48000"
        );
        Ok(())
    }
}

/// Where a training is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Checking,
    /// Downloading uv or Applio (bytes in [`Progress::received`]).
    Downloading,
    /// Setting up Python, PyTorch and Applio's models; only the first time.
    Installing,
    Preparing,
    Extracting,
    Training,
    Indexing,
    Importing,
}

/// A training's progress, shared with the interface.
#[derive(Default)]
pub struct Progress {
    pub cancel: AtomicBool,
    pub received: AtomicU64,
    pub total: AtomicU64,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    stage: Option<Stage>,
    epoch: u32,
    epochs: u32,
    lines: VecDeque<String>,
}

const KEPT_LINES: usize = 200;

impl Progress {
    pub fn stage(&self) -> Option<Stage> {
        self.state.lock().unwrap().stage
    }

    /// The last epoch finished and the epochs asked for.
    pub fn epochs(&self) -> (u32, u32) {
        let state = self.state.lock().unwrap();
        (state.epoch, state.epochs)
    }

    /// The trainer's latest output lines, oldest first.
    pub fn lines(&self, count: usize) -> Vec<String> {
        let state = self.state.lock().unwrap();
        let skip = state.lines.len().saturating_sub(count);
        state.lines.iter().skip(skip).cloned().collect()
    }

    fn set_stage(&self, stage: Stage) {
        log::info!("Training: {stage:?}");
        self.state.lock().unwrap().stage = Some(stage);
    }

    fn line(&self, line: &str) {
        let line = line.trim_end();
        if line.is_empty() {
            return;
        }
        let mut state = self.state.lock().unwrap();
        if let Some(epoch) = epoch_of(line) {
            state.epoch = state.epoch.max(epoch);
        }
        if state.lines.len() == KEPT_LINES {
            state.lines.pop_front();
        }
        state.lines.push_back(line.to_owned());
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

/// The epoch a line of Applio's training output reports
/// (`name | epoch=12 | step=… | …`), if it reports one.
fn epoch_of(line: &str) -> Option<u32> {
    let rest = line.split(" | epoch=").nth(1)?;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// The trainer's folder: `trainer/` beside the datasets and models folders.
pub fn root() -> PathBuf {
    let datasets = crate::dataset::root();
    datasets
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default()
        .join("trainer")
}

fn applio_dir() -> PathBuf {
    root().join(format!("Applio-{APPLIO_VERSION}"))
}

fn ready_marker() -> PathBuf {
    root().join(format!(".ready-applio-{APPLIO_VERSION}"))
}

/// Whether the trainer is set up, so training starts without the big download.
pub fn installed() -> bool {
    ready_marker().is_file()
}

/// Removes the trainer (its Python, PyTorch, Applio and the training files it
/// keeps); the datasets and voices stay.
pub fn remove() -> Result<()> {
    let root = root();
    if root.exists() {
        std::fs::remove_dir_all(&root)
            .with_context(|| format!("Could not remove {}", root.display()))?;
    }
    Ok(())
}

/// The trainer's size on disk, in bytes (0 when it is not set up).
pub fn disk_bytes() -> u64 {
    fn walk(path: &Path) -> u64 {
        let Ok(entries) = std::fs::read_dir(path) else {
            return 0;
        };
        entries
            .flatten()
            .map(|entry| match entry.file_type() {
                Ok(kind) if kind.is_dir() => walk(&entry.path()),
                Ok(kind) if kind.is_file() => entry.metadata().map_or(0, |m| m.len()),
                _ => 0,
            })
            .sum()
    }
    walk(&root())
}

/// The NVIDIA GPU training will use, from `nvidia-smi -L`.
pub fn gpu() -> Result<String> {
    let output = hidden(Command::new("nvidia-smi").arg("-L"))
        .output()
        .ok()
        .filter(|output| output.status.success());
    let Some(output) = output else {
        bail!("Training on this computer needs an NVIDIA GPU with its driver installed (nvidia-smi was not found). Train on Google Colab instead.");
    };
    let text = String::from_utf8_lossy(&output.stdout);
    let first = text.lines().next().unwrap_or("").trim();
    ensure!(
        !first.is_empty(),
        "nvidia-smi lists no GPU. Train on Google Colab instead."
    );
    // "GPU 0: NVIDIA GeForce RTX 4060 (UUID: GPU-…)"
    let name = first
        .split_once(": ")
        .map_or(first, |(_, rest)| rest)
        .split(" (UUID")
        .next()
        .unwrap_or(first);
    Ok(name.to_owned())
}

/// No console window for the trainer's processes on Windows.
fn hidden(command: &mut Command) -> &mut Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    command
}

/// Makes `command`'s process end with this app, so closing the app (or Ctrl+C on
/// the command line) never leaves the trainer running on the GPU. On Linux the
/// child gets SIGKILL when the thread that started it ends.
fn tie_to_app(command: &mut Command) -> &mut Command {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        // SAFETY: prctl is async-signal-safe, as pre_exec requires.
        unsafe {
            command.pre_exec(|| {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }
    command
}

/// On Windows, puts `child` (and the processes it starts) in a job that is
/// closed, killing them, when this app exits.
#[cfg(windows)]
fn tie_child(child: &std::process::Child) -> Result<()> {
    use std::os::windows::io::AsRawHandle;
    use std::sync::OnceLock;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    // One job for the app's lifetime; its handle closes when the app exits.
    static JOB: OnceLock<usize> = OnceLock::new();
    let job = *JOB.get_or_init(|| {
        // SAFETY: plain Win32 calls with valid arguments; the handle is kept.
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                return 0;
            }
            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &limits as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            job as usize
        }
    });
    ensure!(job != 0, "Could not create a job for the trainer");
    // SAFETY: both handles are valid while the child runs.
    let assigned = unsafe { AssignProcessToJobObject(job as _, child.as_raw_handle() as _) };
    ensure!(assigned != 0, "Could not tie the trainer to the app");
    Ok(())
}

/// Trains a voice from `options` and imports it; returns the voice's manifest.
pub fn train(options: &Options, progress: &Progress) -> Result<PathBuf> {
    options.check()?;
    let takes = crate::dataset::takes(&options.dataset);
    let seconds = crate::dataset::total_seconds(&takes);
    ensure!(
        seconds >= crate::dataset::MINIMUM_SECONDS,
        "Only {seconds:.0} seconds recorded; record at least {:.0} (10 minutes is recommended)",
        crate::dataset::MINIMUM_SECONDS
    );
    progress.set_stage(Stage::Checking);
    let gpu = gpu()?;
    progress.line(&format!("GPU: {gpu}"));
    let root = root();
    std::fs::create_dir_all(&root).with_context(|| format!("Cannot create {}", root.display()))?;
    let log = Arc::new(Mutex::new(
        std::fs::File::create(root.join("last-training.log")).ok(),
    ));
    let runner = Runner { progress, log };
    let uv = uv(&runner)?;
    install(&runner, &uv)?;
    run_training(&runner, options)
}

struct Runner<'a> {
    // Shared with the threads that read the trainer's output.
    progress: &'a Progress,
    log: Arc<Mutex<Option<std::fs::File>>>,
}

impl Runner<'_> {
    fn note(&self, line: &str) {
        let line = line.trim_end();
        if line.is_empty() {
            return;
        }
        self.progress.line(line);
        if let Some(file) = self.log.lock().unwrap().as_mut() {
            let _ = writeln!(file, "{line}");
        }
    }

    /// Runs `command`, passing its output to the progress and the log, until it
    /// ends or the training is stopped.
    fn run(&self, what: &str, command: &mut Command) -> Result<()> {
        ensure!(!self.progress.cancelled(), "Training stopped");
        self.note(&format!("$ {what}"));
        let root = root();
        command
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .env("UV_CACHE_DIR", root.join("uv-cache"))
            .env("UV_PYTHON_INSTALL_DIR", root.join("python"))
            .env("UV_NO_PROGRESS", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = tie_to_app(hidden(command))
            .spawn()
            .with_context(|| format!("Could not start {what}"))?;
        #[cfg(windows)]
        if let Err(error) = tie_child(&child) {
            let _ = child.kill();
            return Err(error);
        }
        let streams: Vec<Box<dyn Read + Send>> = [
            child
                .stdout
                .take()
                .map(|s| Box::new(s) as Box<dyn Read + Send>),
            child
                .stderr
                .take()
                .map(|s| Box::new(s) as Box<dyn Read + Send>),
        ]
        .into_iter()
        .flatten()
        .collect();
        let status = std::thread::scope(|scope| -> Result<std::process::ExitStatus> {
            for stream in streams {
                scope.spawn(move || {
                    let mut reader = std::io::BufReader::new(stream);
                    let mut buffer = Vec::new();
                    loop {
                        buffer.clear();
                        match read_line(&mut reader, &mut buffer) {
                            Ok(0) | Err(_) => break,
                            Ok(_) => self.note(&String::from_utf8_lossy(&buffer)),
                        }
                    }
                });
            }
            // The readers end when the process does; the scope waits for them.
            loop {
                if self.progress.cancelled() {
                    let _ = child.kill();
                    let _ = child.wait();
                    bail!("Training stopped; start it again to continue from the last save");
                }
                if let Some(status) = child.try_wait()? {
                    return Ok(status);
                }
                std::thread::sleep(Duration::from_millis(200));
            }
        })?;
        if !status.success() {
            let tail = self.progress.lines(6).join("\n");
            bail!("{what} failed ({status}):\n{tail}");
        }
        Ok(())
    }
}

/// Reads up to a `\n` or `\r` (progress bars redraw with `\r`).
fn read_line(reader: &mut impl BufRead, buffer: &mut Vec<u8>) -> std::io::Result<usize> {
    let mut read = 0;
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(read);
        }
        match available.iter().position(|b| *b == b'\n' || *b == b'\r') {
            Some(end) => {
                buffer.extend_from_slice(&available[..end]);
                reader.consume(end + 1);
                return Ok(read + end + 1);
            }
            None => {
                let length = available.len();
                buffer.extend_from_slice(available);
                reader.consume(length);
                read += length;
            }
        }
    }
}

/// uv: from `PATH` if it is there, else the pinned release in `trainer/uv/`.
fn uv(runner: &Runner) -> Result<PathBuf> {
    let on_path = hidden(Command::new("uv").arg("--version"))
        .output()
        .is_ok_and(|output| output.status.success());
    if on_path {
        return Ok(PathBuf::from("uv"));
    }
    let folder = root().join("uv");
    let binary = folder.join(if cfg!(windows) { "uv.exe" } else { "uv" });
    if binary.is_file() {
        return Ok(binary);
    }
    let Some(release) = UV else {
        bail!("Install uv (https://docs.astral.sh/uv/) to train on this computer; there is no uv download for this system");
    };
    runner.progress.set_stage(Stage::Downloading);
    runner.note(&format!("Downloading uv: {}", release.url));
    let archive = fetch(release.url, release.bytes, release.sha256, runner.progress)?;
    std::fs::create_dir_all(&folder)?;
    let name = if cfg!(windows) { "uv.exe" } else { "uv" };
    let data = if release.url.ends_with(".zip") {
        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(archive))?;
        let mut file = zip.by_name(name).context("The uv download has no uv")?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        data
    } else {
        let tar = gunzip(&archive)?;
        let mut tar = tar::Archive::new(std::io::Cursor::new(tar));
        let mut found = None;
        for entry in tar.entries()? {
            let mut entry = entry?;
            if entry.path()?.file_name().is_some_and(|n| n == name) {
                let mut data = Vec::new();
                entry.read_to_end(&mut data)?;
                found = Some(data);
                break;
            }
        }
        found.context("The uv download has no uv")?
    };
    std::fs::write(&binary, data)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755))?;
    }
    Ok(binary)
}

/// Downloads `url`, checking its size and SHA-256.
fn fetch(url: &str, bytes: u64, sha256: &str, progress: &Progress) -> Result<Vec<u8>> {
    progress.received.store(0, Ordering::Relaxed);
    progress.total.store(bytes, Ordering::Relaxed);
    let response = agent()
        .get(url)
        .call()
        .with_context(|| format!("Could not download {url}"))?;
    let mut reader = response.into_reader();
    let mut data = Vec::with_capacity(bytes as usize);
    let mut buffer = vec![0; 1 << 16];
    let mut hash = Sha256::new();
    loop {
        ensure!(!progress.cancelled(), "Training stopped");
        let read = reader.read(&mut buffer).context("Download interrupted")?;
        if read == 0 {
            break;
        }
        data.extend_from_slice(&buffer[..read]);
        hash.update(&buffer[..read]);
        progress.received.fetch_add(read as u64, Ordering::Relaxed);
    }
    let hex: String = hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    ensure!(
        data.len() as u64 == bytes && hex == sha256,
        "The download from {url} is not the expected file ({} bytes, SHA-256 {hex})",
        data.len()
    );
    Ok(data)
}

/// A gzip file's contents (one member).
fn gunzip(data: &[u8]) -> Result<Vec<u8>> {
    ensure!(
        data.len() > 18 && data[0] == 0x1f && data[1] == 0x8b && data[2] == 8,
        "Not a gzip file"
    );
    let flags = data[3];
    let mut at = 10;
    if flags & 4 != 0 {
        let extra = u16::from_le_bytes([data[at], data[at + 1]]) as usize;
        at += 2 + extra;
    }
    for flag in [8, 16] {
        if flags & flag != 0 {
            at += data[at..]
                .iter()
                .position(|b| *b == 0)
                .context("Bad gzip header")?
                + 1;
        }
    }
    if flags & 2 != 0 {
        at += 2;
    }
    ensure!(at < data.len() - 8, "Bad gzip header");
    miniz_oxide::inflate::decompress_to_vec(&data[at..data.len() - 8])
        .map_err(|error| anyhow::anyhow!("Could not unpack the gzip file: {error:?}"))
}

fn python() -> PathBuf {
    let venv = applio_dir().join(".venv");
    if cfg!(windows) {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python")
    }
}

/// Sets up Applio once: its source, Python 3.12, the engine requirements with
/// PyTorch for CUDA 12.8, and its pretrained models.
fn install(runner: &Runner, uv: &Path) -> Result<()> {
    if installed() {
        return Ok(());
    }
    let applio = applio_dir();
    if !applio.join("core.py").is_file() {
        runner.progress.set_stage(Stage::Downloading);
        runner.note(&format!("Downloading Applio {APPLIO_VERSION}"));
        let archive = fetch(APPLIO_URL, APPLIO_BYTES, APPLIO_SHA256, runner.progress)?;
        let _ = std::fs::remove_dir_all(&applio);
        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(archive))?;
        for i in 0..zip.len() {
            let mut entry = zip.by_index(i)?;
            let Some(path) = entry.enclosed_name() else {
                continue;
            };
            let target = root().join(path);
            if entry.is_dir() {
                std::fs::create_dir_all(&target)?;
            } else {
                if let Some(parent) = target.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let mut file = std::fs::File::create(&target)?;
                std::io::copy(&mut entry, &mut file)?;
            }
        }
        ensure!(
            applio.join("core.py").is_file(),
            "The Applio download did not unpack as expected"
        );
    }
    runner.progress.set_stage(Stage::Installing);
    if !python().is_file() {
        runner.run(
            "uv venv",
            Command::new(uv)
                .args(["venv", ".venv", "--python", "3.12", "--seed"])
                .current_dir(&applio),
        )?;
    }
    runner.run(
        "uv pip install (PyTorch and Applio's requirements)",
        Command::new(uv)
            .args(["pip", "install", "-r", "requirements-engine.txt"])
            .args(["--extra-index-url", TORCH_INDEX])
            .args(["--index-strategy", "unsafe-best-match"])
            .arg("--python")
            .arg(python())
            .current_dir(&applio),
    )?;
    runner.run(
        "Applio prerequisites (pretrained models)",
        Command::new(python())
            .args([
                "core.py",
                "prerequisites",
                "--models",
                "--pretraineds-hifigan",
                "--no-exe",
            ])
            .current_dir(&applio),
    )?;
    let config = applio.join("assets").join("config.json");
    if !config.is_file() {
        std::fs::copy(applio.join("assets").join("config_template.json"), &config)
            .context("Could not write Applio's config.json")?;
    }
    std::fs::write(ready_marker(), APPLIO_VERSION)?;
    Ok(())
}

fn has_file(folder: &Path, wanted: impl Fn(&str) -> bool) -> bool {
    std::fs::read_dir(folder).is_ok_and(|entries| {
        entries
            .flatten()
            .any(|entry| wanted(&entry.file_name().to_string_lossy()))
    })
}

/// The newest saved voice of `model` in Applio's logs (`<model>_<n>e_<n>s.pth`).
fn weights(logs: &Path, model: &str) -> Option<PathBuf> {
    let prefix = format!("{model}_");
    std::fs::read_dir(logs)
        .ok()?
        .flatten()
        .filter(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            name.starts_with(&prefix) && name.ends_with("s.pth") && name.contains("e_")
        })
        .max_by_key(|entry| entry.metadata().and_then(|m| m.modified()).ok())
        .map(|entry| entry.path())
}

fn run_training(runner: &Runner, options: &Options) -> Result<PathBuf> {
    let applio = applio_dir();
    let model = crate::dataset::safe_name(&options.name);
    let logs = applio.join("logs").join(&model);
    let rate = options.sample_rate.to_string();
    let cores = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .clamp(1, 64)
        .to_string();
    let resuming = has_file(&logs, |name| {
        name.starts_with("G_") && name.ends_with(".pth")
    });
    {
        let mut state = runner.progress.state.lock().unwrap();
        state.epochs = options.epochs;
        state.epoch = 0;
    }
    let core = |args: &[&str]| {
        let mut command = Command::new(python());
        command.arg("core.py").args(args).current_dir(&applio);
        command
    };
    if resuming {
        runner.note("Continuing the earlier training.");
    } else {
        runner.progress.set_stage(Stage::Preparing);
        let dataset = options.dataset.to_string_lossy().into_owned();
        runner.run(
            "Applio preprocess",
            &mut core(&[
                "preprocess",
                "--model-name",
                &model,
                "--dataset-path",
                &dataset,
                "--sample-rate",
                &rate,
                "--cpu-cores",
                &cores,
                "--cut-preprocess",
                "Automatic",
                "--chunk-len",
                "3.0",
                "--overlap-len",
                "0.3",
                "--normalization-mode",
                "none",
            ]),
        )?;
        ensure!(
            has_file(&logs.join("sliced_audios"), |n| n.ends_with(".wav")),
            "Preparing the recordings failed; see the trainer's output"
        );
        runner.progress.set_stage(Stage::Extracting);
        runner.run(
            "Applio extract",
            &mut core(&[
                "extract",
                "--model-name",
                &model,
                "--f0-method",
                "rmvpe",
                "--sample-rate",
                &rate,
                "--cpu-cores",
                &cores,
                "--gpu",
                "0",
                "--embedder-model",
                "contentvec",
                "--include-mutes",
                "2",
            ]),
        )?;
        ensure!(
            has_file(&logs.join("extracted"), |n| n.ends_with(".npy")),
            "Extracting features failed; see the trainer's output"
        );
    }
    runner.progress.set_stage(Stage::Training);
    let (epochs, batch) = (options.epochs.to_string(), options.batch_size.to_string());
    runner.run(
        "Applio train",
        &mut core(&[
            "train",
            "--model-name",
            &model,
            "--save-every-epoch",
            "10",
            "--total-epoch",
            &epochs,
            "--sample-rate",
            &rate,
            "--batch-size",
            &batch,
            "--gpu",
            "0",
            "--vocoder",
            "HiFi-GAN",
            "--pretrained",
            "--save-only-latest",
        ]),
    )?;
    let weights = weights(&logs, &model)
        .context("Training ended before the voice was saved; start it again to continue")?;
    runner.progress.set_stage(Stage::Indexing);
    let index = logs.join(format!("{model}.index"));
    if !index.is_file() {
        runner.run(
            "Applio index",
            &mut core(&["index", "--model-name", &model, "--index-algorithm", "Auto"]),
        )?;
    }
    runner.progress.set_stage(Stage::Importing);
    runner.note(&format!(
        "Importing {}",
        weights.file_name().unwrap_or_default().to_string_lossy()
    ));
    crate::import::import_with(
        &weights,
        index.is_file().then_some(index.as_path()),
        Some(&options.name),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epochs_are_read_from_applios_output() {
        assert_eq!(
            epoch_of("aria | epoch=12 | step=480 | 0:00:31 | lowest_value=…"),
            Some(12)
        );
        assert_eq!(epoch_of("aria | epoch=3 | step=99 | batch=4/30"), Some(3));
        assert_eq!(epoch_of("Starting training..."), None);
        let progress = Progress::default();
        progress.line("aria | epoch=7 | step=1");
        progress.line("aria | epoch=6 | step=2");
        assert_eq!(progress.epochs().0, 7);
    }

    #[test]
    fn lines_split_on_carriage_returns_and_newlines() {
        let mut reader = std::io::BufReader::new(&b"one\r two\nthree"[..]);
        let mut lines = Vec::new();
        loop {
            let mut buffer = Vec::new();
            if read_line(&mut reader, &mut buffer).unwrap() == 0 {
                break;
            }
            lines.push(String::from_utf8(buffer).unwrap());
        }
        assert_eq!(lines, ["one", " two", "three"]);
    }

    #[test]
    fn gzip_unpacks() {
        // `printf 'hi\n' | gzip -n`
        let data = [
            0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xcb, 0xc8, 0xe4, 0x02,
            0x00, 0x7a, 0x7a, 0x6f, 0xed, 0x03, 0x00, 0x00, 0x00,
        ];
        assert_eq!(gunzip(&data).unwrap(), b"hi\n");
    }

    #[test]
    fn options_are_checked() {
        let mut options = Options::new("My Voice", PathBuf::from("x"));
        assert!(options.check().is_ok());
        options.sample_rate = 44_100;
        assert!(options.check().is_err());
        options = Options::new("!!!", PathBuf::from("x"));
        assert!(options.check().is_err());
    }
}
