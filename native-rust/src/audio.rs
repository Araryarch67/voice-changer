use crate::{
    config::Settings,
    dsp::{self, StreamResampler},
    engine::{self, Engine},
};
use anyhow::{bail, Context, Result};
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, SampleFormat, SizedSample,
};
use crossbeam_queue::ArrayQueue;
use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

#[derive(Clone, Debug)]
pub struct AudioDevice {
    pub id: String,
    pub name: String,
    pub input: bool,
}
pub fn devices() -> Result<Vec<AudioDevice>> {
    let host = cpal::default_host();
    let mut result = Vec::new();
    for (input, devices) in [
        (true, host.input_devices()?.collect::<Vec<_>>()),
        (false, host.output_devices()?.collect()),
    ] {
        for device in devices {
            result.push(AudioDevice {
                id: device.id()?.to_string(),
                name: device.to_string(),
                input,
            });
        }
    }
    Ok(result)
}

#[derive(Default)]
pub struct Metrics {
    pub running: AtomicBool,
    pub ready: AtomicBool,
    pub input_level: AtomicU32,
    pub output_level: AtomicU32,
    pub processing_ms: AtomicU32,
    pub input_rate: AtomicU32,
    pub output_rate: AtomicU32,
    pub queued_samples: AtomicU64,
    pub dropped_input: AtomicU64,
    pub dropped_output: AtomicU64,
    pub underrun_samples: AtomicU64,
    pub stream_error: AtomicBool,
    pub stream_error_detail: Mutex<Option<String>>,
    /// The monitor device failed; the main output keeps going.
    pub monitor_error: AtomicBool,
    pub error: Mutex<Option<String>>,
    /// Set by the UI; the worker opens and closes the WAV pair to match.
    pub recording: AtomicBool,
    /// Outcome of the last finished recording, taken by the UI.
    pub recorded: Mutex<Option<String>>,
    /// A voice is loading to replace the current one.
    pub voice_loading: AtomicBool,
    /// Why the last voice switch failed, taken by the UI; the old voice keeps playing.
    pub voice_error: Mutex<Option<String>>,
}

pub struct AudioOptions {
    pub input: Option<String>,
    pub output: Option<String>,
    pub manifest: Option<PathBuf>,
    pub settings: Settings,
    /// Folder for REC files.
    pub record_dir: PathBuf,
    /// Send the converted voice to the virtual microphone (built in on Linux,
    /// VB-Cable on Windows) instead of `output`.
    pub virtual_mic: bool,
    /// A second output that plays the converted voice, to hear yourself:
    /// `Some(None)` is the system default device.
    pub monitor: Option<Option<String>>,
}
/// Where audio comes from and goes to; can change while the session runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Devices {
    pub input: Option<String>,
    pub output: Option<String>,
    pub virtual_mic: bool,
    pub monitor: Option<Option<String>>,
}
impl AudioOptions {
    pub fn devices(&self) -> Devices {
        Devices {
            input: self.input.clone(),
            output: self.output.clone(),
            virtual_mic: self.virtual_mic,
            monitor: self.monitor.clone(),
        }
    }
    fn set_devices(&mut self, devices: Devices) {
        self.input = devices.input;
        self.output = devices.output;
        self.virtual_mic = devices.virtual_mic;
        self.monitor = devices.monitor;
    }
}
/// Changes the session makes without stopping.
enum Control {
    /// Load this voice in the background, then switch to it.
    Voice(PathBuf),
    /// Reopen the audio streams on these devices; the voice stays loaded.
    Devices(Devices),
}
pub struct AudioSession {
    pub metrics: Arc<Metrics>,
    stop: Arc<AtomicBool>,
    settings: mpsc::SyncSender<Settings>,
    control: mpsc::Sender<Control>,
}
impl AudioSession {
    pub fn start(options: AudioOptions) -> Result<Self> {
        options.settings.validate()?;
        let metrics = Arc::new(Metrics::default());
        metrics.running.store(true, Ordering::Relaxed);
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::sync_channel(1);
        let (control_tx, control_rx) = mpsc::channel();
        let worker_metrics = metrics.clone();
        let worker_stop = stop.clone();
        thread::Builder::new()
            .name("voice-engine".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run(
                        options,
                        rx,
                        control_rx,
                        &worker_stop,
                        worker_metrics.clone(),
                    )
                }));
                let error = match result {
                    Ok(Ok(())) => None,
                    Ok(Err(error)) => Some(format!("{error:#}")),
                    Err(_) => Some(
                        "Audio worker panicked; inspect ONNX Runtime and model compatibility"
                            .into(),
                    ),
                };
                if let Some(error) = error {
                    *worker_metrics.error.lock().unwrap() = Some(error);
                }
                worker_metrics.ready.store(false, Ordering::Relaxed);
                worker_metrics.running.store(false, Ordering::Release);
            })?;
        Ok(Self {
            metrics,
            stop,
            settings: tx,
            control: control_tx,
        })
    }
    pub fn update(&self, settings: Settings) {
        let _ = self.settings.try_send(settings);
    }
    /// Switches to another voice without stopping: it loads in the background while
    /// the current one keeps playing.
    pub fn switch_voice(&self, manifest: PathBuf) {
        self.metrics.voice_loading.store(true, Ordering::Release);
        let _ = self.control.send(Control::Voice(manifest));
    }
    /// Moves the session to other devices; audio pauses briefly while the streams
    /// reopen, and the voice stays loaded.
    pub fn switch_devices(&self, devices: Devices) {
        let _ = self.control.send(Control::Devices(devices));
    }
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Release);
    }
    pub fn set_recording(&self, on: bool) {
        self.metrics.recording.store(on, Ordering::Release);
    }
    pub fn is_recording(&self) -> bool {
        self.metrics.recording.load(Ordering::Acquire)
    }
    pub fn is_stopping(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }
}
impl Drop for AudioSession {
    fn drop(&mut self) {
        self.stop();
    }
}

/// REC: the microphone and the converted voice as two mono 16-bit WAV files.
struct Recorder {
    input: hound::WavWriter<std::io::BufWriter<std::fs::File>>,
    output: hound::WavWriter<std::io::BufWriter<std::fs::File>>,
    stem: PathBuf,
}
impl Recorder {
    fn create(directory: &Path, rate: u32) -> Result<Self> {
        std::fs::create_dir_all(directory)
            .with_context(|| format!("Create {}", directory.display()))?;
        let seconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_secs());
        let stem = directory.join(format!("voice-{seconds}"));
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let open = |suffix: &str| {
            hound::WavWriter::create(stem.with_file_name(Self::name(&stem, suffix)), spec)
                .context("Create recording")
        };
        Ok(Self {
            input: open("in")?,
            output: open("out")?,
            stem,
        })
    }
    fn name(stem: &Path, suffix: &str) -> String {
        format!(
            "{}-{suffix}.wav",
            stem.file_name().unwrap_or_default().to_string_lossy()
        )
    }
    fn write(&mut self, input: &[f32], output: &[f32]) -> Result<()> {
        for (writer, samples) in [(&mut self.input, input), (&mut self.output, output)] {
            for sample in samples {
                writer.write_sample((sample.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
            }
        }
        Ok(())
    }
    fn finish(
        self,
        metrics: Arc<Metrics>,
        jobs: &mut Vec<thread::JoinHandle<()>>,
        model: Option<String>,
        settings: Settings,
    ) -> Result<String> {
        self.input.finalize()?;
        self.output.finalize()?;
        let dir = self.stem.parent().unwrap_or(Path::new(".")).to_path_buf();
        let in_name = Self::name(&self.stem, "in");
        let out_name = Self::name(&self.stem, "out");
        let in_path = self.stem.with_file_name(&in_name);
        let out_path = self.stem.with_file_name(&out_name);
        let stem = self.stem;
        let base = format!(
            "Recording saved in {}: {in_name} and {out_name}",
            dir.display()
        );
        let fallback = base.clone();

        jobs.retain(|handle| !handle.is_finished());
        if let Ok(handle) = thread::Builder::new()
            .name("voice-analyze".into())
            .spawn(move || {
                let message = match crate::analyze::analyze_wavs_with_context(
                    &in_path,
                    Some(&out_path),
                    &stem,
                    model.as_deref(),
                    Some(&settings),
                ) {
                    Ok((report, json_path, png_path)) => {
                        let json_name = json_path
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into_owned();
                        let plot = png_path
                            .and_then(|p| {
                                p.file_name().map(|n| format!(", {}", n.to_string_lossy()))
                            })
                            .unwrap_or_default();
                        let summary = crate::analyze::format_summary(&report)
                            .lines()
                            .take(3)
                            .collect::<Vec<_>>()
                            .join(" | ");
                        format!("{fallback} (report: {json_name}{plot} — {summary})")
                    }
                    Err(_) => fallback,
                };
                *metrics.recorded.lock().unwrap() = Some(message);
            })
        {
            jobs.push(handle);
            Ok(format!("{base} (analyzing…)"))
        } else {
            Ok(base)
        }
    }
}

/// Where the converted voice plays.
fn voice_output(host: &cpal::Host, options: &AudioOptions) -> Result<cpal::Device> {
    if !options.virtual_mic {
        return select_device(host, &options.output, false);
    }
    // Windows: VB-Cable's speaker, which feeds its microphone.
    #[cfg(windows)]
    {
        crate::windows_mic::output_device(host)
    }
    // Linux: the sound server's default device; the stream moves to the virtual
    // microphone once playing.
    #[cfg(not(windows))]
    {
        select_device(host, &None, false)
    }
}

fn select_device(host: &cpal::Host, id: &Option<String>, input: bool) -> Result<cpal::Device> {
    if let Some(id) = id {
        return host
            .device_by_id(&id.parse()?)
            .context("Selected audio device is unavailable; refresh devices");
    }
    if input {
        host.default_input_device()
    } else {
        host.default_output_device()
    }
    .context("No default audio device available")
}

/// Opens a playback stream under its own name (Linux), so the sound server routes the
/// voice and the monitor separately; the voice targets the virtual microphone from
/// the start when that is on.
fn named_output(
    stream: OutputRole,
    device: &cpal::Device,
    config: cpal::SupportedStreamConfig,
    queue: Arc<ArrayQueue<f32>>,
    metrics: &Arc<Metrics>,
) -> Result<cpal::Stream> {
    let monitor = matches!(stream, OutputRole::Monitor);
    let open = || open_output(device, config, queue, metrics, monitor);
    #[cfg(target_os = "linux")]
    {
        use crate::virtual_mic::{open_stream, Stream};
        open_stream(
            match stream {
                OutputRole::Voice { virtual_mic } => Stream::Output(virtual_mic),
                OutputRole::Monitor => Stream::Monitor,
            },
            open,
        )
    }
    #[cfg(not(target_os = "linux"))]
    open()
}

#[derive(Clone, Copy)]
enum OutputRole {
    Voice {
        #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
        virtual_mic: bool,
    },
    Monitor,
}

/// Waits for the voice stream to reach the virtual microphone, then keeps it there
/// until the session ends.
fn route_to_virtual_mic(stop: Arc<AtomicBool>) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        use crate::virtual_mic::ensure_routed;
        let mut found = false;
        for _ in 0..40 {
            if ensure_routed()? > 0 {
                found = true;
                break;
            }
            thread::sleep(Duration::from_millis(50));
        }
        anyhow::ensure!(
            found,
            "The voice could not be sent to Voice Changer Mic; PipeWire or PulseAudio must be the default output"
        );
        thread::Builder::new()
            .name("virtual-mic-route".into())
            .spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    thread::sleep(Duration::from_secs(2));
                    if !stop.load(Ordering::Acquire) {
                        let _ = ensure_routed();
                    }
                }
            })?;
        Ok(())
    }
    // Windows: the voice already plays into VB-Cable's speaker, which feeds its mic.
    #[cfg(windows)]
    {
        let _ = stop;
        Ok(())
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        let _ = stop;
        bail!("The built-in virtual microphone is only available on Linux and Windows")
    }
}

fn run(
    mut options: AudioOptions,
    updates: mpsc::Receiver<Settings>,
    control: mpsc::Receiver<Control>,
    stop: &AtomicBool,
    metrics: Arc<Metrics>,
) -> Result<()> {
    crate::platform::prepare_audio_session()?;
    let converter = if let Some(path) = &options.manifest {
        Some(engine::load_converter_on(path, options.settings.device)?)
    } else {
        None
    };
    anyhow::ensure!(
        options.settings.bypass || converter.is_some(),
        "Select a model manifest or enable bypass"
    );
    if stop.load(Ordering::Acquire) {
        return Ok(());
    }
    let mut live = Live {
        engine: Engine::new(converter),
        settings: options.settings.clone(),
        model_label: options.manifest.as_ref().map(|p| p.display().to_string()),
        analysis_jobs: Vec::new(),
        loading: None,
    };
    while let Some(devices) = run_devices(&options, &mut live, &updates, &control, stop, &metrics)?
    {
        log::info!("Switching audio devices");
        options.set_devices(devices);
        // The new devices may run at other rates; start the voice afresh.
        live.engine.reset();
        metrics.ready.store(false, Ordering::Release);
        metrics.queued_samples.store(0, Ordering::Relaxed);
    }
    for job in live.analysis_jobs {
        let _ = job.join();
    }
    metrics.recording.store(false, Ordering::Release);
    log::info!("Audio stopped");
    Ok(())
}

type Loaded = Result<Box<dyn engine::Converter>, String>;

/// What a session keeps while its devices change.
struct Live {
    engine: Engine,
    settings: Settings,
    model_label: Option<String>,
    analysis_jobs: Vec<thread::JoinHandle<()>>,
    /// A voice loading in the background, and its label.
    loading: Option<(String, mpsc::Receiver<Loaded>)>,
}
impl Live {
    fn load_voice(&mut self, path: PathBuf, device: crate::config::Device, metrics: &Arc<Metrics>) {
        let label = path.display().to_string();
        log::info!("Loading voice {label} to switch to");
        let (tx, rx) = mpsc::channel();
        let started = thread::Builder::new()
            .name("voice-switch".into())
            .spawn(move || {
                let loaded = std::panic::catch_unwind(|| engine::load_converter_on(&path, device))
                    .map_err(|_| "Loading the voice panicked".to_owned())
                    .and_then(|loaded| loaded.map_err(|error| format!("{error:#}")));
                let _ = tx.send(loaded);
            });
        match started {
            Ok(_) => self.loading = Some((label, rx)),
            Err(error) => {
                metrics.voice_loading.store(false, Ordering::Release);
                *metrics.voice_error.lock().unwrap() = Some(error.to_string());
            }
        }
    }
    /// Switches to a voice that finished loading; returns its label.
    fn finish_voice(&mut self, metrics: &Arc<Metrics>) -> Option<String> {
        let (_, rx) = self.loading.as_ref()?;
        let loaded = match rx.try_recv() {
            Ok(loaded) => loaded,
            Err(mpsc::TryRecvError::Empty) => return None,
            Err(mpsc::TryRecvError::Disconnected) => Err("Loading the voice stopped".into()),
        };
        let (label, _) = self.loading.take().expect("checked above");
        metrics.voice_loading.store(false, Ordering::Release);
        match loaded {
            Ok(converter) => {
                self.engine.set_converter(converter);
                log::info!("Switched to {label}");
                Some(label)
            }
            Err(error) => {
                log::error!("Could not switch voice: {error}");
                *metrics.voice_error.lock().unwrap() = Some(error);
                None
            }
        }
    }
}

/// Opens the streams on `options`' devices and converts until the session stops
/// (`None`) or is moved to other devices (`Some`).
fn run_devices(
    options: &AudioOptions,
    live: &mut Live,
    updates: &mpsc::Receiver<Settings>,
    control: &mpsc::Receiver<Control>,
    stop: &AtomicBool,
    metrics: &Arc<Metrics>,
) -> Result<Option<Devices>> {
    let host = cpal::default_host();
    let input_device = select_device(&host, &options.input, true)?;
    let output_device = voice_output(&host, options)?;
    let input_config = input_device.default_input_config()?;
    let output_config = output_device.default_output_config()?;
    let input_rate = input_config.sample_rate();
    let output_rate = output_config.sample_rate();
    log::info!(
        "Microphone: {} ({} Hz, {} ch, {:?})",
        input_device,
        input_rate,
        input_config.channels(),
        input_config.sample_format()
    );
    log::info!(
        "Output: {} ({} Hz, {} ch, {:?}){}",
        output_device,
        output_rate,
        output_config.channels(),
        output_config.sample_format(),
        if options.virtual_mic {
            ", routed to the virtual microphone"
        } else {
            ""
        }
    );
    metrics.input_rate.store(input_rate, Ordering::Relaxed);
    metrics.output_rate.store(output_rate, Ordering::Relaxed);
    // Both queues are bounded; callbacks never wait on inference, locks, or allocation.
    let input = Arc::new(ArrayQueue::new(input_rate as usize * 2));
    let output = Arc::new(ArrayQueue::new(output_rate as usize * 2));
    let input_stream = match input_config.sample_format() {
        SampleFormat::F32 => input_stream::<f32>(
            &input_device,
            &input_config.into(),
            input.clone(),
            Arc::clone(metrics),
        )?,
        SampleFormat::I16 => input_stream::<i16>(
            &input_device,
            &input_config.into(),
            input.clone(),
            Arc::clone(metrics),
        )?,
        SampleFormat::I32 => input_stream::<i32>(
            &input_device,
            &input_config.into(),
            input.clone(),
            Arc::clone(metrics),
        )?,
        SampleFormat::U16 => input_stream::<u16>(
            &input_device,
            &input_config.into(),
            input.clone(),
            Arc::clone(metrics),
        )?,
        format => bail!("Unsupported input sample format: {format:?}"),
    };
    let output_stream = named_output(
        OutputRole::Voice {
            virtual_mic: options.virtual_mic,
        },
        &output_device,
        output_config,
        output.clone(),
        metrics,
    )?;
    let mut monitor: Option<Monitor> = None;
    // Ends the virtual microphone's route keeper with the session.
    let routing = Arc::new(AtomicBool::new(false));
    struct EndRouting(Arc<AtomicBool>);
    impl Drop for EndRouting {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }
    let _end_routing = EndRouting(routing.clone());
    let block_size = input_rate as usize * live.settings.block_ms as usize / 1000;
    let mut pending = Vec::with_capacity(block_size);
    let mut rate_converter = StreamResampler::new(input_rate, output_rate);
    input_stream.play()?;
    // Output begins when the first processed block is available.
    let mut output_started = false;
    let mut recorder: Option<Recorder> = None;
    metrics.ready.store(true, Ordering::Release);
    log::info!(
        "Live: block {} ms, extra {} ms, crossfade {} ms, pitch {:?}, tune {:+.1}, index {:.0}%, noise {:?}, echo {}{}",
        live.settings.block_ms,
        live.settings.context_ms,
        live.settings.overlap_ms,
        live.settings.f0_detector,
        live.settings.semitones,
        live.settings.index_ratio * 100.0,
        live.settings.noise,
        if live.settings.echo { "on" } else { "off" },
        if live.settings.bypass { ", voice changer off" } else { "" }
    );
    let mut report = (Instant::now(), 0u32, 0f32, 0f32);
    while !stop.load(Ordering::Acquire) {
        if metrics.stream_error.load(Ordering::Relaxed) {
            let detail = metrics.stream_error_detail.lock().unwrap().take();
            match detail {
                Some(detail) => bail!(
                    "Audio stream failed ({detail}); device may have disconnected or microphone permission may be missing"
                ),
                None => bail!(
                    "Audio stream failed; device may have disconnected or microphone permission may be missing"
                ),
            }
        }
        for command in control.try_iter() {
            match command {
                Control::Voice(path) => live.load_voice(path, live.settings.device, metrics),
                Control::Devices(devices) => {
                    finish_recording(
                        &mut recorder,
                        metrics,
                        &mut live.analysis_jobs,
                        live.model_label.clone(),
                        live.settings.clone(),
                    );
                    return Ok(Some(devices));
                }
            }
        }
        if let Some(label) = live.finish_voice(metrics) {
            live.model_label = Some(label);
        }
        for update in updates.try_iter() {
            update.validate()?;
            // Geometry changes are applied on the next session; gain/pitch/bypass are live.
            anyhow::ensure!(
                update.block_ms == live.settings.block_ms
                    && update.context_ms == live.settings.context_ms
                    && update.overlap_ms == live.settings.overlap_ms
                    && update.search_ms == live.settings.search_ms,
                "Stop audio before changing buffer durations"
            );
            live.settings = update;
        }
        // Discard accumulated stale input before selecting the next block.
        if input.len() > block_size * 2 {
            pending.clear();
            while input.len() > block_size {
                if input.pop().is_some() {
                    metrics.dropped_input.fetch_add(1, Ordering::Relaxed);
                }
            }
            live.engine.reset();
        }
        while pending.len() < block_size {
            let Some(sample) = input.pop() else {
                break;
            };
            pending.push(sample);
        }
        if pending.len() < block_size {
            thread::sleep(Duration::from_millis(2));
            continue;
        }
        metrics
            .input_level
            .store(dsp::rms(&pending).to_bits(), Ordering::Relaxed);
        let started = Instant::now();
        let processed = live.engine.process(&pending, input_rate, &live.settings)?;
        let took = started.elapsed().as_secs_f32() * 1000.0;
        metrics
            .processing_ms
            .store(took.to_bits(), Ordering::Relaxed);
        // A short summary every 10 seconds, so the terminal shows how it keeps up.
        report.1 += 1;
        report.2 += took;
        report.3 = report.3.max(took);
        if report.0.elapsed() >= Duration::from_secs(10) {
            log::info!(
                "{} blocks: {:.1} ms average, {:.1} ms worst of {} ms; queued {} ms; dropped {} in / {} out; underrun {} samples",
                report.1,
                report.2 / report.1 as f32,
                report.3,
                live.settings.block_ms,
                output.len() as u64 * 1000 / output_rate.max(1) as u64,
                metrics.dropped_input.load(Ordering::Relaxed),
                metrics.dropped_output.load(Ordering::Relaxed),
                metrics.underrun_samples.load(Ordering::Relaxed)
            );
            report = (Instant::now(), 0, 0.0, 0.0);
        }
        metrics
            .output_level
            .store(dsp::rms(&processed).to_bits(), Ordering::Relaxed);
        match (
            metrics.recording.load(Ordering::Acquire),
            recorder.is_some(),
        ) {
            (true, false) => match Recorder::create(&options.record_dir, input_rate) {
                Ok(created) => recorder = Some(created),
                Err(error) => {
                    metrics.recording.store(false, Ordering::Release);
                    *metrics.recorded.lock().unwrap() =
                        Some(format!("Recording failed: {error:#}"));
                }
            },
            (false, true) => finish_recording(
                &mut recorder,
                metrics,
                &mut live.analysis_jobs,
                live.model_label.clone(),
                live.settings.clone(),
            ),
            _ => {}
        }
        if let Some(active) = &mut recorder {
            if let Err(error) = active.write(&pending, &processed) {
                recorder = None;
                metrics.recording.store(false, Ordering::Release);
                *metrics.recorded.lock().unwrap() = Some(format!("Recording failed: {error:#}"));
            }
        }
        // With the voice changer off, a virtual microphone carries silence rather
        // than the raw voice, so nobody hears the real voice by surprise.
        let processed = if options.virtual_mic && live.settings.bypass {
            vec![0.0; processed.len()]
        } else {
            processed
        };
        if let Some(monitor) = &mut monitor {
            monitor.push(&processed);
        }
        let processed = rate_converter.push(&processed);
        // A quarter block of lead silence, so a block that takes longer than the
        // first one does not run the output dry.
        let lead = output_rate as usize * live.settings.block_ms as usize / 4000;
        if !output_started {
            for _ in 0..lead {
                let _ = output.push(0.0);
            }
        } else if output.len() > lead * 2 {
            // Just before a block is added the queue holds about the lead. More means
            // the output fell behind: it played silence through a slow block, or the
            // output device's clock runs slower than the microphone's. Drop the oldest
            // audio so the delay returns to the lead instead of growing for good.
            while output.len() > lead {
                if output.pop().is_some() {
                    metrics.dropped_output.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        for sample in processed {
            if output.push(sample).is_err() {
                metrics.dropped_output.fetch_add(1, Ordering::Relaxed);
            }
        }
        if !output_started {
            output_stream.play()?;
            output_started = true;
            if options.virtual_mic {
                route_to_virtual_mic(routing.clone())?;
                log::info!("Voice is going to the virtual microphone");
            }
            if let Some(device) = &options.monitor {
                monitor = Some(Monitor::open(&host, device, input_rate, metrics)?);
            }
        }
        if metrics.monitor_error.swap(false, Ordering::Relaxed) && monitor.take().is_some() {
            *metrics.recorded.lock().unwrap() =
                Some("The monitor device stopped; the voice still goes to the output.".into());
        }
        metrics
            .queued_samples
            .store(output.len() as u64, Ordering::Relaxed);
        pending.clear();
    }
    finish_recording(
        &mut recorder,
        metrics,
        &mut live.analysis_jobs,
        live.model_label.clone(),
        live.settings.clone(),
    );
    Ok(None)
}

fn finish_recording(
    recorder: &mut Option<Recorder>,
    metrics: &Arc<Metrics>,
    jobs: &mut Vec<thread::JoinHandle<()>>,
    model: Option<String>,
    settings: Settings,
) {
    if let Some(active) = recorder.take() {
        *metrics.recorded.lock().unwrap() = Some(
            match active.finish(metrics.clone(), jobs, model, settings) {
                Ok(saved) => saved,
                Err(error) => format!("Recording failed: {error:#}"),
            },
        );
    }
}

fn is_recoverable_stream_error(err: &cpal::Error) -> bool {
    matches!(
        err.kind(),
        cpal::ErrorKind::Xrun | cpal::ErrorKind::RealtimeDenied | cpal::ErrorKind::DeviceChanged
    )
}

fn record_stream_error(err: &cpal::Error, direction: &str, metrics: &Metrics, monitor: bool) {
    let stream = if monitor { "monitor" } else { direction };
    if is_recoverable_stream_error(err) {
        log::warn!("Audio {stream}: {err} (recovered)");
        return;
    }
    log::error!("Audio {stream} failed: {err}");
    if monitor {
        metrics.monitor_error.store(true, Ordering::Relaxed);
        return;
    }
    if let Ok(mut detail) = metrics.stream_error_detail.try_lock() {
        if detail.is_none() {
            *detail = Some(format!("{direction}: {err}"));
        }
    }
    metrics.stream_error.store(true, Ordering::Relaxed);
}

fn input_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    queue: Arc<ArrayQueue<f32>>,
    metrics: Arc<Metrics>,
) -> Result<cpal::Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = config.channels as usize;
    let errors = metrics.clone();
    Ok(device.build_input_stream(
        *config,
        move |samples: &[T], _| {
            for frame in samples.chunks_exact(channels) {
                let sample =
                    frame.iter().map(|s| s.to_sample::<f32>()).sum::<f32>() / channels as f32;
                if queue
                    .push(if sample.is_finite() { sample } else { 0.0 })
                    .is_err()
                {
                    metrics.dropped_input.fetch_add(1, Ordering::Relaxed);
                }
            }
        },
        move |err| record_stream_error(&err, "input", &errors, false),
        None,
    )?)
}
/// Builds an output stream in the device's own sample format. A monitor stream does not
/// count underruns, and its failure only stops the monitor.
fn open_output(
    device: &cpal::Device,
    config: cpal::SupportedStreamConfig,
    queue: Arc<ArrayQueue<f32>>,
    metrics: &Arc<Metrics>,
    monitor: bool,
) -> Result<cpal::Stream> {
    let format = config.sample_format();
    let config = config.into();
    let metrics = metrics.clone();
    match format {
        SampleFormat::F32 => output_stream::<f32>(device, &config, queue, metrics, monitor),
        SampleFormat::I16 => output_stream::<i16>(device, &config, queue, metrics, monitor),
        SampleFormat::I32 => output_stream::<i32>(device, &config, queue, metrics, monitor),
        SampleFormat::U16 => output_stream::<u16>(device, &config, queue, metrics, monitor),
        format => bail!("Unsupported output sample format: {format:?}"),
    }
}

fn output_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    queue: Arc<ArrayQueue<f32>>,
    metrics: Arc<Metrics>,
    monitor: bool,
) -> Result<cpal::Stream>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;
    let errors = metrics.clone();
    Ok(device.build_output_stream(
        *config,
        move |samples: &mut [T], _| {
            for frame in samples.chunks_exact_mut(channels) {
                let sample = queue.pop().unwrap_or_else(|| {
                    if !monitor {
                        metrics.underrun_samples.fetch_add(1, Ordering::Relaxed);
                    }
                    0.0
                });
                frame.fill(T::from_sample(sample.clamp(-1.0, 1.0)));
            }
        },
        move |err| record_stream_error(&err, "output", &errors, monitor),
        None,
    )?)
}

/// The second output for hearing the converted voice.
struct Monitor {
    stream: cpal::Stream,
    queue: Arc<ArrayQueue<f32>>,
    resampler: StreamResampler,
    rate: u32,
    started: bool,
}
impl Monitor {
    fn open(
        host: &cpal::Host,
        device: &Option<String>,
        input_rate: u32,
        metrics: &Arc<Metrics>,
    ) -> Result<Self> {
        let device = select_device(host, device, false).context("Monitor device")?;
        let config = device.default_output_config()?;
        let rate = config.sample_rate();
        let queue = Arc::new(ArrayQueue::new(rate as usize * 2));
        log::info!("Monitor: {} ({} Hz)", device, rate);
        let stream = named_output(OutputRole::Monitor, &device, config, queue.clone(), metrics)?;
        stream.play()?;
        Ok(Self {
            stream,
            queue,
            resampler: StreamResampler::new(input_rate, rate),
            rate,
            started: false,
        })
    }
    fn push(&mut self, processed: &[f32]) {
        let _ = &self.stream;
        if !self.started {
            // The same lead silence as the main output.
            for _ in 0..self.rate as usize / 40 {
                let _ = self.queue.push(0.0);
            }
            self.started = true;
        }
        // Keep the monitor close to live; drop the oldest audio if it falls behind.
        let limit = self.rate as usize / 2;
        for sample in self.resampler.push(processed) {
            while self.queue.len() > limit {
                self.queue.pop();
            }
            let _ = self.queue.push(sample);
        }
    }
}
