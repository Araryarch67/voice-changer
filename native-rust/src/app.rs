#[cfg(feature = "audio")]
use crate::audio::{self, AudioDevice, AudioOptions, AudioSession};
use crate::config::{Device, F0Detector, Noise, Settings};
use crate::m3::{self, icons};
use eframe::egui::{self, FontId, Margin, RichText, Vec2};
use serde::{Deserialize, Serialize};
#[cfg(feature = "audio")]
use std::sync::atomic::Ordering;
use std::{path::PathBuf, sync::mpsc};

#[cfg(feature = "catalog")]
mod browse;
#[cfg(feature = "catalog")]
mod setup;

#[derive(Clone, Serialize, Deserialize)]
struct ModelSlot {
    name: String,
    manifest: String,
    #[serde(default)]
    favorite: bool,
    /// This voice's own tuning; `None` until the voice is first used.
    #[serde(default)]
    tune: Option<VoiceTune>,
}
/// The settings that belong to one voice; everything else in `Settings` (devices,
/// buffering, noise handling) is shared by all voices.
#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct VoiceTune {
    semitones: f32,
    speaker_id: i64,
    index_ratio: f32,
    protect: f32,
    f0_detector: F0Detector,
    output_gain: f32,
}
impl Default for VoiceTune {
    fn default() -> Self {
        Self::from(&Settings::default())
    }
}
impl From<&Settings> for VoiceTune {
    fn from(settings: &Settings) -> Self {
        Self {
            semitones: settings.semitones,
            speaker_id: settings.speaker_id,
            index_ratio: settings.index_ratio,
            protect: settings.protect,
            f0_detector: settings.f0_detector,
            output_gain: settings.output_gain,
        }
    }
}
impl VoiceTune {
    fn apply(self, settings: &mut Settings) {
        settings.semitones = self.semitones;
        settings.speaker_id = self.speaker_id;
        settings.index_ratio = self.index_ratio;
        settings.protect = self.protect;
        settings.f0_detector = self.f0_detector;
        settings.output_gain = self.output_gain;
    }
}
#[derive(Serialize, Deserialize)]
#[serde(default)]
struct Saved {
    settings: Settings,
    input: Option<String>,
    output: Option<String>,
    slots: Vec<ModelSlot>,
    selected: usize,
    /// Built-in virtual microphone (Linux).
    virtual_mic: bool,
    /// Also play the converted voice on `monitor_device`, to hear yourself.
    monitor: bool,
    monitor_device: Option<String>,
}
impl Default for Saved {
    fn default() -> Self {
        Self {
            settings: Settings::default(),
            input: None,
            output: None,
            slots: (1..=8)
                .map(|i| ModelSlot {
                    name: format!("Voice {i:02}"),
                    manifest: String::new(),
                    favorite: false,
                    tune: None,
                })
                .collect(),
            selected: 0,
            virtual_mic: false,
            monitor: false,
            monitor_device: None,
        }
    }
}
#[derive(Clone, Copy, PartialEq)]
enum Page {
    Voices,
    Studio,
    Settings,
}
#[derive(Clone, Copy, PartialEq)]
enum Filter {
    All,
    Favorites,
    Ready,
}
pub struct VoiceApp {
    saved: Saved,
    error: Option<String>,
    page: Page,
    filter: Filter,
    search: String,
    edit_model: bool,
    file_dialog: bool,
    file_input: String,
    import_dialog: bool,
    #[cfg(feature = "catalog")]
    browse: browse::Browse,
    /// Downloads the shared models on first start.
    #[cfg(feature = "catalog")]
    setup: setup::Setup,
    import_checkpoint: String,
    import_index: String,
    file_output: String,
    file_analyze: bool,
    work: Option<mpsc::Receiver<Result<String, String>>>,
    message: Option<String>,
    /// The error and message last written to the terminal, so each is logged once.
    logged: (Option<String>, Option<String>),
    /// Voice pictures by manifest path; `None` once a voice is known to have none.
    portraits: std::collections::HashMap<String, Option<egui::TextureHandle>>,
    #[cfg(feature = "audio")]
    devices: Vec<AudioDevice>,
    #[cfg(feature = "audio")]
    session: Option<AudioSession>,
    /// The slot whose tuning is in `saved.settings`.
    tuned_slot: usize,
    /// The devices the running session uses, to move it when they change.
    #[cfg(feature = "audio")]
    live_devices: Option<crate::audio::Devices>,
    /// Watches the "Voice Changer Mic" devices while they exist.
    #[cfg(all(feature = "audio", target_os = "linux"))]
    virtual_mic: Option<crate::virtual_mic::Guard>,
}
impl VoiceApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        m3::install(&cc.egui_ctx);
        let stored: Option<Saved> = cc
            .storage
            .and_then(|s| eframe::get_value(s, "voice-changer"));
        let first_run = stored.is_none();
        let mut saved = stored.unwrap_or_default();
        // A GPU is used from the start when there is one; the choice is kept afterwards.
        if first_run {
            if let Some(gpu) = crate::runtime::preferred_gpu() {
                saved.settings.device = gpu;
            }
        }
        if saved.settings.validate().is_err() {
            saved.settings = Settings::default();
        }
        if saved.slots.is_empty() {
            saved.slots = Saved::default().slots;
        }
        saved.selected = saved.selected.min(saved.slots.len() - 1);
        let was_empty = saved
            .slots
            .iter()
            .all(|slot| slot.manifest.trim().is_empty());
        adopt_models(&mut saved);
        log::info!(
            "Models folder: {}",
            crate::runtime::models_root()
                .map(|root| root.display().to_string())
                .unwrap_or_else(|| {
                    "not found (needs shared/contentvec.onnx and shared/rmvpe.onnx)".into()
                })
        );
        log::info!(
            "{} voices ready",
            saved
                .slots
                .iter()
                .filter(|slot| !slot.manifest.trim().is_empty())
                .count()
        );
        if was_empty && !saved.slots[0].manifest.is_empty() {
            saved.selected = 0;
            saved.settings.bypass = false;
        }
        let mut app = Self {
            saved,
            error: None,
            page: Page::Voices,
            filter: Filter::All,
            search: String::new(),
            edit_model: false,
            file_dialog: false,
            file_input: String::new(),
            import_dialog: false,
            #[cfg(feature = "catalog")]
            browse: Default::default(),
            #[cfg(feature = "catalog")]
            setup: setup::Setup::on_start(),
            import_checkpoint: String::new(),
            import_index: String::new(),
            file_output: String::new(),
            file_analyze: true,
            work: None,
            message: None,
            logged: (None, None),
            portraits: Default::default(),
            #[cfg(feature = "audio")]
            devices: Vec::new(),
            #[cfg(feature = "audio")]
            session: None,
            #[cfg(all(feature = "audio", target_os = "linux"))]
            virtual_mic: None,
            tuned_slot: 0,
            #[cfg(feature = "audio")]
            live_devices: None,
        };
        // Voices saved before per-voice tuning keep the tuning in use for the selected one.
        app.tuned_slot = app.saved.selected;
        let current = VoiceTune::from(&app.saved.settings);
        app.saved.slots[app.tuned_slot].tune.get_or_insert(current);
        crate::gpu_info::detect_in_background();
        #[cfg(feature = "audio")]
        app.refresh_devices();
        #[cfg(all(feature = "audio", target_os = "linux"))]
        app.sync_virtual_mic();
        app
    }
    #[cfg(feature = "audio")]
    fn refresh_devices(&mut self) {
        match audio::devices() {
            Ok(d) => self.devices = d,
            Err(e) => self.error = Some(format!("{e:#}")),
        }
    }
    /// Creates or removes the virtual microphone to match the saved option. A failure
    /// turns the option off and says why.
    #[cfg(all(feature = "audio", target_os = "linux"))]
    fn sync_virtual_mic(&mut self) {
        if !self.saved.virtual_mic {
            self.virtual_mic = None;
            crate::virtual_mic::remove();
            return;
        }
        let ready = crate::virtual_mic::ensure().and_then(|()| match self.virtual_mic.take() {
            Some(guard) => Ok(guard),
            None => crate::virtual_mic::Guard::start(),
        });
        match ready {
            Ok(guard) => self.virtual_mic = Some(guard),
            Err(error) => {
                self.saved.virtual_mic = false;
                crate::virtual_mic::remove();
                self.error = Some(format!("{error:#}"));
            }
        }
    }
    /// Virtual microphone: send the voice to VB-Cable.
    #[cfg(all(windows, feature = "audio"))]
    fn windows_mic_settings(&mut self, ui: &mut egui::Ui) {
        use crate::windows_mic;
        let installed = self
            .devices
            .iter()
            .any(|device| !device.input && windows_mic::is_cable(&device.name));
        m3::field_label(ui, "Virtual microphone");
        if !installed {
            self.saved.virtual_mic = false;
            ui.horizontal(|ui| {
                if ui
                    .add(m3::Button::tonal("Get VB-Cable"))
                    .on_hover_text(windows_mic::VB_CABLE_PAGE)
                    .clicked()
                {
                    windows_mic::open_page(windows_mic::VB_CABLE_PAGE);
                }
                if ui
                    .add(m3::Button::text("Refresh devices").icon(icons::REFRESH))
                    .clicked()
                {
                    self.refresh_devices();
                }
            });
            ui.label(m3::supporting(
                "To use the converted voice in Discord or a game, install VB-Cable (free, from VB-Audio's site): unzip it, run VBCABLE_Setup_x64 as administrator, restart Windows, then press Refresh devices. The app then sends its voice there by itself. For OBS alone nothing is needed: add an Application Audio Capture source for this app.",
            ));
            return;
        }
        ui.horizontal(|ui| {
            m3::switch(ui, &mut self.saved.virtual_mic, "Virtual microphone");
            ui.label(if self.saved.virtual_mic {
                format!("On: the voice goes to {}", windows_mic::SPEAKER)
            } else {
                "Off".into()
            });
        });
        ui.label(m3::supporting(&format!(
            "Pick \"{}\" as the microphone in Discord, OBS or a game. It is silent while the voice changer is stopped.",
            windows_mic::MICROPHONE
        )));
        let buffer = windows_mic::vb_cable_buffer();
        ui.horizontal(|ui| {
            if let Some((samples, rate)) = buffer {
                let ms = samples as f32 * 1000.0 / rate as f32;
                ui.label(
                    egui::RichText::new(format!(
                        "VB-Cable buffer: {samples} samples, up to {ms:.0} ms"
                    ))
                    .color(if samples > 2048 {
                        m3::TERTIARY
                    } else {
                        m3::ON_SURFACE
                    }),
                );
            }
            if ui
                .add(m3::Button::text("VB-Cable settings"))
                .on_hover_text(
                    "Opens VB-Cable's control panel; Windows asks for administrator permission",
                )
                .clicked()
            {
                if let Err(error) = windows_mic::open_vb_cable_settings() {
                    self.error = Some(format!("{error:#}"));
                }
            }
        });
        ui.label(m3::supporting(
            "VB-Cable holds the voice in its buffer before Discord hears it. Its default, 7168 samples, adds about 110 ms; in VB-Cable settings set Max Latency to 2048 (or 1024 if it does not crackle). Most of the remaining delay is Chunk in Studio: lower it as far as your computer keeps up.",
        ))
        ;
    }
    fn running(&self) -> bool {
        #[cfg(feature = "audio")]
        {
            self.session.is_some()
        }
        #[cfg(not(feature = "audio"))]
        {
            false
        }
    }
    #[cfg(feature = "audio")]
    /// The devices chosen in Settings.
    #[cfg(feature = "audio")]
    fn chosen_devices(&self) -> crate::audio::Devices {
        crate::audio::Devices {
            input: self.saved.input.clone(),
            output: self.saved.output.clone(),
            virtual_mic: cfg!(any(target_os = "linux", windows)) && self.saved.virtual_mic,
            monitor: self
                .saved
                .monitor
                .then(|| self.saved.monitor_device.clone()),
        }
    }
    /// Keeps each voice's tuning with it: a newly selected voice brings back its own
    /// (or the defaults the first time), and the running session switches to it.
    fn follow_voice(&mut self) {
        let selected = self.saved.selected.min(self.saved.slots.len() - 1);
        if selected == self.tuned_slot {
            if let Some(slot) = self.saved.slots.get_mut(selected) {
                slot.tune = Some(VoiceTune::from(&self.saved.settings));
            }
            return;
        }
        self.tuned_slot = selected;
        let slot = &self.saved.slots[selected];
        slot.tune
            .unwrap_or_default()
            .apply(&mut self.saved.settings);
        #[cfg(feature = "audio")]
        if let Some(session) = &self.session {
            let manifest = slot.manifest.trim();
            if !manifest.is_empty() {
                log::info!("Switching to \"{}\" while live", slot.name);
                session.switch_voice(PathBuf::from(manifest));
            }
        }
    }
    /// Moves a running session to the devices chosen in Settings when they change.
    #[cfg(feature = "audio")]
    fn follow_devices(&mut self) {
        let Some(session) = &self.session else {
            self.live_devices = None;
            return;
        };
        let chosen = self.chosen_devices();
        if self.live_devices.as_ref() != Some(&chosen) {
            session.switch_devices(chosen.clone());
            self.live_devices = Some(chosen);
        }
    }
    fn start(&mut self) {
        if self.work.is_some() {
            return;
        }
        self.error = None;
        match crate::platform::microphone_ready() {
            Ok(true) => {}
            Ok(false) => {
                self.error = Some("Grant microphone permission, then press Start again.".into());
                return;
            }
            Err(error) => {
                self.error = Some(format!("{error:#}"));
                return;
            }
        }
        // Recreate Voice Changer Mic if something removed it since it was turned on.
        #[cfg(target_os = "linux")]
        if self.saved.virtual_mic {
            self.sync_virtual_mic();
            if !self.saved.virtual_mic {
                return;
            }
        }
        let manifest = self.saved.slots[self.saved.selected].manifest.trim();
        let devices = self.chosen_devices();
        let options = AudioOptions {
            input: devices.input.clone(),
            output: devices.output.clone(),
            manifest: if manifest.is_empty() {
                None
            } else {
                Some(manifest.into())
            },
            settings: self.saved.settings.clone(),
            record_dir: recordings_dir(),
            virtual_mic: devices.virtual_mic,
            monitor: devices.monitor.clone(),
        };
        match AudioSession::start(options) {
            Ok(s) => {
                self.live_devices = Some(devices);
                log::info!(
                    "Start: \"{}\" on {}",
                    self.saved.slots[self.saved.selected].name,
                    self.saved.settings.device.label()
                );
                self.session = Some(s)
            }
            Err(e) => self.error = Some(format!("{e:#}")),
        }
    }
    fn navigation(&mut self, ui: &mut egui::Ui, compact: bool) {
        let pages = [
            (Page::Voices, icons::GRAPHIC_EQ, "Voices"),
            (Page::Studio, icons::TUNE, "Studio"),
            (Page::Settings, icons::SETTINGS, "Settings"),
        ];
        if compact {
            ui.columns(pages.len(), |columns| {
                for (column, (page, glyph, name)) in columns.iter_mut().zip(pages) {
                    column.vertical_centered(|ui| {
                        if m3::nav_item(ui, glyph, name, self.page == page).clicked() {
                            self.page = page;
                        }
                    });
                }
            });
            return;
        }
        ui.vertical_centered(|ui| {
            ui.spacing_mut().item_spacing.y = 12.0;
            ui.add_space(4.0);
            let (rect, _) = ui.allocate_exact_size(Vec2::splat(48.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, 16.0, m3::PRIMARY_CONTAINER);
            m3::paint_icon(
                ui,
                rect.center(),
                icons::GRAPHIC_EQ,
                28.0,
                false,
                m3::ON_PRIMARY_CONTAINER,
            );
            ui.add_space(16.0);
            for (page, glyph, name) in pages {
                if m3::nav_item(ui, glyph, name, self.page == page).clicked() {
                    self.page = page;
                }
            }
        });
    }
    /// The picture for a voice: `portrait.png` (or jpg, jpeg, webp) beside its manifest.
    fn portrait(&mut self, ctx: &egui::Context, index: usize) -> Option<egui::TextureHandle> {
        let manifest = self.saved.slots[index].manifest.trim();
        if manifest.is_empty() {
            return None;
        }
        if let Some(known) = self.portraits.get(manifest) {
            return known.clone();
        }
        let texture = portrait_file(std::path::Path::new(manifest)).and_then(|file| {
            // Scaled down once; cards never show it larger than this.
            let picture = image::open(&file).ok()?.thumbnail(640, 640).to_rgba8();
            let size = [picture.width() as usize, picture.height() as usize];
            let pixels = egui::ColorImage::from_rgba_unmultiplied(size, &picture);
            Some(ctx.load_texture(file.to_string_lossy(), pixels, egui::TextureOptions::LINEAR))
        });
        self.portraits.insert(manifest.to_owned(), texture.clone());
        texture
    }
    /// Copies a picture into the selected voice's folder as its portrait.
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    fn set_portrait(&mut self, picked: PathBuf) {
        let manifest = self.saved.slots[self.saved.selected]
            .manifest
            .trim()
            .to_owned();
        let Some(folder) = std::path::Path::new(&manifest).parent() else {
            return;
        };
        let extension = picked
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        let target = folder.join(format!("portrait.{extension}"));
        if picked.canonicalize().ok() != target.canonicalize().ok() {
            for old in PORTRAIT_TYPES {
                let _ = std::fs::remove_file(folder.join(format!("portrait.{old}")));
            }
            if let Err(error) = std::fs::copy(&picked, &target) {
                self.error = Some(format!("Could not save the picture: {error}"));
                return;
            }
        }
        self.portraits.remove(&manifest);
    }
    #[cfg(any(target_os = "android", target_os = "ios"))]
    fn add_slot(&mut self) {
        self.saved.slots.push(ModelSlot {
            name: format!("Voice {:02}", self.saved.slots.len() + 1),
            manifest: String::new(),
            favorite: false,
            tune: None,
        });
        self.saved.selected = self.saved.slots.len() - 1;
        self.edit_model = true;
    }
    fn import_model(&mut self) {
        #[cfg(feature = "catalog")]
        if !self.shared_models_ready() {
            return;
        }
        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        {
            self.import_dialog = true;
        }
        #[cfg(any(target_os = "android", target_os = "ios"))]
        self.add_slot();
    }
    /// Puts a checkpoint in the import dialog, with the index beside it if the index
    /// field is still empty.
    fn choose_checkpoint(&mut self, path: PathBuf) {
        if self.import_index.trim().is_empty() {
            if let Some(index) = path.parent().and_then(crate::import::find_index) {
                self.import_index = index.to_string_lossy().into_owned();
            }
        }
        self.import_checkpoint = path.to_string_lossy().into_owned();
    }
    /// A dropped `.pth` or `.index` opens the import dialog with it filled in.
    fn drop_file(&mut self, path: PathBuf) {
        let extension = path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        match extension.as_str() {
            "pth" => self.choose_checkpoint(path),
            "index" => self.import_index = path.to_string_lossy().into_owned(),
            _ => {
                self.error = Some("Drop an RVC .pth checkpoint or its .index file.".into());
                return;
            }
        }
        self.error = None;
        self.import_dialog = true;
    }
    /// Converts the chosen checkpoint with the chosen index, or with none.
    fn import_pair(&mut self) {
        if self.work.is_some() {
            self.error = Some("Wait for the current task to finish, then import again.".into());
            return;
        }
        let checkpoint = PathBuf::from(self.import_checkpoint.trim());
        let index = Some(self.import_index.trim())
            .filter(|index| !index.is_empty())
            .map(PathBuf::from);
        log::info!(
            "Importing {} with {}",
            checkpoint.display(),
            index
                .as_deref()
                .map_or("no index".into(), |index| index.display().to_string())
        );
        #[cfg(feature = "export")]
        self.begin_work(move || {
            let started = std::time::Instant::now();
            let manifest = crate::import::import_pair(&checkpoint, index.as_deref())?;
            log::info!(
                "Converted to {} in {:.1} s",
                manifest.parent().unwrap_or(&manifest).display(),
                started.elapsed().as_secs_f32()
            );
            let voice = crate::config::ModelManifest::load(&manifest)?;
            Ok(format!(
                "Imported {}{}.",
                voice.name,
                if voice.feature_bank.is_some() {
                    " with its index"
                } else {
                    " without an index"
                }
            ))
        });
        #[cfg(not(feature = "export"))]
        {
            let _ = (checkpoint, index);
            self.error =
                Some("This build cannot convert checkpoints; import a model.json instead.".into());
        }
    }
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    fn import_dialog(&mut self, ctx: &egui::Context) {
        let busy = self.running() || self.work.is_some();
        let (mut close, mut import) = (false, false);
        let (_, dismissed) = m3::dialog(ctx, "import-dialog", "Import RVC voice", |ui| {
            ui.add_enabled_ui(!busy, |ui| {
                m3::field_label(ui, "Model (.pth)");
                ui.horizontal(|ui| {
                    let width = ui.available_width() - 48.0;
                    ui.add(m3::text_edit(
                        &mut self.import_checkpoint,
                        "/path/to/voice.pth",
                        width,
                    ));
                    if m3::icon_button(
                        ui,
                        icons::FOLDER_OPEN,
                        m3::ON_SURFACE_VARIANT,
                        "Choose model…",
                    )
                    .clicked()
                    {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("RVC checkpoint", &["pth"])
                            .pick_file()
                        {
                            // Suggest the index beside it; the user can still change it.
                            self.choose_checkpoint(path);
                        }
                    }
                });
                ui.label(m3::supporting(
                    "The trained voice. It is converted to ONNX, which takes a few seconds.",
                ));
                m3::field_label(ui, "Index (.index, optional)");
                ui.horizontal(|ui| {
                    let width = ui.available_width() - 96.0;
                    ui.add(m3::text_edit(
                        &mut self.import_index,
                        "/path/to/added_IVF_Flat.index",
                        width,
                    ));
                    if m3::icon_button(
                        ui,
                        icons::FOLDER_OPEN,
                        m3::ON_SURFACE_VARIANT,
                        "Choose index…",
                    )
                    .clicked()
                    {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("RVC feature index", &["index"])
                            .pick_file()
                        {
                            self.import_index = path.to_string_lossy().into_owned();
                        }
                    }
                    if m3::icon_button(ui, icons::CLOSE, m3::ON_SURFACE_VARIANT, "No index")
                        .clicked()
                    {
                        self.import_index.clear();
                    }
                });
                ui.label(m3::supporting(
                    "Makes the voice sound closer to the original speaker; the Index slider sets how much. Leave it empty if you have none.",
                ));
            });
            if busy {
                ui.label(m3::supporting(
                    "Stop live audio, or wait for the current model/file task to finish.",
                ));
            }
            m3::actions(ui, |ui| {
                let ready = !busy && !self.import_checkpoint.trim().is_empty();
                import = ui
                    .add_enabled(ready, m3::Button::filled("Import").icon(icons::ADD))
                    .clicked();
                close = ui.add(m3::Button::text("Cancel")).clicked();
            });
        });
        if import {
            self.import_pair();
            self.import_checkpoint.clear();
            self.import_index.clear();
        }
        self.import_dialog = !(import || close || dismissed);
    }
    fn begin_work<F>(&mut self, work: F)
    where
        F: FnOnce() -> anyhow::Result<String> + Send + 'static,
    {
        let (tx, rx) = mpsc::channel();
        self.message = None;
        self.error = None;
        match std::thread::Builder::new()
            .name("voice-file-worker".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work))
                    .map_err(|_| {
                        "Model worker panicked; check the runtime and model files".to_owned()
                    })
                    .and_then(|result| result.map_err(|error| format!("{error:#}")));
                let _ = tx.send(result);
            }) {
            Ok(_) => self.work = Some(rx),
            Err(error) => self.error = Some(error.to_string()),
        }
    }
    fn check_model(&mut self) {
        let path = PathBuf::from(&self.saved.slots[self.saved.selected].manifest);
        let settings = self.saved.settings.clone();
        self.begin_work(move || {
            let result = crate::benchmark::run(&path, &settings, 4)?;
            Ok(format!("Model test passed. Processing {:.1} ms per {} ms block (RTF {:.2}); first block {:.0} ms. {}",
                result.median_ms, result.block_ms, result.realtime_factor, result.first_block_ms,
                if result.realtime_factor <= 1.0 { "Sustained processing fits the block duration on this machine." } else { "Processing exceeds the block duration; increase block size or reduce context." }))
        });
    }
    fn gallery(&mut self, ui: &mut egui::Ui) {
        header(
            ui,
            "Voices",
            "Pick a voice, then press Start; you can switch voices while it runs. Each voice keeps its own Tune, Index and other voice settings. Import an RVC .pth (and its .index), drop them on the window, or find one online.",
        );
        let total = ui.available_width();
        let side = 320.0;
        if total < 860.0 {
            self.voice_grid(ui, true);
            return;
        }
        // Wide: the gallery beside a detail pane for the selected voice.
        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = 16.0;
            let main = total - side - 16.0;
            ui.allocate_ui_with_layout(
                Vec2::new(main, 0.0),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    ui.set_width(main);
                    self.voice_grid(ui, false);
                },
            );
            ui.allocate_ui_with_layout(
                Vec2::new(side, 0.0),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    ui.set_width(side);
                    self.voice_detail(ui);
                },
            );
        });
    }
    fn test_button(&mut self, ui: &mut egui::Ui) {
        let ready = !self.running()
            && self.work.is_none()
            && !self.saved.slots[self.saved.selected].manifest.is_empty();
        if ui
            .add_enabled(ready, m3::Button::tonal("Test model").icon(icons::SPEED))
            .clicked()
        {
            self.check_model();
        }
    }
    fn edit_button(&mut self, ui: &mut egui::Ui) {
        if ui
            .add_enabled(
                !self.running(),
                m3::Button::outlined("Edit").icon(icons::EDIT),
            )
            .clicked()
        {
            self.edit_model = true;
        }
    }
    /// Search, actions, filters and the card grid. `actions` adds the per-voice
    /// buttons that otherwise live in the detail pane.
    fn voice_grid(&mut self, ui: &mut egui::Ui, actions: bool) {
        let running = self.running();
        ui.horizontal_wrapped(|ui| {
            egui::Frame::new()
                .fill(m3::SURFACE_CONTAINER_HIGH)
                .corner_radius(20)
                .inner_margin(Margin::symmetric(12, 0))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.set_height(40.0);
                        ui.label(m3::icon(icons::SEARCH, 20.0, m3::ON_SURFACE_VARIANT));
                        ui.add(
                            egui::TextEdit::singleline(&mut self.search)
                                .hint_text("Search voices")
                                .frame(false)
                                .desired_width(190.0),
                        );
                    });
                });
            if ui
                .add_enabled(
                    !running,
                    m3::Button::filled("Import model").icon(icons::ADD),
                )
                .clicked()
            {
                self.import_model();
            }
            #[cfg(feature = "catalog")]
            self.browse_button(ui);
            if actions {
                self.test_button(ui);
                self.edit_button(ui);
            }
        });
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            for (filter, title) in [
                (Filter::All, "All voices"),
                (Filter::Favorites, "Favorites"),
                (Filter::Ready, "Ready"),
            ] {
                if m3::filter_chip(ui, title, self.filter == filter).clicked() {
                    self.filter = filter;
                }
            }
        });
        ui.add_space(8.0);
        let search = self.search.to_lowercase();
        let visible: Vec<_> = self
            .saved
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                s.name.to_lowercase().contains(&search)
                    && match self.filter {
                        Filter::All => true,
                        Filter::Favorites => s.favorite,
                        Filter::Ready => !s.manifest.trim().is_empty(),
                    }
            })
            .map(|(i, _)| i)
            .collect();
        let gap = 12.0;
        let available = ui.available_width();
        let columns = ((available + gap) / (168.0 + gap)).floor().max(1.0);
        let width = ((available - gap * (columns - 1.0)) / columns).floor();
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = Vec2::splat(gap);
            for index in visible.iter().copied() {
                self.voice_card(ui, index, width, running);
            }
        });
        if visible.is_empty() {
            ui.add_space(32.0);
            ui.vertical_centered(|ui| {
                ui.label(m3::icon(icons::SEARCH, 40.0, m3::OUTLINE));
                ui.label(m3::title("No voices here"));
                ui.label(m3::supporting(
                    "Change the filter or search, or import a model.",
                ));
            });
        }
        ui.add_space(16.0);
        ui.label(m3::supporting(
            "Empty cards are model slots; pretrained character voices are not bundled.",
        ));
    }
    /// The selected voice: portrait, model file and the controls used most while live.
    fn voice_detail(&mut self, ui: &mut egui::Ui) {
        let index = self.saved.selected;
        m3::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            let image = self.portrait(ui.ctx(), index);
            let slot = &self.saved.slots[index];
            let configured = !slot.manifest.trim().is_empty();
            let (tile, _) = ui
                .allocate_exact_size(Vec2::new(ui.available_width(), 132.0), egui::Sense::hover());
            paint_portrait(ui, tile, 12.0, image.as_ref(), index, &slot.name, configured);
            ui.add_space(4.0);
            ui.add(
                egui::Label::new(RichText::new(&slot.name).size(22.0).color(m3::ON_SURFACE))
                    .truncate(),
            );
            let detail = if configured {
                let index = crate::config::ModelManifest::load(std::path::Path::new(
                    slot.manifest.trim(),
                ))
                .ok()
                .map(|model| model.feature_bank.is_some());
                match index {
                    Some(true) => "RVC · ONNX · with index".to_owned(),
                    Some(false) => "RVC · ONNX · no index".to_owned(),
                    None => "RVC · ONNX · files missing".to_owned(),
                }
            } else {
                "Empty slot. Import an RVC .pth to use it.".to_owned()
            };
            ui.add(egui::Label::new(m3::supporting(&detail)).truncate());
            ui.add_space(8.0);
            if configured {
                let s = &mut self.saved.settings;
                m3::setting(
                    ui,
                    "Tune", "Makes the voice higher or lower. +12 is one octave up, -12 one octave down. Male to female voice: about +8 to +12; female to male: about -8 to -12. A natural range sounds most real.",
                    format!("{:+.1} semitones", s.semitones),
                    &mut s.semitones,
                    -24.0..=24.0,
                    0.1,
                );
                m3::setting(
                    ui,
                    "Index", "How much of the model's accent and tone to copy from its .index file. 50-75% usually sounds most like the character; higher can add a metallic edge. No effect on voices without an index.",
                    format!("{:.0} %", s.index_ratio * 100.0),
                    &mut s.index_ratio,
                    0.0..=1.0,
                    0.01,
                );
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    self.test_button(ui);
                    self.edit_button(ui);
                });
            } else if ui
                .add_enabled(
                    !self.running() && self.work.is_none(),
                    m3::Button::filled("Import model").icon(icons::ADD),
                )
                .clicked()
            {
                self.import_model();
            }
        });
    }
    fn voice_card(&mut self, ui: &mut egui::Ui, index: usize, width: f32, running: bool) {
        let (rect, response) =
            ui.allocate_exact_size(Vec2::new(width, 164.0), egui::Sense::click());
        let image = self.portrait(ui.ctx(), index);
        let slot = &self.saved.slots[index];
        let selected = self.saved.selected == index;
        let configured = !slot.manifest.trim().is_empty();
        let favorite = slot.favorite;
        let painter = ui.painter();
        painter.rect_filled(
            rect,
            12.0,
            if selected {
                m3::SECONDARY_CONTAINER
            } else {
                m3::SURFACE_CONTAINER
            },
        );
        if response.hovered() {
            painter.rect_filled(rect, 12.0, m3::ON_SURFACE.gamma_multiply(0.08));
        }
        if selected {
            painter.rect_stroke(
                rect,
                12.0,
                egui::Stroke::new(2.0_f32, m3::PRIMARY),
                egui::StrokeKind::Inside,
            );
        }
        let tile =
            egui::Rect::from_min_size(rect.min + Vec2::splat(8.0), Vec2::new(width - 16.0, 92.0));
        let content = paint_portrait(ui, tile, 8.0, image.as_ref(), index, &slot.name, configured);
        let name = m3::elided(
            ui,
            &slot.name,
            FontId::new(14.0, m3::medium()),
            m3::ON_SURFACE,
            width - 24.0,
        );
        let left = rect.left() + 12.0;
        painter.galley(egui::pos2(left, tile.bottom() + 10.0), name, m3::ON_SURFACE);
        painter.text(
            egui::pos2(left, tile.bottom() + 33.0),
            egui::Align2::LEFT_TOP,
            if configured {
                "RVC · ONNX"
            } else {
                "Empty slot"
            },
            FontId::proportional(12.0),
            if configured {
                m3::SUCCESS
            } else {
                m3::ON_SURFACE_VARIANT
            },
        );
        if selected {
            let badge = tile.left_top() + Vec2::splat(18.0);
            painter.circle_filled(badge, 11.0, m3::PRIMARY);
            m3::paint_icon(ui, badge, icons::CHECK, 16.0, false, m3::ON_PRIMARY);
        }
        // Empty slots only show the heart once they are favourites.
        if configured || favorite {
            let heart = egui::Rect::from_center_size(
                tile.right_top() + Vec2::new(-20.0, 20.0),
                Vec2::splat(36.0),
            );
            let toggle = ui
                .interact(heart, response.id.with("favorite"), egui::Sense::click())
                .on_hover_cursor(egui::CursorIcon::PointingHand)
                .on_hover_text(if favorite {
                    "Remove from favorites"
                } else {
                    "Add to favorites"
                });
            if image.is_some() {
                ui.painter().circle_filled(
                    heart.center(),
                    16.0,
                    egui::Color32::from_black_alpha(110),
                );
            }
            if toggle.hovered() {
                ui.painter()
                    .circle_filled(heart.center(), 18.0, content.gamma_multiply(0.12));
            }
            m3::paint_icon(
                ui,
                heart.center(),
                icons::FAVORITE,
                20.0,
                favorite,
                if favorite { m3::TERTIARY } else { content },
            );
            if toggle.clicked() {
                self.saved.slots[index].favorite = !favorite;
            }
        }
        let empty = self.saved.slots[index].manifest.trim().is_empty();
        // A filled voice can be chosen while running: it loads and takes over.
        if response.clicked() && !(running && empty) {
            self.saved.selected = index;
            // An empty card is the same as Import model.
            if empty && self.work.is_none() {
                self.import_model();
            }
        }
        if response.double_clicked() && !running && !empty {
            self.saved.selected = index;
            self.edit_model = true;
        }
    }
    fn voice_controls(&mut self, ui: &mut egui::Ui) {
        header(ui, "Voice Studio", "Shape the voice and tune latency.");
        ui.horizontal_wrapped(|ui| {
            m3::chip(
                ui,
                icons::VOICE,
                &self.saved.slots[self.saved.selected].name,
            );
            if ui
                .add(m3::Button::outlined("Reset to defaults").icon(icons::REFRESH))
                .on_hover_text("Put every option on this page back to its original value")
                .clicked()
            {
                self.reset_settings();
            }
        });
        ui.add_space(8.0);
        ui.spacing_mut().item_spacing = Vec2::splat(12.0);
        let width = ui.available_width();
        if width > 1080.0 {
            ui.columns(3, |columns| {
                self.voice_settings(&mut columns[0]);
                self.level_settings(&mut columns[1]);
                self.buffer_settings(&mut columns[2]);
            });
        } else if width > 640.0 {
            ui.columns(2, |columns| {
                self.voice_settings(&mut columns[0]);
                self.level_settings(&mut columns[0]);
                self.buffer_settings(&mut columns[1]);
            });
        } else {
            self.voice_settings(ui);
            self.level_settings(ui);
            self.buffer_settings(ui);
        }
    }
    /// Restores the Studio options. Bypass and the device are not Studio options, and
    /// buffer durations cannot change while audio runs, so those are kept.
    fn reset_settings(&mut self) {
        let current = &self.saved.settings;
        let mut fresh = Settings {
            bypass: current.bypass,
            device: current.device,
            ..Settings::default()
        };
        if self.running() {
            fresh.block_ms = current.block_ms;
            fresh.context_ms = current.context_ms;
            fresh.overlap_ms = current.overlap_ms;
            fresh.search_ms = current.search_ms;
        }
        self.saved.settings = fresh;
        self.message = Some(if self.running() {
            "Options reset to defaults. Buffering is kept until audio is stopped.".into()
        } else {
            "Options reset to defaults.".into()
        });
    }
    fn voice_settings(&mut self, ui: &mut egui::Ui) {
        m3::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing = Vec2::splat(8.0);
            m3::card_title(ui, icons::VOICE, "Voice");
            let s = &mut self.saved.settings;
            m3::setting(
                ui,
                "Tune", "Makes the voice higher or lower. +12 is one octave up, -12 one octave down. Male to female voice: about +8 to +12; female to male: about -8 to -12. A natural range sounds most real.",
                format!("{:+.1} semitones", s.semitones),
                &mut s.semitones,
                -24.0..=24.0,
                0.1,
            );
            m3::setting(
                ui,
                "Index", "How much of the model's accent and tone to copy from its .index file. 50-75% usually sounds most like the character; higher can add a metallic edge. No effect on voices without an index.",
                format!("{:.0} %", s.index_ratio * 100.0),
                &mut s.index_ratio,
                0.0..=1.0,
                0.01,
            );
            m3::setting(
                ui,
                "Protect", "Keeps breaths and s, t, k sounds natural when Index is on. Around 0.33 if they sound lispy or robotic; lower protects more; 0.50 turns it off.",
                format!("{:.2}", s.protect),
                &mut s.protect,
                0.0..=0.5,
                0.01,
            );
            choice(
                ui,
                "F0 detector",
                "How your pitch is measured. RMVPE is the best all-rounder and the most natural for speech and singing. CREPE full and Harvest are accurate but heavy; CREPE tiny, DIO and YIN are light but can crack on high notes.",
                &mut s.f0_detector,
                &[
                    (F0Detector::Rmvpe, "RMVPE"),
                    (F0Detector::CrepeFull, "CREPE full"),
                    (F0Detector::CrepeTiny, "CREPE tiny"),
                    (F0Detector::Harvest, "Harvest"),
                    (F0Detector::Dio, "DIO"),
                    (F0Detector::Yin, "YIN"),
                ],
            );
            ui.allocate_ui_with_layout(
                Vec2::new(ui.available_width(), 40.0),
                egui::Layout::right_to_left(egui::Align::Center),
                |ui| {
                    ui.add(egui::DragValue::new(&mut s.speaker_id).range(0..=10000));
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        ui.label("Speaker ID");
                    });
                },
            );
            ui.label(m3::supporting(
                "Only for models that contain several voices. Leave at 0 otherwise.",
            ));
        });
    }
    fn level_settings(&mut self, ui: &mut egui::Ui) {
        m3::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing = Vec2::splat(8.0);
            m3::card_title(ui, icons::MIC, "Levels");
            let s = &mut self.saved.settings;
            m3::setting(
                ui,
                "Input gain", "Microphone volume before conversion. Raise it if you sound too quiet.",
                format!("{:.2}×", s.input_gain),
                &mut s.input_gain,
                0.0..=4.0,
                0.01,
            );
            m3::setting(
                ui,
                "Output gain", "Volume of the converted voice.",
                format!("{:.2}×", s.output_gain),
                &mut s.output_gain,
                0.0..=4.0,
                0.01,
            );
            // The engine gates on block RMS; show it in dBFS, where -100 is effectively off.
            let mut gate = (20.0 * s.silence_threshold.max(0.00001).log10()).round();
            let shown = if gate <= -100.0 {
                "Off".to_owned()
            } else {
                format!("{gate:.0} dB")
            };
            if m3::setting(
                ui,
                "Noise gate (S.Thresh)", "Below this nothing is converted, so room noise stays out and the GPU rests while you are silent. -50 dB suits most microphones; lower it if quiet words are cut. The pink tick on the microphone meter shows it.",
                shown,
                &mut gate,
                -100.0..=-20.0,
                1.0,
            )
            .changed()
            {
                s.silence_threshold = 10f32.powf(gate / 20.0);
            }
            choice(
                ui,
                "Noise suppression", "Removes background noise. Sup1 cleans the microphone. Sup2 also fades out everything that is not speech and cleans the output; it adds a little delay.",
                &mut s.noise,
                &[
                    (Noise::Off, "Off"),
                    (Noise::Sup1, "Sup1"),
                    (Noise::Sup2, "Sup2"),
                ],
            );
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                m3::switch(ui, &mut s.echo, "Echo cancellation");
                ui.label("Echo cancellation");
            });
            ui.label(m3::supporting(
                "Removes your converted voice from the microphone when you listen on speakers. Leave it off with headphones.",
            ));
        });
    }
    fn buffer_settings(&mut self, ui: &mut egui::Ui) {
        let running = self.running();
        m3::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing = Vec2::splat(8.0);
            m3::card_title(ui, icons::SPEED, "Buffering");
            ui.add_enabled_ui(!running, |ui| {
                let s = &mut self.saved.settings;
                m3::setting(
                    ui,
                    "Chunk", "How much audio is converted at once, and most of the delay. Lower it while Load at the bottom stays under about 70%; raise it if the voice stutters. 160-250 ms suits a good GPU, 320 ms or more the CPU.",
                    format!("{} ms", s.block_ms),
                    &mut s.block_ms,
                    20..=500,
                    1.0,
                );
                s.overlap_ms = s.overlap_ms.min(s.block_ms);
                m3::setting(
                    ui,
                    "Extra", "Past audio the model hears for context, so chunk edges sound continuous. It costs little: about 500 ms is a good default; lower it only if Load is too high.",
                    format!("{} ms", s.context_ms),
                    &mut s.context_ms,
                    0..=2000,
                    1.0,
                );
                let max_overlap = s.block_ms.min(100);
                m3::setting(
                    ui,
                    "Crossfade", "Blends each chunk into the next to hide the seams. Raise it to 60-80 ms if you hear a flutter every chunk.",
                    format!("{} ms", s.overlap_ms),
                    &mut s.overlap_ms,
                    1..=max_overlap,
                    1.0,
                );
                m3::setting(
                    ui,
                    "SOLA search", "How far to look for the best place to join chunks. Leave it unless you hear clicks.",
                    format!("{} ms", s.search_ms),
                    &mut s.search_ms,
                    0..=40,
                    1.0,
                );
            });
            ui.label(m3::supporting("Stop audio to change buffer durations."));
        });
        ui.add_space(4.0);
        m3::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            m3::card_title(ui, icons::INFO, "Tips");
            for (glyph, title, text) in [
                (
                    icons::VOICE,
                    "More like the character",
                    "Use the voice's index (Index 50-75%) and keep Tune in a natural range for the voice. Speak in your normal voice; the model copies your expression.",
                ),
                (
                    icons::GRAPHIC_EQ,
                    "Smoother",
                    "If the voice stutters, raise Chunk until Load stays under 70%. If you hear a flutter at every chunk, raise Crossfade.",
                ),
                (
                    icons::SPEED,
                    "Less delay",
                    "Lower Chunk while Load stays low, and use a GPU. On Windows, set VB-Cable's Max Latency to 2048 in Settings.",
                ),
                (
                    icons::MEMORY,
                    "Less GPU",
                    "Keep the noise gate around -50 dB: nothing is converted while you are silent. Noise suppression Sup2 costs a little more.",
                ),
            ] {
                ui.horizontal_top(|ui| {
                    ui.label(m3::icon(glyph, 20.0, m3::PRIMARY));
                    ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing.y = 2.0;
                        ui.label(m3::title(title));
                        ui.add(egui::Label::new(m3::supporting(text)).wrap());
                    });
                });
                ui.add_space(4.0);
            }
        });
    }
    fn settings(&mut self, ui: &mut egui::Ui) {
        header(ui, "Settings", "Audio devices and runtime.");
        #[cfg(feature = "audio")]
        {
            m3::card().show(ui, |ui| {
                ui.set_width(ui.available_width());
                m3::card_title(ui, icons::ROUTING, "Audio routing");
                // Devices can change while running; the session moves to them.
                ui.scope(|ui| {
                    device_picker(ui, "Microphone", true, &self.devices, &mut self.saved.input);
                    #[cfg(target_os = "linux")]
                    {
                        m3::field_label(ui, "Virtual microphone");
                        ui.horizontal(|ui| {
                            if m3::switch(ui, &mut self.saved.virtual_mic, "Virtual microphone")
                                .changed()
                            {
                                self.sync_virtual_mic();
                            }
                            ui.label(if self.virtual_mic.is_some() {
                                "On: Voice Changer Mic is ready"
                            } else {
                                "Off"
                            });
                        });
                        ui.label(m3::supporting(
                            "Adds a \"Voice Changer Mic\" input. Pick it as the microphone in Discord, OBS or a game to send them your converted voice. No extra software to install. It stays while you are logged in, even with the app closed, so those apps keep it selected; it carries silence while the voice changer is off, never your own voice. Turning this off removes it.",
                        ));
                    }
                    #[cfg(windows)]
                    self.windows_mic_settings(ui);
                    let virtual_mic =
                        cfg!(any(target_os = "linux", windows)) && self.saved.virtual_mic;
                    ui.add_enabled_ui(!virtual_mic, |ui| {
                        device_picker(
                            ui,
                            "Output / virtual cable",
                            false,
                            &self.devices,
                            &mut self.saved.output,
                        );
                    });
                    if virtual_mic {
                        ui.label(m3::supporting(
                            "The voice goes to the virtual microphone. Turn on Monitor below to hear it too.",
                        ));
                    }
                    m3::field_label(ui, "Monitor");
                    ui.horizontal(|ui| {
                        m3::switch(ui, &mut self.saved.monitor, "Monitor");
                        ui.label(if self.saved.monitor { "On" } else { "Off" });
                    });
                    ui.label(m3::supporting(
                        "Plays the converted voice on a second device, usually your headphones, so you hear how you sound while the output goes to Discord or a virtual cable. Use headphones: speakers feed the voice back into the microphone.",
                    ));
                    if self.saved.monitor {
                        device_picker(
                            ui,
                            "Monitor device",
                            false,
                            &self.devices,
                            &mut self.saved.monitor_device,
                        );
                    }
                    ui.add_space(8.0);
                    if ui
                        .add(m3::Button::outlined("Refresh devices").icon(icons::REFRESH))
                        .clicked()
                    {
                        self.refresh_devices();
                    }
                });
                ui.label(m3::supporting("Choose headphones for monitoring, or an installed virtual device to send audio to another app. Devices can be changed while the voice changer runs; audio pauses for a moment while it moves."));
            });
            ui.add_space(4.0);
        }
        let running = self.running();
        m3::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            m3::card_title(ui, icons::MEMORY, "Processing");
            let available = crate::runtime::available_devices();
            let current = self.saved.settings.device;
            // A saved choice stays visible even when this runtime cannot use it.
            let options: Vec<(Device, &str)> = available
                .iter()
                .copied()
                .chain((!available.contains(&current)).then_some(current))
                .map(|device| (device, device.label()))
                .collect();
            ui.add_enabled_ui(!running && options.len() > 1, |ui| {
                choice(
                    ui,
                    "Device",
                    "Where the model runs. A GPU is much faster than the CPU. CUDA is the fastest on NVIDIA; DirectML works on any DirectX 12 GPU (NVIDIA, AMD, Intel) without installing anything.",
                    &mut self.saved.settings.device,
                    &options,
                );
            });
            if let Some(name) = crate::gpu_info::name_for(current) {
                ui.horizontal(|ui| {
                    ui.label(m3::icon(icons::MEMORY, 18.0, m3::PRIMARY));
                    ui.label(egui::RichText::new(format!("Runs on {name}")).color(m3::ON_SURFACE));
                });
            }
            ui.label(m3::supporting(if !available.contains(&current) {
                "The selected device is not available with this ONNX Runtime; choose another one."
            } else if available.len() > 1 {
                "The choice applies the next time you press Start."
            } else {
                "Only the CPU is available with this ONNX Runtime. On Windows, the standard download includes DirectML for any GPU."
            }));
            #[cfg(feature = "catalog")]
            self.setup_status(ui);
        });
        ui.add_space(4.0);
        m3::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            m3::card_title(ui, icons::INFO, "About");
            m3::list_item(
                ui,
                icons::DEVICES,
                "Native Rust application",
                "egui / CPAL / ONNX Runtime. No Python process or network server is used.",
            );
            m3::list_item(
                ui,
                icons::MEMORY,
                "Inference",
                "CPU or CUDA GPU, float32 ONNX exports",
            );
            m3::list_item(
                ui,
                icons::SETTINGS,
                "Runtime",
                "ONNX Runtime 1.22 shared library (ORT_DYLIB_PATH)",
            );
        });
    }
    fn bottom_bar(&mut self, ui: &mut egui::Ui, compact: bool) {
        let index = self.saved.selected;
        #[allow(unused_mut)]
        let mut status = if self.saved.settings.bypass {
            "Ready · voice changer off (bypass)".to_owned()
        } else {
            "Ready · model loads on Start".to_owned()
        };
        #[cfg(feature = "audio")]
        if let Some(session) = &self.session {
            status = if session.metrics.voice_loading.load(Ordering::Acquire) {
                "Live · loading this voice, the last one plays meanwhile…".to_owned()
            } else if session.metrics.ready.load(Ordering::Acquire) {
                let device = self.saved.settings.device;
                match crate::gpu_info::name_for(device) {
                    Some(name) => format!("Live · {name}"),
                    None => format!("Live · {}", device.label()),
                }
            } else {
                "Loading model…".to_owned()
            };
        }
        ui.allocate_ui_with_layout(
            Vec2::new(ui.available_width(), 56.0),
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| {
                #[cfg(feature = "audio")]
                if self.running() {
                    let session = self.session.as_ref().unwrap();
                    let stopping = session.is_stopping();
                    let label = match (compact, stopping) {
                        (true, _) => "",
                        (false, true) => "Stopping…",
                        (false, false) => "Stop",
                    };
                    if ui
                        .add_enabled(!stopping, m3::Button::fab_stop(label).icon(icons::STOP))
                        .clicked()
                    {
                        log::info!("Stop pressed");
                        session.stop();
                    }
                    if !session.metrics.ready.load(Ordering::Acquire) {
                        ui.add(egui::Spinner::new().color(m3::PRIMARY));
                    }
                    let recording = session.is_recording();
                    if m3::icon_button(
                        ui,
                        icons::RECORD,
                        if recording {
                            m3::ERROR
                        } else {
                            m3::ON_SURFACE_VARIANT
                        },
                        if recording {
                            "Stop recording and save WAV + diagnostic report"
                        } else {
                            "Record microphone and converted voice to WAV files with pitch/spectrogram diagnostics"
                        },
                    )
                    .clicked()
                    {
                        session.set_recording(!recording);
                    }
                } else if ui
                    .add_enabled(
                        self.work.is_none(),
                        m3::Button::fab(if compact { "" } else { "Start" }).icon(icons::PLAY),
                    )
                    .clicked()
                {
                    self.start();
                }
                ui.add_space(4.0);
                if m3::icon_button(
                    ui,
                    icons::AUDIO_FILE,
                    m3::ON_SURFACE_VARIANT,
                    "Convert a WAV file",
                )
                .clicked()
                {
                    self.file_dialog = true;
                }
                if m3::icon_button(ui, icons::ROUTING, m3::ON_SURFACE_VARIANT, "Audio routing")
                    .clicked()
                {
                    self.page = Page::Settings;
                }
                ui.add_space(4.0);
                let mut apply = !self.saved.settings.bypass;
                m3::switch(ui, &mut apply, "Apply voice changer")
                    .on_hover_text("Apply voice changer");
                self.saved.settings.bypass = !apply;
                if !compact {
                    ui.label(RichText::new("Voice changer").color(m3::ON_SURFACE_VARIANT));
                    ui.add_space(8.0);
                }
                // Whatever is left on the leading side shows the selected voice.
                let (rect, _) = ui.allocate_exact_size(
                    Vec2::new(ui.available_width(), 56.0),
                    egui::Sense::hover(),
                );
                let image = self.portrait(ui.ctx(), index);
                let slot = &self.saved.slots[index];
                let circle = egui::Rect::from_center_size(
                    egui::pos2(rect.left() + 20.0, rect.center().y),
                    Vec2::splat(40.0),
                );
                paint_portrait(
                    ui,
                    circle,
                    20.0,
                    image.as_ref(),
                    index,
                    &slot.name,
                    !slot.manifest.trim().is_empty(),
                );
                let width = rect.width() - 60.0;
                if width > 24.0 {
                    let x = rect.left() + 52.0;
                    let name = m3::elided(
                        ui,
                        &slot.name,
                        FontId::new(16.0, m3::medium()),
                        m3::ON_SURFACE,
                        width,
                    );
                    let detail = m3::elided(
                        ui,
                        &status,
                        FontId::proportional(12.0),
                        m3::ON_SURFACE_VARIANT,
                        width,
                    );
                    ui.painter()
                        .galley(egui::pos2(x, rect.top() + 9.0), name, m3::ON_SURFACE);
                    ui.painter().galley(
                        egui::pos2(x, rect.top() + 32.0),
                        detail,
                        m3::ON_SURFACE_VARIANT,
                    );
                }
            },
        );
        #[cfg(feature = "audio")]
        self.live_stats(ui);
    }
    /// Live meters and chips under the bottom bar while audio runs: levels, how much
    /// delay the voice has, how hard the processor works, and any glitches.
    #[cfg(feature = "audio")]
    fn live_stats(&self, ui: &mut egui::Ui) {
        let Some(session) = &self.session else {
            return;
        };
        let m = &session.metrics;
        let load = |v: &std::sync::atomic::AtomicU32| f32::from_bits(v.load(Ordering::Relaxed));
        let settings = &self.saved.settings;
        let gate_db = 20.0 * settings.silence_threshold.max(1e-6).log10();
        let converting = !settings.bypass && m.ready.load(Ordering::Acquire);
        let input_db = m3::level_db(load(&m.input_level));
        let silent = converting && gate_db > m3::METER_FLOOR_DB && input_db < gate_db;
        let block = settings.block_ms as f32;
        let processing = load(&m.processing_ms);
        let rate = m.output_rate.load(Ordering::Relaxed);
        let queued = if rate == 0 {
            0.0
        } else {
            m.queued_samples.load(Ordering::Relaxed) as f32 * 1000.0 / rate as f32
        };
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = Vec2::new(8.0, 6.0);
            m3::meter(
                ui,
                icons::MIC,
                load(&m.input_level),
                (!settings.bypass).then_some(gate_db),
                "Microphone level. The pink tick is the noise gate (Studio > Noise gate): below it nothing is converted and the GPU rests.",
            );
            m3::meter(
                ui,
                icons::HEADPHONES,
                load(&m.output_level),
                None,
                "Converted voice level. Red near 0 dB means it is clipping; lower the output gain.",
            );
            if !converting {
                return;
            }
            if silent {
                m3::stat(
                    ui,
                    icons::INFO,
                    "Silent",
                    "GPU idle",
                    m3::OUTLINE,
                    "You are quieter than the noise gate, so nothing is converted. Speak, or lower the gate in Studio if your voice is cut.",
                );
            }
            // A sound waits for its chunk to fill, is converted, then plays after the
            // audio already queued. The queue is measured right after a chunk is added,
            // so it holds that chunk plus the safety lead: the delay is converting time
            // plus the queue.
            let delay = processing + queued;
            let lead = (queued - block).max(0.0);
            let (tone, advice) = if delay < 250.0 {
                (m3::SUCCESS, "Low delay.")
            } else if delay < 450.0 {
                (m3::TERTIARY, "Noticeable delay. Lower Chunk in Studio if the load stays under 70%.")
            } else {
                (m3::ERROR, "High delay. Lower Chunk in Studio; use a GPU if the load is high.")
            };
            m3::stat(
                ui,
                icons::SPEED,
                "Delay",
                &format!("~{delay:.0} ms"),
                tone,
                &format!(
                    "{advice}\nChunk {block:.0} ms + converting {processing:.0} ms + safety buffer {lead:.0} ms. A virtual cable such as VB-Cable adds its own buffer on top."
                ),
            );
            let usage = if block > 0.0 { processing / block } else { 0.0 };
            let (tone, advice) = if usage < 0.6 {
                (m3::SUCCESS, "Plenty of headroom: you can lower Chunk for less delay.")
            } else if usage < 0.9 {
                (m3::TERTIARY, "Busy but keeping up.")
            } else {
                (m3::ERROR, "Cannot keep up, so the voice stutters: raise Chunk, lower Extra, or use a GPU.")
            };
            m3::stat(
                ui,
                icons::MEMORY,
                "Load",
                &format!("{:.0}%", usage * 100.0),
                tone,
                &format!(
                    "{advice}\nConverting one {block:.0} ms chunk takes {processing:.0} ms on {}.",
                    crate::gpu_info::name_for(settings.device)
                        .unwrap_or_else(|| settings.device.label())
                ),
            );
            let glitches = m.dropped_input.load(Ordering::Relaxed)
                + m.dropped_output.load(Ordering::Relaxed)
                + m.underrun_samples.load(Ordering::Relaxed);
            if glitches > 0 {
                m3::stat(
                    ui,
                    icons::ERROR,
                    "Glitches",
                    &format!(
                        "{} / {} / {}",
                        m.dropped_input.load(Ordering::Relaxed),
                        m.dropped_output.load(Ordering::Relaxed),
                        m.underrun_samples.load(Ordering::Relaxed)
                    ),
                    m3::TERTIARY,
                    "Samples dropped from the microphone / dropped from the output / missing when the output ran dry. Occasional ones are harmless; if they keep rising, raise Chunk.",
                );
            }
        });
    }
    fn model_dialog(&mut self, ctx: &egui::Context) {
        let running = self.running();
        let removable = self.saved.slots.len() > 1;
        let (mut remove, mut done) = (false, false);
        #[allow(unused_mut)]
        let mut picture: Option<PathBuf> = None;
        // Some(None) removes the index, Some(Some(path)) replaces it.
        #[allow(unused_mut)]
        let mut index_change: Option<Option<PathBuf>> = None;
        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        let current_index = crate::config::ModelManifest::load(std::path::Path::new(
            &self.saved.slots[self.saved.selected].manifest,
        ))
        .ok()
        .and_then(|model| model.feature_bank)
        .and_then(|bank| bank.file_name().map(|n| n.to_string_lossy().into_owned()));
        let (_, dismissed) = m3::dialog(ctx, "model-dialog", "Configure voice", |ui| {
            ui.add_enabled_ui(!running, |ui| {
                let slot = &mut self.saved.slots[self.saved.selected];
                m3::field_label(ui, "Display name");
                ui.add(m3::text_edit(&mut slot.name, "Voice name", f32::INFINITY));
                // Phones have no file picker for checkpoints; there the manifest path
                // is typed in.
                #[cfg(any(target_os = "android", target_os = "ios"))]
                {
                    m3::field_label(ui, "Model manifest (.json)");
                    ui.add(m3::text_edit(
                        &mut slot.manifest,
                        "/path/to/model.json",
                        f32::INFINITY,
                    ));
                }
                #[cfg(not(any(target_os = "android", target_os = "ios")))]
                if !slot.manifest.trim().is_empty() {
                    m3::field_label(ui, "Index (.index)");
                    ui.label(
                        egui::RichText::new(current_index.as_deref().unwrap_or("No index"))
                            .color(m3::ON_SURFACE),
                    );
                    ui.horizontal(|ui| {
                        if ui
                            .add(m3::Button::outlined("Change index").icon(icons::FOLDER_OPEN))
                            .clicked()
                        {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("RVC feature index", &["index"])
                                .pick_file()
                            {
                                index_change = Some(Some(path));
                            }
                        }
                        if current_index.is_some()
                            && ui.add(m3::Button::text("Remove index")).clicked()
                        {
                            index_change = Some(None);
                        }
                    });
                    ui.label(m3::supporting(
                        "Makes the voice sound closer to the original speaker; the Index slider sets how much. The file is copied into the voice's folder.",
                    ));
                }
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    m3::switch(ui, &mut slot.favorite, "Favorite");
                    ui.label("Favorite");
                });
                #[cfg(not(any(target_os = "android", target_os = "ios")))]
                {
                    m3::field_label(ui, "Picture");
                    if ui
                        .add_enabled(
                            !slot.manifest.trim().is_empty(),
                            m3::Button::outlined("Choose picture").icon(icons::FOLDER_OPEN),
                        )
                        .clicked()
                    {
                        picture = rfd::FileDialog::new()
                            .add_filter("Picture", &PORTRAIT_TYPES)
                            .pick_file();
                    }
                    ui.label(m3::supporting(
                        "Shown on the voice card. It is copied into the voice's folder as portrait.*",
                    ));
                }
                #[cfg(any(target_os = "android", target_os = "ios"))]
                ui.label(m3::supporting(
                    "The manifest connects RVC, ContentVec and RMVPE. See examples/model.json.",
                ));
            });
            m3::actions(ui, |ui| {
                done = ui.add(m3::Button::text("Done")).clicked();
                if removable
                    && ui
                        .add_enabled(
                            !running,
                            m3::Button::danger("Remove slot").icon(icons::DELETE),
                        )
                        .clicked()
                {
                    remove = true;
                }
            });
        });
        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        if let Some(picked) = picture {
            self.set_portrait(picked);
        }
        if let Some(index) = index_change {
            let manifest = PathBuf::from(&self.saved.slots[self.saved.selected].manifest);
            match crate::config::set_feature_bank(&manifest, index.as_deref()) {
                Ok(()) => {
                    self.message = Some(match index {
                        Some(_) => {
                            "Index updated; it applies the next time you press Start.".into()
                        }
                        None => "Index removed.".into(),
                    })
                }
                Err(error) => self.error = Some(format!("{error:#}")),
            }
        }
        if remove {
            self.saved.slots.remove(self.saved.selected);
            self.saved.selected = self.saved.selected.min(self.saved.slots.len() - 1);
        }
        self.edit_model = !(remove || done || dismissed);
    }
    fn file_conversion_dialog(&mut self, ctx: &egui::Context) {
        let busy = self.running() || self.work.is_some();
        let mut close = false;
        let (_, dismissed) = m3::dialog(ctx, "file-dialog", "Convert a WAV file", |ui| {
            m3::chip(
                ui,
                icons::VOICE,
                &self.saved.slots[self.saved.selected].name,
            );
            let browse = cfg!(not(any(target_os = "android", target_os = "ios")));
            let trailing = if browse { 48.0 } else { 0.0 };
            let mut convert = false;
            ui.add_enabled_ui(!busy, |ui| {
                m3::field_label(ui, "Input WAV");
                ui.horizontal(|ui| {
                    let width = ui.available_width() - trailing;
                    ui.add(m3::text_edit(&mut self.file_input, "input.wav", width));
                    #[cfg(not(any(target_os = "android", target_os = "ios")))]
                    if m3::icon_button(
                        ui,
                        icons::FOLDER_OPEN,
                        m3::ON_SURFACE_VARIANT,
                        "Choose input WAV…",
                    )
                    .clicked()
                    {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("WAV audio", &["wav"])
                            .pick_file()
                        {
                            self.file_input = path.to_string_lossy().into_owned();
                            let stem = path.file_stem().unwrap_or_default().to_string_lossy();
                            self.file_output = path
                                .with_file_name(format!("{stem}-converted.wav"))
                                .to_string_lossy()
                                .into_owned();
                        }
                    }
                });
                m3::field_label(ui, "Output WAV");
                ui.horizontal(|ui| {
                    let width = ui.available_width() - trailing;
                    ui.add(m3::text_edit(&mut self.file_output, "converted.wav", width));
                    #[cfg(not(any(target_os = "android", target_os = "ios")))]
                    if m3::icon_button(
                        ui,
                        icons::FOLDER_OPEN,
                        m3::ON_SURFACE_VARIANT,
                        "Choose output…",
                    )
                    .clicked()
                    {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("WAV audio", &["wav"])
                            .set_file_name("converted.wav")
                            .save_file()
                        {
                            self.file_output = path.to_string_lossy().into_owned();
                        }
                    }
                });
                ui.add_space(8.0);
                let s = &mut self.saved.settings;
                m3::setting(
                    ui,
                    "Tune",
                    "Makes the voice higher or lower. +12 is one octave up, -12 one octave down.",
                    format!("{:+.1} semitones", s.semitones),
                    &mut s.semitones,
                    -24.0..=24.0,
                    0.1,
                );
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    m3::switch(ui, &mut self.file_analyze, "Save diagnostic report");
                    ui.label("Save diagnostic report (.json + .png)");
                });
                ui.label(m3::supporting(
                    "Writes pitch, spectrogram, and artifact diagnostics beside the converted WAV.",
                ));
            });
            ui.label(m3::supporting(if busy {
                "Stop live audio, or wait for the current model/file task to finish."
            } else {
                "The selected voice is applied to the entire file. Microphone capture is stopped during file conversion."
            }));
            m3::actions(ui, |ui| {
                let ready = !busy
                    && !self.file_input.trim().is_empty()
                    && !self.file_output.trim().is_empty();
                convert = ui
                    .add_enabled(ready, m3::Button::filled("Convert").icon(icons::AUDIO_FILE))
                    .clicked();
                close = ui.add(m3::Button::text("Close")).clicked();
            });
            if convert {
                let manifest = PathBuf::from(&self.saved.slots[self.saved.selected].manifest);
                let input = PathBuf::from(self.file_input.trim());
                let output = PathBuf::from(self.file_output.trim());
                let with_report = self.file_analyze;
                let settings = Settings {
                    bypass: false,
                    ..self.saved.settings.clone()
                };
                self.begin_work(move || {
                    anyhow::ensure!(
                        input.is_file(),
                        "Input WAV does not exist: {}",
                        input.display()
                    );
                    let converter = crate::engine::load_converter_on(&manifest, settings.device)?;
                    let mut engine = crate::engine::Engine::new(Some(converter));
                    crate::wav::convert(&mut engine, &input, &output, &settings)?;
                    if with_report {
                        let stem = output.file_stem().unwrap_or_default().to_string_lossy();
                        let prefix = output.with_file_name(stem.as_ref());
                        let model_label = manifest.display().to_string();
                        let (report, json, png) = crate::analyze::analyze_wavs_with_context(
                            &input,
                            Some(&output),
                            &prefix,
                            Some(&model_label),
                            Some(&settings),
                        )?;
                        let plot = png
                            .map(|p| format!(" and {}", p.display()))
                            .unwrap_or_default();
                        let headline = crate::analyze::format_summary(&report)
                            .lines()
                            .take(3)
                            .collect::<Vec<_>>()
                            .join(" | ");
                        Ok(format!(
                            "Conversion complete: {} (report: {}{} — {headline})",
                            output.display(),
                            json.display(),
                            plot
                        ))
                    } else {
                        Ok(format!("Conversion complete: {}", output.display()))
                    }
                });
            }
        });
        self.file_dialog = !(close || dismissed);
    }
}
impl eframe::App for VoiceApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Whatever the window shows as an error or a message also goes to the terminal.
        if self.error != self.logged.0 {
            if let Some(error) = &self.error {
                log::error!("{error}");
            }
            self.logged.0 = self.error.clone();
        }
        if self.message != self.logged.1 {
            if let Some(message) = &self.message {
                log::info!("{message}");
            }
            self.logged.1 = self.message.clone();
        }
        let completed = self.work.as_ref().and_then(|work| match work.try_recv() {
            Ok(result) => Some(result),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => Some(Err("Model worker disconnected".into())),
        });
        if let Some(result) = completed {
            self.work = None;
            #[cfg(feature = "catalog")]
            {
                self.browse.download = None;
            }
            match result {
                Ok(message) => {
                    // An import leaves a new voice folder behind; select it.
                    if let Some(added) = adopt_models(&mut self.saved) {
                        if !self.running() {
                            self.saved.selected = added;
                            self.saved.settings.bypass = false;
                        }
                    }
                    self.message = Some(message);
                }
                Err(error) => self.error = Some(error),
            }
        }
        #[cfg(all(feature = "audio", target_os = "linux"))]
        if let Some(problem) = self
            .virtual_mic
            .as_ref()
            .and_then(|guard| guard.error.lock().unwrap().take())
        {
            self.error = Some(problem);
        }
        self.follow_voice();
        #[cfg(feature = "audio")]
        self.follow_devices();
        #[cfg(feature = "audio")]
        if let Some(session) = &self.session {
            if let Some(recorded) = session.metrics.recorded.lock().unwrap().take() {
                self.message = Some(recorded);
            }
            if let Some(problem) = session.metrics.voice_error.lock().unwrap().take() {
                self.error = Some(format!("Could not switch voice: {problem}"));
            }
            if !session.metrics.running.load(Ordering::Acquire) {
                self.error = session.metrics.error.lock().unwrap().clone();
                self.session = None;
            } else {
                session.update(self.saved.settings.clone());
            }
        }
        #[cfg(all(feature = "audio", any(target_os = "android", target_os = "ios")))]
        if !ctx.input(|i| i.focused) {
            if let Some(session) = &self.session {
                session.stop();
            }
        }
        // Dropping a checkpoint or index on the window opens the import dialog with it.
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        if !self.running() {
            for path in dropped {
                self.drop_file(path);
            }
        }
        let compact = ctx.screen_rect().width() < 650.0;
        let bar = |margin: Margin| {
            egui::Frame::new()
                .fill(m3::SURFACE_CONTAINER)
                .inner_margin(margin)
        };
        // Wide: navigation rail beside a rounded content sheet. Compact: navigation bar.
        if compact {
            egui::TopBottomPanel::bottom("navigation-bar")
                .frame(bar(Margin::symmetric(8, 8)))
                .show_separator_line(false)
                .show(ctx, |ui| {
                    self.navigation(ui, true);
                });
        } else {
            egui::SidePanel::left("navigation")
                .exact_width(88.0)
                .resizable(false)
                .frame(bar(Margin::symmetric(8, 12)))
                .show_separator_line(false)
                .show(ctx, |ui| {
                    self.navigation(ui, false);
                });
        }
        egui::TopBottomPanel::bottom("transport")
            .frame(bar(Margin::symmetric(16, 12)))
            .show_separator_line(false)
            .show(ctx, |ui| {
                self.bottom_bar(ui, compact);
            });
        egui::CentralPanel::default()
            .frame(bar(if compact {
                Margin::ZERO
            } else {
                Margin {
                    left: 0,
                    right: 12,
                    top: 12,
                    bottom: 0,
                }
            }))
            .show(ctx, |ui| {
                egui::Frame::new()
                    .fill(m3::SURFACE)
                    .corner_radius(if compact { 0 } else { 24 })
                    .inner_margin(if compact { 16 } else { 28 })
                    .show(ui, |ui| {
                        egui::ScrollArea::vertical()
                            .auto_shrink(false)
                            .show(ui, |ui| {
                                self.notices(ui);
                                match self.page {
                                    Page::Voices => self.gallery(ui),
                                    Page::Studio => self.voice_controls(ui),
                                    Page::Settings => self.settings(ui),
                                }
                            });
                    });
            });
        if self.edit_model {
            self.model_dialog(ctx);
        }
        if self.file_dialog {
            self.file_conversion_dialog(ctx);
        }
        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        if self.import_dialog {
            self.import_dialog(ctx);
        }
        #[cfg(feature = "catalog")]
        if self.browse.open {
            self.browse_dialog(ctx);
        }
        #[cfg(feature = "catalog")]
        {
            self.poll_setup(ctx);
            if self.setup.open {
                self.setup_dialog(ctx);
            }
        }
        if self.running() || self.work.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(33));
        }
    }
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, "voice-changer", &self.saved);
    }
}
impl VoiceApp {
    /// Progress, result and error strips above the page content.
    fn notices(&mut self, ui: &mut egui::Ui) {
        #[cfg(feature = "catalog")]
        self.setup_notice(ui);
        if self.work.is_some() {
            egui::Frame::new()
                .fill(m3::SURFACE_CONTAINER_HIGH)
                .corner_radius(12)
                .inner_margin(Margin::symmetric(16, 8))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        ui.add(egui::Spinner::new().color(m3::PRIMARY));
                        ui.label("Processing model/audio…");
                    });
                });
            ui.add_space(4.0);
        }
        if self.message.as_deref().is_some_and(|message| {
            m3::banner(
                ui,
                icons::CHECK_CIRCLE,
                m3::SECONDARY_CONTAINER,
                m3::ON_SECONDARY_CONTAINER,
                message,
            )
        }) {
            self.message = None;
        }
        if self.error.as_deref().is_some_and(|error| {
            m3::banner(
                ui,
                icons::ERROR,
                m3::ERROR_CONTAINER,
                m3::ON_ERROR_CONTAINER,
                error,
            )
        }) {
            self.error = None;
        }
        if self.message.is_some() || self.error.is_some() {
            ui.add_space(8.0);
        }
    }
}
/// Single-choice row of filter chips under a field label.
fn choice<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    label: &str,
    hint: &str,
    value: &mut T,
    options: &[(T, &str)],
) {
    m3::field_label(ui, label);
    ui.horizontal_wrapped(|ui| {
        for (option, name) in options {
            if m3::filter_chip(ui, name, *value == *option).clicked() {
                *value = *option;
            }
        }
    });
    ui.label(m3::supporting(hint));
}
const PORTRAIT_TYPES: [&str; 4] = ["png", "jpg", "jpeg", "webp"];
fn portrait_file(manifest: &std::path::Path) -> Option<PathBuf> {
    let folder = manifest.parent()?;
    PORTRAIT_TYPES
        .iter()
        .map(|extension| folder.join(format!("portrait.{extension}")))
        .find(|file| file.is_file())
}
/// Paints a voice's picture cropped to cover `rect`, or its monogram tile when it has
/// none. Returns the colour for anything drawn on top.
fn paint_portrait(
    ui: &egui::Ui,
    rect: egui::Rect,
    radius: f32,
    image: Option<&egui::TextureHandle>,
    index: usize,
    name: &str,
    configured: bool,
) -> egui::Color32 {
    let Some(texture) = image else {
        return m3::avatar(ui, rect, radius, index, name, configured);
    };
    let size = texture.size_vec2();
    let scale = (rect.width() / size.x).max(rect.height() / size.y);
    let shown = rect.size() / scale;
    // Centred sideways, biased to the top where a character's face usually is.
    let corner = egui::pos2((size.x - shown.x) / 2.0, (size.y - shown.y) * 0.12);
    let uv = egui::Rect::from_min_size(
        egui::pos2(corner.x / size.x, corner.y / size.y),
        Vec2::new(shown.x / size.x, shown.y / size.y),
    );
    egui::Image::from_texture(texture)
        .uv(uv)
        .corner_radius(radius)
        .paint_at(ui, rect);
    egui::Color32::WHITE
}
/// Gives every discovered voice that has no slot yet one, reusing empty slots first.
/// Returns the slot of the last voice added.
fn adopt_models(saved: &mut Saved) -> Option<usize> {
    let mut added = None;
    for (name, path) in crate::runtime::discover_models() {
        let manifest = path.to_string_lossy().into_owned();
        if saved.slots.iter().any(|s| s.manifest == manifest) {
            continue;
        }
        let slot = ModelSlot {
            name,
            manifest,
            favorite: false,
            tune: None,
        };
        if let Some(empty) = saved
            .slots
            .iter()
            .position(|s| s.manifest.trim().is_empty())
        {
            saved.slots[empty] = slot;
            added = Some(empty);
        } else {
            saved.slots.push(slot);
            added = Some(saved.slots.len() - 1);
        }
    }
    added
}
fn header(ui: &mut egui::Ui, title: &str, detail: &str) {
    ui.label(m3::headline(title));
    ui.label(m3::supporting(detail));
    ui.add_space(12.0);
}
/// REC files go to ~/Music/VoiceChanger, or the temporary folder where there is no home.
#[cfg(feature = "audio")]
fn recordings_dir() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|home| PathBuf::from(home).join("Music").join("VoiceChanger"))
        .unwrap_or_else(std::env::temp_dir)
}
#[cfg(feature = "audio")]
fn device_picker(
    ui: &mut egui::Ui,
    label: &str,
    input: bool,
    devices: &[AudioDevice],
    selected: &mut Option<String>,
) {
    let name = selected
        .as_ref()
        .map(|id| {
            devices
                .iter()
                .find(|d| &d.id == id && d.input == input)
                .map(|d| d.name.as_str())
                .unwrap_or("Unavailable device")
        })
        .unwrap_or("System default");
    m3::field_label(ui, label);
    egui::ComboBox::from_id_salt(label)
        .selected_text(name)
        .width(ui.available_width().min(440.0))
        .show_ui(ui, |ui| {
            ui.selectable_value(selected, None, "System default");
            for device in devices.iter().filter(|d| d.input == input) {
                ui.selectable_value(selected, Some(device.id.clone()), &device.name);
            }
        });
}
