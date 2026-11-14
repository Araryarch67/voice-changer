#[cfg(feature = "audio")]
use crate::audio::{self, AudioDevice, AudioOptions, AudioSession};
use crate::config::{Device, F0Detector, Noise, Settings};
use crate::i18n::{t, tf};
use crate::m3::{self, icons};
use eframe::egui::{self, FontId, Margin, RichText, Vec2};
use serde::{Deserialize, Serialize};
#[cfg(feature = "audio")]
use std::sync::atomic::Ordering;
use std::{path::PathBuf, sync::mpsc};

#[cfg(feature = "catalog")]
mod browse;
#[cfg(feature = "audio")]
mod clone;
#[cfg(all(feature = "catalog", any(feature = "onnx", feature = "onnx-static")))]
mod cuda;
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
    /// Interface language; the system's on first start.
    language: crate::i18n::Lang,
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
            language: crate::i18n::system(),
        }
    }
}
#[derive(Clone, Copy, PartialEq)]
enum Page {
    Voices,
    Studio,
    Convert,
    #[cfg(feature = "audio")]
    Clone,
    Settings,
}
/// What a voice card's menu does with its slot.
#[derive(Clone, Copy)]
enum SlotAction {
    Earlier,
    Later,
    Remove,
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
    file_input: String,
    import_dialog: bool,
    #[cfg(feature = "catalog")]
    browse: browse::Browse,
    /// Downloads the shared models on first start.
    #[cfg(feature = "catalog")]
    setup: setup::Setup,
    /// Downloads NVIDIA's CUDA libraries when GPU (CUDA) is missing them.
    #[cfg(all(feature = "catalog", any(feature = "onnx", feature = "onnx-static")))]
    cuda: cuda::CudaSetup,
    #[cfg(feature = "audio")]
    clone: clone::CloneState,
    import_checkpoint: String,
    import_index: String,
    file_output: String,
    file_analyze: bool,
    work: Option<mpsc::Receiver<Result<String, String>>>,
    message: Option<String>,
    /// The voice waiting for the user to confirm its removal.
    removing: Option<usize>,
    /// The snackbar: its text, whether it is an error, and when it goes away by
    /// itself (set when first shown; errors stay until closed).
    toast: Option<(String, bool, Option<f64>)>,
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
            file_input: String::new(),
            import_dialog: false,
            #[cfg(feature = "catalog")]
            browse: Default::default(),
            #[cfg(feature = "catalog")]
            setup: setup::Setup::on_start(),
            #[cfg(all(feature = "catalog", any(feature = "onnx", feature = "onnx-static")))]
            cuda: Default::default(),
            #[cfg(feature = "audio")]
            clone: Default::default(),
            import_checkpoint: String::new(),
            import_index: String::new(),
            file_output: String::new(),
            file_analyze: true,
            work: None,
            message: None,
            removing: None,
            toast: None,
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
        // VOICE_CHANGER_LANG=en|id|ja picks the language at start (also for screenshots).
        if let Some(lang) = std::env::var("VOICE_CHANGER_LANG")
            .ok()
            .filter(|v| !v.is_empty())
        {
            app.saved.language = crate::i18n::Lang::from_locale(&lang);
        }
        crate::i18n::set(app.saved.language);
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
        m3::field_label(ui, t("Virtual microphone"));
        if !installed {
            self.saved.virtual_mic = false;
            ui.horizontal(|ui| {
                if ui
                    .add(m3::Button::tonal(t("Get VB-Cable")))
                    .on_hover_text(windows_mic::VB_CABLE_PAGE)
                    .clicked()
                {
                    windows_mic::open_page(windows_mic::VB_CABLE_PAGE);
                }
                if ui
                    .add(m3::Button::text(t("Refresh devices")).icon(icons::REFRESH))
                    .clicked()
                {
                    self.refresh_devices();
                }
            });
            ui.label(m3::supporting(
                t("To use the converted voice in Discord or a game, install VB-Cable (free, from VB-Audio's site): unzip it, run VBCABLE_Setup_x64 as administrator, restart Windows, then press Refresh devices. The app then sends its voice there by itself. For OBS alone nothing is needed: add an Application Audio Capture source for this app."),
            ));
            return;
        }
        ui.horizontal(|ui| {
            m3::switch(ui, &mut self.saved.virtual_mic, t("Virtual microphone"));
            ui.label(if self.saved.virtual_mic {
                tf("On: the voice goes to {}", &[&windows_mic::SPEAKER])
            } else {
                t("Off").into()
            });
        });
        ui.label(m3::supporting(&tf(
            "Pick \"{}\" as the microphone in Discord, OBS or a game. It is silent while the voice changer is stopped.",
            &[&windows_mic::MICROPHONE],
        )));
        let buffer = windows_mic::vb_cable_buffer();
        ui.horizontal(|ui| {
            if let Some((samples, rate)) = buffer {
                let ms = samples as f32 * 1000.0 / rate as f32;
                ui.label(
                    egui::RichText::new(tf(
                        "VB-Cable buffer: {} samples, up to {} ms",
                        &[&samples, &format!("{ms:.0}")],
                    ))
                    .color(if samples > 2048 {
                        m3::TERTIARY
                    } else {
                        m3::ON_SURFACE
                    }),
                );
            }
            if ui
                .add(m3::Button::text(t("VB-Cable settings")))
                .on_hover_text(t(
                    "Opens VB-Cable's control panel; Windows asks for administrator permission",
                ))
                .clicked()
            {
                if let Err(error) = windows_mic::open_vb_cable_settings() {
                    self.error = Some(format!("{error:#}"));
                }
            }
        });
        ui.label(m3::supporting(
            t("VB-Cable holds the voice in its buffer before Discord hears it. Its default, 7168 samples, adds about 110 ms; in VB-Cable settings set Max Latency to 2048 (or 1024 if it does not crackle). Most of the remaining delay is Chunk in Studio: lower it as far as your computer keeps up."),
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
                self.error = Some(t("Grant microphone permission, then press Start again.").into());
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
        self.start_session();
    }
    /// A build without live audio still offers file conversion; Start says so.
    #[cfg(not(feature = "audio"))]
    fn start_session(&mut self) {
        self.error =
            Some(t("This build has no live audio; use Convert a WAV file instead.").into());
    }
    #[cfg(feature = "audio")]
    fn start_session(&mut self) {
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
            (Page::Voices, icons::GRAPHIC_EQ, t("Voices")),
            (Page::Studio, icons::TUNE, t("Studio")),
            (Page::Convert, icons::AUDIO_FILE, t("Convert")),
            #[cfg(feature = "audio")]
            (Page::Clone, icons::VOICE, t("Clone")),
            (Page::Settings, icons::SETTINGS, t("Settings")),
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
                self.error = Some(tf("Could not save the picture: {}", &[&error]));
                return;
            }
        }
        self.portraits.remove(&manifest);
    }
    /// Adds an empty slot at the end and selects it; there is no limit to how many.
    fn add_slot(&mut self) {
        self.saved.slots.push(ModelSlot {
            name: format!("Voice {:02}", self.saved.slots.len() + 1),
            manifest: String::new(),
            favorite: false,
            tune: None,
        });
        if !self.running() {
            self.saved.selected = self.saved.slots.len() - 1;
        }
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
        {
            self.add_slot();
            self.edit_model = true;
        }
    }
    /// Puts a checkpoint in the import dialog, with the index beside it if the index
    /// field is still empty.
    fn choose_checkpoint(&mut self, path: PathBuf) {
        // Importing needs the exporter; a build without it only loads model.json files.
        #[cfg(feature = "export")]
        if self.import_index.trim().is_empty() {
            if let Some(index) = path.parent().and_then(crate::import::find_index) {
                self.import_index = index.to_string_lossy().into_owned();
            }
        }
        self.import_checkpoint = path.to_string_lossy().into_owned();
    }
    /// A dropped `.pth` or `.index` opens the import dialog with it filled in; a
    /// `.wav` opens Convert with it as the input.
    fn drop_file(&mut self, path: PathBuf) {
        let extension = path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        match extension.as_str() {
            "pth" => self.choose_checkpoint(path),
            "index" => self.import_index = path.to_string_lossy().into_owned(),
            "wav" => {
                self.choose_input_wav(&path);
                self.error = None;
                self.page = Page::Convert;
                return;
            }
            _ => {
                self.error = Some(
                    t("Drop an RVC .pth checkpoint or its .index file, or a .wav file to convert.")
                        .into(),
                );
                return;
            }
        }
        self.error = None;
        self.import_dialog = true;
    }
    /// Converts the chosen checkpoint with the chosen index, or with none.
    fn import_pair(&mut self) {
        if self.work.is_some() {
            self.error = Some(t("Wait for the current task to finish, then import again.").into());
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
            Ok(tf(
                "Imported {}{}.",
                &[
                    &voice.name,
                    &if voice.feature_bank.is_some() {
                        t(" with its index")
                    } else {
                        t(" without an index")
                    },
                ],
            ))
        });
        #[cfg(not(feature = "export"))]
        {
            let _ = (checkpoint, index);
            self.error = Some(
                t("This build cannot convert checkpoints; import a model.json instead.").into(),
            );
        }
    }
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    fn import_dialog(&mut self, ctx: &egui::Context) {
        let busy = self.running() || self.work.is_some();
        let (mut close, mut import) = (false, false);
        let (_, dismissed) = m3::dialog(ctx, "import-dialog", t("Import RVC voice"), |ui| {
            ui.add_enabled_ui(!busy, |ui| {
                m3::field_label(ui, t("Model (.pth)"));
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
                        t("Choose model…"),
                    )
                    .clicked()
                    {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter(t("RVC checkpoint"), &["pth"])
                            .pick_file()
                        {
                            // Suggest the index beside it; the user can still change it.
                            self.choose_checkpoint(path);
                        }
                    }
                });
                ui.label(m3::supporting(
                    t("The trained voice. It is converted to ONNX, which takes a few seconds."),
                ));
                m3::field_label(ui, t("Index (.index, optional)"));
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
                        t("Choose index…"),
                    )
                    .clicked()
                    {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter(t("RVC feature index"), &["index"])
                            .pick_file()
                        {
                            self.import_index = path.to_string_lossy().into_owned();
                        }
                    }
                    if m3::icon_button(ui, icons::CLOSE, m3::ON_SURFACE_VARIANT, t("No index"))
                        .clicked()
                    {
                        self.import_index.clear();
                    }
                });
                ui.label(m3::supporting(
                    t("Makes the voice sound closer to the original speaker; the Index slider sets how much. Leave it empty if you have none."),
                ));
            });
            if busy {
                ui.label(m3::supporting(t(
                    "Stop live audio, or wait for the current model/file task to finish.",
                )));
            }
            m3::actions(ui, |ui| {
                let ready = !busy && !self.import_checkpoint.trim().is_empty();
                import = ui
                    .add_enabled(ready, m3::Button::filled(t("Import")).icon(icons::ADD))
                    .clicked();
                close = ui.add(m3::Button::text(t("Cancel"))).clicked();
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
                        t("Model worker panicked; check the runtime and model files").to_owned()
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
            Ok(tf("Model test passed. Processing {} ms per {} ms block (RTF {}); first block {} ms. {}",
                &[&format!("{:.1}", result.median_ms), &result.block_ms, &format!("{:.2}", result.realtime_factor), &format!("{:.0}", result.first_block_ms),
                &if result.realtime_factor <= 1.0 { t("Sustained processing fits the block duration on this machine.") } else { t("Processing exceeds the block duration; increase block size or reduce context.") }]))
        });
    }
    fn gallery(&mut self, ui: &mut egui::Ui) {
        header(
            ui,
            t("Voices"),
            t("Pick a voice, then press Start; you can switch voices while it runs. Each voice keeps its own Tune, Index and other voice settings. Import an RVC .pth (and its .index), drop them on the window, or find one online."),
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
            .add_enabled(ready, m3::Button::tonal(t("Test model")).icon(icons::SPEED))
            .clicked()
        {
            self.check_model();
        }
    }
    fn edit_button(&mut self, ui: &mut egui::Ui) {
        if ui
            .add_enabled(
                !self.running(),
                m3::Button::outlined(t("Edit")).icon(icons::EDIT),
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
                                .hint_text(t("Search voices"))
                                .frame(false)
                                .desired_width(190.0),
                        );
                    });
                });
            if ui
                .add_enabled(
                    !running,
                    m3::Button::filled(t("Import model")).icon(icons::ADD),
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
                (Filter::All, t("All voices")),
                (Filter::Favorites, t("Favorites")),
                (Filter::Ready, t("Ready")),
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
            // The slots never run out: one more is a click away.
            if self.filter == Filter::All && search.is_empty() && add_slot_card(ui, width) {
                self.add_slot();
            }
        });
        if visible.is_empty() {
            ui.add_space(32.0);
            ui.vertical_centered(|ui| {
                ui.label(m3::icon(icons::SEARCH, 40.0, m3::OUTLINE));
                ui.label(m3::title(t("No voices here")));
                ui.label(m3::supporting(t(
                    "Change the filter or search, or import a model.",
                )));
            });
        }
        ui.add_space(16.0);
        ui.label(m3::supporting(t(
            "Empty cards are model slots; pretrained character voices are not bundled. Right-click a card, or press and hold it, to move or remove it.",
        )));
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
                    Some(true) => t("RVC · ONNX · with index").to_owned(),
                    Some(false) => t("RVC · ONNX · no index").to_owned(),
                    None => t("RVC · ONNX · files missing").to_owned(),
                }
            } else {
                t("Empty slot. Import an RVC .pth to use it.").to_owned()
            };
            ui.add(egui::Label::new(m3::supporting(&detail)).truncate());
            ui.add_space(8.0);
            if configured {
                let s = &mut self.saved.settings;
                m3::setting(
                    ui,
                    t("Tune"), t("Makes the voice higher or lower. +12 is one octave up, -12 one octave down. Male to female voice: about +8 to +12; female to male: about -8 to -12. A natural range sounds most real."),
                    m3::Format::new(t(" semitones"))
                    .tip(t("Recommended: +12 for a female voice, -2.5 for a male voice."))
                    .decimals(1)
                    .signed()
                    .presets(&[-12.0, -8.0, -5.0, 0.0, 5.0, 8.0, 12.0]),
                    &mut s.semitones,
                    -24.0..=24.0,
                    0.1,
                );
                m3::setting(
                    ui,
                    t("Index"), t("How much of the model's accent and tone to copy from its .index file. 50-75% usually sounds most like the character; higher can add a metallic edge. No effect on voices without an index."),
                    m3::Format::new(" %").scale(100.0).presets(&[0.0, 0.25, 0.5, 0.75, 1.0]),
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
                    m3::Button::filled(t("Import model")).icon(icons::ADD),
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
                t("Empty slot")
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
                    t("Remove from favorites")
                } else {
                    t("Add to favorites")
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
        // Right-click, or press and hold on a touch screen: move or remove the slot.
        let last = self.saved.slots.len() - 1;
        let remove = if empty {
            t("Remove slot")
        } else {
            t("Remove voice")
        };
        let mut action = None;
        response.context_menu(|ui| {
            ui.set_width(220.0);
            ui.spacing_mut().item_spacing = Vec2::ZERO;
            for (label, enabled, chosen) in [
                (t("Move earlier"), index > 0, SlotAction::Earlier),
                (t("Move later"), index < last, SlotAction::Later),
                (remove, !running && (last > 0 || !empty), SlotAction::Remove),
            ] {
                if ui
                    .add_enabled_ui(enabled, |ui| m3::menu_item(ui, label, false))
                    .inner
                    .clicked()
                {
                    action = Some(chosen);
                    ui.close();
                }
            }
        });
        match action {
            Some(SlotAction::Earlier) => self.move_slot(index, index - 1),
            Some(SlotAction::Later) => self.move_slot(index, index + 1),
            Some(SlotAction::Remove) => self.ask_to_remove(index),
            None => {}
        }
    }
    /// Moves slot `from` to position `to`, keeping the same voice selected.
    fn move_slot(&mut self, from: usize, to: usize) {
        let selected = self.saved.selected;
        let slot = self.saved.slots.remove(from);
        self.saved.slots.insert(to, slot);
        self.saved.selected = index_after_move(selected, from, to);
        self.tuned_slot = index_after_move(self.tuned_slot, from, to);
    }
    /// An empty slot goes at once; a voice asks first, as its files may be deleted.
    fn ask_to_remove(&mut self, index: usize) {
        if self.saved.slots[index].manifest.trim().is_empty() {
            self.remove_slot(index);
        } else {
            self.removing = Some(index);
        }
    }
    /// Asks before removing a voice: its folder in the models folder is deleted from
    /// disk (it would come back on the next start otherwise); files anywhere else stay.
    fn removal_dialog(&mut self, ctx: &egui::Context) {
        let Some(index) = self.removing.filter(|&i| i < self.saved.slots.len()) else {
            self.removing = None;
            return;
        };
        let slot = &self.saved.slots[index];
        let manifest = PathBuf::from(slot.manifest.trim());
        let folder = crate::runtime::voice_folder(&manifest, &crate::runtime::model_directories());
        let detail = match &folder {
            Some(folder) => tf(
                "{} and its folder are deleted from disk: {}. This cannot be undone.",
                &[&slot.name, &plain_path(folder)],
            ),
            None => tf(
                "{} is removed from the gallery. Its files are outside the models folder and stay where they are: {}",
                &[&slot.name, &manifest.display()],
            ),
        };
        let (mut confirm, mut cancel) = (false, false);
        let (_, dismissed) = m3::dialog(ctx, "removal-dialog", t("Remove this voice?"), |ui| {
            ui.add(egui::Label::new(RichText::new(&detail).color(m3::ON_SURFACE_VARIANT)).wrap());
            m3::actions(ui, |ui| {
                let label = if folder.is_some() {
                    t("Delete")
                } else {
                    t("Remove")
                };
                confirm = ui
                    .add_enabled(
                        !self.running(),
                        m3::Button::danger(label).icon(icons::DELETE),
                    )
                    .clicked();
                cancel = ui.add(m3::Button::text(t("Cancel"))).clicked();
            });
        });
        if confirm {
            self.removing = None;
            self.edit_model = false;
            let name = self.saved.slots[index].name.clone();
            match crate::runtime::delete_voice(&manifest) {
                Ok(deleted) => {
                    let gone = self.saved.slots[index].manifest.trim().to_owned();
                    self.portraits.remove(&gone);
                    if deleted.is_some() {
                        // Other slots holding the same voice would point at nothing.
                        for (other, slot) in self.saved.slots.iter_mut().enumerate() {
                            if other != index && slot.manifest.trim() == gone {
                                *slot = empty_slot(other);
                            }
                        }
                    }
                    if self.saved.slots.len() < 2 {
                        // The only slot stays, emptied, so the gallery is never bare.
                        self.saved.slots[index] = empty_slot(index);
                        self.tuned_slot = usize::MAX;
                    } else {
                        self.remove_slot(index);
                    }
                    self.message = Some(match deleted {
                        Some(_) => tf("Deleted {}.", &[&name]),
                        None => tf("Removed {} from the gallery.", &[&name]),
                    });
                }
                Err(error) => {
                    self.error = Some(tf("Could not delete {}: {}", &[&name, &error]));
                }
            }
        } else if cancel || dismissed {
            self.removing = None;
        }
    }
    /// Removes slot `index` from the gallery; see `removal_dialog` for its files.
    fn remove_slot(&mut self, index: usize) {
        if self.saved.slots.len() < 2 {
            return;
        }
        self.saved.slots.remove(index);
        let selected = self.saved.selected;
        self.saved.selected = if selected > index {
            selected - 1
        } else {
            selected.min(self.saved.slots.len() - 1)
        };
        if self.tuned_slot == index {
            // The tuned voice is gone: the newly selected one brings back its own.
            self.tuned_slot = usize::MAX;
        } else if self.tuned_slot > index {
            self.tuned_slot -= 1;
        }
    }
    fn voice_controls(&mut self, ui: &mut egui::Ui) {
        header(
            ui,
            t("Voice Studio"),
            t("Shape the voice and tune latency."),
        );
        ui.horizontal_wrapped(|ui| {
            m3::chip(
                ui,
                icons::VOICE,
                &self.saved.slots[self.saved.selected].name,
            );
            if ui
                .add(m3::Button::outlined(t("Reset to defaults")).icon(icons::REFRESH))
                .on_hover_text(t(
                    "Put every option on this page back to its original value",
                ))
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
            t("Options reset to defaults. Buffering is kept until audio is stopped.").into()
        } else {
            t("Options reset to defaults.").into()
        });
    }
    fn voice_settings(&mut self, ui: &mut egui::Ui) {
        #[cfg(feature = "catalog")]
        let mut offer = None;
        m3::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing = Vec2::splat(8.0);
            m3::card_title(ui, icons::VOICE, t("Voice"));
            let s = &mut self.saved.settings;
            m3::setting(
                ui,
                t("Tune"), t("Makes the voice higher or lower. +12 is one octave up, -12 one octave down. Male to female voice: about +8 to +12; female to male: about -8 to -12. A natural range sounds most real."),
                m3::Format::new(t(" semitones"))
                    .tip(t("Recommended: +12 for a female voice, -2.5 for a male voice."))
                    .decimals(1)
                    .signed()
                    .presets(&[-12.0, -8.0, -5.0, 0.0, 5.0, 8.0, 12.0]),
                &mut s.semitones,
                -24.0..=24.0,
                0.1,
            );
            m3::setting(
                ui,
                t("Index"), t("How much of the model's accent and tone to copy from its .index file. 50-75% usually sounds most like the character; higher can add a metallic edge. No effect on voices without an index."),
                m3::Format::new(" %").scale(100.0).presets(&[0.0, 0.25, 0.5, 0.75, 1.0]),
                &mut s.index_ratio,
                0.0..=1.0,
                0.01,
            );
            m3::setting(
                ui,
                t("Protect"), t("Keeps breaths and s, t, k sounds natural when Index is on. Around 0.33 if they sound lispy or robotic; lower protects more; 0.50 turns it off."),
                m3::Format::new("").decimals(2).presets(&[0.1, 0.2, 0.33, 0.5]),
                &mut s.protect,
                0.0..=0.5,
                0.01,
            );
            m3::setting(
                ui,
                t("Pitch correction"),
                t("Gently guides wobbly or slightly off-key singing onto musical notes while keeping vibrato natural. Leave at 0 % for natural speech."),
                m3::Format::new(" %").scale(100.0).presets(&[0.0, 0.25, 0.5, 0.75, 1.0]),
                &mut s.pitch_correct,
                0.0..=1.0,
                0.01,
            );
            choice(
                ui,
                t("F0 detector"),
                t("How your pitch is measured. RMVPE is recommended: the best all-rounder and the most natural for speech and singing. FCPE is about as accurate and lighter. CREPE full and Harvest are accurate but heavy; CREPE tiny, DIO and YIN are light but can crack on high notes."),
                &mut s.f0_detector,
                &[
                    (F0Detector::Rmvpe, t("RMVPE (recommended)")),
                    (F0Detector::CrepeFull, "CREPE full"),
                    (F0Detector::CrepeTiny, "CREPE tiny"),
                    (F0Detector::Fcpe, "FCPE"),
                    (F0Detector::Harvest, "Harvest"),
                    (F0Detector::Dio, "DIO"),
                    (F0Detector::Yin, "YIN"),
                ],
            );
            // A CREPE detector needs its own model; offer it instead of failing on Start.
            #[cfg(feature = "catalog")]
            if !setup::detector_ready(s.f0_detector) {
                ui.horizontal_wrapped(|ui| {
                    ui.label(m3::icon(icons::ERROR, 20.0, m3::ERROR));
                    ui.label(t("This detector's model is not installed yet, so Start will fail."));
                    if ui.add(m3::Button::tonal(t("Download"))).clicked() {
                        offer = Some(s.f0_detector);
                    }
                });
            }
            ui.allocate_ui_with_layout(
                Vec2::new(ui.available_width(), 40.0),
                egui::Layout::right_to_left(egui::Align::Center),
                |ui| {
                    ui.add(egui::DragValue::new(&mut s.speaker_id).range(0..=10000));
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        ui.label(t("Speaker ID"));
                    });
                },
            );
            ui.label(m3::supporting(
                t("Only for models that contain several voices. Leave at 0 otherwise."),
            ));
        });
        #[cfg(feature = "catalog")]
        if let Some(detector) = offer {
            self.offer_detector(detector);
        }
    }
    fn level_settings(&mut self, ui: &mut egui::Ui) {
        m3::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing = Vec2::splat(8.0);
            m3::card_title(ui, icons::MIC, t("Levels"));
            let s = &mut self.saved.settings;
            m3::setting(
                ui,
                t("Input gain"), t("Microphone volume before conversion. Raise it if you sound too quiet."),
                m3::Format::new("×").decimals(2).presets(&[0.5, 0.75, 1.0, 1.5, 2.0]),
                &mut s.input_gain,
                0.0..=4.0,
                0.01,
            );
            m3::setting(
                ui,
                t("Output gain"), t("Volume of the converted voice."),
                m3::Format::new("×").decimals(2).presets(&[0.5, 0.75, 1.0, 1.5, 2.0]),
                &mut s.output_gain,
                0.0..=4.0,
                0.01,
            );
            m3::setting(
                ui,
                t("Vocal leveling"),
                t("Smoothly evens out quiet and loud singing phrases without boosting room noise in pauses."),
                m3::Format::new(" %").scale(100.0).presets(&[0.0, 0.25, 0.5, 0.75, 1.0]),
                &mut s.vocal_leveling,
                0.0..=1.0,
                0.01,
            );
            // The engine gates on block RMS; show it in dBFS, where -100 is effectively off.
            let mut gate = (20.0 * s.silence_threshold.max(0.00001).log10()).round();
            if m3::setting(
                ui,
                t("Noise gate (S.Thresh)"), t("Below this nothing is converted, so room noise stays out and the GPU rests while you are silent. -50 dB suits most microphones; lower it if quiet words are cut. The pink tick on the microphone meter shows it."),
                m3::Format::new(" dB")
                    .off_at_min()
                    .presets(&[-100.0, -60.0, -55.0, -50.0, -45.0, -40.0]),
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
                t("Noise suppression"), t("Removes background noise. Sup2 is recommended: it also fades out everything that is not speech and cleans the output, for a little more delay. Sup1 only cleans the microphone."),
                &mut s.noise,
                &[
                    (Noise::Off, t("Off")),
                    (Noise::Sup1, "Sup1"),
                    (Noise::Sup2, t("Sup2 (recommended)")),
                ],
            );
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                m3::switch(ui, &mut s.echo, t("Echo cancellation"));
                ui.label(t("Echo cancellation"));
            });
            ui.label(m3::supporting(
                t("Removes your converted voice from the microphone when you listen on speakers. Leave it off with headphones."),
            ));
        });
    }
    fn buffer_settings(&mut self, ui: &mut egui::Ui) {
        let running = self.running();
        m3::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing = Vec2::splat(8.0);
            m3::card_title(ui, icons::SPEED, t("Buffering"));
            ui.add_enabled_ui(!running, |ui| {
                let s = &mut self.saved.settings;
                m3::setting(
                    ui,
                    t("Chunk"), t("How much audio is converted at once, and most of the delay. Lower it while Load at the bottom stays under about 70%; raise it if the voice stutters. 160-250 ms suits a good GPU, 320 ms or more the CPU."),
                    m3::Format::new(" ms").presets(&CHUNK_PRESETS).preset_label(chunk_label),
                    &mut s.block_ms,
                    20..=crate::config::MAX_BLOCK_MS,
                    1.0,
                );
                s.overlap_ms = s.overlap_ms.min(s.block_ms);
                m3::setting(
                    ui,
                    t("Extra"), t("Past audio the model hears for context, so chunk edges sound continuous. It costs little: about 500 ms is a good default; lower it only if Load is too high."),
                    m3::Format::new(" ms").presets(&EXTRA_PRESETS).preset_label(extra_label),
                    &mut s.context_ms,
                    0..=3000,
                    1.0,
                );
                let max_overlap = s.block_ms.min(100);
                m3::setting(
                    ui,
                    t("Crossfade"), t("Blends each chunk into the next to hide the seams. Raise it to 60-80 ms if you hear a flutter every chunk."),
                    m3::Format::new(" ms").presets(&[10.0, 20.0, 40.0, 60.0, 80.0, 100.0]),
                    &mut s.overlap_ms,
                    1..=max_overlap,
                    1.0,
                );
                m3::setting(
                    ui,
                    t("SOLA search"), t("How far to look for the best place to join chunks. Leave it unless you hear clicks."),
                    m3::Format::new(" ms").presets(&[0.0, 6.0, 12.0, 20.0]),
                    &mut s.search_ms,
                    0..=40,
                    1.0,
                );
            });
            ui.label(m3::supporting(t("Stop audio to change buffer durations.")));
        });
        ui.add_space(4.0);
        m3::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            m3::card_title(ui, icons::INFO, t("Tips"));
            for (glyph, title, text) in [
                (
                    icons::VOICE,
                    t("More like the character"),
                    t("Use the voice's index (Index 50-75%) and keep Tune in a natural range for the voice. Speak in your normal voice; the model copies your expression."),
                ),
                (
                    icons::GRAPHIC_EQ,
                    t("Smoother"),
                    t("If the voice stutters, raise Chunk until Load stays under 70%. If you hear a flutter at every chunk, raise Crossfade."),
                ),
                (
                    icons::SPEED,
                    t("Less delay"),
                    t("Lower Chunk while Load stays low, and use a GPU. On Windows, set VB-Cable's Max Latency to 2048 in Settings."),
                ),
                (
                    icons::MEMORY,
                    t("Less GPU"),
                    t("Keep the noise gate around -50 dB: nothing is converted while you are silent. Noise suppression Sup2 costs a little more."),
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
        header(ui, t("Settings"), t("Audio devices and runtime."));
        m3::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            m3::card_title(ui, icons::INFO, t("Language"));
            let options: Vec<(crate::i18n::Lang, &str)> = crate::i18n::Lang::ALL
                .iter()
                .map(|lang| (*lang, lang.name()))
                .collect();
            choice(
                ui,
                t("Interface language"),
                t("The language of the app's text. Error details and logs stay in English."),
                &mut self.saved.language,
                &options,
            );
        });
        ui.add_space(4.0);
        #[cfg(feature = "audio")]
        {
            m3::card().show(ui, |ui| {
                ui.set_width(ui.available_width());
                m3::card_title(ui, icons::ROUTING, t("Audio routing"));
                // Devices can change while running; the session moves to them.
                ui.scope(|ui| {
                    device_picker(ui, t("Microphone"), true, &self.devices, &mut self.saved.input);
                    #[cfg(target_os = "linux")]
                    {
                        m3::field_label(ui, t("Virtual microphone"));
                        ui.horizontal(|ui| {
                            if m3::switch(ui, &mut self.saved.virtual_mic, t("Virtual microphone"))
                                .changed()
                            {
                                self.sync_virtual_mic();
                            }
                            ui.label(if self.virtual_mic.is_some() {
                                t("On: Voice Changer Mic is ready")
                            } else {
                                t("Off")
                            });
                        });
                        ui.label(m3::supporting(
                            t("Adds a \"Voice Changer Mic\" input. Pick it as the microphone in Discord, OBS or a game to send them your converted voice. No extra software to install. It stays while you are logged in, even with the app closed, so those apps keep it selected; it carries silence while the voice changer is off, never your own voice. Turning this off removes it."),
                        ));
                    }
                    #[cfg(windows)]
                    self.windows_mic_settings(ui);
                    let virtual_mic =
                        cfg!(any(target_os = "linux", windows)) && self.saved.virtual_mic;
                    ui.add_enabled_ui(!virtual_mic, |ui| {
                        device_picker(
                            ui,
                            t("Output / virtual cable"),
                            false,
                            &self.devices,
                            &mut self.saved.output,
                        );
                    });
                    if virtual_mic {
                        ui.label(m3::supporting(
                            t("The voice goes to the virtual microphone. Turn on Monitor below to hear it too."),
                        ));
                    }
                    m3::field_label(ui, t("Monitor"));
                    ui.horizontal(|ui| {
                        m3::switch(ui, &mut self.saved.monitor, t("Monitor"));
                        ui.label(if self.saved.monitor { t("On") } else { t("Off") });
                    });
                    ui.label(m3::supporting(
                        t("Plays the converted voice on a second device, usually your headphones, so you hear how you sound while the output goes to Discord or a virtual cable. Use headphones: speakers feed the voice back into the microphone."),
                    ));
                    if self.saved.monitor {
                        device_picker(
                            ui,
                            t("Monitor device"),
                            false,
                            &self.devices,
                            &mut self.saved.monitor_device,
                        );
                    }
                    ui.add_space(8.0);
                    if ui
                        .add(m3::Button::outlined(t("Refresh devices")).icon(icons::REFRESH))
                        .clicked()
                    {
                        self.refresh_devices();
                    }
                });
                ui.label(m3::supporting(t("Choose headphones for monitoring, or an installed virtual device to send audio to another app. Devices can be changed while the voice changer runs; audio pauses for a moment while it moves.")));
            });
            ui.add_space(4.0);
        }
        let running = self.running();
        m3::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            m3::card_title(ui, icons::MEMORY, t("Processing"));
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
                    t("Device"),
                    t("Where the model runs. A GPU is much faster than the CPU. CUDA is the fastest on NVIDIA; DirectML works on any DirectX 12 GPU (NVIDIA, AMD, Intel) without installing anything."),
                    &mut self.saved.settings.device,
                    &options,
                );
            });
            if let Some(name) = crate::gpu_info::name_for(current) {
                ui.horizontal(|ui| {
                    ui.label(m3::icon(icons::MEMORY, 18.0, m3::PRIMARY));
                    ui.label(egui::RichText::new(tf("Runs on {}", &[&name])).color(m3::ON_SURFACE));
                });
            }
            ui.label(m3::supporting(if !available.contains(&current) {
                t("The selected device is not available with this ONNX Runtime; choose another one.")
            } else if available.len() > 1 {
                t("The choice applies the next time you press Start.")
            } else {
                t("Only the CPU is available with this ONNX Runtime. On Windows, the standard download includes DirectML for any GPU.")
            }));
            #[cfg(all(feature = "catalog", any(feature = "onnx", feature = "onnx-static")))]
            self.cuda_status(ui);
            #[cfg(feature = "catalog")]
            self.setup_status(ui);
        });
        ui.add_space(4.0);
        m3::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            m3::card_title(ui, icons::INFO, t("About"));
            m3::list_item(
                ui,
                icons::DEVICES,
                t("Native Rust application"),
                t("egui / CPAL / ONNX Runtime. No Python process or network server is used."),
            );
            m3::list_item(
                ui,
                icons::MEMORY,
                t("Inference"),
                t("CPU or CUDA GPU, float32 ONNX exports"),
            );
            m3::list_item(
                ui,
                icons::SETTINGS,
                t("Runtime"),
                t("ONNX Runtime 1.22 shared library (ORT_DYLIB_PATH)"),
            );
        });
    }
    fn bottom_bar(&mut self, ui: &mut egui::Ui, compact: bool) {
        let index = self.saved.selected;
        #[allow(unused_mut)]
        let mut status = if self.saved.settings.bypass {
            t("Ready · voice changer off (bypass)").to_owned()
        } else {
            t("Ready · model loads on Start").to_owned()
        };
        #[cfg(feature = "audio")]
        if let Some(session) = &self.session {
            status = if session.metrics.voice_loading.load(Ordering::Acquire) {
                t("Live · loading this voice, the last one plays meanwhile…").to_owned()
            } else if session.metrics.ready.load(Ordering::Acquire) {
                let device = self.saved.settings.device;
                match crate::gpu_info::name_for(device) {
                    Some(name) => tf("Live · {}", &[&name]),
                    None => tf("Live · {}", &[&device.label()]),
                }
            } else {
                t("Loading model…").to_owned()
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
                        (false, true) => t("Stopping…"),
                        (false, false) => t("Stop"),
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
                            t("Stop recording and save WAV + diagnostic report")
                        } else {
                            t("Record microphone and converted voice to WAV files with pitch/spectrogram diagnostics")
                        },
                    )
                    .clicked()
                    {
                        session.set_recording(!recording);
                    }
                } else if ui
                    .add_enabled(
                        self.work.is_none(),
                        m3::Button::fab(if compact { "" } else { t("Start") }).icon(icons::PLAY),
                    )
                    .clicked()
                {
                    self.start();
                }
                // Without live audio Start stays, and says to convert a file instead.
                #[cfg(not(feature = "audio"))]
                if ui
                    .add_enabled(
                        self.work.is_none(),
                        m3::Button::fab(if compact { "" } else { t("Start") }).icon(icons::PLAY),
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
                    t("Convert a WAV file"),
                )
                .clicked()
                {
                    self.page = Page::Convert;
                }
                if m3::icon_button(ui, icons::ROUTING, m3::ON_SURFACE_VARIANT, t("Audio routing"))
                    .clicked()
                {
                    self.page = Page::Settings;
                }
                ui.add_space(4.0);
                let mut apply = !self.saved.settings.bypass;
                m3::switch(ui, &mut apply, t("Apply voice changer"))
                    .on_hover_text(t("Apply voice changer"));
                self.saved.settings.bypass = !apply;
                if !compact {
                    ui.label(RichText::new(t("Voice changer")).color(m3::ON_SURFACE_VARIANT));
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
                t("Microphone level. The pink tick is the noise gate (Studio > Noise gate): below it nothing is converted and the GPU rests."),
            );
            m3::meter(
                ui,
                icons::HEADPHONES,
                load(&m.output_level),
                None,
                t("Converted voice level. Red near 0 dB means it is clipping; lower the output gain."),
            );
            if !converting {
                return;
            }
            if silent {
                m3::stat(
                    ui,
                    icons::INFO,
                    t("Silent"),
                    t("GPU idle"),
                    m3::OUTLINE,
                    t("You are quieter than the noise gate, so nothing is converted. Speak, or lower the gate in Studio if your voice is cut."),
                );
            }
            // A sound waits for its chunk to fill, is converted, then plays after the
            // audio already queued. The queue is measured right after a chunk is added,
            // so it holds that chunk plus the safety lead: the delay is converting time
            // plus the queue.
            let delay = processing + queued;
            let lead = (queued - block).max(0.0);
            let (tone, advice) = if delay < 250.0 {
                (m3::SUCCESS, t("Low delay."))
            } else if delay < 450.0 {
                (m3::TERTIARY, t("Noticeable delay. Lower Chunk in Studio if the load stays under 70%."))
            } else {
                (m3::ERROR, t("High delay. Lower Chunk in Studio; use a GPU if the load is high."))
            };
            m3::stat(
                ui,
                icons::SPEED,
                t("Delay"),
                &format!("~{delay:.0} ms"),
                tone,
                &format!(
                    "{advice}\n{}",
                    tf(
                        "Chunk {} ms + converting {} ms + safety buffer {} ms. A virtual cable such as VB-Cable adds its own buffer on top.",
                        &[
                            &format!("{block:.0}"),
                            &format!("{processing:.0}"),
                            &format!("{lead:.0}"),
                        ],
                    )
                ),
            );
            let usage = if block > 0.0 { processing / block } else { 0.0 };
            let (tone, advice) = if usage < 0.6 {
                (m3::SUCCESS, t("Plenty of headroom: you can lower Chunk for less delay."))
            } else if usage < 0.9 {
                (m3::TERTIARY, t("Busy but keeping up."))
            } else {
                (m3::ERROR, t("Cannot keep up, so the voice stutters: raise Chunk, lower Extra, or use a GPU."))
            };
            m3::stat(
                ui,
                icons::MEMORY,
                t("Load"),
                &format!("{:.0}%", usage * 100.0),
                tone,
                &format!(
                    "{advice}\n{}",
                    tf(
                        "Converting one {} ms chunk takes {} ms on {}.",
                        &[
                            &format!("{block:.0}"),
                            &format!("{processing:.0}"),
                            &crate::gpu_info::name_for(settings.device)
                                .unwrap_or_else(|| settings.device.label()),
                        ],
                    )
                ),
            );
            let glitches = m.dropped_input.load(Ordering::Relaxed)
                + m.dropped_output.load(Ordering::Relaxed)
                + m.underrun_samples.load(Ordering::Relaxed);
            if glitches > 0 {
                m3::stat(
                    ui,
                    icons::ERROR,
                    t("Glitches"),
                    &format!(
                        "{} / {} / {}",
                        m.dropped_input.load(Ordering::Relaxed),
                        m.dropped_output.load(Ordering::Relaxed),
                        m.underrun_samples.load(Ordering::Relaxed)
                    ),
                    m3::TERTIARY,
                    t("Samples dropped from the microphone / dropped from the output / missing when the output ran dry. Occasional ones are harmless; if they keep rising, raise Chunk."),
                );
            }
        });
    }
    fn model_dialog(&mut self, ctx: &egui::Context) {
        let running = self.running();
        let removable = self.saved.slots.len() > 1
            || !self.saved.slots[self.saved.selected]
                .manifest
                .trim()
                .is_empty();
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
        let (_, dismissed) = m3::dialog(ctx, "model-dialog", t("Configure voice"), |ui| {
            ui.add_enabled_ui(!running, |ui| {
                let slot = &mut self.saved.slots[self.saved.selected];
                m3::field_label(ui, t("Display name"));
                ui.add(m3::text_edit(&mut slot.name, t("Voice name"), f32::INFINITY));
                // Phones have no file picker for checkpoints; there the manifest path
                // is typed in.
                #[cfg(any(target_os = "android", target_os = "ios"))]
                {
                    m3::field_label(ui, t("Model manifest (.json)"));
                    ui.add(m3::text_edit(
                        &mut slot.manifest,
                        "/path/to/model.json",
                        f32::INFINITY,
                    ));
                }
                #[cfg(not(any(target_os = "android", target_os = "ios")))]
                if !slot.manifest.trim().is_empty() {
                    m3::field_label(ui, t("Index (.index)"));
                    ui.label(
                        egui::RichText::new(current_index.as_deref().unwrap_or(t("No index")))
                            .color(m3::ON_SURFACE),
                    );
                    ui.horizontal(|ui| {
                        if ui
                            .add(m3::Button::outlined(t("Change index")).icon(icons::FOLDER_OPEN))
                            .clicked()
                        {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter(t("RVC feature index"), &["index"])
                                .pick_file()
                            {
                                index_change = Some(Some(path));
                            }
                        }
                        if current_index.is_some()
                            && ui.add(m3::Button::text(t("Remove index"))).clicked()
                        {
                            index_change = Some(None);
                        }
                    });
                    ui.label(m3::supporting(
                        t("Makes the voice sound closer to the original speaker; the Index slider sets how much. The file is copied into the voice's folder."),
                    ));
                }
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    m3::switch(ui, &mut slot.favorite, t("Favorite"));
                    ui.label(t("Favorite"));
                });
                #[cfg(not(any(target_os = "android", target_os = "ios")))]
                {
                    m3::field_label(ui, t("Picture"));
                    if ui
                        .add_enabled(
                            !slot.manifest.trim().is_empty(),
                            m3::Button::outlined(t("Choose picture")).icon(icons::FOLDER_OPEN),
                        )
                        .clicked()
                    {
                        picture = rfd::FileDialog::new()
                            .add_filter(t("Picture"), &PORTRAIT_TYPES)
                            .pick_file();
                    }
                    ui.label(m3::supporting(
                        t("Shown on the voice card. It is copied into the voice's folder as portrait.*"),
                    ));
                }
                #[cfg(any(target_os = "android", target_os = "ios"))]
                ui.label(m3::supporting(
                    t("The manifest connects RVC, ContentVec and RMVPE. See examples/model.json."),
                ));
            });
            let remove_label = if self.saved.slots[self.saved.selected]
                .manifest
                .trim()
                .is_empty()
            {
                t("Remove slot")
            } else {
                t("Remove voice")
            };
            m3::actions(ui, |ui| {
                done = ui.add(m3::Button::text(t("Done"))).clicked();
                if removable
                    && ui
                        .add_enabled(
                            !running,
                            m3::Button::danger(remove_label).icon(icons::DELETE),
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
                            t("Index updated; it applies the next time you press Start.").into()
                        }
                        None => t("Index removed.").into(),
                    })
                }
                Err(error) => self.error = Some(format!("{error:#}")),
            }
        }
        if remove {
            self.ask_to_remove(self.saved.selected);
        }
        self.edit_model = !(remove || done || dismissed);
    }
    /// Uses `path` as the input and puts the output beside it.
    fn choose_input_wav(&mut self, path: &std::path::Path) {
        self.file_input = path.to_string_lossy().into_owned();
        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        self.file_output = path
            .with_file_name(format!("{stem}-converted.wav"))
            .to_string_lossy()
            .into_owned();
    }
    /// Converts a recorded WAV file with the selected voice.
    fn convert_page(&mut self, ui: &mut egui::Ui) {
        header(
            ui,
            t("Convert a WAV file"),
            t("Changes the voice in a recording with one of your voices and saves the result as a new WAV file. Drop a .wav on the window to start here."),
        );
        let busy = self.running() || self.work.is_some();
        m3::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing = Vec2::splat(8.0);
            m3::field_label(ui, t("Voice"));
            let ready: Vec<usize> = (0..self.saved.slots.len())
                .filter(|&i| !self.saved.slots[i].manifest.trim().is_empty())
                .collect();
            let mut picked = None;
            ui.add_enabled_ui(!busy && !ready.is_empty(), |ui| {
                let width = ui.available_width().min(440.0);
                let current = &self.saved.slots[self.saved.selected];
                let name = if current.manifest.trim().is_empty() {
                    t("Choose a voice")
                } else {
                    current.name.as_str()
                };
                m3::select(ui, name, width, |ui| {
                    for &index in &ready {
                        let slot = &self.saved.slots[index];
                        if m3::menu_item(ui, &slot.name, index == self.saved.selected).clicked() {
                            picked = Some(index);
                        }
                    }
                });
            });
            if let Some(index) = picked {
                self.saved.selected = index;
            }
            ui.label(m3::supporting(if ready.is_empty() {
                t("Import a voice first; the file is converted with it.")
            } else {
                t("The voice the whole file is converted with. It is also the voice selected for live use.")
            }));
            self.file_conversion(ui, busy);
        });
    }
    fn file_conversion(&mut self, ui: &mut egui::Ui, busy: bool) {
        {
            let browse = cfg!(not(any(target_os = "android", target_os = "ios")));
            let trailing = if browse { 48.0 } else { 0.0 };
            let mut convert = false;
            ui.add_enabled_ui(!busy, |ui| {
                m3::field_label(ui, t("Input WAV"));
                ui.horizontal(|ui| {
                    let width = ui.available_width() - trailing;
                    ui.add(m3::text_edit(&mut self.file_input, "input.wav", width));
                    #[cfg(not(any(target_os = "android", target_os = "ios")))]
                    if m3::icon_button(
                        ui,
                        icons::FOLDER_OPEN,
                        m3::ON_SURFACE_VARIANT,
                        t("Choose input WAV…"),
                    )
                    .clicked()
                    {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter(t("WAV audio"), &["wav"])
                            .pick_file()
                        {
                            self.choose_input_wav(&path);
                        }
                    }
                });
                m3::field_label(ui, t("Output WAV"));
                ui.horizontal(|ui| {
                    let width = ui.available_width() - trailing;
                    ui.add(m3::text_edit(&mut self.file_output, "converted.wav", width));
                    #[cfg(not(any(target_os = "android", target_os = "ios")))]
                    if m3::icon_button(
                        ui,
                        icons::FOLDER_OPEN,
                        m3::ON_SURFACE_VARIANT,
                        t("Choose output…"),
                    )
                    .clicked()
                    {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter(t("WAV audio"), &["wav"])
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
                    t("Tune"),
                    t("Makes the voice higher or lower. +12 is one octave up, -12 one octave down."),
                    m3::Format::new(t(" semitones"))
                    .tip(t("Recommended: +12 for a female voice, -2.5 for a male voice."))
                    .decimals(1)
                    .signed()
                    .presets(&[-12.0, -8.0, -5.0, 0.0, 5.0, 8.0, 12.0]),
                    &mut s.semitones,
                    -24.0..=24.0,
                    0.1,
                );
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    m3::switch(ui, &mut self.file_analyze, t("Save diagnostic report"));
                    ui.label(t("Save diagnostic report (.json + .png)"));
                });
                ui.label(m3::supporting(
                    t("Writes pitch, spectrogram, and artifact diagnostics beside the converted WAV."),
                ));
            });
            ui.label(m3::supporting(if busy {
                t("Stop live audio, or wait for the current model/file task to finish.")
            } else {
                t("The selected voice is applied to the entire file. Microphone capture is stopped during file conversion.")
            }));
            m3::actions(ui, |ui| {
                let ready = !busy
                    && !self.saved.slots[self.saved.selected]
                        .manifest
                        .trim()
                        .is_empty()
                    && !self.file_input.trim().is_empty()
                    && !self.file_output.trim().is_empty();
                convert = ui
                    .add_enabled(
                        ready,
                        m3::Button::filled(t("Convert")).icon(icons::AUDIO_FILE),
                    )
                    .clicked();
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
                            .map(|p| tf(" and {}", &[&p.display()]))
                            .unwrap_or_default();
                        let headline = crate::analyze::format_summary(&report)
                            .lines()
                            .take(3)
                            .collect::<Vec<_>>()
                            .join(" | ");
                        Ok(tf(
                            "Conversion complete: {} (report: {}{} — {})",
                            &[&output.display(), &json.display(), &plot, &headline],
                        ))
                    } else {
                        Ok(tf("Conversion complete: {}", &[&output.display()]))
                    }
                });
            }
        }
    }
}
impl eframe::App for VoiceApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        crate::i18n::set(self.saved.language);
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
            Err(mpsc::TryRecvError::Disconnected) => {
                Some(Err(t("Model worker disconnected").into()))
            }
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
                self.error = Some(tf("Could not switch voice: {}", &[&problem]));
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
                            .scroll_bar_visibility(
                                egui::scroll_area::ScrollBarVisibility::AlwaysHidden,
                            )
                            .auto_shrink(false)
                            .show(ui, |ui| {
                                self.notices(ui);
                                match self.page {
                                    Page::Voices => self.gallery(ui),
                                    Page::Studio => self.voice_controls(ui),
                                    Page::Convert => self.convert_page(ui),
                                    #[cfg(feature = "audio")]
                                    Page::Clone => self.clone_page(ui),
                                    Page::Settings => self.settings(ui),
                                }
                            });
                    });
            });
        if self.edit_model {
            self.model_dialog(ctx);
        }
        if self.removing.is_some() {
            self.removal_dialog(ctx);
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
        #[cfg(all(feature = "catalog", any(feature = "onnx", feature = "onnx-static")))]
        {
            self.poll_cuda_setup(ctx);
            if self.cuda.open {
                self.cuda_dialog(ctx);
            }
        }
        self.show_toast(ctx);
        if self.running() || self.work.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(33));
        }
    }
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, "voice-changer", &self.saved);
    }
}
impl VoiceApp {
    /// Shows `text` in a snackbar over whatever page or dialog is open, and writes it
    /// to the terminal. A message goes away by itself; an error stays until closed.
    #[cfg(feature = "catalog")]
    fn toast(&mut self, text: String, error: bool) {
        if error {
            log::error!("{text}");
        } else {
            log::info!("{text}");
        }
        self.toast = Some((text, error, None));
    }

    fn show_toast(&mut self, ctx: &egui::Context) {
        const SHOWN_FOR: f64 = 6.0;
        let Some((text, error, until)) = &mut self.toast else {
            return;
        };
        let now = ctx.input(|i| i.time);
        let closed = m3::snackbar(ctx, text, *error);
        let expired = !*error && now >= *until.get_or_insert(now + SHOWN_FOR);
        if closed || expired {
            self.toast = None;
        } else if let Some(until) = until.filter(|_| !*error) {
            ctx.request_repaint_after(std::time::Duration::from_secs_f64(
                (until - now).max(0.0) + 0.05,
            ));
        }
    }

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
                        ui.label(t("Processing model/audio…"));
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
        #[cfg(all(feature = "catalog", any(feature = "onnx", feature = "onnx-static")))]
        self.cuda_offer(ui);
        if self.message.is_some() || self.error.is_some() {
            ui.add_space(8.0);
        }
    }
}
/// Single-choice row of filter chips under a field label.
/// The reference client's CHUNK choices (units of 128 samples at 48 kHz) in ms, up
/// to `config::MAX_BLOCK_MS`.
const CHUNK_PRESETS: [f64; 25] = {
    const UNITS: [f64; 25] = [
        8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 64.0, 80.0, 96.0, 112.0, 128.0, 192.0, 256.0, 320.0,
        384.0, 448.0, 512.0, 576.0, 640.0, 704.0, 768.0, 832.0, 896.0, 960.0, 1024.0,
    ];
    let mut ms = [0.0; 25];
    let mut i = 0;
    while i < 25 {
        ms[i] = UNITS[i] * 128.0 * 1000.0 / 48000.0;
        i += 1;
    }
    ms
};
const _: () = assert!(CHUNK_PRESETS[24] <= crate::config::MAX_BLOCK_MS as f64);
/// "128 (341.3 ms, 16384)", as the reference client lists CHUNK.
fn chunk_label(ms: f64) -> String {
    let units = (ms * 48.0 / 128.0).round();
    format!("{units} ({ms:.1} ms, {})", units * 128.0)
}
/// The reference client's EXTRA choices (samples at 48 kHz) in ms.
const EXTRA_PRESETS: [f64; 6] = [
    4096.0 / 48.0,
    8192.0 / 48.0,
    16384.0 / 48.0,
    32768.0 / 48.0,
    65536.0 / 48.0,
    131072.0 / 48.0,
];
/// "16384 (341 ms)", the reference's sample count with its duration.
fn extra_label(ms: f64) -> String {
    format!("{} ({ms:.0} ms)", (ms * 48.0).round())
}
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
/// `path` as people write it: Windows' long-path prefix (`\?\`) left off.
fn plain_path(path: &std::path::Path) -> String {
    let text = path.display().to_string();
    text.strip_prefix(r"\\?\")
        .map(str::to_owned)
        .unwrap_or(text)
}
/// An empty slot named for `index`, as new slots are.
fn empty_slot(index: usize) -> ModelSlot {
    ModelSlot {
        name: format!("Voice {:02}", index + 1),
        manifest: String::new(),
        favorite: false,
        tune: None,
    }
}
/// Where the slot at `position` ends up when the slot at `from` moves to `to`.
fn index_after_move(position: usize, from: usize, to: usize) -> usize {
    if position == from {
        to
    } else if from < to && position > from && position <= to {
        position - 1
    } else if to < from && position >= to && position < from {
        position + 1
    } else {
        position
    }
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
/// The tile after the last voice card that adds another empty slot.
fn add_slot_card(ui: &mut egui::Ui, width: f32) -> bool {
    let (rect, response) = ui.allocate_exact_size(Vec2::new(width, 164.0), egui::Sense::click());
    let hint = t("Add another empty voice slot; there is no limit");
    response
        .widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, t("Add slot")));
    let painter = ui.painter();
    if response.hovered() {
        painter.rect_filled(rect, 12.0, m3::ON_SURFACE.gamma_multiply(0.08));
    }
    painter.rect_stroke(
        rect,
        12.0,
        egui::Stroke::new(1.0_f32, m3::OUTLINE_VARIANT),
        egui::StrokeKind::Inside,
    );
    m3::paint_icon(
        ui,
        rect.center() - Vec2::new(0.0, 12.0),
        icons::ADD,
        28.0,
        false,
        m3::PRIMARY,
    );
    painter.text(
        rect.center() + Vec2::new(0.0, 22.0),
        egui::Align2::CENTER_CENTER,
        t("Add slot"),
        FontId::new(14.0, m3::medium()),
        m3::PRIMARY,
    );
    response
        .on_hover_cursor(egui::CursorIcon::PointingHand)
        .on_hover_text(hint)
        .clicked()
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
                .unwrap_or(t("Unavailable device"))
        })
        .unwrap_or(t("System default"));
    m3::field_label(ui, label);
    let mut picked = None;
    let width = ui.available_width().min(440.0);
    m3::select(ui, name, width, |ui| {
        if m3::menu_item(ui, t("System default"), selected.is_none()).clicked() {
            picked = Some(None);
        }
        for device in devices.iter().filter(|d| d.input == input) {
            let on = selected.as_ref() == Some(&device.id);
            if m3::menu_item(ui, &device.name, on).clicked() {
                picked = Some(Some(device.id.clone()));
            }
        }
    });
    if let Some(device) = picked {
        *selected = device;
    }
}

#[cfg(test)]
mod tests {
    use super::index_after_move;

    #[test]
    fn every_slot_follows_a_move() {
        for (from, to) in [(0, 3), (3, 0), (2, 3), (3, 2), (1, 1)] {
            let mut slots: Vec<usize> = (0..5).collect();
            let slot = slots.remove(from);
            slots.insert(to, slot);
            for position in 0..5 {
                assert_eq!(
                    slots[index_after_move(position, from, to)],
                    position,
                    "{from}->{to}"
                );
            }
        }
    }
}
