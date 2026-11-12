#[cfg(feature = "audio")]
use crate::audio::{self, AudioDevice, AudioOptions, AudioSession};
use crate::config::{Device, F0Detector, Noise, Settings};
use crate::m3::{self, icons};
use eframe::egui::{self, FontId, Margin, RichText, Vec2};
use serde::{Deserialize, Serialize};
#[cfg(feature = "audio")]
use std::sync::atomic::Ordering;
use std::{path::PathBuf, sync::mpsc};

#[derive(Clone, Serialize, Deserialize)]
struct ModelSlot {
    name: String,
    manifest: String,
    #[serde(default)]
    favorite: bool,
}
#[derive(Serialize, Deserialize)]
#[serde(default)]
struct Saved {
    settings: Settings,
    input: Option<String>,
    output: Option<String>,
    slots: Vec<ModelSlot>,
    selected: usize,
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
                })
                .collect(),
            selected: 0,
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
    file_output: String,
    work: Option<mpsc::Receiver<Result<String, String>>>,
    message: Option<String>,
    /// Voice pictures by manifest path; `None` once a voice is known to have none.
    portraits: std::collections::HashMap<String, Option<egui::TextureHandle>>,
    #[cfg(feature = "audio")]
    devices: Vec<AudioDevice>,
    #[cfg(feature = "audio")]
    session: Option<AudioSession>,
}
impl VoiceApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        m3::install(&cc.egui_ctx);
        let mut saved: Saved = cc
            .storage
            .and_then(|s| eframe::get_value(s, "voice-changer"))
            .unwrap_or_default();
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
            file_output: String::new(),
            work: None,
            message: None,
            portraits: Default::default(),
            #[cfg(feature = "audio")]
            devices: Vec::new(),
            #[cfg(feature = "audio")]
            session: None,
        };
        #[cfg(feature = "audio")]
        app.refresh_devices();
        app
    }
    #[cfg(feature = "audio")]
    fn refresh_devices(&mut self) {
        match audio::devices() {
            Ok(d) => self.devices = d,
            Err(e) => self.error = Some(format!("{e:#}")),
        }
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
        let manifest = self.saved.slots[self.saved.selected].manifest.trim();
        let options = AudioOptions {
            input: self.saved.input.clone(),
            output: self.saved.output.clone(),
            manifest: if manifest.is_empty() {
                None
            } else {
                Some(manifest.into())
            },
            settings: self.saved.settings.clone(),
            record_dir: recordings_dir(),
        };
        match AudioSession::start(options) {
            Ok(s) => self.session = Some(s),
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
    fn add_slot(&mut self) {
        self.saved.slots.push(ModelSlot {
            name: format!("Voice {:02}", self.saved.slots.len() + 1),
            manifest: String::new(),
            favorite: false,
        });
        self.saved.selected = self.saved.slots.len() - 1;
        self.edit_model = true;
    }
    fn import_model(&mut self) {
        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        if let Some(path) = rfd::FileDialog::new()
            .add_filter(
                "RVC voice (.pth, .zip) or manifest (.json)",
                &["pth", "zip", "json"],
            )
            .pick_file()
        {
            self.import_file(path);
        }
        #[cfg(any(target_os = "android", target_os = "ios"))]
        self.add_slot();
    }
    /// A manifest is linked as it is; an RVC checkpoint or archive is converted into a
    /// new voice folder in the background and picked up when that finishes.
    fn import_file(&mut self, path: PathBuf) {
        if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("json"))
        {
            match crate::config::ModelManifest::load(&path) {
                Ok(model) => {
                    self.saved.slots.push(ModelSlot {
                        name: model.name,
                        manifest: path.to_string_lossy().into_owned(),
                        favorite: false,
                    });
                    self.saved.selected = self.saved.slots.len() - 1;
                    self.saved.settings.bypass = false;
                    self.error = None;
                }
                Err(error) => self.error = Some(format!("{error:#}")),
            }
            return;
        }
        if self.work.is_some() {
            self.error = Some("Wait for the current task to finish, then import again.".into());
            return;
        }
        #[cfg(feature = "export")]
        self.begin_work(move || {
            let manifest = crate::import::import_voice(&path)?;
            let voice = crate::config::ModelManifest::load(&manifest)?;
            Ok(format!(
                "Imported {}{}.",
                voice.name,
                if voice.feature_bank.is_some() {
                    " with its index"
                } else {
                    " (no index found)"
                }
            ))
        });
        #[cfg(not(feature = "export"))]
        {
            self.error =
                Some("This build cannot convert checkpoints; import a model.json instead.".into());
        }
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
            "Pick a voice, then press Start. Import an RVC .pth or .zip, or drop one on the window.",
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
            if actions {
                self.test_button(ui);
                self.edit_button(ui);
            }
            if ui
                .add_enabled(
                    !running && self.work.is_none(),
                    m3::Button::text("Empty slot").icon(icons::ADD),
                )
                .clicked()
            {
                self.add_slot();
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
            let file = PathBuf::from(slot.manifest.trim());
            let folder = file
                .parent()
                .and_then(|p| p.file_name())
                .map(|name| format!("{}/", name.to_string_lossy()))
                .unwrap_or_default();
            let detail = if configured {
                format!(
                    "RVC · ONNX · {folder}{}",
                    file.file_name().unwrap_or_default().to_string_lossy()
                )
            } else {
                "Empty slot. Connect a model manifest to use it.".to_owned()
            };
            ui.add(egui::Label::new(m3::supporting(&detail)).truncate());
            ui.add_space(8.0);
            if configured {
                let s = &mut self.saved.settings;
                m3::setting(
                    ui,
                    "Tune", "Makes the voice higher or lower. +12 is one octave up, -12 one octave down.",
                    format!("{:+.1} semitones", s.semitones),
                    &mut s.semitones,
                    -24.0..=24.0,
                    0.1,
                );
                m3::setting(
                    ui,
                    "Index", "How much of the model's accent and tone to copy. Higher sounds more like the model but can add artifacts.",
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
                    !self.running(),
                    m3::Button::filled("Connect model").icon(icons::ADD),
                )
                .clicked()
            {
                self.edit_model = true;
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
        if response.clicked() && !running {
            self.saved.selected = index;
        }
        if response.double_clicked() && !running {
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
                "Tune", "Makes the voice higher or lower. +12 is one octave up, -12 one octave down.",
                format!("{:+.1} semitones", s.semitones),
                &mut s.semitones,
                -24.0..=24.0,
                0.1,
            );
            m3::setting(
                ui,
                "Index", "How much of the model's accent and tone to copy. Higher sounds more like the model but can add artifacts.",
                format!("{:.0} %", s.index_ratio * 100.0),
                &mut s.index_ratio,
                0.0..=1.0,
                0.01,
            );
            m3::setting(
                ui,
                "Protect", "Keeps breaths and consonants natural. Lower protects more; 0.50 turns it off.",
                format!("{:.2}", s.protect),
                &mut s.protect,
                0.0..=0.5,
                0.01,
            );
            choice(
                ui,
                "F0 detector",
                "How your pitch is measured. RMVPE is the best all-rounder. CREPE full and Harvest are accurate but heavy; CREPE tiny, DIO and YIN are light.",
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
                "Noise gate (S.Thresh)", "Silences everything quieter than this, so room noise is not converted. Try around -45 dB.",
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
                    "Chunk", "How much audio is converted at once. Bigger is smoother and lighter on the CPU, but adds delay.",
                    format!("{} ms", s.block_ms),
                    &mut s.block_ms,
                    20..=500,
                    1.0,
                );
                s.overlap_ms = s.overlap_ms.min(s.block_ms);
                m3::setting(
                    ui,
                    "Extra", "Past audio the model hears for context. More improves quality but needs more CPU.",
                    format!("{} ms", s.context_ms),
                    &mut s.context_ms,
                    0..=2000,
                    1.0,
                );
                let max_overlap = s.block_ms.min(100);
                m3::setting(
                    ui,
                    "Crossfade", "Blends each chunk into the next to hide the seams.",
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
    }
    fn settings(&mut self, ui: &mut egui::Ui) {
        header(ui, "Settings", "Audio devices and runtime.");
        #[cfg(feature = "audio")]
        {
            let running = self.running();
            m3::card().show(ui, |ui| {
                ui.set_width(ui.available_width());
                m3::card_title(ui, icons::ROUTING, "Audio routing");
                ui.add_enabled_ui(!running, |ui| {
                    device_picker(ui, "Microphone", true, &self.devices, &mut self.saved.input);
                    device_picker(
                        ui,
                        "Output / virtual cable",
                        false,
                        &self.devices,
                        &mut self.saved.output,
                    );
                    ui.add_space(8.0);
                    if ui
                        .add(m3::Button::outlined("Refresh devices").icon(icons::REFRESH))
                        .clicked()
                    {
                        self.refresh_devices();
                    }
                });
                ui.label(m3::supporting("Choose headphones for monitoring, or an installed virtual device to send audio to another app."));
            });
            ui.add_space(4.0);
        }
        let running = self.running();
        m3::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            m3::card_title(ui, icons::MEMORY, "Processing");
            let gpu = crate::runtime::gpu_available();
            ui.add_enabled_ui(!running && (gpu || self.saved.settings.device == Device::Gpu), |ui| {
                choice(
                    ui,
                    "Device", "Where the model runs. A GPU is much faster than the CPU.",
                    &mut self.saved.settings.device,
                    &[(Device::Cpu, "CPU"), (Device::Gpu, "GPU (CUDA)")],
                );
            });
            ui.label(m3::supporting(if gpu {
                "GPU is available. The choice applies the next time you press Start."
            } else {
                "This ONNX Runtime has no CUDA provider, so only CPU is available. Install a GPU build of ONNX Runtime with matching CUDA and cuDNN libraries to enable GPU."
            }));
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
            status = if session.metrics.ready.load(Ordering::Acquire) {
                "Live".to_owned()
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
                            "Stop recording"
                        } else {
                            "Record the microphone and the converted voice to WAV files"
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
        if let Some(session) = &self.session {
            let m = &session.metrics;
            let load = |v: &std::sync::atomic::AtomicU32| f32::from_bits(v.load(Ordering::Relaxed));
            ui.horizontal_wrapped(|ui| {
                m3::meter(ui, icons::MIC, load(&m.input_level));
                m3::meter(ui, icons::HEADPHONES, load(&m.output_level));
                let rate = m.output_rate.load(Ordering::Relaxed);
                let queued = if rate == 0 {
                    0.0
                } else {
                    m.queued_samples.load(Ordering::Relaxed) as f32 * 1000.0 / rate as f32
                };
                ui.label(m3::supporting(&format!(
                    "{:.1} ms processing · {queued:.0} ms queued",
                    load(&m.processing_ms)
                )));
                ui.label(m3::supporting(&format!(
                    "Dropped {} / {} · underrun {}",
                    m.dropped_input.load(Ordering::Relaxed),
                    m.dropped_output.load(Ordering::Relaxed),
                    m.underrun_samples.load(Ordering::Relaxed)
                )));
            });
        }
    }
    fn model_dialog(&mut self, ctx: &egui::Context) {
        let running = self.running();
        let removable = self.saved.slots.len() > 1;
        let (mut remove, mut done) = (false, false);
        #[allow(unused_mut)]
        let mut picture: Option<PathBuf> = None;
        let (_, dismissed) = m3::dialog(ctx, "model-dialog", "Configure voice", |ui| {
            ui.add_enabled_ui(!running, |ui| {
                let slot = &mut self.saved.slots[self.saved.selected];
                m3::field_label(ui, "Display name");
                ui.add(m3::text_edit(&mut slot.name, "Voice name", f32::INFINITY));
                m3::field_label(ui, "Model manifest (.json)");
                ui.horizontal(|ui| {
                    let browse = cfg!(not(any(target_os = "android", target_os = "ios")));
                    let width = ui.available_width() - if browse { 48.0 } else { 0.0 };
                    ui.add(m3::text_edit(
                        &mut slot.manifest,
                        "/path/to/model.json",
                        width,
                    ));
                    #[cfg(not(any(target_os = "android", target_os = "ios")))]
                    if m3::icon_button(
                        ui,
                        icons::FOLDER_OPEN,
                        m3::ON_SURFACE_VARIANT,
                        "Browse manifest…",
                    )
                    .clicked()
                    {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Voice model manifest", &["json"])
                            .pick_file()
                        {
                            slot.manifest = path.to_string_lossy().into_owned();
                        }
                    }
                });
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
                    Ok(format!("Conversion complete: {}", output.display()))
                });
            }
        });
        self.file_dialog = !(close || dismissed);
    }
}
impl eframe::App for VoiceApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let completed = self.work.as_ref().and_then(|work| match work.try_recv() {
            Ok(result) => Some(result),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => Some(Err("Model worker disconnected".into())),
        });
        if let Some(result) = completed {
            self.work = None;
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
        #[cfg(feature = "audio")]
        if let Some(session) = &self.session {
            if let Some(recorded) = session.metrics.recorded.lock().unwrap().take() {
                self.message = Some(recorded);
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
        // Dropping a checkpoint, archive or manifest on the window imports it.
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        if !self.running() {
            for path in dropped {
                self.import_file(path);
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
