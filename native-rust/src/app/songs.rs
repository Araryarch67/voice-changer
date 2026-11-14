//! Split and Cover pages: separate a song's vocals from its music
//! (`crate::separate`), and sing a song in one of the user's voices.
use super::{header, icons, m3, Page, VoiceApp};
use crate::i18n::{t, tf};
use crate::separate::{self, Progress};
use eframe::egui::{self, Vec2};
use std::{
    path::{Path, PathBuf},
    sync::{atomic::Ordering, Arc},
};

pub(super) struct SongsState {
    split_input: String,
    /// Empty: beside the song.
    split_folder: String,
    cover_input: String,
    cover_output: String,
    vocal_gain: f32,
    music_gain: f32,
    keep_stems: bool,
    /// The cover's own Tune: the song's singer is not the person at the
    /// microphone, so Studio's Tune does not suit it.
    semitones: f32,
    /// Choose Tune from the song's pitch for a voice in `register`.
    auto_tune: bool,
    register: separate::Register,
    /// The running split or cover.
    progress: Option<Arc<Progress>>,
}
impl Default for SongsState {
    fn default() -> Self {
        Self {
            split_input: String::new(),
            split_folder: String::new(),
            cover_input: String::new(),
            cover_output: String::new(),
            vocal_gain: 1.0,
            music_gain: 1.0,
            keep_stems: false,
            semitones: 0.0,
            auto_tune: true,
            register: separate::Register::Female,
            progress: None,
        }
    }
}

/// Song files a picker offers and a drop accepts.
pub(super) fn is_song(path: &Path) -> bool {
    path.extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .is_some_and(|e| crate::audio_file::EXTENSIONS.contains(&e.as_str()))
}

fn cover_path(song: &Path) -> PathBuf {
    let stem = song.file_stem().unwrap_or_default().to_string_lossy();
    song.with_file_name(format!("{stem}-cover.wav"))
}

#[cfg(not(any(target_os = "android", target_os = "ios")))]
fn pick_song() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter(t("Songs"), &crate::audio_file::EXTENSIONS)
        .pick_file()
}

impl VoiceApp {
    /// A dropped song fills the Split or Cover page, whichever is open (Cover
    /// otherwise).
    pub(super) fn drop_song(&mut self, path: &Path) {
        if self.page == Page::Split {
            self.songs.split_input = path.to_string_lossy().into_owned();
        } else {
            self.songs.cover_input = path.to_string_lossy().into_owned();
            self.songs.cover_output = cover_path(path).to_string_lossy().into_owned();
            self.page = Page::Cover;
        }
        self.error = None;
    }

    /// Clears the progress once its job is over.
    pub(super) fn poll_songs(&mut self) {
        if self.work.is_none() {
            self.songs.progress = None;
        }
    }

    /// The model notice, the progress of a running job, and whether a job can start.
    fn song_status(&mut self, ui: &mut egui::Ui) -> bool {
        let installed = separate::model_path().is_some();
        if !installed {
            egui::Frame::new()
                .fill(m3::SURFACE_CONTAINER_HIGH)
                .corner_radius(12)
                .inner_margin(egui::Margin::symmetric(16, 10))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal_wrapped(|ui| {
                        ui.label(m3::icon(icons::ERROR, 20.0, m3::ERROR));
                        ui.label(t(
                            "The vocal separation model (Kim Vocal 2, 64 MB) is not installed yet.",
                        ));
                        #[cfg(feature = "catalog")]
                        if ui.add(m3::Button::tonal(t("Download"))).clicked() {
                            self.offer_model(separate::MODEL_FILE);
                        }
                    });
                });
        }
        if let Some(progress) = self.songs.progress.clone() {
            let stage = match progress.stage.load(Ordering::Relaxed) {
                separate::SEPARATING => t("Separating the vocals…"),
                separate::CONVERTING => t("Singing it in the new voice…"),
                separate::MEASURING => t("Measuring how high the song is…"),
                _ => t("Saving…"),
            };
            let fraction = progress.fraction();
            ui.horizontal(|ui| {
                ui.add(egui::Spinner::new().color(m3::PRIMARY));
                ui.label(format!("{stage} {:.0}%", fraction * 100.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.add(m3::Button::text(t("Cancel"))).clicked() {
                        progress.cancel.store(true, Ordering::Relaxed);
                    }
                });
            });
            m3::linear_progress(ui, fraction);
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_millis(200));
        }
        installed && !self.running() && self.work.is_none()
    }

    /// A song field with a picker.
    fn song_field(ui: &mut egui::Ui, value: &mut String) -> bool {
        let browse = cfg!(not(any(target_os = "android", target_os = "ios")));
        ui.horizontal(|ui| {
            let width = ui.available_width() - if browse { 48.0 } else { 0.0 };
            ui.add(m3::text_edit(value, "song.mp3", width));
            // The desktop file picker; on phones the path is typed.
            #[cfg(not(any(target_os = "android", target_os = "ios")))]
            let picked = m3::icon_button(
                ui,
                icons::FOLDER_OPEN,
                m3::ON_SURFACE_VARIANT,
                t("Choose a song…"),
            )
            .clicked()
                && pick_song()
                    .map(|path| *value = path.to_string_lossy().into_owned())
                    .is_some();
            #[cfg(any(target_os = "android", target_os = "ios"))]
            let picked = false;
            picked
        })
        .inner
    }

    pub(super) fn split_page(&mut self, ui: &mut egui::Ui) {
        header(
            ui,
            t("Split vocals"),
            t("Separates a song into its vocals and its music, as two WAV files. MP3, FLAC, OGG, M4A and WAV songs work. Drop a song on the window to start here."),
        );
        m3::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing = Vec2::splat(8.0);
            let ready = self.song_status(ui);
            ui.add_enabled_ui(self.work.is_none(), |ui| {
                m3::field_label(ui, t("Song"));
                Self::song_field(ui, &mut self.songs.split_input);
                m3::field_label(ui, t("Save to folder"));
                ui.horizontal(|ui| {
                    let browse = cfg!(not(any(target_os = "android", target_os = "ios")));
                    let width = ui.available_width() - if browse { 48.0 } else { 0.0 };
                    ui.add(m3::text_edit(
                        &mut self.songs.split_folder,
                        t("Beside the song"),
                        width,
                    ));
                    #[cfg(not(any(target_os = "android", target_os = "ios")))]
                    if m3::icon_button(
                        ui,
                        icons::FOLDER_OPEN,
                        m3::ON_SURFACE_VARIANT,
                        t("Choose a folder…"),
                    )
                    .clicked()
                    {
                        if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                            self.songs.split_folder = folder.to_string_lossy().into_owned();
                        }
                    }
                });
            });
            let song = PathBuf::from(self.songs.split_input.trim());
            let folder = Some(self.songs.split_folder.trim())
                .filter(|folder| !folder.is_empty())
                .map(PathBuf::from);
            let (vocals, instrumental) = separate::stem_paths(&song, folder.as_deref());
            ui.label(m3::supporting(&if self.songs.split_input.trim().is_empty() {
                t("The vocals and the music are saved as two WAV files (44.1 kHz stereo). Works best on studio recordings; backing vocals and reverb may stay with the vocals.").to_owned()
            } else {
                tf(
                    "Saves {} and {}.",
                    &[
                        &vocals.file_name().unwrap_or_default().to_string_lossy(),
                        &instrumental.file_name().unwrap_or_default().to_string_lossy(),
                    ],
                )
            }));
            let mut start = false;
            m3::actions(ui, |ui| {
                start = ui
                    .add_enabled(
                        ready && !self.songs.split_input.trim().is_empty(),
                        m3::Button::filled(t("Split")).icon(icons::CALL_SPLIT),
                    )
                    .clicked();
            });
            if start {
                let Some(model) = separate::model_path() else {
                    return;
                };
                let device = self.saved.settings.device;
                let progress = Arc::new(Progress::default());
                self.songs.progress = Some(progress.clone());
                self.begin_work(move || {
                    anyhow::ensure!(song.is_file(), "The song does not exist: {}", song.display());
                    let seconds =
                        separate::split_file(&model, device, &song, &vocals, &instrumental, &progress)?;
                    Ok(tf(
                        "Split {} s of audio: {} and {}",
                        &[
                            &format!("{seconds:.0}"),
                            &vocals.display(),
                            &instrumental.display(),
                        ],
                    ))
                });
            }
        });
    }

    pub(super) fn cover_page(&mut self, ui: &mut egui::Ui) {
        header(
            ui,
            t("Cover a song"),
            t("Sings a song in one of your voices: its vocals are separated, converted with the voice, and mixed back over the music. Drop a song on the window to start here."),
        );
        m3::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing = Vec2::splat(8.0);
            let ready = self.song_status(ui);
            let busy = self.work.is_some();
            self.voice_picker(ui, !busy);
            ui.add_enabled_ui(!busy, |ui| {
                m3::field_label(ui, t("Song"));
                if Self::song_field(ui, &mut self.songs.cover_input) {
                    self.songs.cover_output = cover_path(Path::new(self.songs.cover_input.trim()))
                        .to_string_lossy()
                        .into_owned();
                }
                m3::field_label(ui, t("Save the cover as"));
                ui.horizontal(|ui| {
                    let browse = cfg!(not(any(target_os = "android", target_os = "ios")));
                    let width = ui.available_width() - if browse { 48.0 } else { 0.0 };
                    ui.add(m3::text_edit(&mut self.songs.cover_output, "song-cover.wav", width));
                    #[cfg(not(any(target_os = "android", target_os = "ios")))]
                    if m3::icon_button(ui, icons::FOLDER_OPEN, m3::ON_SURFACE_VARIANT, t("Choose output…"))
                        .clicked()
                    {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter(t("WAV audio"), &["wav"])
                            .set_file_name("cover.wav")
                            .save_file()
                        {
                            self.songs.cover_output = path.to_string_lossy().into_owned();
                        }
                    }
                });
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    m3::switch(ui, &mut self.songs.auto_tune, t("Automatic Tune"));
                    ui.label(t("Automatic Tune"));
                });
                ui.label(m3::supporting(t(
                    "Measures how high the song is sung and moves it by whole octaves to suit the voice, so it stays in key with the music.",
                )));
                if self.songs.auto_tune {
                    m3::field_label(ui, t("The chosen voice (RVC model) is"));
                    ui.horizontal_wrapped(|ui| {
                        for (register, name) in [
                            (separate::Register::Female, t("A female voice")),
                            (separate::Register::Male, t("A male voice")),
                        ] {
                            if m3::filter_chip(ui, name, self.songs.register == register).clicked() {
                                self.songs.register = register;
                            }
                        }
                    });
                    ui.label(m3::supporting(t(
                        "Pick the voice you sing with, not the song's singer: the singer's pitch is measured from the song.",
                    )));
                }
                ui.add_enabled_ui(!self.songs.auto_tune, |ui| m3::setting(
                    ui,
                    t("Tune"),
                    t("Moves the song for the voice; Studio's Tune is not used here. The music is not moved, so 0, +12 (one octave up) and -12 (one octave down) keep the voice in key with it."),
                    m3::Format::new(t(" semitones"))
                        .tip(t("A man's song in a female voice: try +12. A woman's song in a male voice: try -12. High notes break up or go quiet when they end up too high for the voice, so if they vanish, go lower."))
                        .decimals(1)
                        .signed()
                        .presets(&[-24.0, -12.0, 0.0, 12.0, 24.0]),
                    &mut self.songs.semitones,
                    -24.0..=24.0,
                    0.1,
                ));
                m3::setting(
                    ui,
                    t("Voice volume"),
                    t("How loud the new voice is over the music. Raise it if the voice is buried."),
                    m3::Format::new("×").decimals(2).presets(&[0.5, 0.75, 1.0, 1.25, 1.5]),
                    &mut self.songs.vocal_gain,
                    0.0..=2.0,
                    0.01,
                );
                m3::setting(
                    ui,
                    t("Music volume"),
                    t("How loud the music is under the voice."),
                    m3::Format::new("×").decimals(2).presets(&[0.5, 0.75, 1.0, 1.25, 1.5]),
                    &mut self.songs.music_gain,
                    0.0..=2.0,
                    0.01,
                );
                ui.horizontal(|ui| {
                    m3::switch(ui, &mut self.songs.keep_stems, t("Keep the separated parts"));
                    ui.label(t("Keep the separated parts"));
                });
                ui.label(m3::supporting(t(
                    "Also saves the original vocals, the music and the new vocals beside the cover, to mix them yourself.",
                )));
            });
            ui.label(m3::supporting(t(
                "Index, Protect and the F0 detector from Studio are used too; RMVPE suits singing best.",
            )));
            let voice_ready = !self.saved.slots[self.saved.selected]
                .manifest
                .trim()
                .is_empty();
            let mut start = false;
            m3::actions(ui, |ui| {
                start = ui
                    .add_enabled(
                        ready
                            && voice_ready
                            && !self.songs.cover_input.trim().is_empty()
                            && !self.songs.cover_output.trim().is_empty(),
                        m3::Button::filled(t("Make cover")).icon(icons::LIBRARY_MUSIC),
                    )
                    .clicked();
            });
            if start {
                let Some(model) = separate::model_path() else {
                    return;
                };
                let voice = PathBuf::from(self.saved.slots[self.saved.selected].manifest.trim());
                let song = PathBuf::from(self.songs.cover_input.trim());
                let output = PathBuf::from(self.songs.cover_output.trim());
                let settings = crate::config::Settings {
                    semitones: self.songs.semitones,
                    ..self.saved.settings.clone()
                };
                let auto_tune = self.songs.auto_tune.then_some(self.songs.register);
                let (vocal_gain, music_gain, keep_stems) = (
                    self.songs.vocal_gain,
                    self.songs.music_gain,
                    self.songs.keep_stems,
                );
                let progress = Arc::new(Progress::default());
                self.songs.progress = Some(progress.clone());
                self.begin_work(move || {
                    anyhow::ensure!(song.is_file(), "The song does not exist: {}", song.display());
                    let files = separate::cover(
                        &separate::CoverJob {
                            separation_model: &model,
                            voice: &voice,
                            settings: &settings,
                            input: &song,
                            output: &output,
                            vocal_gain,
                            music_gain,
                            keep_stems,
                            auto_tune,
                        },
                        &progress,
                    )?;
                    Ok(match files.pitch {
                        Some(pitch) => tf(
                            "Cover saved: {} (the song is sung around {} Hz, so Tune {})",
                            &[
                                &files.cover.display(),
                                &format!("{:.0}", pitch.median_hz),
                                &format!("{:+.0}", files.semitones),
                            ],
                        ),
                        None => tf("Cover saved: {}", &[&files.cover.display()]),
                    })
                });
            }
        });
    }
}
