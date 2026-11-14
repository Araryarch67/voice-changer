//! Clone page: records the user's voice into a dataset (`crate::dataset`), packs it
//! for the Colab notebook that trains an RVC v2 voice, and imports the result.
use super::{header, icons, m3, VoiceApp};
use crate::dataset::{self, Take};
use crate::i18n::{t, tf, Lang};
use crate::recorder::Recorder;
use eframe::egui::{self, RichText, Vec2};
use std::path::{Path, PathBuf};

#[derive(Default)]
pub(super) struct CloneState {
    /// The new voice's name, as typed.
    name: String,
    recorder: Option<Recorder>,
    /// Which sentence to read next.
    prompt: usize,
    /// The language of the sentences; the app's language until one is picked. The
    /// voice learns the accent of what is read, so it is chosen apart from the app's.
    reading: Option<Lang>,
    /// The takes of `listed`, read again when the name changes or a take is saved.
    takes: Vec<Take>,
    listed: Option<PathBuf>,
    /// What the last saved take showed: text and whether it is a warning.
    note: Option<(String, bool)>,
    /// The last packed ZIP.
    packed: Option<PathBuf>,
    /// The name was filled in from the most recent dataset.
    started: bool,
}

/// The dataset folder of `name`; `None` while it has no usable name.
fn folder(name: &str) -> Option<PathBuf> {
    let name = dataset::safe_name(name);
    (!name.is_empty()).then(|| dataset::root().join(name))
}

/// The most recently changed dataset folder's name, to continue it.
fn latest_dataset() -> Option<String> {
    std::fs::read_dir(dataset::root())
        .ok()?
        .flatten()
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| Some((entry.metadata().ok()?.modified().ok()?, entry.file_name())))
        .max_by_key(|(modified, _)| *modified)
        .map(|(_, name)| name.to_string_lossy().into_owned())
}

/// `seconds` as m:ss.
fn clock(seconds: f64) -> String {
    let seconds = seconds.max(0.0).round() as u64;
    format!("{}:{:02}", seconds / 60, seconds % 60)
}

/// Opens `path` (a folder or a web page) with the system's default program.
pub(super) fn open_external(path: &str) {
    let opened = if cfg!(target_os = "windows") {
        std::process::Command::new("explorer").arg(path).spawn()
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(path).spawn()
    } else {
        std::process::Command::new("xdg-open").arg(path).spawn()
    };
    if let Err(error) = opened {
        log::warn!("Could not open {path}: {error}");
    }
}

/// Sentences to read aloud in `lang`, chosen to cover the sounds of the language.
fn prompts(lang: Lang) -> &'static [&'static str] {
    match lang {
        Lang::En => &[
            "The quick brown fox jumps over the lazy dog near the riverbank.",
            "She sells seashells by the seashore every summer morning.",
            "Could you please bring me a glass of water and a warm towel?",
            "Thirty-three thousand people watched the final match last Thursday.",
            "I can't believe it's already seven o'clock! We're going to be late.",
            "The weather today is cloudy, with a chance of light rain this evening.",
            "Which way is the train station? I think I took a wrong turn.",
            "My favourite foods are fresh bread, spicy noodles, and sweet mangoes.",
            "Every voice is unique: the pitch, the rhythm, and the little pauses.",
            "Hello everyone, welcome back to the stream. How is your day going?",
            "Please speak clearly, take a breath, and don't rush the words.",
            "Yesterday we visited the museum, then walked along the old harbour.",
            "Oh no, the battery is almost empty. Do you have a charger?",
            "Numbers like four, eight, fifteen, sixteen, twenty-three and forty-two.",
            "Whisper softly, then laugh out loud, then ask a curious question?",
            "The children played joyfully in the garden until the sun went down.",
            "Really? That's amazing news! Congratulations on your new job.",
            "Measure twice and cut once, as the old carpenter used to say.",
            "Let's sing a little: la, la, la, la, la, and hold the last note.",
            "Thank you for listening. See you next time, and take care.",
        ],
        Lang::Id => &[
            "Selamat pagi semuanya, apa kabar hari ini? Semoga sehat selalu.",
            "Saya suka makan nasi goreng pedas dengan telur mata sapi.",
            "Kemarin hujan deras sekali, jadi kami menunggu di depan toko.",
            "Tolong ambilkan segelas air putih dan handuk yang bersih, ya.",
            "Kereta berangkat pukul tujuh lewat lima belas menit dari stasiun.",
            "Wah, ini luar biasa! Aku nggak percaya kita bisa menang.",
            "Kucing kecil itu melompat ke atas meja lalu tidur nyenyak.",
            "Angka satu, dua, tiga, empat, lima, enam, tujuh, delapan, sembilan, sepuluh.",
            "Bagaimana caranya pergi ke pantai yang paling dekat dari sini?",
            "Jangan lupa matikan lampu dan kunci pintu sebelum tidur.",
            "Halo teman-teman, selamat datang kembali di siaran langsungku.",
            "Gunung itu tinggi sekali, puncaknya tertutup awan putih.",
            "Ssst, pelan-pelan saja bicaranya, adik sedang belajar.",
            "Harga sayur di pasar naik, tapi buahnya masih murah.",
            "Ayo kita nyanyi sebentar: la, la, la, la, la, tahan nada terakhir.",
            "Ngomong-ngomong, kamu sudah nonton film yang baru itu belum?",
            "Pak guru menjelaskan pelajaran dengan sabar dan jelas.",
            "Hmm, sebentar ya, aku pikir-pikir dulu jawabannya.",
            "Burung-burung bernyanyi riang di pohon mangga depan rumah.",
            "Terima kasih sudah mendengarkan, sampai jumpa lagi!",
        ],
        Lang::Ja => &[
            "おはようございます。今日はとても良い天気ですね。",
            "駅までの道を教えていただけますか？",
            "昨日は友達と一緒にラーメンを食べに行きました。",
            "すみません、もう一度ゆっくり話してください。",
            "東京から大阪まで新幹線で二時間半かかります。",
            "わあ、すごい！本当におめでとうございます！",
            "猫が窓のそばで気持ちよさそうに寝ています。",
            "一、二、三、四、五、六、七、八、九、十。",
            "雨が降りそうなので、傘を持って行きましょう。",
            "みなさん、こんばんは。今日の配信へようこそ。",
            "冷たいお茶と温かいコーヒー、どちらがいいですか？",
            "週末は山に登って、きれいな景色を見ました。",
            "ちょっと待って、それは本当なの？",
            "小さな子どもたちが公園で楽しそうに遊んでいます。",
            "ら、ら、ら、ら、ら、と歌って、最後の音を伸ばしてください。",
            "この本はとても面白くて、一晩で読んでしまいました。",
            "えーと、少し考えさせてください。",
            "春になると、桜の花がいっぱい咲きます。",
            "ゆっくり深呼吸して、落ち着いて話しましょう。",
            "聞いてくれてありがとう。また会いましょう！",
        ],
    }
}

impl VoiceApp {
    pub(super) fn clone_page(&mut self, ui: &mut egui::Ui) {
        if !self.clone.started {
            self.clone.started = true;
            if let Some(name) = latest_dataset() {
                self.clone.name = name;
            }
        }
        let folder = folder(&self.clone.name);
        if self.clone.listed != folder {
            self.clone.takes = folder.as_deref().map(dataset::takes).unwrap_or_default();
            self.clone.listed = folder.clone();
            self.clone.packed = None;
        }
        header(
            ui,
            t("Clone your voice"),
            t("Record yourself, train an RVC v2 voice on Google Colab's free GPU, and import it here. Only clone voices you have the right to use."),
        );
        self.record_card(ui, folder.as_deref());
        ui.add_space(16.0);
        self.train_card(ui, folder.as_deref());
        ui.add_space(16.0);
        self.import_trained_card(ui);
    }

    fn record_card(&mut self, ui: &mut egui::Ui, folder: Option<&Path>) {
        let recording = self.clone.recorder.is_some();
        let live = self.running();
        m3::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing = Vec2::splat(8.0);
            m3::card_title(ui, icons::MIC, t("1. Record"));
            m3::field_label(ui, t("Voice name"));
            ui.add_enabled(
                !recording,
                m3::text_edit(
                    &mut self.clone.name,
                    t("my-voice"),
                    ui.available_width().min(440.0),
                ),
            );
            ui.label(m3::supporting(t(
                "The name of your new voice. Recordings are kept in the datasets folder under this name; the same name continues an earlier recording.",
            )));
            ui.add_space(4.0);
            let reading = self.clone.reading.unwrap_or_else(crate::i18n::current);
            m3::field_label(ui, t("Sentence language"));
            ui.horizontal_wrapped(|ui| {
                for lang in Lang::ALL {
                    if m3::filter_chip(ui, lang.name(), lang == reading).clicked() {
                        self.clone.reading = Some(lang);
                    }
                }
            });
            ui.label(m3::supporting(t(
                "Read in the language you will speak through the voice: it learns your accent from what you read.",
            )));
            let sentences = prompts(reading);
            let sentence = sentences[self.clone.prompt % sentences.len()];
            m3::field_label(ui, t("Read aloud"));
            egui::Frame::new()
                .fill(m3::SURFACE_CONTAINER_HIGH)
                .corner_radius(12)
                .inner_margin(16)
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.add(
                        egui::Label::new(RichText::new(sentence).size(20.0).color(m3::ON_SURFACE))
                            .wrap(),
                    );
                });
            ui.label(m3::supporting(t(
                "Read the sentences at your normal speaking level, a hand's width from the microphone, in a quiet room. Talking freely, telling a story or singing works too: what matters is 10 minutes or more of only your voice.",
            )));
            if let Some(recorder) = &self.clone.recorder {
                ui.horizontal(|ui| {
                    m3::meter(
                        ui,
                        icons::MIC,
                        recorder.level(),
                        None,
                        t("Microphone level. Aim for the loudest words to reach about -6 dB."),
                    );
                    ui.label(
                        RichText::new(tf("Recording {}", &[&clock(recorder.seconds())]))
                            .color(m3::ERROR),
                    );
                });
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_millis(50));
            }
            let (mut start, mut stop, mut discard, mut next) = (false, false, false, false);
            m3::actions(ui, |ui| {
                if recording {
                    stop = ui
                        .add(m3::Button::filled(t("Stop and save")).icon(icons::STOP))
                        .clicked();
                    discard = ui.add(m3::Button::text(t("Discard"))).clicked();
                } else {
                    start = ui
                        .add_enabled(
                            folder.is_some() && !live,
                            m3::Button::filled(t("Record")).icon(icons::RECORD),
                        )
                        .clicked();
                }
                next = ui
                    .add(m3::Button::text(t("Next sentence")).icon(icons::REFRESH))
                    .clicked();
            });
            if live {
                ui.label(m3::supporting(t(
                    "Stop live audio to record; both use the microphone.",
                )));
            } else if folder.is_none() {
                ui.label(m3::supporting(t("Give the voice a name to start recording.")));
            }
            if next {
                self.clone.prompt += 1;
            }
            if start {
                match Recorder::start(&self.saved.input) {
                    Ok(recorder) => {
                        self.clone.recorder = Some(recorder);
                        self.clone.note = None;
                    }
                    Err(error) => {
                        self.error = Some(tf("Could not start recording: {}", &[&error]));
                    }
                }
            }
            // A take stops by itself at its longest.
            let full = self
                .clone
                .recorder
                .as_ref()
                .is_some_and(|recorder| recorder.seconds() >= 600.0);
            if stop || full {
                self.save_take(folder);
            }
            if discard {
                if let Some(recorder) = self.clone.recorder.take() {
                    let _ = recorder.finish();
                }
            }
            self.dataset_summary(ui);
        });
    }

    fn save_take(&mut self, folder: Option<&Path>) {
        let (Some(recorder), Some(folder)) = (self.clone.recorder.take(), folder) else {
            return;
        };
        let (rate, samples) = recorder.finish();
        self.clone.note = Some(match dataset::save_take(folder, rate, &samples) {
            Ok(report) => {
                self.clone.prompt += 1;
                if report.clipped {
                    (t("The take was saved, but it clipped: lower the microphone gain or move back a little, and record it again.").to_owned(), true)
                } else if report.quiet {
                    (t("The take was saved, but it is very quiet: raise the microphone gain or move closer.").to_owned(), true)
                } else {
                    (
                        tf("Saved {} of voice.", &[&clock(report.take.seconds)]),
                        false,
                    )
                }
            }
            Err(error) => (format!("{error:#}"), true),
        });
        self.clone.listed = None;
        self.clone.packed = None;
    }

    /// Total recorded against the goal, the warning of the last take, and the takes.
    fn dataset_summary(&mut self, ui: &mut egui::Ui) {
        let total = dataset::total_seconds(&self.clone.takes);
        ui.add_space(4.0);
        ui.label(
            RichText::new(tf(
                "{} recorded of the {} recommended",
                &[&clock(total), &clock(dataset::RECOMMENDED_SECONDS)],
            ))
            .color(m3::ON_SURFACE),
        );
        progress(ui, (total / dataset::RECOMMENDED_SECONDS) as f32);
        if let Some((note, warning)) = &self.clone.note {
            let color = if *warning {
                m3::ERROR
            } else {
                m3::ON_SURFACE_VARIANT
            };
            ui.add(egui::Label::new(RichText::new(note).color(color)).wrap());
        }
        let recording = self.clone.recorder.is_some();
        let mut remove = None;
        for (index, take) in self.clone.takes.iter().enumerate().rev() {
            ui.horizontal(|ui| {
                let name = take
                    .path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                ui.label(RichText::new(name).color(m3::ON_SURFACE));
                ui.label(m3::supporting(&clock(take.seconds)));
                ui.add_enabled_ui(!recording, |ui| {
                    if m3::icon_button(ui, icons::DELETE, m3::ON_SURFACE_VARIANT, t("Delete take"))
                        .clicked()
                    {
                        remove = Some(index);
                    }
                });
            });
        }
        if let Some(index) = remove {
            let path = self.clone.takes[index].path.clone();
            if let Err(error) = std::fs::remove_file(&path) {
                self.error = Some(tf("Could not delete {}: {}", &[&path.display(), &error]));
            }
            self.clone.listed = None;
            self.clone.packed = None;
        }
        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        {
            let folder = self.clone.listed.clone();
            let mut add = false;
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                add = ui
                    .add_enabled(
                        folder.is_some() && !recording,
                        m3::Button::outlined(t("Add WAV recordings…")).icon(icons::ADD),
                    )
                    .clicked();
            });
            ui.label(m3::supporting(t(
                "Adds recordings you already have of the same voice. Only that voice should be heard: no music, no other speakers.",
            )));
            if add {
                if let (Some(folder), Some(paths)) = (
                    folder,
                    rfd::FileDialog::new()
                        .add_filter(t("WAV audio"), &["wav"])
                        .pick_files(),
                ) {
                    let mut failed = Vec::new();
                    for path in &paths {
                        if let Err(error) = dataset::add_file(&folder, path) {
                            failed.push(format!("{}: {error:#}", path.display()));
                        }
                    }
                    self.clone.listed = None;
                    self.clone.packed = None;
                    if !failed.is_empty() {
                        self.error = Some(failed.join("\n"));
                    }
                }
            }
        }
    }

    fn train_card(&mut self, ui: &mut egui::Ui, folder: Option<&Path>) {
        let total = dataset::total_seconds(&self.clone.takes);
        let enough = total >= dataset::MINIMUM_SECONDS;
        let busy = self.work.is_some() || self.clone.recorder.is_some();
        m3::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing = Vec2::splat(8.0);
            m3::card_title(ui, icons::MEMORY, t("2. Train on Google Colab"));
            ui.add(
                egui::Label::new(m3::supporting(t(
                    "Pack the recordings into a ZIP and open the notebook. In Colab choose Runtime > Change runtime type > T4 GPU, run the four steps in order and upload the ZIP in step 2. Training takes about 1–3 hours; step 4 downloads your voice. You need a Google account.",
                )))
                .wrap(),
            );
            let (mut pack, mut open, mut show) = (false, false, false);
            m3::actions(ui, |ui| {
                pack = ui
                    .add_enabled(
                        enough && !busy && folder.is_some(),
                        m3::Button::filled(t("Pack for Colab")).icon(icons::AUDIO_FILE),
                    )
                    .clicked();
                open = ui
                    .add(m3::Button::tonal(t("Open Colab notebook")).icon(icons::ROUTING))
                    .clicked();
                show = ui
                    .add_enabled(
                        self.clone.packed.is_some(),
                        m3::Button::text(t("Show ZIP")).icon(icons::FOLDER_OPEN),
                    )
                    .clicked();
            });
            if !enough {
                ui.label(m3::supporting(&tf(
                    "Record at least {} first; {} or more gives a much better voice.",
                    &[
                        &clock(dataset::MINIMUM_SECONDS),
                        &clock(dataset::RECOMMENDED_SECONDS),
                    ],
                )));
            } else if let Some(packed) = &self.clone.packed {
                ui.label(m3::supporting(&tf(
                    "Ready to upload: {}",
                    &[&packed.display()],
                )));
            }
            if pack {
                self.pack_dataset();
            }
            if open {
                open_external(&dataset::colab_url());
            }
            if show {
                if let Some(parent) = self.clone.packed.as_ref().and_then(|p| p.parent()) {
                    open_external(&parent.to_string_lossy());
                }
            }
        });
    }

    #[cfg(feature = "export")]
    fn pack_dataset(&mut self) {
        let Some(folder) = self.clone.listed.clone() else {
            return;
        };
        let name = dataset::safe_name(&self.clone.name);
        let output = dataset::root().join(format!("{name}.zip"));
        self.clone.packed = Some(output.clone());
        self.begin_work(move || {
            let seconds = dataset::pack(&folder, &name, &output)?;
            Ok(tf(
                "Packed {} of voice into {}. Upload it in step 2 of the Colab notebook.",
                &[&clock(seconds), &output.display()],
            ))
        });
    }
    #[cfg(not(feature = "export"))]
    fn pack_dataset(&mut self) {
        self.error = Some(t("This build cannot pack datasets.").into());
    }

    fn import_trained_card(&mut self, ui: &mut egui::Ui) {
        let busy = self.work.is_some() || self.running();
        m3::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing = Vec2::splat(8.0);
            m3::card_title(ui, icons::ADD, t("3. Import the trained voice"));
            ui.label(m3::supporting(t(
                "Choose the <name>-rvc.zip that step 4 downloaded. It becomes a new voice in Voices, ready to use.",
            )));
            let mut import = false;
            m3::actions(ui, |ui| {
                import = ui
                    .add_enabled(
                        !busy,
                        m3::Button::filled(t("Import trained voice…")).icon(icons::ADD),
                    )
                    .clicked();
            });
            if import {
                self.import_trained();
            }
        });
    }

    fn import_trained(&mut self) {
        #[cfg(feature = "catalog")]
        if !self.shared_models_ready() {
            return;
        }
        #[cfg(all(feature = "export", not(any(target_os = "android", target_os = "ios"))))]
        if let Some(path) = rfd::FileDialog::new()
            .add_filter(t("Trained voice"), &["zip", "pth"])
            .pick_file()
        {
            self.begin_work(move || {
                let manifest = crate::import::import_voice(&path)?;
                Ok(tf("Imported {}", &[&manifest.display()]))
            });
        }
    }
}

/// A thin bar filled to `fraction`.
fn progress(ui: &mut egui::Ui, fraction: f32) {
    let (rect, _) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 8.0), egui::Sense::hover());
    let track = egui::Rect::from_center_size(rect.center(), Vec2::new(rect.width(), 4.0));
    ui.painter()
        .rect_filled(track, 2.0, m3::SURFACE_CONTAINER_HIGHEST);
    let mut filled = track;
    filled.set_right(egui::lerp(
        track.left()..=track.right(),
        fraction.clamp(0.0, 1.0),
    ));
    ui.painter().rect_filled(filled, 2.0, m3::PRIMARY);
}
