//! Voice engine dialog: picks and downloads the shared models (ContentVec and RMVPE,
//! required; CREPE tiny and full, optional) with M3 checkboxes, a recommendation for
//! each, and a progress bar. Opens on first start while the required ones are missing.
use super::{icons, m3, VoiceApp};
use crate::i18n::{t, tf};
use crate::{catalog::Progress, setup};
use eframe::egui::{self, FontId, Rect, Sense, Vec2};
use std::sync::{atomic::Ordering, mpsc, Arc};

pub(super) struct Setup {
    pub open: bool,
    progress: Option<Arc<Progress>>,
    done: Option<mpsc::Receiver<Result<(), String>>>,
    problem: Option<String>,
    /// Which of `setup::SHARED` to download.
    chosen: [bool; setup::SHARED.len()],
}
impl Default for Setup {
    fn default() -> Self {
        Self {
            open: false,
            progress: None,
            done: None,
            problem: None,
            chosen: setup::SHARED.map(|model| model.required || model.recommended),
        }
    }
}

impl Setup {
    /// Opens on start when the required models are missing.
    pub fn on_start() -> Self {
        Self {
            open: crate::runtime::models_root().is_none(),
            ..Default::default()
        }
    }
    pub fn downloading(&self) -> bool {
        self.done.is_some()
    }
    /// Bytes received, the total, and the fraction done.
    fn amounts(&self) -> Option<(u64, u64, f32)> {
        let progress = self.progress.as_ref()?;
        let received = progress.received.load(Ordering::Relaxed);
        let total = progress.total.load(Ordering::Relaxed).max(1);
        Some((received, total, received as f32 / total as f32))
    }
}

fn megabytes(bytes: u64) -> String {
    format!("{:.0} MB", bytes as f64 / 1_048_576.0)
}

/// The folder the models are in once installed.
fn shared_folder() -> std::path::PathBuf {
    crate::runtime::models_root()
        .unwrap_or_else(setup::target_root)
        .join("shared")
}

/// Whether the CREPE model for `detector` is installed (other detectors need none).
pub(super) fn detector_ready(detector: crate::config::F0Detector) -> bool {
    use crate::config::F0Detector;
    let file = match detector {
        F0Detector::CrepeFull => "crepe_full.onnx",
        F0Detector::CrepeTiny => "crepe_tiny.onnx",
        _ => return true,
    };
    shared_folder().join(file).is_file()
}

impl VoiceApp {
    /// Opens the setup dialog instead of failing when the shared models are missing.
    /// Returns true when they are there.
    pub(super) fn shared_models_ready(&mut self) -> bool {
        if crate::runtime::models_root().is_some() {
            return true;
        }
        self.setup.open = true;
        false
    }

    /// Opens the dialog with the CREPE model for `detector` ticked.
    pub(super) fn offer_detector(&mut self, detector: crate::config::F0Detector) {
        use crate::config::F0Detector;
        let file = match detector {
            F0Detector::CrepeFull => "crepe_full.onnx",
            F0Detector::CrepeTiny => "crepe_tiny.onnx",
            _ => return,
        };
        for (model, chosen) in setup::SHARED.iter().zip(&mut self.setup.chosen) {
            if model.file == file {
                *chosen = true;
            }
        }
        self.setup.open = true;
    }

    fn start_setup(&mut self, ctx: &egui::Context) {
        let chosen: Vec<&'static str> = setup::SHARED
            .iter()
            .zip(self.setup.chosen)
            .filter(|(model, chosen)| model.required || *chosen)
            .map(|(model, _)| model.file)
            .collect();
        let progress = Arc::new(Progress::default());
        let shared = progress.clone();
        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();
        let root = crate::runtime::models_root().unwrap_or_else(setup::target_root);
        let started = std::thread::Builder::new()
            .name("shared-models".into())
            .spawn(move || {
                let result =
                    setup::download(&root, &chosen, &shared).map_err(|error| format!("{error:#}"));
                let _ = tx.send(result);
                ctx.request_repaint();
            });
        match started {
            Ok(_) => {
                self.setup.progress = Some(progress);
                self.setup.done = Some(rx);
                self.setup.problem = None;
            }
            Err(error) => self.setup.problem = Some(error.to_string()),
        }
    }

    /// Runs every frame, so the download finishes while the dialog is hidden too.
    pub(super) fn poll_setup(&mut self, ctx: &egui::Context) {
        if !self.setup.downloading() {
            return;
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
        let Some(result) = self.setup.done.as_ref().and_then(|rx| rx.try_recv().ok()) else {
            return;
        };
        self.setup.done = None;
        self.setup.progress = None;
        match result {
            Ok(()) if crate::runtime::models_root().is_some() => {
                self.setup.open = false;
                self.toast(
                    t("Setup finished. Import a voice or press Find online to add one.").into(),
                    false,
                );
            }
            Ok(()) => {
                self.setup.problem = Some(tf(
                    "The models were saved to {}, but the app does not look there. Start it from its install folder.",
                    &[&setup::target_root().display()],
                ));
            }
            Err(problem) => {
                // Models that failed their check are not kept; say which, wherever
                // the user is looking.
                self.toast(
                    tf(
                        "Download failed, nothing was installed for: {}",
                        &[&problem],
                    ),
                    true,
                );
                self.setup.problem = Some(problem);
            }
        }
    }

    pub(super) fn setup_dialog(&mut self, ctx: &egui::Context) {
        let downloading = self.setup.downloading();
        let folder = shared_folder();
        let (mut start, mut later, mut cancel, mut hide) = (false, false, false, false);
        let (_, dismissed) = m3::dialog(ctx, "setup-dialog", t("Set up the voice engine"), |ui| {
            ui.add(
                egui::Label::new(
                    egui::RichText::new(t(
                        "Every voice uses two shared models to understand your speech. Download them once and all voices you add will work.",
                    ))
                    .color(m3::ON_SURFACE_VARIANT),
                )
                .wrap(),
            );
            ui.add_space(8.0);
            let glyphs = [icons::VOICE, icons::GRAPHIC_EQ, icons::SPEED, icons::MEMORY];
            let mut size = 0;
            for ((model, glyph), chosen) in
                setup::SHARED.iter().zip(glyphs).zip(&mut self.setup.chosen)
            {
                let ready = setup::installed(&folder, model);
                if model.required {
                    *chosen = true;
                }
                model_row(ui, glyph, model, ready, chosen, downloading);
                if *chosen && !ready {
                    size += model.bytes;
                }
            }
            ui.add_space(8.0);
            if let Some((received, total, fraction)) = self.setup.amounts() {
                linear_progress(ui, fraction);
                ui.horizontal(|ui| {
                    ui.label(m3::supporting(&tf(
                        "Downloading… {} of {}",
                        &[&megabytes(received), &megabytes(total)],
                    )));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(m3::supporting(&format!("{:.0}%", fraction * 100.0)));
                    });
                });
            } else if let Some(problem) = &self.setup.problem {
                egui::Frame::new()
                    .fill(m3::ERROR_CONTAINER)
                    .corner_radius(12)
                    .inner_margin(egui::Margin::symmetric(16, 10))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.label(m3::icon(icons::ERROR, 20.0, m3::ON_ERROR_CONTAINER));
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(problem).color(m3::ON_ERROR_CONTAINER),
                                )
                                .wrap(),
                            );
                        });
                    });
            } else if size > 0 {
                ui.label(m3::supporting(&tf(
                    "{} in total, from Hugging Face. Saved in {}.",
                    &[&megabytes(size), &folder.display()],
                )));
            } else {
                ui.label(m3::supporting(t("Everything you picked is installed.")));
            }
            ui.label(m3::supporting(t(
                "Each file is checked against its published size and SHA-256 before it is installed.",
            )));
            m3::actions(ui, |ui| {
                if downloading {
                    hide = ui
                        .add(m3::Button::tonal(t("Hide")))
                        .on_hover_text(t("Keep downloading while you use the app"))
                        .clicked();
                    cancel = ui.add(m3::Button::text(t("Cancel"))).clicked();
                } else {
                    let label = if self.setup.problem.is_some() {
                        t("Try again")
                    } else {
                        t("Download")
                    };
                    start = ui
                        .add_enabled(size > 0, m3::Button::filled(label))
                        .clicked();
                    later = ui
                        .add(m3::Button::text(if size > 0 {
                            t("Later")
                        } else {
                            t("Close")
                        }))
                        .on_hover_text(t("You can finish this from Settings before adding a voice"))
                        .clicked();
                }
            });
        });
        if start {
            self.start_setup(ctx);
        }
        if cancel {
            if let Some(progress) = &self.setup.progress {
                progress.cancel.store(true, Ordering::Relaxed);
            }
        }
        if later || hide || dismissed {
            self.setup.open = false;
        }
    }

    /// Progress strip above the page while the download runs with the dialog hidden.
    pub(super) fn setup_notice(&mut self, ui: &mut egui::Ui) {
        if self.setup.open {
            return;
        }
        let Some((received, total, fraction)) = self.setup.amounts() else {
            return;
        };
        let mut show = false;
        egui::Frame::new()
            .fill(m3::SURFACE_CONTAINER_HIGH)
            .corner_radius(12)
            .inner_margin(egui::Margin {
                left: 16,
                right: 8,
                top: 4,
                bottom: 12,
            })
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(m3::icon(icons::GRAPHIC_EQ, 20.0, m3::PRIMARY));
                    ui.label(tf(
                        "Downloading the voice engine… {} of {} ({}%)",
                        &[
                            &megabytes(received),
                            &megabytes(total),
                            &format!("{:.0}", fraction * 100.0),
                        ],
                    ));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        show = ui.add(m3::Button::text(t("Show"))).clicked();
                    });
                });
                linear_progress(ui, fraction);
            });
        ui.add_space(4.0);
        if show {
            self.setup.open = true;
        }
    }

    /// Settings card entry for the shared models.
    pub(super) fn setup_status(&mut self, ui: &mut egui::Ui) {
        let ready = crate::runtime::models_root().is_some();
        m3::field_label(ui, t("Voice engine"));
        ui.horizontal(|ui| {
            if let Some((_, _, fraction)) = self.setup.amounts() {
                ui.add(egui::Spinner::new().color(m3::PRIMARY));
                ui.label(tf(
                    "Downloading… {}%",
                    &[&format!("{:.0}", fraction * 100.0)],
                ));
                if ui.add(m3::Button::text(t("Show"))).clicked() {
                    self.setup.open = true;
                }
            } else if ready {
                ui.label(m3::icon(icons::CHECK_CIRCLE, 20.0, m3::SUCCESS));
                ui.label(t("ContentVec and RMVPE are installed"));
                if ui.add(m3::Button::text(t("Manage models"))).clicked() {
                    self.setup.open = true;
                }
            } else {
                ui.label(m3::icon(icons::ERROR, 20.0, m3::ERROR));
                ui.label(t("Not installed"));
                if ui.add(m3::Button::tonal(t("Download"))).clicked() {
                    self.setup.open = true;
                }
            }
        });
        ui.label(m3::supporting(t(
            "The two shared models every voice needs: one hears your words, the other follows your pitch. Manage models also adds the optional CREPE pitch detectors.",
        )));
    }
}

/// A model with its checkbox, icon, name and size, what it does, and its badges
/// (required, recommended, installed).
fn model_row(
    ui: &mut egui::Ui,
    glyph: char,
    model: &setup::SharedModel,
    ready: bool,
    chosen: &mut bool,
    downloading: bool,
) {
    ui.horizontal(|ui| {
        ui.add_enabled_ui(!model.required && !ready && !downloading, |ui| {
            let mut ticked = *chosen || ready;
            if m3::checkbox(ui, &mut ticked, model.title).changed() {
                *chosen = ticked;
            }
        });
        let (tile, _) = ui.allocate_exact_size(Vec2::splat(40.0), Sense::hover());
        ui.painter()
            .rect_filled(tile, 12.0, m3::SECONDARY_CONTAINER);
        m3::paint_icon(
            ui,
            tile.center(),
            glyph,
            24.0,
            false,
            m3::ON_SECONDARY_CONTAINER,
        );
        ui.add_space(4.0);
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 2.0;
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!("{} · {}", model.title, megabytes(model.bytes)))
                        .font(FontId::new(15.0, m3::medium()))
                        .color(m3::ON_SURFACE),
                );
                if ready {
                    m3::badge(
                        ui,
                        t("Installed"),
                        m3::SUCCESS.gamma_multiply(0.2),
                        m3::SUCCESS,
                    );
                } else if model.required {
                    m3::badge(
                        ui,
                        t("Required"),
                        m3::PRIMARY_CONTAINER,
                        m3::ON_PRIMARY_CONTAINER,
                    );
                } else if model.recommended {
                    m3::badge(
                        ui,
                        t("Recommended"),
                        m3::SECONDARY_CONTAINER,
                        m3::ON_SECONDARY_CONTAINER,
                    );
                } else {
                    m3::badge(
                        ui,
                        t("Optional"),
                        m3::SURFACE_CONTAINER_HIGHEST,
                        m3::ON_SURFACE_VARIANT,
                    );
                }
            });
            ui.add(egui::Label::new(m3::supporting(t(model.purpose))).wrap());
        });
    });
    ui.add_space(2.0);
}

/// M3 linear progress indicator: active track, a gap, the inactive track and its
/// stop dot.
fn linear_progress(ui: &mut egui::Ui, fraction: f32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 8.0), Sense::hover());
    let track = Rect::from_center_size(rect.center(), Vec2::new(rect.width(), 4.0));
    let x = egui::lerp(track.left()..=track.right(), fraction.clamp(0.0, 1.0));
    let painter = ui.painter();
    if x + 4.0 < track.right() {
        painter.rect_filled(
            Rect::from_min_max(
                egui::pos2((x + 4.0).max(track.left()), track.top()),
                track.max,
            ),
            2.0,
            m3::SECONDARY_CONTAINER,
        );
        painter.circle_filled(
            egui::pos2(track.right() - 2.0, track.center().y),
            2.0,
            m3::PRIMARY,
        );
    }
    if x > track.left() {
        painter.rect_filled(
            Rect::from_min_max(track.min, egui::pos2(x, track.bottom())),
            2.0,
            m3::PRIMARY,
        );
    }
}
