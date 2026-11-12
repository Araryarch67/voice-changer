//! First-start dialog that downloads the shared ContentVec and RMVPE models.
use super::{icons, m3, VoiceApp};
use crate::{catalog::Progress, setup};
use eframe::egui::{self, FontId, Rect, Sense, Vec2};
use std::sync::{atomic::Ordering, mpsc, Arc};

#[derive(Default)]
pub(super) struct Setup {
    pub open: bool,
    progress: Option<Arc<Progress>>,
    done: Option<mpsc::Receiver<Result<(), String>>>,
    problem: Option<String>,
}

impl Setup {
    /// Opens on start when the shared models are missing.
    pub fn on_start() -> Self {
        Self {
            open: crate::runtime::models_root().is_none(),
            ..Default::default()
        }
    }
    fn downloading(&self) -> bool {
        self.done.is_some()
    }
}

fn megabytes(bytes: u64) -> String {
    format!("{:.0} MB", bytes as f64 / 1_048_576.0)
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

    fn start_setup(&mut self, ctx: &egui::Context) {
        let progress = Arc::new(Progress::default());
        let shared = progress.clone();
        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();
        let root = setup::target_root();
        let started = std::thread::Builder::new()
            .name("shared-models".into())
            .spawn(move || {
                let result = setup::download(&root, &shared).map_err(|error| format!("{error:#}"));
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

    pub(super) fn setup_dialog(&mut self, ctx: &egui::Context) {
        if let Some(result) = self.setup.done.as_ref().and_then(|rx| rx.try_recv().ok()) {
            self.setup.done = None;
            self.setup.progress = None;
            match result {
                Ok(()) if crate::runtime::models_root().is_some() => {
                    self.setup.open = false;
                    self.message = Some(
                        "Setup finished. Import a voice or press Find online to add one.".into(),
                    );
                    return;
                }
                Ok(()) => {
                    self.setup.problem = Some(format!(
                        "The models were saved to {}, but the app does not look there. Start it from its install folder.",
                        setup::target_root().display()
                    ))
                }
                Err(problem) => self.setup.problem = Some(problem),
            }
        }
        let downloading = self.setup.downloading();
        let (mut start, mut later, mut cancel) = (false, false, false);
        let (_, dismissed) = m3::dialog(ctx, "setup-dialog", "Set up the voice engine", |ui| {
            ui.add(
                egui::Label::new(
                    egui::RichText::new(
                        "Every voice uses two shared models to understand your speech. Download them once and all voices you add will work.",
                    )
                    .color(m3::ON_SURFACE_VARIANT),
                )
                .wrap(),
            );
            ui.add_space(8.0);
            let folder = setup::target_root().join("shared");
            for (model, glyph) in setup::SHARED.iter().zip([icons::VOICE, icons::GRAPHIC_EQ]) {
                let ready = folder.join(model.file).is_file();
                model_row(ui, glyph, model, ready);
            }
            ui.add_space(8.0);
            if let Some(progress) = &self.setup.progress {
                let received = progress.received.load(Ordering::Relaxed);
                let total = progress.total.load(Ordering::Relaxed).max(1);
                let fraction = received as f32 / total as f32;
                linear_progress(ui, fraction);
                ui.horizontal(|ui| {
                    ui.label(m3::supporting(&format!(
                        "Downloading… {} of {}",
                        megabytes(received),
                        megabytes(total)
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
            } else {
                ui.label(m3::supporting(&format!(
                    "{} in total, from Hugging Face. Saved in {}.",
                    megabytes(setup::total_bytes()),
                    folder.display()
                )));
            }
            m3::actions(ui, |ui| {
                if downloading {
                    cancel = ui.add(m3::Button::text("Cancel")).clicked();
                } else {
                    let label = if self.setup.problem.is_some() {
                        "Try again"
                    } else {
                        "Download"
                    };
                    start = ui.add(m3::Button::filled(label)).clicked();
                    later = ui
                        .add(m3::Button::text("Later"))
                        .on_hover_text("You can finish this from Settings before adding a voice")
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
        if downloading {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        } else if later || dismissed {
            self.setup.open = false;
        }
    }

    /// Settings card entry for the shared models.
    pub(super) fn setup_status(&mut self, ui: &mut egui::Ui) {
        let ready = crate::runtime::models_root().is_some();
        m3::field_label(ui, "Voice engine");
        ui.horizontal(|ui| {
            if ready {
                ui.label(m3::icon(icons::CHECK_CIRCLE, 20.0, m3::SUCCESS));
                ui.label("ContentVec and RMVPE are installed");
            } else {
                ui.label(m3::icon(icons::ERROR, 20.0, m3::ERROR));
                ui.label("Not installed");
                if ui.add(m3::Button::tonal("Download")).clicked() {
                    self.setup.open = true;
                }
            }
        });
        ui.label(m3::supporting(
            "The two shared models every voice needs: one hears your words, the other follows your pitch.",
        ));
    }
}

/// Two-line row: an icon tile, the model's name and size, what it does, and a check
/// once it is there.
fn model_row(ui: &mut egui::Ui, glyph: char, model: &setup::SharedModel, ready: bool) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 64.0), Sense::hover());
    let tile = Rect::from_min_size(
        egui::pos2(rect.left(), rect.center().y - 20.0),
        Vec2::splat(40.0),
    );
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
    let x = tile.right() + 16.0;
    let width = rect.right() - x - 36.0;
    let name = m3::elided(
        ui,
        &format!("{} · {}", model.title, megabytes(model.bytes)),
        FontId::new(16.0, m3::medium()),
        m3::ON_SURFACE,
        width,
    );
    let purpose = m3::elided(
        ui,
        model.purpose,
        FontId::proportional(12.0),
        m3::ON_SURFACE_VARIANT,
        width,
    );
    ui.painter()
        .galley(egui::pos2(x, rect.top() + 12.0), name, m3::ON_SURFACE);
    ui.painter().galley(
        egui::pos2(x, rect.top() + 34.0),
        purpose,
        m3::ON_SURFACE_VARIANT,
    );
    if ready {
        m3::paint_icon(
            ui,
            egui::pos2(rect.right() - 12.0, rect.center().y),
            icons::CHECK_CIRCLE,
            24.0,
            true,
            m3::SUCCESS,
        );
    }
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
