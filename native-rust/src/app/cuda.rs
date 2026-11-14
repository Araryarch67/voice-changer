//! CUDA setup dialog: when GPU (CUDA) cannot start because NVIDIA's CUDA 12 or
//! cuDNN 9 libraries are missing, offers to download them (`crate::cuda_setup`)
//! with NVIDIA's license note, a progress bar, and a check that CUDA starts after.
use super::setup::{linear_progress, megabytes};
use super::{icons, m3, VoiceApp};
use crate::cuda_setup::{self, CudaProgress};
use crate::i18n::{t, tf};
use eframe::egui;
use std::sync::{atomic::Ordering, mpsc, Arc};

/// How a finished setup went.
enum Outcome {
    /// CUDA starts now.
    Ready,
    /// The libraries are in place, but this process cannot use them until it starts
    /// again (the provider failed earlier in a way it keeps).
    Restart,
    Failed(String),
}

#[derive(Default)]
pub(super) struct CudaSetup {
    pub open: bool,
    /// Whether the download is offered; checked once, and again after a setup.
    needed: Option<bool>,
    progress: Option<Arc<CudaProgress>>,
    done: Option<mpsc::Receiver<Outcome>>,
    outcome: Option<Outcome>,
}
impl CudaSetup {
    fn downloading(&self) -> bool {
        self.done.is_some()
    }
    /// Bytes received, the total, the fraction done, and whether it is unpacking.
    fn amounts(&self) -> Option<(u64, u64, f32, bool)> {
        let progress = self.progress.as_ref()?;
        let received = progress.download.received.load(Ordering::Relaxed);
        let total = progress.download.total.load(Ordering::Relaxed).max(1);
        let unpacking = progress.unpacking.load(Ordering::Relaxed);
        Some((received, total, received as f32 / total as f32, unpacking))
    }
}

/// Whether `error` is CUDA failing to start, which this setup can fix.
fn cuda_error(error: &str) -> bool {
    error.contains("CUDA could not start")
}

impl VoiceApp {
    /// Whether CUDA libraries are missing and can be downloaded. Checked once.
    pub(super) fn cuda_needed(&mut self) -> bool {
        if self.cuda.downloading() || matches!(self.cuda.outcome, Some(Outcome::Restart)) {
            return true;
        }
        *self.cuda.needed.get_or_insert_with(|| {
            let needed = cuda_setup::needed();
            if needed {
                log::info!(
                    "CUDA libraries missing: {}",
                    crate::runtime::missing_cuda_libraries().join(", ")
                );
            }
            needed
        })
    }

    fn start_cuda_setup(&mut self, ctx: &egui::Context) {
        let Some(folder) = cuda_setup::folder() else {
            return;
        };
        let progress = Arc::new(CudaProgress::default());
        let shared = progress.clone();
        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();
        let started = std::thread::Builder::new()
            .name("cuda-setup".into())
            .spawn(move || {
                let outcome = match cuda_setup::download(&folder, &shared) {
                    Err(error) => Outcome::Failed(format!("{error:#}")),
                    Ok(()) => match crate::runtime::check_cuda() {
                        Ok(()) => Outcome::Ready,
                        Err(error) => {
                            log::warn!("CUDA still does not start: {error:#}");
                            if crate::runtime::missing_cuda_libraries().is_empty() {
                                Outcome::Restart
                            } else {
                                Outcome::Failed(format!("{error:#}"))
                            }
                        }
                    },
                };
                let _ = tx.send(outcome);
                ctx.request_repaint();
            });
        match started {
            Ok(_) => {
                self.cuda.progress = Some(progress);
                self.cuda.done = Some(rx);
                self.cuda.outcome = None;
            }
            Err(error) => self.cuda.outcome = Some(Outcome::Failed(error.to_string())),
        }
    }

    /// Runs every frame, so the download finishes while the dialog is hidden too.
    pub(super) fn poll_cuda_setup(&mut self, ctx: &egui::Context) {
        if !self.cuda.downloading() {
            return;
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
        let Some(outcome) = self.cuda.done.as_ref().and_then(|rx| rx.try_recv().ok()) else {
            return;
        };
        self.cuda.done = None;
        self.cuda.progress = None;
        match &outcome {
            Outcome::Ready => {
                self.cuda.needed = Some(false);
                if self.error.as_deref().is_some_and(cuda_error) {
                    self.error = None;
                }
                self.toast(
                    t("CUDA is ready. Press Start to use the GPU.").into(),
                    false,
                );
            }
            Outcome::Restart => self.toast(
                t("CUDA is installed. Close and reopen the app to use it.").into(),
                false,
            ),
            Outcome::Failed(problem) => {
                self.toast(tf("CUDA setup failed: {}", &[problem]), true);
            }
        }
        self.cuda.outcome = Some(outcome);
    }

    pub(super) fn cuda_dialog(&mut self, ctx: &egui::Context) {
        let downloading = self.cuda.downloading();
        let folder = cuda_setup::folder();
        let remaining = folder.as_deref().map_or(0, cuda_setup::remaining_bytes);
        let (mut start, mut close, mut cancel) = (false, false, false);
        let (_, dismissed) = m3::dialog(ctx, "cuda-dialog", t("Set up CUDA"), |ui| {
            ui.add(
                egui::Label::new(
                    egui::RichText::new(t(
                        "GPU (CUDA) needs NVIDIA's CUDA 12 and cuDNN 9 libraries, and some are missing. The app can download just the parts it uses from NVIDIA and keep them in its own folder; nothing else on the computer changes.",
                    ))
                    .color(m3::ON_SURFACE_VARIANT),
                )
                .wrap(),
            );
            ui.add_space(8.0);
            for package in cuda_setup::packages() {
                ui.horizontal(|ui| {
                    ui.label(m3::icon(icons::MEMORY, 20.0, m3::PRIMARY));
                    ui.label(format!("{} · {}", package.title, megabytes(package.bytes)));
                });
            }
            ui.add_space(8.0);
            egui::Frame::new()
                .fill(m3::SURFACE_CONTAINER_HIGHEST)
                .corner_radius(12)
                .inner_margin(egui::Margin::symmetric(16, 10))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        ui.label(m3::icon(icons::INFO, 20.0, m3::ON_SURFACE_VARIANT));
                        ui.add(
                            egui::Label::new(m3::supporting(t(
                                "These libraries are NVIDIA software, downloaded from NVIDIA's servers. By downloading them you accept NVIDIA's license agreements:",
                            )))
                            .wrap(),
                        );
                    });
                    for (name, url) in cuda_setup::LICENSES {
                        ui.hyperlink_to(name, url);
                    }
                });
            ui.add_space(8.0);
            match (self.cuda.amounts(), &self.cuda.outcome) {
                (Some((received, total, fraction, unpacking)), _) => {
                    linear_progress(ui, fraction);
                    ui.label(m3::supporting(&if unpacking {
                        t("Unpacking…").to_owned()
                    } else {
                        tf(
                            "Downloading… {} of {}",
                            &[&megabytes(received), &megabytes(total)],
                        )
                    }));
                }
                (None, Some(Outcome::Ready)) => {
                    ui.horizontal(|ui| {
                        ui.label(m3::icon(icons::CHECK_CIRCLE, 20.0, m3::SUCCESS));
                        ui.label(t("CUDA is ready. Press Start to use the GPU."));
                    });
                }
                (None, Some(Outcome::Restart)) => {
                    ui.horizontal(|ui| {
                        ui.label(m3::icon(icons::REFRESH, 20.0, m3::PRIMARY));
                        ui.add(
                            egui::Label::new(t(
                                "CUDA is installed. Close and reopen the app to use it.",
                            ))
                            .wrap(),
                        );
                    });
                }
                (None, Some(Outcome::Failed(problem))) => {
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
                }
                (None, None) => {}
            }
            if let Some(folder) = &folder {
                ui.label(m3::supporting(&tf(
                    "{} to download, about twice that on disk. Saved in {}; delete that folder to remove them.",
                    &[&megabytes(remaining), &folder.display()],
                )));
            }
            ui.label(m3::supporting(t(
                "Each archive is checked against NVIDIA's published size and SHA-256 before anything is taken from it. CPU and DirectML keep working without CUDA.",
            )));
            m3::actions(ui, |ui| {
                if downloading {
                    close = ui
                        .add(m3::Button::tonal(t("Hide")))
                        .on_hover_text(t("Keep downloading while you use the app"))
                        .clicked();
                    cancel = ui.add(m3::Button::text(t("Cancel"))).clicked();
                } else if remaining > 0 && folder.is_some() {
                    start = ui
                        .add(m3::Button::filled(t("Accept and download")))
                        .on_hover_text(t(
                            "Downloads NVIDIA's CUDA libraries for the GPU (CUDA) device",
                        ))
                        .clicked();
                    close = ui
                        .add(m3::Button::text(t("Later")))
                        .on_hover_text(t("You can set up CUDA later from Settings"))
                        .clicked();
                } else {
                    close = ui.add(m3::Button::text(t("Close"))).clicked();
                }
            });
        });
        if start {
            self.start_cuda_setup(ctx);
        }
        if cancel {
            if let Some(progress) = &self.cuda.progress {
                progress.download.cancel.store(true, Ordering::Relaxed);
            }
        }
        if close || dismissed {
            self.cuda.open = false;
        }
    }

    /// Settings entry, under the device choice, while CUDA libraries are missing or
    /// being set up.
    pub(super) fn cuda_status(&mut self, ui: &mut egui::Ui) {
        if !self.cuda_needed() {
            return;
        }
        m3::field_label(ui, t("CUDA libraries"));
        ui.horizontal(|ui| {
            if let Some((_, _, fraction, _)) = self.cuda.amounts() {
                ui.add(egui::Spinner::new().color(m3::PRIMARY));
                ui.label(tf(
                    "Downloading… {}%",
                    &[&format!("{:.0}", fraction * 100.0)],
                ));
                if ui.add(m3::Button::text(t("Show"))).clicked() {
                    self.cuda.open = true;
                }
            } else if matches!(self.cuda.outcome, Some(Outcome::Restart)) {
                ui.label(m3::icon(icons::REFRESH, 20.0, m3::PRIMARY));
                ui.label(t("Installed; restart the app to use them"));
            } else {
                ui.label(m3::icon(icons::ERROR, 20.0, m3::ERROR));
                ui.label(t("Not installed"));
                if ui.add(m3::Button::tonal(t("Set up CUDA"))).clicked() {
                    self.cuda.open = true;
                }
            }
        });
        ui.label(m3::supporting(t(
            "GPU (CUDA) needs NVIDIA's CUDA 12 and cuDNN 9 libraries. Set up CUDA downloads them for this app (over 1 GB), or choose another device.",
        )));
    }

    /// Under the error banner, when the error is CUDA failing for missing libraries.
    pub(super) fn cuda_offer(&mut self, ui: &mut egui::Ui) {
        if !self.error.as_deref().is_some_and(cuda_error) || !self.cuda_needed() {
            return;
        }
        ui.horizontal(|ui| {
            if ui
                .add(m3::Button::filled(t("Set up CUDA")).icon(icons::MEMORY))
                .on_hover_text(t(
                    "Downloads NVIDIA's CUDA libraries for the GPU (CUDA) device",
                ))
                .clicked()
            {
                self.cuda.open = true;
            }
            ui.label(m3::supporting(t(
                "CUDA's NVIDIA libraries are missing; the app can download them.",
            )));
        });
        ui.add_space(4.0);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn recognises_the_cuda_start_error() {
        // rvc.rs words the error this way; the offer looks for it.
        assert!(super::cuda_error(
            "Load RVC ONNX: GPU (CUDA) is selected but CUDA could not start; ..."
        ));
        assert!(!super::cuda_error(
            "GPU (DirectML) is selected but DirectML could not start"
        ));
    }
}
