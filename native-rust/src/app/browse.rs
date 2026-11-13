//! "Find voices online": search voice-models.com and add a voice in one step
//! (download, unpack the .pth and .index, convert).
use super::{icons, m3, VoiceApp};
use crate::catalog::{self, Entry, Progress};
use crate::i18n::{t, tf};
use eframe::egui;
use std::sync::{atomic::Ordering, mpsc, Arc};

type Found = Result<(catalog::Page, bool), String>;

#[derive(Default)]
pub(super) struct Browse {
    pub open: bool,
    query: String,
    /// The query the results belong to.
    searched: String,
    page: u32,
    results: Vec<Entry>,
    more: bool,
    pending: Option<mpsc::Receiver<Found>>,
    problem: Option<String>,
    /// The voice being added and its download progress.
    pub download: Option<(String, Arc<Progress>)>,
    listed_once: bool,
}

impl VoiceApp {
    pub(super) fn browse_button(&mut self, ui: &mut egui::Ui) {
        if ui
            .add_enabled(
                !self.running(),
                m3::Button::outlined(t("Find online")).icon(icons::SEARCH),
            )
            .on_hover_text(t("Search voice-models.com and add a voice in one step"))
            .clicked()
            && self.shared_models_ready()
        {
            self.browse.open = true;
        }
    }

    fn search_online(&mut self, ctx: &egui::Context, append: bool) {
        let browse = &mut self.browse;
        if !append {
            browse.page = 1;
            browse.searched = browse.query.trim().to_owned();
        } else {
            browse.page += 1;
        }
        let (query, page) = (browse.searched.clone(), browse.page);
        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();
        let started = std::thread::Builder::new()
            .name("voice-search".into())
            .spawn(move || {
                let found = catalog::search(&query, page)
                    .map(|found| (found, append))
                    .map_err(|error| format!("{error:#}"));
                let _ = tx.send(found);
                ctx.request_repaint();
            });
        match started {
            Ok(_) => {
                browse.pending = Some(rx);
                browse.problem = None;
            }
            Err(error) => browse.problem = Some(error.to_string()),
        }
    }

    fn add_online_voice(&mut self, entry: &Entry) {
        if !entry.downloadable() {
            return;
        }
        let link = entry.link.clone();
        if self.work.is_some() {
            self.error = Some(t("Wait for the current task to finish, then add the voice.").into());
            return;
        }
        let name = catalog::voice_name(&entry.title);
        let progress = Arc::new(Progress::default());
        let shared = progress.clone();
        let title = name.clone();
        self.begin_work(move || {
            let root = crate::runtime::models_root().ok_or_else(|| {
                anyhow::anyhow!("The voice engine (shared/contentvec.onnx and shared/rmvpe.onnx) is not set up yet; download it in Settings under Processing")
            })?;
            let folder = root.join(".downloads");
            let url = catalog::resolve(&link)?;
            let file = catalog::download(&url, &folder, &shared)?;
            let imported = crate::import::import_download(&file, &title);
            let _ = std::fs::remove_file(&file);
            let _ = std::fs::remove_dir(&folder);
            let voice = crate::config::ModelManifest::load(&imported?)?;
            Ok(tf(
                "Added {}{}.",
                &[
                    &voice.name,
                    &if voice.feature_bank.is_some() {
                        t(" with its index")
                    } else {
                        t(" (it came without an index)")
                    },
                ],
            ))
        });
        if self.work.is_some() {
            self.browse.download = Some((name, progress));
        }
    }

    pub(super) fn browse_dialog(&mut self, ctx: &egui::Context) {
        if !self.browse.listed_once && self.browse.pending.is_none() {
            // Opening shows the newest voices, so the list is never empty.
            self.browse.listed_once = true;
            self.search_online(ctx, false);
        }
        if let Some(found) = self
            .browse
            .pending
            .as_ref()
            .and_then(|rx| rx.try_recv().ok())
        {
            self.browse.pending = None;
            match found {
                Ok((page, append)) => {
                    if !append {
                        self.browse.results.clear();
                    }
                    self.browse.results.extend(page.entries);
                    self.browse.more = page.has_more;
                }
                Err(problem) => self.browse.problem = Some(problem),
            }
        }
        let busy = self.work.is_some();
        let (mut search, mut more, mut close) = (false, false, false);
        let mut add: Option<Entry> = None;
        let mut open_page: Option<String> = None;
        let (_, dismissed) = m3::dialog(ctx, "browse-dialog", t("Find voices online"), |ui| {
            let browse = &mut self.browse;
            ui.horizontal(|ui| {
                let width = ui.available_width() - 110.0;
                let field = ui.add(m3::text_edit(
                    &mut browse.query,
                    t("Character, singer, game…"),
                    width,
                ));
                let entered = field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                let clicked = ui
                    .add_enabled(
                        browse.pending.is_none(),
                        m3::Button::filled(t("Search")).icon(icons::SEARCH),
                    )
                    .clicked();
                search = (entered || clicked) && browse.pending.is_none();
            });
            if let Some(problem) = &browse.problem {
                ui.label(egui::RichText::new(problem).color(m3::ERROR));
            } else if browse.pending.is_some() && browse.results.is_empty() {
                ui.horizontal(|ui| {
                    ui.add(egui::Spinner::new().color(m3::PRIMARY));
                    ui.label(m3::supporting(t("Searching voice-models.com…")));
                });
            } else if browse.results.is_empty() {
                ui.label(m3::supporting(t("No voices found. Try another name.")));
            } else {
                ui.label(m3::supporting(&if browse.searched.is_empty() {
                    tf("{} newest voices", &[&browse.results.len()])
                } else {
                    tf(
                        "{} voices for \"{}\"",
                        &[&browse.results.len(), &browse.searched],
                    )
                }));
            }
            egui::ScrollArea::vertical()
                .max_height(400.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    for entry in &browse.results {
                        ui.add_space(4.0);
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(&entry.title).color(m3::ON_SURFACE),
                            )
                            .wrap(),
                        );
                        let mut details: Vec<String> = Vec::new();
                        details.extend(entry.size.clone());
                        details.extend(entry.uploader.as_ref().map(|by| tf("by {}", &[by])));
                        details.extend(entry.unsupported().map(str::to_owned));
                        ui.horizontal(|ui| {
                            ui.label(m3::supporting(&details.join(" · ")));
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .add(m3::Button::text(t("Page")))
                                        .on_hover_text(&entry.page)
                                        .clicked()
                                    {
                                        open_page = Some(entry.page.clone());
                                    }
                                    if entry.unsupported().is_none()
                                        && ui
                                            .add_enabled(
                                                !busy,
                                                m3::Button::tonal(t("Add voice")).icon(icons::ADD),
                                            )
                                            .clicked()
                                    {
                                        add = Some(entry.clone());
                                    }
                                },
                            );
                        });
                        ui.separator();
                    }
                    if browse.more {
                        more = ui
                            .add_enabled(
                                browse.pending.is_none(),
                                m3::Button::text(t("More results")),
                            )
                            .clicked();
                    }
                });
            if let Some((name, progress)) = &browse.download {
                ui.add_space(6.0);
                let (received, total) = (
                    progress.received.load(Ordering::Relaxed),
                    progress.total.load(Ordering::Relaxed),
                );
                let mb = |bytes: u64| bytes as f32 / 1_048_576.0;
                let (fraction, text) = if total > 0 && received >= total {
                    (1.0, tf("Converting {}…", &[name]))
                } else if total > 0 {
                    (
                        received as f32 / total as f32,
                        tf(
                            "Downloading {}: {} of {} MB",
                            &[
                                name,
                                &format!("{:.0}", mb(received)),
                                &format!("{:.0}", mb(total)),
                            ],
                        ),
                    )
                } else {
                    (
                        0.0,
                        tf(
                            "Downloading {}: {} MB",
                            &[name, &format!("{:.0}", mb(received))],
                        ),
                    )
                };
                ui.add(egui::ProgressBar::new(fraction).text(text));
                if received < total.max(1)
                    && ui.add(m3::Button::text(t("Cancel download"))).clicked()
                {
                    progress.cancel.store(true, Ordering::Relaxed);
                }
            }
            ui.label(m3::supporting(
                t("Voices come from voice-models.com and are hosted by their uploaders. Many imitate real people; use them only where you have the right to. Hugging Face, Google Drive and MediaFire links download here; Mega links open in the browser."),
            ));
            m3::actions(ui, |ui| {
                close = ui.add(m3::Button::text(t("Close"))).clicked();
            });
        });
        if search {
            self.search_online(ctx, false);
        }
        if more {
            self.search_online(ctx, true);
        }
        if let Some(entry) = add {
            self.add_online_voice(&entry);
        }
        if let Some(page) = open_page {
            open_in_browser(&page);
        }
        if self.browse.pending.is_some() || self.browse.download.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        self.browse.open = !(close || dismissed);
    }
}

/// Opens `url` in the default browser.
fn open_in_browser(url: &str) {
    let opened = if cfg!(target_os = "windows") {
        std::process::Command::new("explorer").arg(url).spawn()
    } else {
        std::process::Command::new("xdg-open").arg(url).spawn()
    };
    if let Err(error) = opened {
        log::warn!("Could not open {url}: {error}");
    }
}
