//! Development-only screenshot runner: cargo run --example capture -- output.png [width height [click_x click_y]...]
//! A click whose x starts with `r` (r318 290) uses the right button, for context menus.
#[cfg(feature = "gui")]
fn main() -> eframe::Result {
    use eframe::egui;
    struct Capture {
        app: voice_changer::app::VoiceApp,
        path: String,
        size: egui::Vec2,
        frames: usize,
        clicks: Vec<(egui::Pos2, egui::PointerButton)>,
    }
    impl eframe::App for Capture {
        // Optional clicks before the screenshot, one every eight frames, to capture
        // another page, a dialog or an open menu.
        fn raw_input_hook(&mut self, _ctx: &egui::Context, raw: &mut egui::RawInput) {
            let Some(step) = self.frames.checked_sub(3) else {
                return;
            };
            let Some(&(pos, button)) = self.clicks.get(step / 8) else {
                return;
            };
            match step % 8 {
                0 => raw.events.push(egui::Event::PointerMoved(pos)),
                pressed @ (1 | 2) => raw.events.push(egui::Event::PointerButton {
                    pos,
                    button,
                    pressed: pressed == 1,
                    modifiers: Default::default(),
                }),
                _ => {}
            }
        }
        fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
            // A window size saved by an earlier run wins over the one asked for at
            // creation, so ask again.
            if self.frames == 0 {
                ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(self.size));
            }
            self.app.update(ctx, frame);
            for event in ctx.input(|i| i.events.clone()) {
                if let egui::Event::Screenshot { image, .. } = event {
                    image::save_buffer(
                        &self.path,
                        image.as_raw(),
                        image.width() as u32,
                        image.height() as u32,
                        image::ColorType::Rgba8,
                    )
                    .unwrap();
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
            self.frames += 1;
            // CAPTURE_FRAMES waits longer, e.g. for an online search to come back.
            let wait = std::env::var("CAPTURE_FRAMES")
                .ok()
                .and_then(|frames| frames.parse().ok())
                .unwrap_or(40);
            if self.frames == wait {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
            }
            ctx.request_repaint();
        }
    }
    let args: Vec<_> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "preview.png".into());
    let width = args
        .get(2)
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1100.0);
    let height = args
        .get(3)
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(760.0);
    let coordinate = |i: usize| {
        args.get(i)
            .and_then(|s| s.trim_start_matches('r').parse::<f32>().ok())
    };
    let button = |i: usize| {
        if args[i].starts_with('r') {
            egui::PointerButton::Secondary
        } else {
            egui::PointerButton::Primary
        }
    };
    let clicks: Vec<(egui::Pos2, egui::PointerButton)> = (4..args.len())
        .step_by(2)
        .filter_map(|i| {
            coordinate(i)
                .zip(coordinate(i + 1))
                .map(|(x, y)| (egui::pos2(x, y), button(i)))
        })
        .collect();
    eframe::run_native(
        "Voice Changer Preview",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([width, height]),
            persist_window: false,
            ..Default::default()
        },
        Box::new(move |cc| {
            Ok(Box::new(Capture {
                app: voice_changer::app::VoiceApp::new(cc),
                path,
                size: egui::vec2(width, height),
                frames: 0,
                clicks,
            }))
        }),
    )
}
#[cfg(not(feature = "gui"))]
fn main() {
    eprintln!("Enable the gui feature");
}
