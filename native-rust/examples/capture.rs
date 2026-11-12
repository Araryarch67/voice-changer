//! Development-only screenshot runner: cargo run --example capture -- output.png [width height [click_x click_y]]
#[cfg(feature = "gui")]
fn main() -> eframe::Result {
    use eframe::egui;
    struct Capture {
        app: voice_changer_native::app::VoiceApp,
        path: String,
        frames: usize,
        click: Option<egui::Pos2>,
    }
    impl eframe::App for Capture {
        // Optional click before the screenshot, to capture another page or a dialog.
        fn raw_input_hook(&mut self, _ctx: &egui::Context, raw: &mut egui::RawInput) {
            let Some(pos) = self.click else { return };
            match self.frames {
                3 => raw.events.push(egui::Event::PointerMoved(pos)),
                4 | 5 => raw.events.push(egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: self.frames == 4,
                    modifiers: Default::default(),
                }),
                _ => {}
            }
        }
        fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
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
            if self.frames == 40 {
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
    let coordinate = |i: usize| args.get(i).and_then(|s| s.parse::<f32>().ok());
    let click = coordinate(4)
        .zip(coordinate(5))
        .map(|(x, y)| egui::pos2(x, y));
    eframe::run_native(
        "Voice Changer Preview",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([width, height]),
            ..Default::default()
        },
        Box::new(move |cc| {
            Ok(Box::new(Capture {
                app: voice_changer_native::app::VoiceApp::new(cc),
                path,
                frames: 0,
                click,
            }))
        }),
    )
}
#[cfg(not(feature = "gui"))]
fn main() {
    eprintln!("Enable the gui feature");
}
