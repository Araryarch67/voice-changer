//! Material 3 dark theme: colour roles, type, icons and the components the app uses.
use crate::i18n::t;
use eframe::egui::{
    self, emath::Numeric, Align2, Color32, CornerRadius, FontFamily, FontId, Frame, Margin, Rect,
    Response, RichText, Sense, Stroke, StrokeKind, Ui, Vec2, WidgetInfo, WidgetType,
};
use std::{ops::RangeInclusive, sync::Arc};

pub const PRIMARY: Color32 = Color32::from_rgb(208, 188, 255);
pub const ON_PRIMARY: Color32 = Color32::from_rgb(56, 30, 114);
pub const PRIMARY_CONTAINER: Color32 = Color32::from_rgb(79, 55, 139);
pub const ON_PRIMARY_CONTAINER: Color32 = Color32::from_rgb(234, 221, 255);
pub const SECONDARY_CONTAINER: Color32 = Color32::from_rgb(74, 68, 88);
pub const ON_SECONDARY_CONTAINER: Color32 = Color32::from_rgb(232, 222, 248);
pub const TERTIARY: Color32 = Color32::from_rgb(239, 184, 200);
pub const SUCCESS: Color32 = Color32::from_rgb(160, 214, 180);
pub const ERROR: Color32 = Color32::from_rgb(242, 184, 181);
pub const ON_ERROR: Color32 = Color32::from_rgb(96, 20, 16);
pub const ERROR_CONTAINER: Color32 = Color32::from_rgb(140, 29, 24);
pub const ON_ERROR_CONTAINER: Color32 = Color32::from_rgb(249, 222, 220);
pub const SURFACE: Color32 = Color32::from_rgb(20, 18, 24);
pub const SURFACE_CONTAINER_LOW: Color32 = Color32::from_rgb(29, 27, 32);
pub const SURFACE_CONTAINER: Color32 = Color32::from_rgb(33, 31, 38);
pub const SURFACE_CONTAINER_HIGH: Color32 = Color32::from_rgb(43, 41, 48);
pub const SURFACE_CONTAINER_HIGHEST: Color32 = Color32::from_rgb(54, 52, 59);
pub const ON_SURFACE: Color32 = Color32::from_rgb(230, 224, 233);
pub const ON_SURFACE_VARIANT: Color32 = Color32::from_rgb(202, 196, 208);
pub const OUTLINE: Color32 = Color32::from_rgb(147, 143, 153);
pub const OUTLINE_VARIANT: Color32 = Color32::from_rgb(73, 69, 79);

/// Material Symbols Rounded codepoints present in the bundled subset.
pub mod icons {
    pub const GRAPHIC_EQ: char = '\u{e1b8}';
    pub const LIBRARY_MUSIC: char = '\u{e030}';
    pub const TUNE: char = '\u{e429}';
    pub const SETTINGS: char = '\u{e8b8}';
    pub const SEARCH: char = '\u{e8b6}';
    pub const ADD: char = '\u{e145}';
    pub const EDIT: char = '\u{e3c9}';
    pub const FAVORITE: char = '\u{e87d}';
    pub const CHECK: char = '\u{e5ca}';
    pub const PLAY: char = '\u{e037}';
    pub const STOP: char = '\u{e047}';
    pub const MIC: char = '\u{e029}';
    pub const HEADPHONES: char = '\u{f01f}';
    pub const AUDIO_FILE: char = '\u{eb82}';
    pub const SPEED: char = '\u{e9e4}';
    pub const DELETE: char = '\u{e872}';
    pub const REFRESH: char = '\u{e5d5}';
    pub const CLOSE: char = '\u{e5cd}';
    pub const ERROR: char = '\u{e000}';
    pub const CHECK_CIRCLE: char = '\u{e86c}';
    pub const FOLDER_OPEN: char = '\u{e2c8}';
    pub const ROUTING: char = '\u{e8c1}';
    pub const MEMORY: char = '\u{e322}';
    pub const INFO: char = '\u{e88e}';
    pub const VOICE: char = '\u{e91f}';
    pub const DEVICES: char = '\u{e1b1}';
    pub const RECORD: char = '\u{e061}';
}

pub fn medium() -> FontFamily {
    FontFamily::Name("medium".into())
}
fn symbols(filled: bool) -> FontFamily {
    FontFamily::Name(if filled { "icons-filled" } else { "icons" }.into())
}
pub fn headline(text: &str) -> RichText {
    RichText::new(text).size(28.0).color(ON_SURFACE)
}
pub fn title(text: &str) -> RichText {
    RichText::new(text)
        .family(medium())
        .size(16.0)
        .color(ON_SURFACE)
}
pub fn supporting(text: &str) -> RichText {
    RichText::new(text).size(12.0).color(ON_SURFACE_VARIANT)
}
pub fn icon(glyph: char, size: f32, color: Color32) -> RichText {
    RichText::new(glyph)
        .family(symbols(false))
        .size(size)
        .color(color)
}

pub fn install(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    let embedded: [(&str, &'static [u8]); 5] = [
        (
            "roboto",
            include_bytes!("../assets/fonts/Roboto-Regular-subset.ttf"),
        ),
        (
            "roboto-medium",
            include_bytes!("../assets/fonts/Roboto-Medium-subset.ttf"),
        ),
        (
            "icons",
            include_bytes!("../assets/fonts/MaterialSymbolsRounded-subset.ttf"),
        ),
        // Japanese interface text (and kana in voice or device names).
        (
            "noto-jp",
            include_bytes!("../assets/fonts/NotoSansJP-Regular-subset.ttf"),
        ),
        (
            "icons-filled",
            include_bytes!("../assets/fonts/MaterialSymbolsRounded-Filled-subset.ttf"),
        ),
    ];
    for (name, bytes) in embedded {
        fonts
            .font_data
            .insert(name.into(), Arc::new(egui::FontData::from_static(bytes)));
    }
    // egui's own fonts stay as fallbacks for scripts outside the Roboto subset.
    let fallback = fonts.families[&FontFamily::Proportional].clone();
    let mut regular = vec!["roboto".to_owned(), "noto-jp".to_owned()];
    regular.extend(fallback.iter().cloned());
    let mut strong = vec!["roboto-medium".to_owned(), "noto-jp".to_owned()];
    strong.extend(fallback);
    fonts.families.insert(FontFamily::Proportional, regular);
    fonts.families.insert(medium(), strong);
    fonts
        .families
        .insert(symbols(false), vec!["icons".to_owned()]);
    fonts
        .families
        .insert(symbols(true), vec!["icons-filled".to_owned()]);
    ctx.set_fonts(fonts);

    ctx.set_visuals(egui::Visuals::dark());
    let mut style = (*ctx.style()).clone();
    style.text_styles = [
        (egui::TextStyle::Heading, FontId::proportional(28.0)),
        (egui::TextStyle::Body, FontId::proportional(14.0)),
        (egui::TextStyle::Button, FontId::new(14.0, medium())),
        (egui::TextStyle::Small, FontId::proportional(12.0)),
        (egui::TextStyle::Monospace, FontId::monospace(13.0)),
    ]
    .into();
    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    style.spacing.button_padding = egui::vec2(16.0, 10.0);
    style.spacing.interact_size.y = 40.0;
    style.spacing.menu_margin = Margin::same(8);
    let v = &mut style.visuals;
    v.panel_fill = SURFACE;
    v.window_fill = SURFACE_CONTAINER_HIGH;
    v.window_stroke = Stroke::NONE;
    v.window_corner_radius = CornerRadius::same(28);
    v.menu_corner_radius = CornerRadius::same(12);
    v.extreme_bg_color = SURFACE_CONTAINER_HIGHEST;
    v.faint_bg_color = SURFACE_CONTAINER_LOW;
    v.hyperlink_color = PRIMARY;
    v.selection.bg_fill = PRIMARY_CONTAINER;
    v.selection.stroke = Stroke::new(2.0_f32, PRIMARY);
    v.text_cursor.stroke = Stroke::new(2.0_f32, PRIMARY);
    let shadow = egui::Shadow {
        offset: [0, 4],
        blur: 16,
        spread: 0,
        color: Color32::from_black_alpha(90),
    };
    v.window_shadow = shadow;
    v.popup_shadow = shadow;
    let w = &mut v.widgets;
    for state in [
        &mut w.noninteractive,
        &mut w.inactive,
        &mut w.hovered,
        &mut w.active,
        &mut w.open,
    ] {
        state.corner_radius = CornerRadius::same(8);
        state.expansion = 0.0;
        state.fg_stroke = Stroke::new(1.0_f32, ON_SURFACE);
        state.bg_fill = SURFACE_CONTAINER_HIGHEST;
        state.weak_bg_fill = Color32::TRANSPARENT;
        state.bg_stroke = Stroke::new(1.0_f32, OUTLINE);
    }
    w.noninteractive.bg_fill = SURFACE_CONTAINER;
    w.noninteractive.bg_stroke = Stroke::new(1.0_f32, OUTLINE_VARIANT);
    w.hovered.weak_bg_fill = ON_SURFACE.gamma_multiply(0.08);
    w.hovered.bg_stroke = Stroke::new(1.0_f32, ON_SURFACE);
    w.active.weak_bg_fill = ON_SURFACE.gamma_multiply(0.12);
    w.active.bg_stroke = Stroke::new(2.0_f32, PRIMARY);
    w.open.bg_stroke = Stroke::new(2.0_f32, PRIMARY);
    ctx.set_style(style);
}

/// M3 state layer: the content colour at 8 % on hover/focus and 12 % while pressed.
fn state_layer(response: &Response, tint: Color32) -> Color32 {
    if response.is_pointer_button_down_on() {
        tint.gamma_multiply(0.12)
    } else if response.hovered() || response.has_focus() {
        tint.gamma_multiply(0.08)
    } else {
        Color32::TRANSPARENT
    }
}
pub fn paint_icon(
    ui: &Ui,
    center: egui::Pos2,
    glyph: char,
    size: f32,
    filled: bool,
    color: Color32,
) {
    ui.painter().text(
        center,
        Align2::CENTER_CENTER,
        glyph,
        FontId::new(size, symbols(filled)),
        color,
    );
}
/// Single-line text cut with an ellipsis at `width`.
pub fn elided(ui: &Ui, text: &str, font: FontId, color: Color32, width: f32) -> Arc<egui::Galley> {
    let mut job = egui::text::LayoutJob::simple_singleline(text.to_owned(), font, color);
    job.wrap = egui::text::TextWrapping::truncate_at_width(width);
    ui.fonts(|fonts| fonts.layout_job(job))
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Filled,
    Tonal,
    Outlined,
    Text,
    Danger,
    Fab,
    FabStop,
}
pub struct Button<'a> {
    kind: Kind,
    glyph: Option<char>,
    label: &'a str,
}
impl<'a> Button<'a> {
    fn new(kind: Kind, label: &'a str) -> Self {
        Self {
            kind,
            glyph: None,
            label,
        }
    }
    pub fn filled(label: &'a str) -> Self {
        Self::new(Kind::Filled, label)
    }
    pub fn tonal(label: &'a str) -> Self {
        Self::new(Kind::Tonal, label)
    }
    pub fn outlined(label: &'a str) -> Self {
        Self::new(Kind::Outlined, label)
    }
    pub fn text(label: &'a str) -> Self {
        Self::new(Kind::Text, label)
    }
    /// Text button in the error colour, for destructive actions.
    pub fn danger(label: &'a str) -> Self {
        Self::new(Kind::Danger, label)
    }
    /// Extended floating action button; an empty label gives the icon-only FAB.
    pub fn fab(label: &'a str) -> Self {
        Self::new(Kind::Fab, label)
    }
    pub fn fab_stop(label: &'a str) -> Self {
        Self::new(Kind::FabStop, label)
    }
    pub fn icon(mut self, glyph: char) -> Self {
        self.glyph = Some(glyph);
        self
    }
}
impl egui::Widget for Button<'_> {
    fn ui(self, ui: &mut Ui) -> Response {
        let fab = matches!(self.kind, Kind::Fab | Kind::FabStop);
        let text_only = matches!(self.kind, Kind::Text | Kind::Danger);
        let (height, radius, icon_size) = if fab {
            (56.0, 16.0, 24.0)
        } else {
            (40.0, 20.0, 18.0)
        };
        let galley = ui.painter().layout_no_wrap(
            self.label.to_owned(),
            FontId::new(if fab { 16.0 } else { 14.0 }, medium()),
            Color32::PLACEHOLDER,
        );
        let side = if fab {
            16.0
        } else if text_only {
            12.0
        } else {
            24.0
        };
        let (left, gap) = match (self.glyph.is_some(), self.label.is_empty()) {
            (true, true) => (side, 0.0),
            (true, false) => (if fab { side } else { side - 8.0 }, 8.0),
            _ => (side, 0.0),
        };
        let icon_width = if self.glyph.is_some() { icon_size } else { 0.0 };
        let right = if self.label.is_empty() {
            side
        } else if fab {
            20.0
        } else {
            side
        };
        let width = left + icon_width + gap + galley.size().x + right;
        let (rect, response) = ui.allocate_exact_size(Vec2::new(width, height), Sense::click());
        let enabled = ui.is_enabled();
        response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, enabled, self.label));
        if ui.is_rect_visible(rect) {
            let (fill, content, stroke) = match self.kind {
                Kind::Filled | Kind::Fab => (PRIMARY, ON_PRIMARY, Stroke::NONE),
                Kind::FabStop => (ERROR, ON_ERROR, Stroke::NONE),
                Kind::Tonal => (SECONDARY_CONTAINER, ON_SECONDARY_CONTAINER, Stroke::NONE),
                Kind::Outlined => (Color32::TRANSPARENT, PRIMARY, Stroke::new(1.0_f32, OUTLINE)),
                Kind::Text => (Color32::TRANSPARENT, PRIMARY, Stroke::NONE),
                Kind::Danger => (Color32::TRANSPARENT, ERROR, Stroke::NONE),
            };
            let painter = ui.painter();
            painter.rect(rect, radius, fill, stroke, StrokeKind::Inside);
            painter.rect_filled(rect, radius, state_layer(&response, content));
            let mut x = rect.left() + left;
            if let Some(glyph) = self.glyph {
                let center = egui::pos2(x + icon_size / 2.0, rect.center().y);
                paint_icon(ui, center, glyph, icon_size, fab, content);
                x += icon_size + gap;
            }
            let y = rect.center().y - galley.size().y / 2.0;
            ui.painter().galley(egui::pos2(x, y), galley, content);
        }
        response.on_hover_cursor(egui::CursorIcon::PointingHand)
    }
}

/// Standard 40 px icon button; `hint` is the tooltip and accessible name.
pub fn icon_button(ui: &mut Ui, glyph: char, color: Color32, hint: &str) -> Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(40.0), Sense::click());
    let enabled = ui.is_enabled();
    response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, enabled, hint));
    ui.painter()
        .circle_filled(rect.center(), 20.0, state_layer(&response, color));
    paint_icon(ui, rect.center(), glyph, 24.0, false, color);
    response
        .on_hover_cursor(egui::CursorIcon::PointingHand)
        .on_hover_text(hint)
}

/// A small info icon that shows `tip` while it is held (touch screens have no
/// hover) or hovered.
pub fn info_button(ui: &mut Ui, tip: &str) -> Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(28.0), Sense::click());
    let enabled = ui.is_enabled();
    response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, enabled, tip));
    let held = response.is_pointer_button_down_on();
    ui.painter()
        .circle_filled(rect.center(), 14.0, state_layer(&response, PRIMARY));
    paint_icon(
        ui,
        rect.center(),
        icons::INFO,
        20.0,
        held,
        if held || response.hovered() {
            PRIMARY
        } else {
            ON_SURFACE_VARIANT
        },
    );
    if held {
        response.show_tooltip_text(tip);
    }
    response
        .on_hover_cursor(egui::CursorIcon::Help)
        .on_hover_text(tip)
}

pub fn filter_chip(ui: &mut Ui, label: &str, selected: bool) -> Response {
    let galley = ui.painter().layout_no_wrap(
        label.to_owned(),
        FontId::new(14.0, medium()),
        Color32::PLACEHOLDER,
    );
    let left = if selected { 34.0 } else { 16.0 };
    let size = Vec2::new(left + galley.size().x + 16.0, 32.0);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    let enabled = ui.is_enabled();
    response.widget_info(|| WidgetInfo::selected(WidgetType::Button, enabled, selected, label));
    let content = if selected {
        ON_SECONDARY_CONTAINER
    } else {
        ON_SURFACE_VARIANT
    };
    let painter = ui.painter();
    if selected {
        painter.rect_filled(rect, 8.0, SECONDARY_CONTAINER);
        paint_icon(
            ui,
            egui::pos2(rect.left() + 17.0, rect.center().y),
            icons::CHECK,
            18.0,
            false,
            content,
        );
    } else {
        painter.rect_stroke(
            rect,
            8.0,
            Stroke::new(1.0_f32, OUTLINE_VARIANT),
            StrokeKind::Inside,
        );
    }
    painter.rect_filled(rect, 8.0, state_layer(&response, content));
    let y = rect.center().y - galley.size().y / 2.0;
    painter.galley(egui::pos2(rect.left() + left, y), galley, content);
    response.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// Read-only assist chip: an icon and a label in an outlined pill.
pub fn chip(ui: &mut Ui, glyph: char, label: &str) {
    let width = (ui.available_width() - 50.0).max(40.0);
    let galley = elided(ui, label, FontId::new(14.0, medium()), ON_SURFACE, width);
    let size = Vec2::new(8.0 + 18.0 + 8.0 + galley.size().x + 16.0, 32.0);
    let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
    ui.painter().rect_stroke(
        rect,
        8.0,
        Stroke::new(1.0_f32, OUTLINE_VARIANT),
        StrokeKind::Inside,
    );
    paint_icon(
        ui,
        egui::pos2(rect.left() + 17.0, rect.center().y),
        glyph,
        18.0,
        false,
        PRIMARY,
    );
    let y = rect.center().y - galley.size().y / 2.0;
    ui.painter()
        .galley(egui::pos2(rect.left() + 34.0, y), galley, ON_SURFACE);
}

pub fn switch(ui: &mut Ui, on: &mut bool, label: &str) -> Response {
    let (rect, mut response) = ui.allocate_exact_size(Vec2::new(52.0, 32.0), Sense::click());
    if response.clicked() {
        *on = !*on;
        response.mark_changed();
    }
    let (enabled, checked) = (ui.is_enabled(), *on);
    response.widget_info(|| WidgetInfo::selected(WidgetType::Checkbox, enabled, checked, label));
    let t = ui.ctx().animate_bool_responsive(response.id, *on);
    let painter = ui.painter();
    painter.rect(
        rect,
        16.0,
        SURFACE_CONTAINER_HIGHEST.lerp_to_gamma(PRIMARY, t),
        Stroke::new(2.0_f32, OUTLINE.gamma_multiply(1.0 - t)),
        StrokeKind::Inside,
    );
    let center = egui::pos2(
        egui::lerp(rect.left() + 16.0..=rect.right() - 16.0, t),
        rect.center().y,
    );
    let pressed = response.is_pointer_button_down_on();
    let radius = if pressed {
        14.0
    } else {
        egui::lerp(8.0..=12.0, t)
    };
    painter.circle_filled(
        center,
        20.0,
        state_layer(&response, if *on { PRIMARY } else { ON_SURFACE }),
    );
    painter.circle_filled(center, radius, OUTLINE.lerp_to_gamma(ON_PRIMARY, t));
    if t > 0.5 {
        paint_icon(ui, center, icons::CHECK, 16.0, false, PRIMARY);
    }
    response.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// M3 slider: a 12 px track split around a bar handle. Arrow keys step while focused.
pub fn slider<N: Numeric>(
    ui: &mut Ui,
    value: &mut N,
    range: RangeInclusive<N>,
    step: f64,
) -> Response {
    let (low, high) = (range.start().to_f64(), range.end().to_f64());
    let size = Vec2::new(ui.available_width(), 36.0);
    let (rect, mut response) = ui.allocate_exact_size(size, Sense::click_and_drag());
    let travel = rect.shrink2(Vec2::new(2.0, 0.0));
    let current = value.to_f64();
    let mut next = current;
    let mut touched = false;
    if let Some(pointer) = response.interact_pointer_pos() {
        let t = ((pointer.x - travel.left()) / travel.width()).clamp(0.0, 1.0);
        next = low + f64::from(t) * (high - low);
        touched = true;
        response.request_focus();
    }
    if response.has_focus() {
        ui.memory_mut(|memory| {
            memory.set_focus_lock_filter(
                response.id,
                egui::EventFilter {
                    horizontal_arrows: true,
                    vertical_arrows: true,
                    ..Default::default()
                },
            );
        });
        let (down, up) = ui.input(|input| {
            (
                input.key_pressed(egui::Key::ArrowLeft) || input.key_pressed(egui::Key::ArrowDown),
                input.key_pressed(egui::Key::ArrowRight) || input.key_pressed(egui::Key::ArrowUp),
            )
        });
        if down != up {
            next += if up { step } else { -step };
            touched = true;
        }
    }
    if touched {
        let snapped = N::from_f64(((next / step).round() * step).clamp(low, high));
        if snapped.to_f64() != current {
            *value = snapped;
            response.mark_changed();
        }
    }
    let next = value.to_f64();
    let enabled = ui.is_enabled();
    response.widget_info(|| WidgetInfo::slider(enabled, next, ""));
    if ui.is_rect_visible(rect) {
        let t = if high > low {
            ((next - low) / (high - low)) as f32
        } else {
            0.0
        };
        let x = egui::lerp(travel.left()..=travel.right(), t);
        let (top, bottom) = (rect.center().y - 6.0, rect.center().y + 6.0);
        let painter = ui.painter();
        if x - 6.0 > rect.left() {
            painter.rect_filled(
                Rect::from_min_max(egui::pos2(rect.left(), top), egui::pos2(x - 6.0, bottom)),
                CornerRadius {
                    nw: 6,
                    sw: 6,
                    ne: 2,
                    se: 2,
                },
                PRIMARY,
            );
        }
        if x + 6.0 < rect.right() {
            painter.rect_filled(
                Rect::from_min_max(egui::pos2(x + 6.0, top), egui::pos2(rect.right(), bottom)),
                CornerRadius {
                    nw: 2,
                    sw: 2,
                    ne: 6,
                    se: 6,
                },
                SECONDARY_CONTAINER,
            );
            if x + 14.0 < rect.right() {
                painter.circle_filled(
                    egui::pos2(rect.right() - 6.0, rect.center().y),
                    2.0,
                    PRIMARY,
                );
            }
        }
        let held = response.is_pointer_button_down_on();
        let handle = Rect::from_center_size(
            egui::pos2(x, rect.center().y),
            Vec2::new(if held { 2.0 } else { 4.0 }, 32.0),
        );
        painter.rect_filled(handle, 2.0, PRIMARY);
        if response.hovered() || response.has_focus() {
            painter.rect_filled(handle.expand(1.0), 3.0, PRIMARY.gamma_multiply(0.25));
        }
    }
    response.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// How a setting's number reads and is typed: the stored value times `scale`, with
/// `decimals` places and a `suffix` ("ms", "%", ...).
#[derive(Clone, Copy)]
pub struct Format {
    scale: f64,
    decimals: usize,
    suffix: &'static str,
    signed: bool,
    off_at_min: bool,
    /// Common values offered in a drop-down beside the number, in stored units.
    presets: &'static [f64],
    /// How a preset reads in the drop-down; `None` shows it like the number.
    preset_label: Option<fn(f64) -> String>,
    /// Advice behind an info button beside the name, shown while it is held or
    /// hovered.
    tip: Option<&'static str>,
}
impl Format {
    pub const fn new(suffix: &'static str) -> Self {
        Self {
            scale: 1.0,
            decimals: 0,
            suffix,
            signed: false,
            off_at_min: false,
            presets: &[],
            preset_label: None,
            tip: None,
        }
    }
    /// Advice shown while the info button beside the name is held or hovered.
    pub const fn tip(mut self, tip: &'static str) -> Self {
        self.tip = Some(tip);
        self
    }
    /// Common values to pick from a drop-down; typing and the slider still work.
    pub const fn presets(mut self, presets: &'static [f64]) -> Self {
        self.presets = presets;
        self
    }
    /// How each preset reads, for values known by another name (CHUNK 128).
    pub const fn preset_label(mut self, label: fn(f64) -> String) -> Self {
        self.preset_label = Some(label);
        self
    }
    /// Shown and typed as the value times `scale` (0.5 as 50 %).
    pub const fn scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }
    pub const fn decimals(mut self, decimals: usize) -> Self {
        self.decimals = decimals;
        self
    }
    /// Positive values get a "+".
    pub const fn signed(mut self) -> Self {
        self.signed = true;
        self
    }
    /// The lowest value reads "Off", and typing "off" (in English or the current
    /// language) sets it.
    pub const fn off_at_min(mut self) -> Self {
        self.off_at_min = true;
        self
    }
    fn show(self, value: f64, min: f64) -> String {
        if self.off_at_min && value <= min {
            return t("Off").into();
        }
        let shown = value * self.scale;
        if self.signed {
            format!("{shown:+.*}{}", self.decimals, self.suffix)
        } else {
            format!("{shown:.*}{}", self.decimals, self.suffix)
        }
    }
    /// Reads a typed value: units and spaces are ignored and a comma works as the
    /// decimal point, so "320", "320 ms" and "0,5" all parse.
    fn parse(self, text: &str, min: f64) -> Option<f64> {
        let text = text.trim().to_lowercase();
        if self.off_at_min && (text == "off" || text == t("Off").to_lowercase()) {
            return Some(min);
        }
        let number: String = text
            .replace(',', ".")
            .chars()
            .filter(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | '+'))
            .collect();
        number.parse::<f64>().ok().map(|value| value / self.scale)
    }
}

/// A labelled slider row: the name on the left, an editable number on the right
/// (click to type, or drag sideways), the slider, then a plain-language hint for
/// people who do not know the term.
pub fn setting<N: Numeric>(
    ui: &mut Ui,
    label: &str,
    hint: &str,
    format: Format,
    value: &mut N,
    range: RangeInclusive<N>,
    step: f64,
) -> Response {
    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing.y = 0.0;
        let (low, high) = (range.start().to_f64(), range.end().to_f64());
        let typed = ui
            .allocate_ui_with_layout(
                Vec2::new(ui.available_width(), 30.0),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    ui.label(RichText::new(label).size(14.0).color(ON_SURFACE));
                    if let Some(tip) = format.tip {
                        info_button(ui, tip);
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let mut picked = None;
                        if !format.presets.is_empty() {
                            let current = value.to_f64();
                            // A value between two presets opens the list at the nearer one.
                            let nearest = format
                                .presets
                                .iter()
                                .enumerate()
                                .min_by(|a, b| {
                                    (a.1 - current).abs().total_cmp(&(b.1 - current).abs())
                                })
                                .map(|(index, _)| index);
                            menu_button(ui, t("Common values"), 220.0, |ui| {
                                for (index, &preset) in format.presets.iter().enumerate() {
                                    let text = match format.preset_label {
                                        Some(name) => name(preset),
                                        None => format.show(preset, low),
                                    };
                                    let on = (preset - current).abs() < step.max(1e-6) / 2.0
                                        || (preset.round() - current).abs() < 0.5 && step >= 1.0;
                                    let item = menu_item(ui, &text, on);
                                    if Some(index) == nearest && !on && menu_opening(ui) {
                                        item.scroll_to_me(Some(egui::Align::Center));
                                    }
                                    if item.clicked() {
                                        picked = Some(preset);
                                    }
                                }
                            });
                        }
                        if let Some(preset) = picked {
                            *value = N::from_f64(((preset / step).round() * step).clamp(low, high));
                        }
                        ui.spacing_mut().interact_size = Vec2::new(64.0, 26.0);
                        ui.spacing_mut().button_padding = egui::vec2(8.0, 2.0);
                        ui.style_mut().override_font_id = Some(FontId::new(14.0, medium()));
                        ui.add(
                            egui::DragValue::new(value)
                                .range(range.clone())
                                .speed(step)
                                .custom_formatter(move |n, _| format.show(n, low))
                                .custom_parser(move |text| {
                                    format
                                        .parse(text, low)
                                        .map(|n| ((n / step).round() * step).clamp(low, high))
                                }),
                        )
                        .on_hover_text(t("Click to type a value, or drag sideways"))
                    })
                    .inner
                },
            )
            .inner;
        ui.add_space(2.0);
        let mut response = slider(ui, value, range, step);
        if typed.changed() {
            response.mark_changed();
        }
        if !hint.is_empty() {
            ui.label(supporting(hint));
            ui.add_space(4.0);
        }
        response
    })
    .inner
}

pub fn card() -> Frame {
    Frame::new()
        .fill(SURFACE_CONTAINER)
        .corner_radius(16)
        .inner_margin(20)
}
pub fn card_title(ui: &mut Ui, glyph: char, text: &str) {
    ui.horizontal(|ui| {
        ui.set_height(24.0);
        ui.label(icon(glyph, 20.0, PRIMARY));
        ui.label(title(text));
    });
    ui.add_space(4.0);
}
/// Two-line list item with a leading icon.
pub fn list_item(ui: &mut Ui, glyph: char, headline: &str, detail: &str) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 52.0), Sense::hover());
    paint_icon(
        ui,
        egui::pos2(rect.left() + 12.0, rect.center().y),
        glyph,
        24.0,
        false,
        ON_SURFACE_VARIANT,
    );
    let width = rect.width() - 40.0;
    let first = elided(ui, headline, FontId::proportional(16.0), ON_SURFACE, width);
    let second = elided(
        ui,
        detail,
        FontId::proportional(12.0),
        ON_SURFACE_VARIANT,
        width,
    );
    let x = rect.left() + 40.0;
    ui.painter()
        .galley(egui::pos2(x, rect.top() + 6.0), first, ON_SURFACE);
    ui.painter()
        .galley(egui::pos2(x, rect.top() + 29.0), second, ON_SURFACE_VARIANT);
}

/// Lowest level a meter shows, in dBFS.
pub const METER_FLOOR_DB: f32 = -60.0;

/// RMS level in dBFS, clamped to the meter's range.
pub fn level_db(level: f32) -> f32 {
    (20.0 * level.max(1e-6).log10()).clamp(METER_FLOOR_DB, 0.0)
}

/// Level meter: an icon, a rounded track filled to the level on a -60..0 dB scale,
/// an optional tick at `gate_db` (the noise gate), and the level in dB.
pub fn meter(ui: &mut Ui, glyph: char, level: f32, gate_db: Option<f32>, hint: &str) -> Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::new(196.0, 32.0), Sense::hover());
    let db = level_db(level);
    let open = gate_db.is_none_or(|gate| db >= gate);
    paint_icon(
        ui,
        egui::pos2(rect.left() + 10.0, rect.center().y),
        glyph,
        20.0,
        false,
        if open { ON_SURFACE } else { ON_SURFACE_VARIANT },
    );
    let track = Rect::from_min_max(
        egui::pos2(rect.left() + 28.0, rect.center().y - 4.0),
        egui::pos2(rect.right() - 52.0, rect.center().y + 4.0),
    );
    let x = |db: f32| {
        egui::lerp(
            track.left()..=track.right(),
            (db - METER_FLOOR_DB) / -METER_FLOOR_DB,
        )
    };
    let painter = ui.painter();
    painter.rect_filled(track, 4.0, SECONDARY_CONTAINER);
    let mut active = track;
    active.set_right(x(db).max(track.left() + 8.0));
    let fill = if db > -3.0 {
        ERROR
    } else if open {
        PRIMARY
    } else {
        OUTLINE
    };
    painter.rect_filled(active, 4.0, fill);
    if let Some(gate) = gate_db.filter(|gate| *gate > METER_FLOOR_DB) {
        let gx = x(gate);
        painter.line_segment(
            [
                egui::pos2(gx, track.top() - 4.0),
                egui::pos2(gx, track.bottom() + 4.0),
            ],
            Stroke::new(2.0_f32, TERTIARY),
        );
    }
    painter.text(
        egui::pos2(rect.right(), rect.center().y),
        Align2::RIGHT_CENTER,
        if db <= METER_FLOOR_DB {
            t("silent").to_owned()
        } else {
            format!("{db:.0} dB")
        },
        FontId::new(12.0, medium()),
        ON_SURFACE_VARIANT,
    );
    response.on_hover_text(hint)
}

/// Status chip: a tonal pill with an icon, a short label and a value, tinted by
/// `tone` (PRIMARY when fine, TERTIARY for a caution, ERROR for a problem).
pub fn stat(
    ui: &mut Ui,
    glyph: char,
    label: &str,
    value: &str,
    tone: Color32,
    hint: &str,
) -> Response {
    let label_galley = ui.painter().layout_no_wrap(
        label.to_owned(),
        FontId::proportional(12.0),
        ON_SURFACE_VARIANT,
    );
    let value_galley =
        ui.painter()
            .layout_no_wrap(value.to_owned(), FontId::new(13.0, medium()), ON_SURFACE);
    let width = 10.0 + 18.0 + 8.0 + label_galley.size().x + 6.0 + value_galley.size().x + 12.0;
    let (rect, response) = ui.allocate_exact_size(Vec2::new(width, 32.0), Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 8.0, tone.gamma_multiply(0.14));
    painter.rect_stroke(
        rect,
        8.0,
        Stroke::new(1.0_f32, tone.gamma_multiply(0.45)),
        StrokeKind::Inside,
    );
    paint_icon(
        ui,
        egui::pos2(rect.left() + 19.0, rect.center().y),
        glyph,
        18.0,
        false,
        tone,
    );
    let mut x = rect.left() + 36.0;
    let y = rect.center().y;
    painter.galley(
        egui::pos2(x, y - label_galley.size().y / 2.0),
        label_galley.clone(),
        ON_SURFACE_VARIANT,
    );
    x += label_galley.size().x + 6.0;
    painter.galley(
        egui::pos2(x, y - value_galley.size().y / 2.0),
        value_galley,
        ON_SURFACE,
    );
    response.on_hover_text(hint)
}

const TONES: [(Color32, Color32); 6] = [
    (PRIMARY_CONTAINER, ON_PRIMARY_CONTAINER),
    (
        Color32::from_rgb(99, 59, 72),
        Color32::from_rgb(255, 216, 228),
    ),
    (
        Color32::from_rgb(30, 78, 80),
        Color32::from_rgb(178, 235, 236),
    ),
    (
        Color32::from_rgb(43, 70, 120),
        Color32::from_rgb(216, 226, 255),
    ),
    (
        Color32::from_rgb(52, 80, 46),
        Color32::from_rgb(200, 238, 190),
    ),
    (
        Color32::from_rgb(98, 64, 28),
        Color32::from_rgb(255, 221, 181),
    ),
];
/// Monogram tile for a voice; empty slots get a plus. Returns the content colour.
pub fn avatar(
    ui: &Ui,
    rect: Rect,
    radius: f32,
    index: usize,
    name: &str,
    configured: bool,
) -> Color32 {
    let painter = ui.painter().with_clip_rect(rect.intersect(ui.clip_rect()));
    if !configured {
        painter.rect_filled(rect, radius, SURFACE_CONTAINER_HIGHEST);
        paint_icon(
            ui,
            rect.center(),
            icons::ADD,
            (rect.height() * 0.4).clamp(18.0, 32.0),
            false,
            OUTLINE,
        );
        return ON_SURFACE_VARIANT;
    }
    let (fill, content) = TONES[index % TONES.len()];
    painter.rect_filled(rect, radius, fill);
    if rect.height() > 60.0 {
        // Decorative equaliser bars along the bottom edge.
        let count = (rect.width() / 9.0) as usize;
        for bar in 0..count {
            let height = 6.0 + ((index * 31 + bar * 17 + bar * bar * 7) % 22) as f32;
            let x = rect.left() + 6.0 + bar as f32 * 9.0;
            painter.rect_filled(
                Rect::from_min_max(
                    egui::pos2(x, rect.bottom() - height),
                    egui::pos2(x + 4.0, rect.bottom() + 2.0),
                ),
                2.0,
                content.gamma_multiply(0.12),
            );
        }
    }
    let letter = name
        .chars()
        .find(|c| c.is_alphanumeric())
        .map(|c| c.to_uppercase().to_string())
        .unwrap_or_default();
    painter.text(
        rect.center(),
        Align2::CENTER_CENTER,
        letter,
        FontId::new((rect.height() * 0.42).clamp(16.0, 40.0), medium()),
        content,
    );
    content
}

/// Navigation rail / bar destination: icon in a pill indicator with a label below.
pub fn nav_item(ui: &mut Ui, glyph: char, label: &str, active: bool) -> Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::new(72.0, 56.0), Sense::click());
    let enabled = ui.is_enabled();
    response.widget_info(|| WidgetInfo::selected(WidgetType::Button, enabled, active, label));
    let t = ui.ctx().animate_bool_responsive(response.id, active);
    let center = egui::pos2(rect.center().x, rect.top() + 16.0);
    let pill = |width: f32| Rect::from_center_size(center, Vec2::new(width, 32.0));
    let painter = ui.painter();
    painter.rect_filled(
        pill(egui::lerp(32.0..=56.0, t)),
        16.0,
        SECONDARY_CONTAINER.gamma_multiply(t),
    );
    painter.rect_filled(pill(56.0), 16.0, state_layer(&response, ON_SURFACE));
    let content = if active {
        ON_SECONDARY_CONTAINER
    } else {
        ON_SURFACE_VARIANT
    };
    paint_icon(ui, center, glyph, 24.0, active, content);
    painter.text(
        egui::pos2(rect.center().x, rect.top() + 46.0),
        Align2::CENTER_CENTER,
        label,
        FontId::new(12.0, medium()),
        if active {
            ON_SURFACE
        } else {
            ON_SURFACE_VARIANT
        },
    );
    response.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// M3 checkbox: an 18 px box, filled with the primary colour and a check mark when
/// on, in a 40 px touch target. A disabled one is drawn faded and cannot change.
pub fn checkbox(ui: &mut Ui, on: &mut bool, label: &str) -> Response {
    let (rect, mut response) = ui.allocate_exact_size(Vec2::splat(40.0), Sense::click());
    let enabled = ui.is_enabled();
    if response.clicked() && enabled {
        *on = !*on;
        response.mark_changed();
    }
    let checked = *on;
    response.widget_info(|| WidgetInfo::selected(WidgetType::Checkbox, enabled, checked, label));
    let painter = ui.painter();
    let alpha = if enabled { 1.0 } else { 0.38 };
    painter.circle_filled(rect.center(), 20.0, state_layer(&response, PRIMARY));
    let square = Rect::from_center_size(rect.center(), Vec2::splat(18.0));
    if checked {
        painter.rect_filled(square, 2.0, PRIMARY.gamma_multiply(alpha));
        paint_icon(
            ui,
            square.center(),
            icons::CHECK,
            16.0,
            false,
            ON_PRIMARY.gamma_multiply(alpha),
        );
    } else {
        painter.rect_stroke(
            square,
            2.0,
            Stroke::new(2.0_f32, ON_SURFACE_VARIANT.gamma_multiply(alpha)),
            StrokeKind::Inside,
        );
    }
    if enabled {
        response.on_hover_cursor(egui::CursorIcon::PointingHand)
    } else {
        response
    }
}

/// Small tonal label such as "Recommended" or "Installed".
pub fn badge(ui: &mut Ui, text: &str, fill: Color32, content: Color32) {
    let galley = ui
        .painter()
        .layout_no_wrap(text.to_owned(), FontId::new(11.0, medium()), content);
    let size = Vec2::new(galley.size().x + 16.0, 20.0);
    let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
    ui.painter().rect_filled(rect, 10.0, fill);
    ui.painter().galley(
        egui::pos2(rect.left() + 8.0, rect.center().y - galley.size().y / 2.0),
        galley,
        content,
    );
}

/// M3 snackbar ("toast"): a short message at the bottom of the window, above the
/// bottom bar, with a close button. Errors use the error container colours. Returns
/// true when closed.
pub fn snackbar(ctx: &egui::Context, text: &str, error: bool) -> bool {
    let (fill, content) = if error {
        (ERROR_CONTAINER, ON_ERROR_CONTAINER)
    } else {
        (
            Color32::from_rgb(230, 224, 233),
            Color32::from_rgb(50, 47, 53),
        )
    };
    let width = (ctx.screen_rect().width() - 32.0).min(560.0);
    let mut closed = false;
    egui::Area::new(egui::Id::new("snackbar"))
        .order(egui::Order::Foreground)
        .anchor(Align2::CENTER_BOTTOM, Vec2::new(0.0, -96.0))
        .show(ctx, |ui| {
            Frame::new()
                .fill(fill)
                .corner_radius(8)
                .inner_margin(Margin {
                    left: 16,
                    right: 4,
                    top: 6,
                    bottom: 6,
                })
                .shadow(egui::Shadow {
                    offset: [0, 3],
                    blur: 10,
                    spread: 0,
                    color: Color32::from_black_alpha(110),
                })
                .show(ui, |ui| {
                    ui.set_width(width - 20.0);
                    ui.horizontal(|ui| {
                        if error {
                            ui.label(icon(icons::ERROR, 20.0, content));
                        }
                        let text_width = (ui.available_width() - 48.0).max(80.0);
                        ui.allocate_ui_with_layout(
                            Vec2::new(text_width, 20.0),
                            egui::Layout::top_down(egui::Align::Min),
                            |ui| {
                                ui.set_width(text_width);
                                ui.add(egui::Label::new(RichText::new(text).color(content)).wrap());
                            },
                        );
                        closed = icon_button(ui, icons::CLOSE, content, t("Dismiss")).clicked();
                    });
                });
        });
    closed
}

/// Tonal message strip with a dismiss button; returns true when dismissed.
pub fn banner(ui: &mut Ui, glyph: char, fill: Color32, content: Color32, text: &str) -> bool {
    let mut dismissed = false;
    Frame::new()
        .fill(fill)
        .corner_radius(12)
        .inner_margin(Margin::symmetric(16, 8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(icon(glyph, 24.0, content));
                let width = (ui.available_width() - 56.0).max(80.0);
                ui.allocate_ui_with_layout(
                    Vec2::new(width, 20.0),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        ui.set_width(width);
                        ui.add(egui::Label::new(RichText::new(text).color(content)).wrap());
                    },
                );
                dismissed = icon_button(ui, icons::CLOSE, content, t("Dismiss")).clicked();
            });
        });
    dismissed
}

/// Basic dialog over a scrim. Returns the body's value and whether it was dismissed.
pub fn dialog<R>(
    ctx: &egui::Context,
    id: &str,
    heading: &str,
    body: impl FnOnce(&mut Ui) -> R,
) -> (R, bool) {
    let width = (ctx.screen_rect().width() - 48.0).min(520.0) - 48.0;
    let response = egui::Modal::new(egui::Id::new(id))
        .frame(
            Frame::new()
                .fill(SURFACE_CONTAINER_HIGH)
                .corner_radius(28)
                .inner_margin(24),
        )
        .backdrop_color(Color32::from_black_alpha(140))
        .show(ctx, |ui| {
            ui.set_width(width);
            ui.label(RichText::new(heading).size(24.0).color(ON_SURFACE));
            ui.add_space(8.0);
            body(ui)
        });
    let dismissed = response.should_close();
    (response.inner, dismissed)
}
/// Right-aligned row for dialog actions; add the confirming action first.
pub fn actions(ui: &mut Ui, add: impl FnOnce(&mut Ui)) {
    ui.add_space(16.0);
    ui.allocate_ui_with_layout(
        Vec2::new(ui.available_width(), 40.0),
        egui::Layout::right_to_left(egui::Align::Center),
        add,
    );
}
pub fn field_label(ui: &mut Ui, text: &str) {
    ui.add_space(4.0);
    ui.label(
        RichText::new(text)
            .family(medium())
            .size(12.0)
            .color(ON_SURFACE_VARIANT),
    );
}
/// Outlined text field.
/// A chevron pointing down, or up while its menu is open.
fn paint_chevron(ui: &Ui, center: egui::Pos2, open: bool, color: Color32) {
    let tip = if open { -2.5 } else { 2.5 };
    ui.painter().add(egui::Shape::line(
        vec![
            center + Vec2::new(-5.0, -tip),
            center + Vec2::new(0.0, tip),
            center + Vec2::new(5.0, -tip),
        ],
        Stroke::new(2.0_f32, color),
    ));
}

fn menu_open(anchor: &Response) -> bool {
    egui::Popup::is_id_open(&anchor.ctx, egui::Popup::default_response_id(anchor))
}

/// M3 menu below `anchor`: opens when the anchor is clicked and closes on a choice
/// or a click elsewhere. Fill it with [`menu_item`]s; long lists scroll.
pub fn menu(anchor: &Response, width: f32, add: impl FnOnce(&mut Ui)) {
    // How many frames it has been open: for the first few (the first one only
    // measures it) the chosen item scrolls into view, see `menu_item`.
    let frames_key = egui::Popup::default_response_id(anchor).with("frames-open");
    let ctx = anchor.ctx.clone();
    let shown = egui::Popup::menu(anchor)
        .gap(4.0)
        .width(width)
        .frame(
            Frame::new()
                .fill(SURFACE_CONTAINER_HIGH)
                .corner_radius(12)
                .inner_margin(Margin::symmetric(0, 8))
                .shadow(anchor.ctx.style().visuals.popup_shadow),
        )
        .show(|ui| {
            let frames: u32 = ctx.data(|data| data.get_temp(frames_key).unwrap_or(0));
            ctx.data_mut(|data| data.insert_temp(frames_key, frames.saturating_add(1)));
            let opening = frames < 3;
            ui.set_width(width);
            egui::ScrollArea::vertical()
                .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                .max_height(304.0)
                .show(ui, |ui| {
                    ui.set_width(width);
                    ui.spacing_mut().item_spacing = Vec2::ZERO;
                    ui.data_mut(|data| data.insert_temp(egui::Id::new(MENU_OPENING), opening));
                    add(ui);
                });
        });
    if shown.is_none() {
        ctx.data_mut(|data| data.remove::<u32>(frames_key));
    }
}

/// Whether the [`menu`] being drawn has just opened.
const MENU_OPENING: &str = "m3-menu-opening";

/// Inside a [`menu`]: whether it has just opened, so an item can scroll into view.
pub fn menu_opening(ui: &Ui) -> bool {
    ui.data(|data| data.get_temp(egui::Id::new(MENU_OPENING)).unwrap_or(false))
}

/// A row of a [`menu`]; the chosen one is tinted and ticked, and scrolled into view
/// when the menu opens, so a long list starts at the current value.
pub fn menu_item(ui: &mut Ui, text: &str, selected: bool) -> Response {
    let (rect, response) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 40.0), Sense::click());
    if selected && menu_opening(ui) {
        response.scroll_to_me(Some(egui::Align::Center));
    }
    let enabled = ui.is_enabled();
    response.widget_info(|| WidgetInfo::selected(WidgetType::Button, enabled, selected, text));
    let row = rect.shrink2(Vec2::new(8.0, 0.0));
    let content = if selected {
        ON_SECONDARY_CONTAINER
    } else {
        ON_SURFACE
    };
    if selected {
        ui.painter().rect_filled(row, 8.0, SECONDARY_CONTAINER);
        paint_icon(
            ui,
            egui::pos2(row.left() + 20.0, row.center().y),
            icons::CHECK,
            18.0,
            false,
            content,
        );
    }
    ui.painter()
        .rect_filled(row, 8.0, state_layer(&response, content));
    let galley = elided(
        ui,
        text,
        FontId::new(14.0, FontFamily::Proportional),
        content,
        row.width() - 52.0,
    );
    ui.painter().galley(
        egui::pos2(row.left() + 40.0, row.center().y - galley.size().y / 2.0),
        galley,
        content,
    );
    response.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// M3 select field: shows `text` with a chevron and opens a [`menu`] of the same
/// width, filled by `add`.
pub fn select(ui: &mut Ui, text: &str, width: f32, add: impl FnOnce(&mut Ui)) -> Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::new(width, 44.0), Sense::click());
    let enabled = ui.is_enabled();
    response.widget_info(|| WidgetInfo::labeled(WidgetType::ComboBox, enabled, text));
    let open = menu_open(&response);
    let content = if enabled {
        ON_SURFACE
    } else {
        ON_SURFACE.gamma_multiply(0.38)
    };
    let painter = ui.painter();
    painter.rect_filled(rect, 12.0, SURFACE_CONTAINER_HIGHEST);
    painter.rect_filled(rect, 12.0, state_layer(&response, ON_SURFACE));
    if open {
        painter.rect_stroke(
            rect,
            12.0,
            Stroke::new(2.0_f32, PRIMARY),
            egui::StrokeKind::Inside,
        );
    }
    let galley = elided(
        ui,
        text,
        FontId::new(14.0, FontFamily::Proportional),
        content,
        width - 60.0,
    );
    painter.galley(
        egui::pos2(rect.left() + 16.0, rect.center().y - galley.size().y / 2.0),
        galley,
        content,
    );
    paint_chevron(
        ui,
        egui::pos2(rect.right() - 22.0, rect.center().y),
        open,
        if open { PRIMARY } else { ON_SURFACE_VARIANT },
    );
    menu(&response, width, add);
    response.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// A small chevron button that opens a [`menu`] of `width`, filled by `add`.
pub fn menu_button(ui: &mut Ui, hint: &str, width: f32, add: impl FnOnce(&mut Ui)) -> Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::new(32.0, 28.0), Sense::click());
    let enabled = ui.is_enabled();
    response.widget_info(|| WidgetInfo::labeled(WidgetType::ComboBox, enabled, hint));
    let open = menu_open(&response);
    let painter = ui.painter();
    painter.rect_filled(rect, 8.0, SURFACE_CONTAINER_HIGHEST);
    painter.rect_filled(rect, 8.0, state_layer(&response, ON_SURFACE));
    paint_chevron(
        ui,
        rect.center(),
        open,
        if open { PRIMARY } else { ON_SURFACE_VARIANT },
    );
    menu(&response, width, add);
    response
        .on_hover_cursor(egui::CursorIcon::PointingHand)
        .on_hover_text(hint)
}

pub fn text_edit<'a>(text: &'a mut String, hint: &str, width: f32) -> egui::TextEdit<'a> {
    egui::TextEdit::singleline(text)
        .hint_text(hint)
        .desired_width(width - 28.0)
        .margin(Margin::symmetric(14, 10))
        .background_color(Color32::TRANSPARENT)
}

#[cfg(test)]
mod tests {
    use super::Format;

    #[test]
    fn numbers_read_and_type_in_their_units() {
        let percent = Format::new(" %").scale(100.0);
        assert_eq!(percent.show(0.5, 0.0), "50 %");
        assert_eq!(percent.parse("75%", 0.0), Some(0.75));
        let tune = Format::new(" semitones").decimals(1).signed();
        assert_eq!(tune.show(3.0, -24.0), "+3.0 semitones");
        assert_eq!(tune.parse(" -7,5 ", -24.0), Some(-7.5));
        let gate = Format::new(" dB").off_at_min();
        assert_eq!(gate.show(-100.0, -100.0), "Off");
        assert_eq!(gate.show(-50.0, -100.0), "-50 dB");
        assert_eq!(gate.parse("off", -100.0), Some(-100.0));
        assert_eq!(Format::new(" ms").parse("320 ms", 20.0), Some(320.0));
        assert_eq!(Format::new(" ms").parse("abc", 20.0), None);
    }
}
