//! Material 3 dark theme: colour roles, type, icons and the components the app uses.
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
    let embedded: [(&str, &'static [u8]); 4] = [
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
    let mut regular = vec!["roboto".to_owned()];
    regular.extend(fallback.iter().cloned());
    let mut strong = vec!["roboto-medium".to_owned()];
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

/// A labelled slider row: name on the left, formatted value on the right, the slider,
/// then a plain-language hint for people who do not know the term.
pub fn setting<N: Numeric>(
    ui: &mut Ui,
    label: &str,
    hint: &str,
    shown: String,
    value: &mut N,
    range: RangeInclusive<N>,
    step: f64,
) -> Response {
    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing.y = 0.0;
        let (rect, _) =
            ui.allocate_exact_size(Vec2::new(ui.available_width(), 22.0), Sense::hover());
        ui.painter().text(
            rect.left_center(),
            Align2::LEFT_CENTER,
            label,
            FontId::proportional(14.0),
            ON_SURFACE,
        );
        ui.painter().text(
            rect.right_center(),
            Align2::RIGHT_CENTER,
            shown,
            FontId::new(14.0, medium()),
            ON_SURFACE_VARIANT,
        );
        let response = slider(ui, value, range, step);
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

/// Level meter: an icon followed by a rounded linear indicator.
pub fn meter(ui: &mut Ui, glyph: char, level: f32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(150.0, 24.0), Sense::hover());
    paint_icon(
        ui,
        egui::pos2(rect.left() + 10.0, rect.center().y),
        glyph,
        20.0,
        false,
        ON_SURFACE_VARIANT,
    );
    let track = Rect::from_min_max(
        egui::pos2(rect.left() + 28.0, rect.center().y - 3.0),
        egui::pos2(rect.right(), rect.center().y + 3.0),
    );
    ui.painter().rect_filled(track, 3.0, SECONDARY_CONTAINER);
    let mut active = track;
    active.set_width((track.width() * level.clamp(0.0, 1.0)).max(6.0));
    ui.painter().rect_filled(active, 3.0, PRIMARY);
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
                dismissed = icon_button(ui, icons::CLOSE, content, "Dismiss").clicked();
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
pub fn text_edit<'a>(text: &'a mut String, hint: &str, width: f32) -> egui::TextEdit<'a> {
    egui::TextEdit::singleline(text)
        .hint_text(hint)
        .desired_width(width - 28.0)
        .margin(Margin::symmetric(14, 10))
        .background_color(Color32::TRANSPARENT)
}
