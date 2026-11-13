//! Interface languages. Text is written in English in the code and looked up in
//! [`table::TABLE`]; text without an entry stays English. Backend errors and logs
//! stay English, so reports and searches keep working.
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fmt::Display,
    sync::{
        atomic::{AtomicU8, Ordering},
        OnceLock,
    },
};

mod table;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Lang {
    #[default]
    En,
    Id,
    Ja,
}
impl Lang {
    pub const ALL: [Lang; 3] = [Lang::En, Lang::Id, Lang::Ja];
    /// The language's own name, so it can be found whatever language is shown.
    pub fn name(self) -> &'static str {
        match self {
            Lang::En => "English",
            Lang::Id => "Bahasa Indonesia",
            Lang::Ja => "日本語",
        }
    }
    /// From a locale name such as "id_ID.UTF-8", "ja-JP" or "en-US".
    pub fn from_locale(locale: &str) -> Lang {
        let lower = locale.to_ascii_lowercase();
        if lower.starts_with("id") || lower.starts_with("in_") || lower.starts_with("in-") {
            Lang::Id
        } else if lower.starts_with("ja") {
            Lang::Ja
        } else {
            Lang::En
        }
    }
}

static CURRENT: AtomicU8 = AtomicU8::new(0);

pub fn set(lang: Lang) {
    CURRENT.store(lang as u8, Ordering::Relaxed);
}
pub fn current() -> Lang {
    match CURRENT.load(Ordering::Relaxed) {
        1 => Lang::Id,
        2 => Lang::Ja,
        _ => Lang::En,
    }
}

fn lookup() -> &'static HashMap<&'static str, (&'static str, &'static str)> {
    static MAP: OnceLock<HashMap<&'static str, (&'static str, &'static str)>> = OnceLock::new();
    MAP.get_or_init(|| {
        table::TABLE
            .iter()
            .map(|&(en, id, ja)| (en, (id, ja)))
            .collect()
    })
}

/// `text` in the current language.
pub fn t(text: &'static str) -> &'static str {
    let lang = current();
    if lang == Lang::En {
        return text;
    }
    match lookup().get(text) {
        Some(&(id, ja)) => match lang {
            Lang::Id if !id.is_empty() => id,
            Lang::Ja if !ja.is_empty() => ja,
            _ => text,
        },
        None => text,
    }
}

/// `template` in the current language with each `{}` replaced by the next argument.
pub fn tf(template: &'static str, args: &[&dyn Display]) -> String {
    let mut out = String::new();
    let mut args = args.iter();
    let mut rest = t(template);
    while let Some(at) = rest.find("{}") {
        out.push_str(&rest[..at]);
        if let Some(arg) = args.next() {
            out.push_str(&arg.to_string());
        }
        rest = &rest[at + 2..];
    }
    out.push_str(rest);
    out
}

/// The system's language, for the first start.
pub fn system() -> Lang {
    for name in ["LC_ALL", "LC_MESSAGES", "LANG"] {
        if let Some(value) = std::env::var_os(name) {
            let value = value.to_string_lossy();
            if !value.is_empty() && value != "C" && value != "POSIX" {
                return Lang::from_locale(&value);
            }
        }
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Globalization::GetUserDefaultLocaleName;
        let mut buffer = [0u16; 85];
        let len = unsafe { GetUserDefaultLocaleName(buffer.as_mut_ptr(), buffer.len() as i32) };
        if len > 1 {
            return Lang::from_locale(&String::from_utf16_lossy(&buffer[..len as usize - 1]));
        }
    }
    Lang::En
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `t("…")` and `tf("…")` in the interface has a translation (also when
    /// the text starts on the line after the bracket), and each translation keeps
    /// the template's `{}` placeholders.
    #[test]
    fn every_interface_text_is_translated() {
        let sources = [
            include_str!("app.rs"),
            include_str!("app/browse.rs"),
            include_str!("app/setup.rs"),
            include_str!("m3.rs"),
        ];
        let map = lookup();
        let mut missing = Vec::new();
        for source in sources {
            for call in ["t(", "tf("] {
                let mut rest = source;
                while let Some(at) = rest.find(call) {
                    let preceded = rest[..at].chars().last();
                    rest = &rest[at + call.len()..];
                    if preceded.is_some_and(|c| c.is_alphanumeric() || c == '_') {
                        continue;
                    }
                    let Some(literal) = rest.trim_start().strip_prefix('"') else {
                        continue;
                    };
                    rest = literal;
                    let mut text = String::new();
                    let mut chars = rest.chars();
                    while let Some(c) = chars.next() {
                        match c {
                            '"' => break,
                            '\\' => match chars.next() {
                                Some('n') => text.push('\n'),
                                Some(other) => text.push(other),
                                None => {}
                            },
                            c => text.push(c),
                        }
                    }
                    if !map.contains_key(text.as_str()) {
                        missing.push(text);
                    }
                }
            }
        }
        // Text that reaches `t` through a variable.
        #[cfg(feature = "catalog")]
        for model in &crate::setup::SHARED {
            if !map.contains_key(model.purpose) {
                missing.push(model.purpose.to_owned());
            }
        }
        missing.sort();
        missing.dedup();
        assert!(
            missing.is_empty(),
            "No translation for:\n{}",
            missing.join("\n")
        );
        for &(en, id, ja) in table::TABLE {
            let holes = en.matches("{}").count();
            assert_eq!(id.matches("{}").count(), holes, "{en} -> {id}");
            assert_eq!(ja.matches("{}").count(), holes, "{en} -> {ja}");
            assert!(!id.is_empty() && !ja.is_empty(), "Incomplete: {en}");
        }
        let mut seen = std::collections::HashSet::new();
        for &(en, _, _) in table::TABLE {
            assert!(seen.insert(en), "Listed twice: {en}");
        }
    }

    #[test]
    fn languages_switch_and_fill_placeholders() {
        assert_eq!(Lang::from_locale("id_ID.UTF-8"), Lang::Id);
        assert_eq!(Lang::from_locale("ja-JP"), Lang::Ja);
        assert_eq!(Lang::from_locale("en-US"), Lang::En);
        set(Lang::En);
        assert_eq!(tf("{} of {}", &[&1, &2]), "1 of 2");
        assert_eq!(t("Not in the table"), "Not in the table");
        set(Lang::En);
    }
}
