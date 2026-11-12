//! Searches voice-models.com and downloads a voice from it.
//!
//! The site has no public API; this reads the same `fetch_data.php` endpoint its own
//! search box uses, so it can break if the site changes. Requests stay polite: one per
//! search or page, never per keystroke. The files themselves are hosted elsewhere
//! (mostly Hugging Face); only links that point at a file are downloaded.
use anyhow::{bail, Context, Result};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const SITE: &str = "https://voice-models.com";
const USER_AGENT: &str = concat!("voice-changer-native/", env!("CARGO_PKG_VERSION"));

#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub title: String,
    pub uploader: Option<String>,
    /// As the site writes it, e.g. "88.5 MB".
    pub size: Option<String>,
    /// The model's page on voice-models.com.
    pub page: String,
    /// The file link as listed.
    pub link: String,
}
impl Entry {
    /// Whether the app can fetch this voice's file from its host.
    pub fn downloadable(&self) -> bool {
        source(&self.link).is_some()
    }
    /// Why this one cannot be downloaded from the app, if it cannot.
    pub fn unsupported(&self) -> Option<&'static str> {
        let lower = self.title.to_lowercase();
        if lower.contains("gpt-sovits") || lower.contains("gptsovits") || lower.contains("so-vits")
        {
            return Some("Not an RVC voice");
        }
        if !self.downloadable() {
            return Some("Opens on its host's page");
        }
        None
    }
}

pub struct Page {
    pub entries: Vec<Entry>,
    pub has_more: bool,
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .user_agent(USER_AGENT)
        .timeout_connect(std::time::Duration::from_secs(15))
        .timeout_read(std::time::Duration::from_secs(60))
        .build()
}

/// One page of results (from 1); an empty query lists the newest voices.
pub fn search(query: &str, page: u32) -> Result<Page> {
    log::info!("Searching voice-models.com for \"{query}\" (page {page})");
    let response = agent()
        .post(&format!("{SITE}/fetch_data.php"))
        .send_form(&[("page", &page.to_string()), ("search", query.trim())])
        .context("voice-models.com did not answer; check the internet connection")?;
    let body: serde_json::Value = response
        .into_string()
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .context("voice-models.com sent an unexpected reply; its search may have changed")?;
    let table = body["table"].as_str().unwrap_or_default();
    let pagination = body["pagination"].as_str().unwrap_or_default();
    let entries = parse_table(table);
    log::info!("{} results", entries.len());
    Ok(Page {
        has_more: has_page(pagination, page + 1),
        entries,
    })
}

fn parse_table(table: &str) -> Vec<Entry> {
    table
        .split("<tr>")
        .skip(1)
        .filter_map(|row| {
            let row = row.split("</tr>").next()?;
            let (slug, title) = between(row, "href='/model/", "</a>")?.split_once('>')?;
            let slug = slug.split('\'').next()?;
            let link = attribute(row, "data-clipboard-text=")
                .or_else(|| attribute(row, "run?url=").map(|url| percent_decode(&url)))?;
            Some(Entry {
                title: clean_text(title),
                uploader: between(row, "title='Uploaded by ", "'").map(clean_text),
                size: between(row, "bg-secondary text-white ms-2'>", "<").map(clean_text),
                page: format!("{SITE}/model/{slug}"),
                link,
            })
        })
        .filter(|entry| !entry.title.is_empty())
        .collect()
}

fn between<'a>(text: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let from = text.find(start)? + start.len();
    let to = text[from..].find(end)? + from;
    Some(&text[from..to])
}

/// The value of `name='…'` (or `"…"`), or of a URL parameter ending at a quote.
fn attribute(row: &str, name: &str) -> Option<String> {
    let from = row.find(name)? + name.len();
    let rest = &row[from..];
    let (quote, rest) = match rest.chars().next()? {
        q @ ('\'' | '"') => (q, &rest[1..]),
        _ => ('\'', rest),
    };
    let value = &rest[..rest.find([quote, '"', '\''])?];
    (!value.is_empty()).then(|| decode_entities(value))
}

/// Text without tags (the site highlights search words with spans), entities decoded.
fn clean_text(html: &str) -> String {
    let mut text = String::new();
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            c if !in_tag => text.push(c),
            _ => {}
        }
    }
    decode_entities(
        text.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .as_str(),
    )
}

fn decode_entities(text: &str) -> String {
    text.replace("&#039;", "'")
        .replace("&#39;", "'")
        .replace("&quot;", "\"")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(&text[i + 1..i + 3], 16) {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn has_page(pagination: &str, page: u32) -> bool {
    [
        format!("fetchData({page},"),
        format!("fetchData({page} ,"),
        format!(">{page}<"),
    ]
    .iter()
    .any(|needle| pagination.contains(needle.as_str()))
}

/// Where a voice's file comes from.
#[derive(Debug, PartialEq)]
enum Source {
    /// A link that serves the file itself.
    Direct(String),
    /// A public Google Drive file, by id.
    Drive(String),
    /// A MediaFire file page, which holds the real link.
    MediaFire(String),
}

/// Hugging Face `blob` pages become `resolve` links. Repository pages, Drive folders
/// and Mega (encrypted in the browser) are not files the app can fetch.
fn source(link: &str) -> Option<Source> {
    let link = link.trim();
    if !link.starts_with("https://") {
        return None;
    }
    let path = link.split(['?', '#']).next()?.to_lowercase();
    let is_file = path.ends_with(".zip") || path.ends_with(".pth");
    if link.starts_with("https://huggingface.co/") {
        if link.contains("/tree/") || !is_file {
            return None;
        }
        return Some(Source::Direct(link.replacen("/blob/", "/resolve/", 1)));
    }
    if link.contains("drive.google.com/") || link.contains("drive.usercontent.google.com/") {
        if link.contains("/folders/") {
            return None;
        }
        return drive_id(link).map(Source::Drive);
    }
    if link.contains("mediafire.com/file/") {
        return Some(Source::MediaFire(link.to_owned()));
    }
    let unsupported = ["mega.nz", "mega.co.nz", "pixeldrain.com"];
    (is_file && unsupported.iter().all(|host| !link.contains(host)))
        .then(|| Source::Direct(link.to_owned()))
}

/// The file id in `/file/d/<id>/…`, `?id=<id>` or `&id=<id>` links.
fn drive_id(link: &str) -> Option<String> {
    let id = if let Some(rest) = link.split("/file/d/").nth(1) {
        rest.split(['/', '?', '#']).next()?
    } else {
        let query = link.split_once('?')?.1;
        query
            .split('&')
            .find_map(|pair| pair.strip_prefix("id="))?
            .split('#')
            .next()?
    };
    let valid = !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    valid.then(|| id.to_owned())
}

/// The URL that serves the file, reading the host's page first where needed.
pub fn resolve(link: &str) -> Result<String> {
    match source(link).context("This voice's host does not allow downloads from the app")? {
        Source::Direct(url) => Ok(url),
        // confirm=t skips the "too large to scan for viruses" page.
        Source::Drive(id) => Ok(format!(
            "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"
        )),
        Source::MediaFire(page) => {
            let html = agent()
                .get(&page)
                .call()
                .context("MediaFire did not answer")?
                .into_string()
                .context("MediaFire sent an unreadable page")?;
            mediafire_link(&html)
                .context("MediaFire did not offer a download link; the file may have been removed")
        }
    }
}

/// The download button's link on a MediaFire file page; newer pages store it
/// base64-encoded in `data-scrambled-url`.
fn mediafire_link(html: &str) -> Option<String> {
    let button = &html[html
        .find("id=\"downloadButton\"")
        .or_else(|| html.find("downloadButton"))?
        .saturating_sub(600)..];
    let button = &button[..button.len().min(2000)];
    if let Some(encoded) = between(button, "data-scrambled-url=\"", "\"") {
        if let Some(url) = base64_decode(encoded).and_then(|b| String::from_utf8(b).ok()) {
            if url.starts_with("https://") {
                return Some(url);
            }
        }
    }
    let href = between(button, "href=\"https://download", "\"")?;
    Some(format!("https://download{href}"))
}

fn base64_decode(text: &str) -> Option<Vec<u8>> {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::new();
    let (mut bits, mut count) = (0u32, 0);
    for byte in text.trim().bytes().filter(|b| *b != b'=') {
        let value = ALPHABET.iter().position(|c| *c == byte)? as u32;
        bits = (bits << 6) | value;
        count += 6;
        if count >= 8 {
            count -= 8;
            out.push((bits >> count) as u8);
        }
    }
    Some(out)
}

/// The file name a server gives in `Content-Disposition`.
fn disposition_name(header: &str) -> Option<String> {
    if let Some(encoded) = header.split("filename*=").nth(1) {
        let value = encoded.split(';').next()?.trim().trim_matches('"');
        let value = value.rsplit("''").next()?;
        return Some(percent_decode(value));
    }
    let value = header.split("filename=").nth(1)?.split(';').next()?;
    Some(value.trim().trim_matches('"').to_owned())
}

/// Download progress, shared with the UI.
#[derive(Default)]
pub struct Progress {
    pub received: AtomicU64,
    /// 0 while unknown.
    pub total: AtomicU64,
    pub cancel: AtomicBool,
}

/// Downloads `url` into `folder` and returns the file. Stops early when cancelled.
pub fn download(url: &str, folder: &Path, progress: &Progress) -> Result<PathBuf> {
    log::info!("Downloading {url}");
    let response = agent()
        .get(url)
        .call()
        .with_context(|| format!("Download failed: {url}"))?;
    // A host that refuses (a private Drive file, its daily limit, a removed file)
    // answers with a web page instead of the file.
    if response.content_type().contains("html") {
        bail!(
            "The host sent a web page instead of the voice; the file may be private, removed, or over its download limit. Try Page to open it in the browser."
        );
    }
    let total = response
        .header("Content-Length")
        .and_then(|length| length.parse().ok())
        .unwrap_or(0);
    progress.total.store(total, Ordering::Relaxed);
    let name = response
        .header("Content-Disposition")
        .and_then(disposition_name)
        .or_else(|| {
            url.split(['?', '#'])
                .next()
                .and_then(|path| path.rsplit('/').next())
                .map(percent_decode)
        })
        .filter(|name| !name.is_empty() && name.contains('.'))
        .unwrap_or_else(|| "voice.zip".into());
    // Only the file name is kept, so a crafted URL cannot write elsewhere.
    let name = Path::new(&name)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "voice.zip".into());
    std::fs::create_dir_all(folder)?;
    let target = folder.join(&name);
    let mut file = std::fs::File::create(&target)?;
    let mut reader = response.into_reader();
    let mut buffer = vec![0; 1 << 16];
    let started = std::time::Instant::now();
    loop {
        if progress.cancel.load(Ordering::Relaxed) {
            drop(file);
            let _ = std::fs::remove_file(&target);
            bail!("Download cancelled");
        }
        let read = reader.read(&mut buffer).context("Download interrupted")?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])?;
        progress.received.fetch_add(read as u64, Ordering::Relaxed);
    }
    let received = progress.received.load(Ordering::Relaxed);
    anyhow::ensure!(
        total == 0 || received == total,
        "Download incomplete ({received} of {total} bytes)"
    );
    drop(file);
    // Both .zip archives and .pth checkpoints are zip files.
    let mut magic = [0u8; 2];
    let zipped = std::fs::File::open(&target)
        .and_then(|mut file| file.read_exact(&mut magic))
        .is_ok()
        && &magic == b"PK";
    if !zipped {
        let _ = std::fs::remove_file(&target);
        bail!(
            "The download is not a voice archive (.zip or .pth); open the voice's page to check it"
        );
    }
    log::info!(
        "Downloaded {} ({:.1} MB in {:.1} s)",
        name,
        received as f64 / 1_048_576.0,
        started.elapsed().as_secs_f32()
    );
    Ok(target)
}

/// A readable voice name from a listing title: bracketed tags go, and so do
/// parenthesised training details such as "(RVC V2)" or "(300 epochs)".
pub fn voice_name(title: &str) -> String {
    const DETAILS: [&str; 16] = [
        "rvc", "epoch", "pretrain", "rmvpe", "crepe", "harvest", "steps", "mangio", "ov2", "titan",
        "kits", "v1", "v2", "40k", "48k", "32k",
    ];
    let mut name = String::new();
    let mut rest = title;
    while let Some(open) = rest.find(['[', '(']) {
        let close = if rest[open..].starts_with('[') {
            ']'
        } else {
            ')'
        };
        let Some(end) = rest[open..].find(close).map(|end| open + end) else {
            break;
        };
        let inner = rest[open + 1..end].to_lowercase();
        name.push_str(&rest[..open]);
        if close == ')' && !DETAILS.iter().any(|detail| inner.contains(detail)) {
            name.push_str(&rest[open..=end]);
        }
        rest = &rest[end + 1..];
    }
    name.push_str(rest);
    let name = name.split_whitespace().collect::<Vec<_>>().join(" ");
    let name = name.trim_matches(|c: char| c == '-' || c == '|' || c.is_whitespace());
    if name.is_empty() {
        title.trim().to_owned()
    } else {
        name.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TABLE: &str = "<tr><td><a href='/model/4kz' style='color:#fff;' class='fs-5'><span style='background-color: #063AA4;'>Zeta</span></a> <a href='https://easyaivoice.com/run?url=https%3A%2F%2Fmodels.weights.gg%2Fbbaf.zip' target='_blank'>Run</a></td><td><button class='copy-btn' data-clipboard-text='https://models.weights.gg/bbaf.zip' title='Click to copy URL'></button></td></tr>\
<tr><td><a href='/model/ac6' style='color:#fff;' class='fs-5'>Anthony Girard (Swat 4) [RVC v2] [118 Epochs]</a><a href='/creator/Sgt_Zero' class='badge' title='Uploaded by Sgt_Zero'>Sgt_Zero</a><span class='badge bg-secondary text-white ms-2'>88.5 MB</span> <a href='https://easyaivoice.com/run?url=https%3A%2F%2Fhuggingface.co%2FS%2FP%2Fblob%2Fmain%2FAnthony%2520Girard.zip' target='_blank'>Run</a></td></tr>\
<tr><td><a href='/model/1oI' class='fs-5'>Kamillie (<span>Zeta</span> Gundam) [RVCv2]</a> <a href='https://easyaivoice.com/run?url=https%3A%2F%2Fhuggingface.co%2FHamiltek%2FKamillieBidan%2Ftree%2Fmain'>Run</a></td><td><button class='copy-btn' data-clipboard-text='https://huggingface.co/Hamiltek/KamillieBidan/tree/main'></button></td></tr>";

    #[test]
    fn reads_listing_rows() {
        let entries = parse_table(TABLE);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].title, "Zeta");
        assert_eq!(entries[0].link, "https://models.weights.gg/bbaf.zip");
        assert_eq!(entries[0].page, "https://voice-models.com/model/4kz");
        assert_eq!(
            entries[1].title,
            "Anthony Girard (Swat 4) [RVC v2] [118 Epochs]"
        );
        assert_eq!(entries[1].uploader.as_deref(), Some("Sgt_Zero"));
        assert_eq!(entries[1].size.as_deref(), Some("88.5 MB"));
        assert_eq!(
            source(&entries[1].link),
            Some(Source::Direct(
                "https://huggingface.co/S/P/resolve/main/Anthony%20Girard.zip".into()
            ))
        );
        assert_eq!(entries[2].title, "Kamillie (Zeta Gundam) [RVCv2]");
        assert_eq!(entries[2].unsupported(), Some("Opens on its host's page"));
    }

    #[test]
    fn knows_which_hosts_it_can_fetch() {
        let drive = |id: &str| Some(Source::Drive(id.into()));
        assert_eq!(
            source("https://drive.google.com/file/d/1AbC-_x/view?usp=sharing"),
            drive("1AbC-_x")
        );
        assert_eq!(
            source("https://drive.google.com/uc?export=download&id=1mzA5n"),
            drive("1mzA5n")
        );
        assert_eq!(
            source("https://drive.google.com/open?id=XYZ#x"),
            drive("XYZ")
        );
        assert_eq!(source("https://drive.google.com/drive/folders/1abc"), None);
        assert_eq!(
            source("https://www.mediafire.com/file/0ik8/Polly.zip/file"),
            Some(Source::MediaFire(
                "https://www.mediafire.com/file/0ik8/Polly.zip/file".into()
            ))
        );
        assert_eq!(source("https://mega.nz/file/x#key"), None);
        assert_eq!(source("http://example.com/a.zip"), None);
        assert_eq!(source("https://huggingface.co/a/b/tree/main"), None);
        assert_eq!(
            source("https://huggingface.co/a/b/resolve/main/c.zip?download=true"),
            Some(Source::Direct(
                "https://huggingface.co/a/b/resolve/main/c.zip?download=true".into()
            ))
        );
    }

    #[test]
    fn reads_mediafire_buttons_and_file_names() {
        let plain = r#"<a class="input popsok" aria-label="Download file" href="https://download1234.mediafire.com/abc/Polly.zip" id="downloadButton">"#;
        assert_eq!(
            mediafire_link(plain).as_deref(),
            Some("https://download1234.mediafire.com/abc/Polly.zip")
        );
        // base64 of https://download9.mediafire.com/x/y.zip
        let scrambled = r#"<a href="javascript:void(0)" data-scrambled-url="aHR0cHM6Ly9kb3dubG9hZDkubWVkaWFmaXJlLmNvbS94L3kuemlw" id="downloadButton">"#;
        assert_eq!(
            mediafire_link(scrambled).as_deref(),
            Some("https://download9.mediafire.com/x/y.zip")
        );
        assert_eq!(
            disposition_name("attachment; filename=\"Polly Sparks.zip\"").as_deref(),
            Some("Polly Sparks.zip")
        );
        assert_eq!(
            disposition_name("attachment; filename=\"x\"; filename*=UTF-8''Diao%20Chan.zip")
                .as_deref(),
            Some("Diao Chan.zip")
        );
    }

    #[test]
    fn sees_further_pages() {
        let pagination = "<li class='page-item active'><a onclick=\"fetchData(1, '')\">1</a></li><li><a onclick=\"fetchData(2, '')\">2</a></li>";
        assert!(has_page(pagination, 2));
        assert!(!has_page(pagination, 3));
    }

    #[test]
    fn names_voices_without_tags() {
        assert_eq!(
            voice_name("Anthony Girard (Swat 4) [RVC v2] [RMVPE] [118 Epochs]"),
            "Anthony Girard (Swat 4)"
        );
        assert_eq!(voice_name("[RVC]"), "[RVC]");
        assert_eq!(
            voice_name("Vestia Zeta (Hololive ID) (RVC V2) (300 epochs) (Ov2 Pretrain)"),
            "Vestia Zeta (Hololive ID)"
        );
        assert_eq!(
            voice_name(
                "Kamillie Bidan (Zeta Gundam) (VA:Nobuo Tobita) [RVCv2|mangio-crepe|120epochs]"
            ),
            "Kamillie Bidan (Zeta Gundam) (VA:Nobuo Tobita)"
        );
    }
}
