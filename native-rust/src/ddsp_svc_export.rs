//! Builds a DDSP-SVC voice's ONNX graphs from its checkpoints: the DDSP model's
//! `Unit2Control` (a PCmer, as `crate::diffusion_svc_export` builds Diffusion-SVC's
//! naive model) and, when one is given, the separately trained shallow-diffusion
//! model's conditioning and denoiser. DDSP-SVC keeps each model's settings in the
//! `config.yaml` it was trained with, beside the checkpoint.
use crate::ddsp_svc::{DdspInfo, Synth, VOCODER_HOP, VOCODER_RATE};
use crate::diffusion_svc::Info;
use crate::diffusion_svc_export::{
    diffusion_graphs, pcmer_graph, state_dict, DiffusionNet, PcmerNet, Weights,
};
use crate::pth;
use anyhow::{bail, ensure, Context, Result};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

/// The encoders the reference feeds from its ContentVec (`hubert_base`) file at
/// layer 12, which `contentvec.onnx`'s `unit12` output is.
const ENCODERS: [&str; 2] = ["contentvec768l12", "hubertbase768l12"];

/// A DDSP-SVC `config.yaml`, flattened to `section.key` strings. Only plain
/// `key: value` mappings are read; lists and anything else are skipped.
#[derive(Debug, Default)]
pub(crate) struct Config(HashMap<String, String>);

impl Config {
    pub(crate) fn parse(text: &str) -> Self {
        let mut values = HashMap::new();
        let mut path: Vec<(usize, String)> = Vec::new();
        for line in text.lines() {
            let line = strip_comment(line);
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('-') {
                continue;
            }
            let indent = line.len() - line.trim_start().len();
            let Some((key, value)) = trimmed.split_once(':') else {
                continue;
            };
            while path.last().is_some_and(|(i, _)| *i >= indent) {
                path.pop();
            }
            let key = unquote(key.trim()).to_owned();
            let value = value.trim();
            if value.is_empty() {
                path.push((indent, key));
            } else {
                let mut full: Vec<&str> = path.iter().map(|(_, k)| k.as_str()).collect();
                full.push(&key);
                values.insert(full.join("."), unquote(value).to_owned());
            }
        }
        Self(values)
    }

    fn read(model: &Path) -> Result<Self> {
        let file = model.with_file_name("config.yaml");
        let text = std::fs::read_to_string(&file).with_context(|| {
            format!(
                "DDSP-SVC keeps a model's settings in the config.yaml it was trained with, beside the model; {} is missing",
                file.display()
            )
        })?;
        Ok(Self::parse(&text))
    }

    pub(crate) fn text(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }

    pub(crate) fn int(&self, key: &str) -> Option<i64> {
        let value = self.text(key)?;
        value.parse::<i64>().ok().or_else(|| {
            value
                .parse::<f64>()
                .ok()
                .filter(|v| v.fract() == 0.0)
                .map(|v| v as i64)
        })
    }

    fn need(&self, key: &str) -> Result<usize> {
        let value = self
            .int(key)
            .with_context(|| format!("The model's config.yaml has no {key}"))?;
        ensure!(
            value > 0,
            "The model's config.yaml's {key} must be positive"
        );
        Ok(value as usize)
    }

    fn flag(&self, key: &str) -> bool {
        matches!(
            self.text(key).map(str::to_ascii_lowercase).as_deref(),
            Some("true" | "1" | "yes")
        )
    }
}

fn strip_comment(line: &str) -> &str {
    let mut quote = None;
    for (i, c) in line.char_indices() {
        match (c, quote) {
            ('"' | '\'', None) => quote = Some(c),
            (c, Some(q)) if c == q => quote = None,
            ('#', None) => return &line[..i],
            _ => {}
        }
    }
    line
}

fn unquote(value: &str) -> &str {
    let value = value.trim();
    for q in ['"', '\''] {
        if value.len() >= 2 && value.starts_with(q) && value.ends_with(q) {
            return &value[1..value.len() - 1];
        }
    }
    value
}

/// The data settings both models must share.
fn data(config: &Config) -> Result<(u32, usize, usize, String)> {
    let encoder = config.text("data.encoder").unwrap_or("").to_owned();
    ensure!(
        ENCODERS.contains(&encoder.as_str()),
        "This model uses the {encoder:?} encoder; only contentvec768l12 (ContentVec, 768 channels, layer 12) and hubertbase768l12 are supported"
    );
    Ok((
        config.need("data.sampling_rate")? as u32,
        config.need("data.block_size")?,
        config.need("data.encoder_out_channels")?,
        encoder,
    ))
}

fn load_weights(path: &Path, what: &str) -> Result<Weights> {
    let root = pth::load(path).with_context(|| format!("Read the {what} {}", path.display()))?;
    state_dict(&root, what)
}

/// What [`export`] wrote.
#[derive(Debug, Clone)]
pub struct Exported {
    pub ddsp: PathBuf,
    pub condition: Option<PathBuf>,
    pub denoiser: Option<PathBuf>,
    pub info: DdspInfo,
    pub diffusion: Option<Info>,
}

/// Builds `ddsp.onnx` in `folder` from a DDSP-SVC model (`model_*.pt` with its
/// `config.yaml`), and `condition.onnx` and `denoiser.onnx` from its diffusion
/// model when `diffusion` names one.
pub fn export(model: &Path, diffusion: Option<&Path>, folder: &Path) -> Result<Exported> {
    let config = Config::read(model)?;
    let kind = config.text("model.type").unwrap_or("");
    let synth = match kind {
        "Sins" => Synth::Sins,
        "CombSub" => Synth::CombSub,
        "CombSubFast" => Synth::CombSubFast,
        "Diffusion" => bail!(
            "This is a DDSP-SVC diffusion model; import its DDSP model (Sins, CombSub or CombSubFast) and give this one as its diffusion model"
        ),
        other => bail!(
            "Unsupported DDSP-SVC model type {other:?}; Sins, CombSub and CombSubFast (DDSP-SVC 3.0) are supported"
        ),
    };
    let (sample_rate, block_size, units, encoder) = data(&config)?;
    ensure!(
        sample_rate == VOCODER_RATE && block_size == VOCODER_HOP,
        "This model is for {sample_rate} Hz with {block_size} samples per frame; the NSF-HiFiGAN vocoder needs {VOCODER_RATE} Hz and {VOCODER_HOP}"
    );
    let controls = match synth {
        Synth::Sins => [
            config.need("model.n_harmonics")?,
            config.need("model.n_mag_allpass")?,
            config.need("model.n_mag_noise")?,
        ],
        Synth::CombSub => [
            config.need("model.n_mag_allpass")?,
            config.need("model.n_mag_harmonic")?,
            config.need("model.n_mag_noise")?,
        ],
        Synth::CombSubFast => [block_size + 1; 3],
    };
    let speakers = config.int("model.n_spk").unwrap_or(1).max(1) as usize;
    let weights: Weights = load_weights(model, "DDSP-SVC model")?
        .into_iter()
        .filter_map(|(key, tensor)| {
            key.strip_prefix("unit2ctrl.")
                .map(|key| (key.to_owned(), tensor))
        })
        .collect();
    ensure!(
        !weights.is_empty(),
        "Not a DDSP-SVC model: the checkpoint has no unit2ctrl weights"
    );
    let layers = (0..)
        .take_while(|i| weights.contains_key(&format!("decoder._layers.{i}.norm.weight")))
        .count();
    let mut info = DdspInfo {
        sample_rate,
        block_size,
        units,
        speakers,
        synth,
        controls,
        log10_mel: false,
    };

    std::fs::create_dir_all(folder)?;
    let mut exported = Exported {
        ddsp: folder.join("ddsp.onnx"),
        condition: None,
        denoiser: None,
        info: info.clone(),
        diffusion: None,
    };
    if let Some(diffusion) = diffusion {
        let diff_config = Config::read(diffusion)?;
        let kind = diff_config.text("model.type").unwrap_or("");
        ensure!(
            kind == "Diffusion",
            "The diffusion model's config.yaml says it is a {kind:?} model, not a Diffusion one"
        );
        let (diff_rate, diff_block, diff_units, diff_encoder) = data(&diff_config)?;
        ensure!(
            (diff_rate, diff_block, diff_units, diff_encoder.as_str())
                == (sample_rate, block_size, units, encoder.as_str()),
            "The diffusion model was trained for {diff_rate} Hz, {diff_block} samples a frame and the {diff_encoder} encoder; the DDSP model for {sample_rate} Hz, {block_size} and {encoder}. They must match"
        );
        let vocoder = diff_config.text("vocoder.type").unwrap_or("nsf-hifigan");
        info.log10_mel = match vocoder {
            "nsf-hifigan" => false,
            "nsf-hifigan-log10" => true,
            other => bail!(
                "The diffusion model uses the {other:?} vocoder; only nsf-hifigan is supported"
            ),
        };
        let net = DiffusionNet {
            units,
            sample_rate,
            block_size,
            speakers: diff_config.int("model.n_spk").unwrap_or(1).max(1) as usize,
            layers: diff_config.need("model.n_layers")?,
            chans: diff_config.need("model.n_chans")?,
            hidden: diff_config.need("model.n_hidden")?,
            k_step_max: diff_config.int("model.k_step_max").unwrap_or(0).max(0) as usize,
        };
        if diff_config.flag("model.use_speaker_encoder") {
            bail!("Diffusion models with a speaker encoder are not supported");
        }
        let weights = load_weights(diffusion, "DDSP-SVC diffusion model")?;
        let (condition, denoiser, diffusion_info) = diffusion_graphs(&net, &weights)?;
        let (condition_file, denoiser_file) =
            (folder.join("condition.onnx"), folder.join("denoiser.onnx"));
        std::fs::write(&condition_file, condition)?;
        std::fs::write(&denoiser_file, denoiser)?;
        exported.condition = Some(condition_file);
        exported.denoiser = Some(denoiser_file);
        exported.diffusion = Some(diffusion_info);
        exported.info = info.clone();
    }
    let metadata = serde_json::to_string(&info)?;
    let graph = pcmer_graph(
        &PcmerNet {
            units,
            speakers,
            layers,
            phase: true,
            output: "controls",
            output_size: info.control_size(),
        },
        &weights,
        "ddsp_svc",
        &[("ddsp_svc", &metadata)],
    )?;
    std::fs::write(&exported.ddsp, graph)?;
    Ok(exported)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_yaml_is_flattened() {
        let config = Config::parse(
            "data:\n  sampling_rate: 44100 # the rate\n  encoder: 'contentvec768l12'\n  extensions: # trained on\n    - wav\nmodel:\n  type: \"CombSub\"\n  n_spk: 2\n  use_pitch_aug: true\nenhancer:\n  ckpt: 'pretrain/nsf_hifigan/model'\n  type: nsf-hifigan\n",
        );
        assert_eq!(config.int("data.sampling_rate"), Some(44100));
        assert_eq!(config.text("data.encoder"), Some("contentvec768l12"));
        assert_eq!(config.text("model.type"), Some("CombSub"));
        assert_eq!(config.int("model.n_spk"), Some(2));
        assert!(config.flag("model.use_pitch_aug"));
        assert_eq!(config.text("enhancer.type"), Some("nsf-hifigan"));
        assert_eq!(config.text("data.extensions"), None);
    }

    #[test]
    fn a_missing_config_is_explained() {
        let dir = std::env::temp_dir().join(format!("ddsp-no-config-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("model_1000.pt");
        std::fs::write(&model, b"not a checkpoint").unwrap();
        let error = export(&model, None, &dir.join("out"))
            .unwrap_err()
            .to_string();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(error.contains("config.yaml"), "{error}");
    }
}
