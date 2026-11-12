//! RVC checkpoint (`.pth`) to ONNX export without Python.
//!
//! The graphs in `templates/` are the reference PyTorch exporter's ONNX graphs with every
//! weight removed (regenerated only by `tools/dev/make_rvc_templates.py`). Each records which
//! checkpoint tensor fills each initializer: the tensor itself, its transpose, or the
//! weight-norm product `g * v / ||v||`. Export copies the template's protobuf unchanged
//! except for those initializers and the model metadata.
use crate::pth::{self, Tensor, Value};
use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use std::{collections::HashMap, path::Path};

/// One template per standard architecture, zlib-compressed: (version, sample rate, f0, realtime).
macro_rules! templates {
    ($(($version:literal, $rate:literal, $f0:literal, $realtime:literal, $file:literal)),* $(,)?) => {
        [$(($version, $rate, $f0, $realtime, include_bytes!(concat!("../templates/", $file)).as_slice())),*]
    };
}
type TemplateEntry = (&'static str, u32, bool, bool, &'static [u8]);
const TEMPLATES: [TemplateEntry; 24] = templates![
    ("v1", 32000, true, false, "rvc-v1-32000.template.onnx.zz"),
    (
        "v1",
        32000,
        true,
        true,
        "rvc-v1-32000-realtime.template.onnx.zz"
    ),
    (
        "v1",
        32000,
        false,
        false,
        "rvc-v1-32000-nof0.template.onnx.zz"
    ),
    (
        "v1",
        32000,
        false,
        true,
        "rvc-v1-32000-nof0-realtime.template.onnx.zz"
    ),
    ("v1", 40000, true, false, "rvc-v1-40000.template.onnx.zz"),
    (
        "v1",
        40000,
        true,
        true,
        "rvc-v1-40000-realtime.template.onnx.zz"
    ),
    (
        "v1",
        40000,
        false,
        false,
        "rvc-v1-40000-nof0.template.onnx.zz"
    ),
    (
        "v1",
        40000,
        false,
        true,
        "rvc-v1-40000-nof0-realtime.template.onnx.zz"
    ),
    ("v1", 48000, true, false, "rvc-v1-48000.template.onnx.zz"),
    (
        "v1",
        48000,
        true,
        true,
        "rvc-v1-48000-realtime.template.onnx.zz"
    ),
    (
        "v1",
        48000,
        false,
        false,
        "rvc-v1-48000-nof0.template.onnx.zz"
    ),
    (
        "v1",
        48000,
        false,
        true,
        "rvc-v1-48000-nof0-realtime.template.onnx.zz"
    ),
    ("v2", 32000, true, false, "rvc-v2-32000.template.onnx.zz"),
    (
        "v2",
        32000,
        true,
        true,
        "rvc-v2-32000-realtime.template.onnx.zz"
    ),
    (
        "v2",
        32000,
        false,
        false,
        "rvc-v2-32000-nof0.template.onnx.zz"
    ),
    (
        "v2",
        32000,
        false,
        true,
        "rvc-v2-32000-nof0-realtime.template.onnx.zz"
    ),
    ("v2", 40000, true, false, "rvc-v2-40000.template.onnx.zz"),
    (
        "v2",
        40000,
        true,
        true,
        "rvc-v2-40000-realtime.template.onnx.zz"
    ),
    (
        "v2",
        40000,
        false,
        false,
        "rvc-v2-40000-nof0.template.onnx.zz"
    ),
    (
        "v2",
        40000,
        false,
        true,
        "rvc-v2-40000-nof0-realtime.template.onnx.zz"
    ),
    ("v2", 48000, true, false, "rvc-v2-48000.template.onnx.zz"),
    (
        "v2",
        48000,
        true,
        true,
        "rvc-v2-48000-realtime.template.onnx.zz"
    ),
    (
        "v2",
        48000,
        false,
        false,
        "rvc-v2-48000-nof0.template.onnx.zz"
    ),
    (
        "v2",
        48000,
        false,
        true,
        "rvc-v2-48000-nof0-realtime.template.onnx.zz"
    ),
];

fn inflate(compressed: &[u8]) -> Result<Vec<u8>> {
    miniz_oxide::inflate::decompress_to_vec_zlib(compressed)
        .map_err(|e| anyhow::anyhow!("Corrupt embedded template: {e:?}"))
}
/// Config entries that do not change the inference graph: spec channels and segment size
/// (training-only posterior encoder) and the speaker count (embedding rows).
const GRAPH_INDEPENDENT_CONFIG: [usize; 3] = [0, 1, 15];

#[derive(Deserialize)]
struct Template {
    format: u32,
    config: Vec<serde_json::Value>,
    channels: usize,
    f0: bool,
    initializers: Vec<(String, String, String)>,
    state_keys: Vec<String>,
}

#[derive(Debug, serde::Serialize)]
pub struct Exported {
    pub version: String,
    pub f0: bool,
    pub sample_rate: u32,
    pub channels: usize,
    pub speakers: usize,
}

/// Exports an RVC v1 (256-channel) or v2 (768-channel) checkpoint, with or without f0,
/// at the standard 32/40/48 kHz architectures, like the reference `onnxExporter` classes.
pub fn export_rvc(checkpoint: &Path, output: &Path, realtime: bool) -> Result<Exported> {
    let root = pth::load(checkpoint)?;
    // Matches RVCModelSlotGenerator: a missing or None version is v1.
    let version = root.get("version").and_then(Value::as_str).unwrap_or("v1");
    let f0 = root.get("f0").and_then(Value::as_int).unwrap_or(1) != 0;
    if version != "v1" && version != "v2" {
        bail!("Unsupported RVC checkpoint version {version}; v1 and v2 are supported");
    }
    let config = root
        .get("config")
        .and_then(Value::to_json)
        .and_then(|c| c.as_array().cloned())
        .context("Checkpoint has no config list")?;
    let sample_rate = config
        .last()
        .and_then(serde_json::Value::as_u64)
        .context("Checkpoint config has no sample rate")? as u32;
    let Some(Value::Dict(items)) = root.get("weight") else {
        bail!("Checkpoint has no weight dictionary");
    };
    let mut weights = HashMap::new();
    for (key, value) in items {
        if let (Value::Str(key), Value::Tensor(tensor)) = (key, value) {
            weights.insert(key.as_str(), tensor);
        }
    }
    let speakers = weights
        .get("emb_g.weight")
        .map(|t| t.shape[0])
        .context("Checkpoint has no speaker embedding")?;

    let (.., compressed) = TEMPLATES
        .iter()
        .find(|t| (t.0, t.1, t.2, t.3) == (version, sample_rate, f0, realtime))
        .with_context(|| format!("No RVC {version} template for {sample_rate} Hz; standard rates are 32000, 40000 and 48000"))?;
    let bytes = &inflate(compressed)?;
    let template = read_template(bytes)?;
    ensure!(template.format == 1, "Unsupported template format");
    ensure!(template.f0 == f0, "Template f0 mismatch");
    ensure!(
        template.config.len() == config.len()
            && (0..config.len())
                .filter(|i| !GRAPH_INDEPENDENT_CONFIG.contains(i))
                .all(|i| template.config[i] == config[i]),
        "Checkpoint architecture {config:?} differs from the standard {sample_rate} Hz RVC {version} configuration"
    );
    let missing: Vec<_> = template
        .state_keys
        .iter()
        .filter(|k| !k.starts_with("enc_q.") && !weights.contains_key(k.as_str()))
        .collect();
    let unexpected: Vec<_> = weights
        .keys()
        .filter(|k| !template.state_keys.iter().any(|s| s == *k))
        .collect();
    ensure!(
        missing.is_empty() && unexpected.is_empty(),
        "Checkpoint architecture mismatch: missing={missing:?}, unexpected={unexpected:?}"
    );

    let get = |key: &str| {
        weights
            .get(key)
            .copied()
            .with_context(|| format!("Checkpoint lacks {key}"))
    };
    let mut initializers = HashMap::new();
    for (name, kind, key) in &template.initializers {
        let tensor = match kind.as_str() {
            "sd" => get(key)?.clone(),
            "sd_t" => transpose(get(key)?)?,
            "wn" => weight_norm(get(&format!("{key}_g"))?, get(&format!("{key}_v"))?)?,
            other => bail!("Unknown template source {other}"),
        };
        ensure!(
            tensor.data.iter().all(|v| v.is_finite()),
            "Non-finite weights in {key}"
        );
        initializers.insert(name.as_str(), tensor);
    }
    // v1 uses the projected layer-9 embedding, v2 layer 12 (RVCModelSlotGenerator).
    let v1 = version == "v1";
    let metadata = serde_json::json!({
        "application": "VC_CLIENT", "version": "2.1", "samplingRate": sample_rate,
        "f0": f0, "embChannels": template.channels, "embOutputLayer": if v1 { 9 } else { 12 },
        "useFinalProj": v1, "embedder": "contentvec", "speakers": speakers, "realtimeDecoder": realtime,
    });
    let model = fill_model(bytes, &initializers, &metadata.to_string())?;
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = output.with_extension("partial.onnx");
    std::fs::write(&temporary, model).with_context(|| format!("Write {}", temporary.display()))?;
    std::fs::rename(&temporary, output)?;
    Ok(Exported {
        version: version.to_owned(),
        f0,
        sample_rate,
        channels: template.channels,
        speakers,
    })
}

fn transpose(tensor: &Tensor) -> Result<Tensor> {
    let [rows, cols] = tensor.shape[..] else {
        bail!("Transpose expects a matrix")
    };
    let data = (0..cols * rows)
        .map(|i| tensor.data[(i % rows) * cols + i / rows])
        .collect();
    Ok(Tensor {
        shape: vec![cols, rows],
        data,
    })
}

/// `torch._weight_norm(v, g, dim=0)`: each output channel of `v` scaled to norm `g`.
fn weight_norm(g: &Tensor, v: &Tensor) -> Result<Tensor> {
    let channels = *v.shape.first().context("Empty weight-norm tensor")?;
    ensure!(
        g.data.len() == channels && v.data.len().is_multiple_of(channels),
        "Weight-norm shape mismatch"
    );
    let mut data = v.data.clone();
    for (channel, row) in data.chunks_exact_mut(v.data.len() / channels).enumerate() {
        let norm = row.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let scale = g.data[channel] as f64 / norm;
        row.iter_mut().for_each(|x| *x = (*x as f64 * scale) as f32);
    }
    Ok(Tensor {
        shape: v.shape.clone(),
        data,
    })
}

fn read_template(model: &[u8]) -> Result<Template> {
    for field in fields(model)? {
        if let (14, Wire::Bytes(entry)) = field {
            let (mut key, mut value) = (None, None);
            for (number, wire) in fields(entry)? {
                match (number, wire) {
                    (1, Wire::Bytes(b)) => key = Some(b),
                    (2, Wire::Bytes(b)) => value = Some(b),
                    _ => {}
                }
            }
            if key == Some(b"vc_template".as_slice()) {
                return Ok(serde_json::from_slice(
                    value.context("Empty template recipe")?,
                )?);
            }
        }
    }
    bail!("Template has no vc_template recipe")
}

/// Template ModelProto with weights inserted into GraphProto.initializer and metadata replaced.
fn fill_model(model: &[u8], weights: &HashMap<&str, Tensor>, metadata: &str) -> Result<Vec<u8>> {
    let mut out =
        Vec::with_capacity(model.len() + weights.values().map(|t| t.data.len() * 4).sum::<usize>());
    let mut filled = 0;
    for (number, wire, raw) in fields_raw(model)? {
        match (number, wire) {
            (7, Wire::Bytes(graph)) => {
                let mut graph_out = Vec::new();
                for (number, wire, raw) in fields_raw(graph)? {
                    let (5, Wire::Bytes(tensor)) = (number, wire) else {
                        graph_out.extend_from_slice(raw);
                        continue;
                    };
                    let (name, dims, data_type) = tensor_header(tensor)?;
                    ensure!(data_type == 1, "Template initializer {name} is not float32");
                    let weight = weights
                        .get(name.as_str())
                        .with_context(|| format!("No weights for {name}"))?;
                    // Speaker count is the only shape allowed to differ from the template.
                    ensure!(
                        dims == weight.shape
                            || (name == "emb_g.weight" && dims[1..] == weight.shape[1..]),
                        "Shape mismatch for {name}: template {dims:?}, checkpoint {:?}",
                        weight.shape
                    );
                    let mut message = Vec::with_capacity(weight.data.len() * 4 + name.len() + 32);
                    for dim in &weight.shape {
                        varint_field(&mut message, 1, *dim as u64);
                    }
                    varint_field(&mut message, 2, 1);
                    bytes_field(&mut message, 8, name.as_bytes());
                    let raw_data: Vec<u8> =
                        weight.data.iter().flat_map(|v| v.to_le_bytes()).collect();
                    bytes_field(&mut message, 9, &raw_data);
                    bytes_field(&mut graph_out, 5, &message);
                    filled += 1;
                }
                bytes_field(&mut out, 7, &graph_out);
            }
            // Drop the template recipe; real metadata is appended below.
            (14, _) => {}
            _ => out.extend_from_slice(raw),
        }
    }
    ensure!(
        filled == weights.len(),
        "Template initializers do not match its recipe"
    );
    let mut entry = Vec::new();
    bytes_field(&mut entry, 1, b"metadata");
    bytes_field(&mut entry, 2, metadata.as_bytes());
    bytes_field(&mut out, 14, &entry);
    Ok(out)
}

fn tensor_header(tensor: &[u8]) -> Result<(String, Vec<usize>, u64)> {
    let (mut name, mut dims, mut data_type) = (String::new(), Vec::new(), 0);
    for (number, wire) in fields(tensor)? {
        match (number, wire) {
            (1, Wire::Varint(v)) => dims.push(v as usize),
            (1, Wire::Bytes(packed)) => {
                let mut pos = 0;
                while pos < packed.len() {
                    dims.push(varint(packed, &mut pos)? as usize);
                }
            }
            (2, Wire::Varint(v)) => data_type = v,
            (8, Wire::Bytes(b)) => name = String::from_utf8(b.to_vec())?,
            (9 | 4, _) => bail!("Template initializer {name} still holds data"),
            _ => {}
        }
    }
    Ok((name, dims, data_type))
}

enum Wire<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
    Fixed,
}

fn fields(message: &[u8]) -> Result<Vec<(u64, Wire<'_>)>> {
    Ok(fields_raw(message)?
        .into_iter()
        .map(|(n, w, _)| (n, w))
        .collect())
}

/// Protobuf fields with their decoded payload and complete encoded bytes.
fn fields_raw(message: &[u8]) -> Result<Vec<(u64, Wire<'_>, &[u8])>> {
    let mut fields = Vec::new();
    let mut pos = 0;
    while pos < message.len() {
        let start = pos;
        let tag = varint(message, &mut pos)?;
        let wire = match tag & 7 {
            0 => Wire::Varint(varint(message, &mut pos)?),
            1 | 5 => {
                pos += if tag & 7 == 1 { 8 } else { 4 };
                Wire::Fixed
            }
            2 => {
                let len = varint(message, &mut pos)? as usize;
                let bytes = message
                    .get(pos..pos.saturating_add(len))
                    .context("Truncated protobuf")?;
                pos += len;
                Wire::Bytes(bytes)
            }
            other => bail!("Unsupported protobuf wire type {other}"),
        };
        ensure!(pos <= message.len(), "Truncated protobuf");
        fields.push((tag >> 3, wire, &message[start..pos]));
    }
    Ok(fields)
}

fn varint(bytes: &[u8], pos: &mut usize) -> Result<u64> {
    let mut value = 0;
    for shift in (0..64).step_by(7) {
        let byte = *bytes.get(*pos).context("Truncated protobuf varint")?;
        *pos += 1;
        value |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    bail!("Protobuf varint too long")
}

fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push(value as u8 | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn varint_field(out: &mut Vec<u8>, number: u64, value: u64) {
    write_varint(out, number << 3);
    write_varint(out, value);
}

fn bytes_field(out: &mut Vec<u8>, number: u64, bytes: &[u8]) {
    write_varint(out, (number << 3) | 2);
    write_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_template_has_a_complete_recipe() {
        for (version, rate, f0, realtime, compressed) in TEMPLATES {
            let bytes = &inflate(compressed).unwrap();
            let template = read_template(bytes).unwrap();
            let variant = format!("{version} {rate} f0={f0} realtime={realtime}");
            assert_eq!(
                template.config.last().unwrap().as_u64(),
                Some(rate as u64),
                "{variant}"
            );
            assert_eq!(template.f0, f0, "{variant}");
            assert_eq!(
                template.channels,
                if version == "v1" { 256 } else { 768 },
                "{variant}"
            );
            assert!(template.initializers.len() > 300, "{variant}");
            // No-f0 graphs have no pitch embedding or NSF source module.
            let has_pitch = template
                .initializers
                .iter()
                .any(|(_, _, key)| key == "enc_p.emb_pitch.weight");
            assert_eq!(has_pitch, f0, "{variant}");
            let weights: HashMap<&str, Tensor> = template
                .initializers
                .iter()
                .map(|(name, _, _)| {
                    (
                        name.as_str(),
                        Tensor {
                            shape: vec![],
                            data: vec![],
                        },
                    )
                })
                .collect();
            // Shapes are checked only when filling; an empty fill must fail on shape, not structure.
            let error = fill_model(bytes, &weights, "{}").unwrap_err().to_string();
            assert!(error.contains("Shape mismatch"), "{error}");
        }
    }

    #[test]
    fn weight_norm_and_transpose_match_definitions() {
        let v = Tensor {
            shape: vec![2, 2],
            data: vec![3.0, 4.0, 0.0, 2.0],
        };
        let g = Tensor {
            shape: vec![2, 1, 1],
            data: vec![10.0, 0.5],
        };
        assert_eq!(weight_norm(&g, &v).unwrap().data, vec![6.0, 8.0, 0.0, 0.5]);
        let m = Tensor {
            shape: vec![2, 3],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let t = transpose(&m).unwrap();
        assert_eq!(
            (t.shape, t.data),
            (vec![3, 2], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        );
    }

    #[test]
    fn protobuf_varints_round_trip() {
        for value in [0, 1, 127, 128, 300, u32::MAX as u64, u64::MAX] {
            let mut bytes = Vec::new();
            write_varint(&mut bytes, value);
            let mut pos = 0;
            assert_eq!(varint(&bytes, &mut pos).unwrap(), value);
            assert_eq!(pos, bytes.len());
        }
    }
}
