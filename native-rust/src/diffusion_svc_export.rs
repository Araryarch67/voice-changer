//! Builds a Diffusion-SVC voice's ONNX graphs from a combo checkpoint (`.ptc`),
//! following the reference's `Unit2MelNaive` (PCmer: Performer attention and
//! Conformer convolution), `Unit2Mel`'s conditioning and the `WaveNet` denoiser.
//! The graphs are built for the checkpoint's own configuration, so any layer and
//! channel count works.
use crate::diffusion_svc::{Info, MEL_BINS};
use crate::onnx_build::{layers, Attr, Dim, Graph};
use crate::pth::{self, Tensor, Value};
use anyhow::{bail, ensure, Context, Result};
use std::{collections::HashMap, path::Path};

/// The architecture read from a combo checkpoint's configs.
#[derive(Debug, Clone, PartialEq)]
struct Architecture {
    units: usize,
    sample_rate: u32,
    block_size: usize,
    speakers: usize,
    naive_layers: usize,
    naive_chans: usize,
    diff_layers: usize,
    diff_chans: usize,
    diff_hidden: usize,
    k_step_max: usize,
}

pub(crate) type Weights = HashMap<String, Tensor>;

fn config_int(config: &Value, section: &str, key: &str) -> Option<i64> {
    let value = config.get(section)?.get(key)?;
    match value {
        Value::Float(v) if v.fract() == 0.0 => Some(*v as i64),
        other => other.as_int(),
    }
}

fn config_bool(config: &Value, section: &str, key: &str) -> bool {
    config
        .get(section)
        .and_then(|s| s.get(key))
        .and_then(Value::as_int)
        .is_some_and(|v| v != 0)
}

pub(crate) fn state_dict(model: &Value, what: &str) -> Result<Weights> {
    let Some(Value::Dict(items)) = model.get("model") else {
        bail!("The checkpoint's {what} has no weights");
    };
    let mut weights = HashMap::new();
    for (key, value) in items {
        if let (Value::Str(key), Value::Tensor(tensor)) = (key, value) {
            weights.insert(key.clone(), tensor.clone());
        }
    }
    Ok(weights)
}

/// Reads a combo checkpoint: its architecture and the weights of both models.
fn read(path: &Path) -> Result<(Architecture, Weights, Weights)> {
    let root = pth::load(path)?;
    let diff_config = root
        .get("diff_config_dict")
        .context("Not a Diffusion-SVC combo model (.ptc): no diff_config_dict. Combine the diffusion and naive models with Diffusion-SVC's combo.py first")?;
    let naive_config = root
        .get("naive_config_dict")
        .context("Not a Diffusion-SVC combo model (.ptc): no naive_config_dict")?;
    let model_type = |config: &Value| {
        config
            .get("model")
            .and_then(|m| m.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    };
    ensure!(
        model_type(diff_config) == "Diffusion",
        "Unsupported diffusion model type {:?}; only Diffusion is supported",
        model_type(diff_config)
    );
    ensure!(
        model_type(naive_config) == "Naive",
        "Unsupported naive model type {:?}; only Naive is supported (NaiveFS is not)",
        model_type(naive_config)
    );
    for config in [diff_config, naive_config] {
        ensure!(
            !config_bool(config, "model", "use_speaker_encoder"),
            "Models with a speaker encoder are not supported"
        );
    }
    let encoder = diff_config
        .get("data")
        .and_then(|d| d.get("encoder"))
        .and_then(Value::as_str)
        .unwrap_or("");
    ensure!(
        encoder == "contentvec768l12",
        "This model uses the {encoder:?} encoder; only contentvec768l12 (ContentVec, 768 channels, layer 12) is supported"
    );
    let need = |config: &Value, section: &str, key: &str| -> Result<usize> {
        let value = config_int(config, section, key)
            .with_context(|| format!("The model config has no {section}.{key}"))?;
        ensure!(
            value > 0,
            "The model config's {section}.{key} must be positive"
        );
        Ok(value as usize)
    };
    let architecture = Architecture {
        units: need(diff_config, "data", "encoder_out_channels")?,
        sample_rate: need(diff_config, "data", "sampling_rate")? as u32,
        block_size: need(diff_config, "data", "block_size")?,
        speakers: config_int(diff_config, "model", "n_spk")
            .unwrap_or(1)
            .max(1) as usize,
        naive_layers: need(naive_config, "model", "n_layers")?,
        naive_chans: need(naive_config, "model", "n_chans")?,
        diff_layers: need(diff_config, "model", "n_layers")?,
        diff_chans: need(diff_config, "model", "n_chans")?,
        diff_hidden: need(diff_config, "model", "n_hidden")?,
        k_step_max: need(diff_config, "model", "k_step_max")?,
    };
    ensure!(
        architecture.sample_rate == 44_100 && architecture.block_size == 512,
        "This model is for {} Hz with {} samples per frame; the NSF-HiFiGAN vocoder needs 44100 Hz and 512",
        architecture.sample_rate,
        architecture.block_size
    );
    let diff = state_dict(
        root.get("diff_model")
            .context("No diff_model in the checkpoint")?,
        "diffusion model",
    )?;
    let naive = state_dict(
        root.get("naive_model")
            .context("No naive_model in the checkpoint")?,
        "naive model",
    )?;
    Ok((architecture, naive, diff))
}

pub(crate) fn weight<'a>(weights: &'a Weights, key: &str) -> Result<&'a Tensor> {
    weights
        .get(key)
        .with_context(|| format!("The checkpoint has no {key}"))
}

/// `weight_norm` on a Linear layer: `g * v / ||v||` per output row.
pub(crate) fn weight_norm(g: &Tensor, v: &Tensor) -> Tensor {
    let rows = v.shape[0];
    let cols = v.data.len() / rows;
    let mut data = v.data.clone();
    for r in 0..rows {
        let row = &mut data[r * cols..(r + 1) * cols];
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        let scale = g.data[r] / norm;
        row.iter_mut().for_each(|x| *x *= scale);
    }
    Tensor {
        shape: v.shape.clone(),
        data,
    }
}

/// `log(1 + f0 / 700)` through `prefix`'s f0 embedding, plus the volume
/// embedding: what both models add to their input.
fn f0_and_volume(g: &mut Graph, w: &Weights, f0: &str, volume: &str) -> Result<String> {
    let scaled = layers::scale(g, f0, 1.0 / 700.0);
    let one = g.scalar(1.0);
    let shifted = g.op("Add", &[&scaled, &one], &[]);
    let log = g.op("Log", &[&shifted], &[]);
    let f0_embed = layers::linear(
        g,
        &log,
        weight(w, "f0_embed.weight")?,
        Some(weight(w, "f0_embed.bias")?),
    );
    let volume_embed = layers::linear(
        g,
        volume,
        weight(w, "volume_embed.weight")?,
        Some(weight(w, "volume_embed.bias")?),
    );
    Ok(g.op("Add", &[&f0_embed, &volume_embed], &[]))
}

/// Adds the chosen speaker's embedding (input `speaker`, int64 `[1]`) when the
/// model has several speakers.
fn add_speaker(g: &mut Graph, w: &Weights, x: &str, speakers: usize) -> Result<String> {
    if speakers <= 1 {
        return Ok(x.to_string());
    }
    let table = weight(w, "spk_embed.weight")?;
    let table = g.weight(&table.data, &table.shape);
    let speaker = "speaker".to_string();
    let gathered = g.op("Gather", &[&table, &speaker], &[("axis", Attr::Int(0))]);
    let row = layers::unsqueeze(g, &gathered, &[0]);
    Ok(g.op("Add", &[x, &row], &[]))
}

/// The Performer self-attention of one PCmer layer (8 heads of 64, softmax
/// kernel with the checkpoint's random projection).
fn performer(g: &mut Graph, w: &Weights, x: &str, prefix: &str) -> Result<String> {
    let heads = 8_i64;
    let w_of = |name: &str| weight(w, &format!("{prefix}.attn.{name}"));
    let projection = w_of("fast_attention.projection_matrix")?;
    let (features, dim_head) = (projection.shape[0], projection.shape[1]);
    let split_heads = |g: &mut Graph, t: &str| {
        let shaped = layers::reshape(g, t, &[1, -1, heads, dim_head as i64]);
        layers::transpose(g, &shaped, &[0, 2, 1, 3])
    };
    let q = layers::linear(g, x, w_of("to_q.weight")?, Some(w_of("to_q.bias")?));
    let k = layers::linear(g, x, w_of("to_k.weight")?, Some(w_of("to_k.bias")?));
    let v = layers::linear(g, x, w_of("to_v.weight")?, Some(w_of("to_v.bias")?));
    let (q, k, v) = (split_heads(g, &q), split_heads(g, &k), split_heads(g, &v));
    let normalizer = (dim_head as f32).powf(-0.25);
    let ratio = (features as f32).powf(-0.5);
    let mut projection_t = vec![0.0; features * dim_head];
    for f in 0..features {
        for d in 0..dim_head {
            projection_t[d * features + f] = projection.data[f * dim_head + d];
        }
    }
    let projection_t = g.weight(&projection_t, &[dim_head, features]);
    let kernel = |g: &mut Graph, data: &str, is_query: bool| {
        let scaled = layers::scale(g, data, normalizer);
        let dash = g.op("MatMul", &[&scaled, &projection_t], &[]);
        let squared = g.op("Mul", &[data, data], &[]);
        let axes = g.ints(&[-1]);
        let sum = g.op(
            "ReduceSum",
            &[&squared, &axes],
            &[("keepdims", Attr::Int(1))],
        );
        let diag = layers::scale(g, &sum, 0.5 * normalizer * normalizer);
        let mut exponent = g.op("Sub", &[&dash, &diag], &[]);
        let eps = g.scalar(1e-4);
        if is_query {
            let max = g.op(
                "ReduceMax",
                &[&dash],
                &[("axes", Attr::Ints(vec![-1])), ("keepdims", Attr::Int(1))],
            );
            exponent = g.op("Sub", &[&exponent, &max], &[]);
            let exp = g.op("Exp", &[&exponent], &[]);
            let plus = g.op("Add", &[&exp, &eps], &[]);
            layers::scale(g, &plus, ratio)
        } else {
            // The reference adds eps inside the exponent for keys.
            let shifted = g.op("Add", &[&exponent, &eps], &[]);
            let exp = g.op("Exp", &[&shifted], &[]);
            layers::scale(g, &exp, ratio)
        }
    };
    let q = kernel(g, &q, true);
    let k = kernel(g, &k, false);
    // Linear attention: out = (q @ (k^T v)) / (q . sum(k)).
    let axes = g.ints(&[-2]);
    let k_sum = g.op("ReduceSum", &[&k, &axes], &[("keepdims", Attr::Int(1))]);
    let k_sum_t = layers::transpose(g, &k_sum, &[0, 1, 3, 2]);
    let denominator = g.op("MatMul", &[&q, &k_sum_t], &[]);
    let tiny = g.scalar(1e-8);
    let denominator = g.op("Add", &[&denominator, &tiny], &[]);
    let k_t = layers::transpose(g, &k, &[0, 1, 3, 2]);
    let context = g.op("MatMul", &[&k_t, &v], &[]);
    let numerator = g.op("MatMul", &[&q, &context], &[]);
    let out = g.op("Div", &[&numerator, &denominator], &[]);
    let merged = layers::transpose(g, &out, &[0, 2, 1, 3]);
    let merged = layers::reshape(g, &merged, &[1, -1, heads * dim_head as i64]);
    Ok(layers::linear(
        g,
        &merged,
        w_of("to_out.weight")?,
        Some(w_of("to_out.bias")?),
    ))
}

/// The Conformer convolution module of one PCmer layer.
fn conformer(g: &mut Graph, w: &Weights, x: &str, prefix: &str) -> Result<String> {
    let w_of = |i: usize, name: &str| weight(w, &format!("{prefix}.conformer.net.{i}.{name}"));
    let normed = layers::layer_norm(g, x, w_of(0, "weight")?, w_of(0, "bias")?);
    let channels_first = layers::transpose(g, &normed, &[0, 2, 1]);
    let expanded = layers::conv1d(
        g,
        &channels_first,
        w_of(2, "weight")?,
        Some(w_of(2, "bias")?),
        0,
        1,
        1,
    );
    let inner = w_of(2, "weight")?.shape[0] as i64 / 2;
    let parts = layers::split(g, &expanded, 1, &[inner, inner]);
    let gate = g.op("Sigmoid", &[&parts[1]], &[]);
    let glu = g.op("Mul", &[&parts[0], &gate], &[]);
    let depthwise = weight(w, &format!("{prefix}.conformer.net.4.conv.weight"))?;
    let kernel = depthwise.shape[2] as i64;
    // calc_same_padding: (k / 2, k / 2 - (k + 1) % 2), symmetric for odd kernels.
    ensure!(kernel % 2 == 1, "Even conformer kernels are not supported");
    let conv = layers::conv1d(
        g,
        &glu,
        depthwise,
        Some(weight(w, &format!("{prefix}.conformer.net.4.conv.bias"))?),
        kernel / 2,
        1,
        inner,
    );
    let sigmoid = g.op("Sigmoid", &[&conv], &[]);
    let swish = g.op("Mul", &[&conv, &sigmoid], &[]);
    let projected = layers::conv1d(
        g,
        &swish,
        w_of(6, "weight")?,
        Some(w_of(6, "bias")?),
        0,
        1,
        1,
    );
    Ok(layers::transpose(g, &projected, &[0, 2, 1]))
}

/// `naive.onnx`: units `[1,T,C]`, f0 `[1,T,1]` (Hz), volume `[1,T,1]` (and
/// speaker `[1]` with several speakers) to mel `[1,T,128]`.
fn naive_graph(a: &Architecture, w: &Weights) -> Result<Vec<u8>> {
    pcmer_graph(
        &PcmerNet {
            units: a.units,
            speakers: a.speakers,
            layers: a.naive_layers,
            phase: false,
            output: "mel",
            output_size: MEL_BINS,
        },
        w,
        "diffusion_svc_naive",
        &[],
    )
}

/// A units-to-frames network: a convolution stack, f0 and volume embeddings (and a
/// phase embedding for DDSP-SVC), a PCmer and a weight-normed dense output. Both
/// Diffusion-SVC's naive model and DDSP-SVC's Unit2Control are one.
pub(crate) struct PcmerNet {
    pub units: usize,
    pub speakers: usize,
    pub layers: usize,
    /// Takes a `phase` input `[1,T,1]` (radians), embedded as `phase / pi`.
    pub phase: bool,
    pub output: &'static str,
    pub output_size: usize,
}

pub(crate) fn pcmer_graph(
    net: &PcmerNet,
    w: &Weights,
    name: &str,
    metadata: &[(&str, &str)],
) -> Result<Vec<u8>> {
    let mut g = Graph::new();
    let frames = Dim::Named("frames");
    let units = g.input(
        "units",
        &[Dim::Fixed(1), frames, Dim::Fixed(net.units as i64)],
    );
    let f0 = g.input("f0", &[Dim::Fixed(1), frames, Dim::Fixed(1)]);
    let phase = net
        .phase
        .then(|| g.input("phase", &[Dim::Fixed(1), frames, Dim::Fixed(1)]));
    let volume = g.input("volume", &[Dim::Fixed(1), frames, Dim::Fixed(1)]);
    let channels_first = layers::transpose(&mut g, &units, &[0, 2, 1]);
    let x = layers::conv1d(
        &mut g,
        &channels_first,
        weight(w, "stack.0.weight")?,
        Some(weight(w, "stack.0.bias")?),
        1,
        1,
        1,
    );
    let x = layers::group_norm(
        &mut g,
        &x,
        4,
        weight(w, "stack.1.weight")?,
        weight(w, "stack.1.bias")?,
    );
    let x = g.op("LeakyRelu", &[&x], &[("alpha", Attr::Float(0.01))]);
    let x = layers::conv1d(
        &mut g,
        &x,
        weight(w, "stack.3.weight")?,
        Some(weight(w, "stack.3.bias")?),
        1,
        1,
        1,
    );
    let x = layers::transpose(&mut g, &x, &[0, 2, 1]);
    let embedded = f0_and_volume(&mut g, w, &f0, &volume)?;
    let mut x = g.op("Add", &[&x, &embedded], &[]);
    if let Some(phase) = phase {
        let scaled = layers::scale(&mut g, &phase, std::f32::consts::FRAC_1_PI);
        let phase_embed = layers::linear(
            &mut g,
            &scaled,
            weight(w, "phase_embed.weight")?,
            Some(weight(w, "phase_embed.bias")?),
        );
        x = g.op("Add", &[&x, &phase_embed], &[]);
    }
    if net.speakers > 1 {
        g.input_int64("speaker", &[Dim::Fixed(1)]);
    }
    x = add_speaker(&mut g, w, &x, net.speakers)?;
    for layer in 0..net.layers {
        let prefix = format!("decoder._layers.{layer}");
        let normed = layers::layer_norm(
            &mut g,
            &x,
            weight(w, &format!("{prefix}.norm.weight"))?,
            weight(w, &format!("{prefix}.norm.bias"))?,
        );
        let attention = performer(&mut g, w, &normed, &prefix)?;
        x = g.op("Add", &[&x, &attention], &[]);
        let conv = conformer(&mut g, w, &x, &prefix)?;
        x = g.op("Add", &[&x, &conv], &[]);
    }
    let x = layers::layer_norm(
        &mut g,
        &x,
        weight(w, "norm.weight")?,
        weight(w, "norm.bias")?,
    );
    let dense = weight_norm(
        weight(w, "dense_out.weight_g")?,
        weight(w, "dense_out.weight_v")?,
    );
    ensure!(
        dense.shape[0] == net.output_size,
        "The model's output layer has {} values a frame, not {}",
        dense.shape[0],
        net.output_size
    );
    let out = layers::linear(&mut g, &x, &dense, Some(weight(w, "dense_out.bias")?));
    g.output(
        net.output,
        &out,
        &[Dim::Fixed(1), frames, Dim::Fixed(net.output_size as i64)],
    );
    Ok(g.finish(name, metadata))
}

/// `condition.onnx`: the diffusion model's conditioning, `[1,hidden,T]`, from the
/// same inputs as the naive model.
fn condition_graph(a: &Architecture, w: &Weights) -> Result<Vec<u8>> {
    let mut g = Graph::new();
    let frames = Dim::Named("frames");
    let units = g.input(
        "units",
        &[Dim::Fixed(1), frames, Dim::Fixed(a.units as i64)],
    );
    let f0 = g.input("f0", &[Dim::Fixed(1), frames, Dim::Fixed(1)]);
    let volume = g.input("volume", &[Dim::Fixed(1), frames, Dim::Fixed(1)]);
    let unit_embed = layers::linear(
        &mut g,
        &units,
        weight(w, "unit_embed.weight")?,
        Some(weight(w, "unit_embed.bias")?),
    );
    let embedded = f0_and_volume(&mut g, w, &f0, &volume)?;
    let mut x = g.op("Add", &[&unit_embed, &embedded], &[]);
    if a.speakers > 1 {
        g.input_int64("speaker", &[Dim::Fixed(1)]);
    }
    x = add_speaker(&mut g, w, &x, a.speakers)?;
    let condition = layers::transpose(&mut g, &x, &[0, 2, 1]);
    g.output(
        "condition",
        &condition,
        &[Dim::Fixed(1), Dim::Fixed(a.diff_hidden as i64), frames],
    );
    Ok(g.finish("diffusion_svc_condition", &[]))
}

/// `denoiser.onnx`: one WaveNet step, x `[1,1,128,T]`, step `[1]` (continuous
/// time in model units) and condition `[1,hidden,T]` to the predicted noise
/// `[1,1,128,T]`.
fn denoiser_graph(a: &Architecture, w: &Weights, info: &Info) -> Result<Vec<u8>> {
    let mut g = Graph::new();
    let frames = Dim::Named("frames");
    let mel_dims = [
        Dim::Fixed(1),
        Dim::Fixed(1),
        Dim::Fixed(MEL_BINS as i64),
        frames,
    ];
    let x_in = g.input("x", &mel_dims);
    let step = g.input("step", &[Dim::Fixed(1)]);
    let condition = g.input(
        "condition",
        &[Dim::Fixed(1), Dim::Fixed(a.diff_hidden as i64), frames],
    );
    let p = "decoder.denoise_fn";
    let wt = |name: &str| weight(w, &format!("{p}.{name}"));
    let x = layers::reshape(&mut g, &x_in, &[1, MEL_BINS as i64, -1]);
    let x = layers::conv1d(
        &mut g,
        &x,
        wt("input_projection.weight")?,
        Some(wt("input_projection.bias")?),
        0,
        1,
        1,
    );
    let mut x = g.op("Relu", &[&x], &[]);
    // SinusoidalPosEmb(n_chans) of the step.
    let half = a.diff_chans / 2;
    let rate = (10_000f64).ln() / (half as f64 - 1.0);
    let frequencies: Vec<f32> = (0..half)
        .map(|i| (-(i as f64) * rate).exp() as f32)
        .collect();
    let frequencies = g.weight(&frequencies, &[1, half]);
    let step_column = layers::unsqueeze(&mut g, &step, &[1]);
    let phase = g.op("Mul", &[&step_column, &frequencies], &[]);
    let sin = g.op("Sin", &[&phase], &[]);
    let cos = g.op("Cos", &[&phase], &[]);
    let embedding = g.op("Concat", &[&sin, &cos], &[("axis", Attr::Int(-1))]);
    let hidden = layers::linear(
        &mut g,
        &embedding,
        wt("mlp.0.weight")?,
        Some(wt("mlp.0.bias")?),
    );
    let softplus = g.op("Softplus", &[&hidden], &[]);
    let tanh = g.op("Tanh", &[&softplus], &[]);
    let mish = g.op("Mul", &[&hidden, &tanh], &[]);
    let diffusion_step =
        layers::linear(&mut g, &mish, wt("mlp.2.weight")?, Some(wt("mlp.2.bias")?));
    let n = a.diff_chans as i64;
    let mut skips: Option<String> = None;
    for layer in 0..a.diff_layers {
        let lw = |name: &str| wt(&format!("residual_layers.{layer}.{name}"));
        let projected = layers::linear(
            &mut g,
            &diffusion_step,
            lw("diffusion_projection.weight")?,
            Some(lw("diffusion_projection.bias")?),
        );
        let projected = layers::unsqueeze(&mut g, &projected, &[2]);
        let conditioner = layers::conv1d(
            &mut g,
            &condition,
            lw("conditioner_projection.weight")?,
            Some(lw("conditioner_projection.bias")?),
            0,
            1,
            1,
        );
        let y = g.op("Add", &[&x, &projected], &[]);
        let y = layers::conv1d(
            &mut g,
            &y,
            lw("dilated_conv.weight")?,
            Some(lw("dilated_conv.bias")?),
            1,
            1,
            1,
        );
        let y = g.op("Add", &[&y, &conditioner], &[]);
        let parts = layers::split(&mut g, &y, 1, &[n, n]);
        let gate = g.op("Sigmoid", &[&parts[0]], &[]);
        let filter = g.op("Tanh", &[&parts[1]], &[]);
        let y = g.op("Mul", &[&gate, &filter], &[]);
        let y = layers::conv1d(
            &mut g,
            &y,
            lw("output_projection.weight")?,
            Some(lw("output_projection.bias")?),
            0,
            1,
            1,
        );
        let parts = layers::split(&mut g, &y, 1, &[n, n]);
        let summed = g.op("Add", &[&x, &parts[0]], &[]);
        x = layers::scale(&mut g, &summed, std::f32::consts::FRAC_1_SQRT_2);
        skips = Some(match skips {
            None => parts[1].clone(),
            Some(total) => g.op("Add", &[&total, &parts[1]], &[]),
        });
    }
    let skips = skips.context("The denoiser has no layers")?;
    let x = layers::scale(&mut g, &skips, 1.0 / (a.diff_layers as f32).sqrt());
    let x = layers::conv1d(
        &mut g,
        &x,
        wt("skip_projection.weight")?,
        Some(wt("skip_projection.bias")?),
        0,
        1,
        1,
    );
    let x = g.op("Relu", &[&x], &[]);
    let x = layers::conv1d(
        &mut g,
        &x,
        wt("output_projection.weight")?,
        Some(wt("output_projection.bias")?),
        0,
        1,
        1,
    );
    let noise = layers::unsqueeze(&mut g, &x, &[1]);
    g.output("noise", &noise, &mel_dims);
    let info = serde_json::to_string(info)?;
    Ok(g.finish("diffusion_svc_denoiser", &[("diffusion_svc", &info)]))
}

/// What [`export`] wrote.
#[derive(Debug, Clone)]
pub struct Exported {
    pub naive: std::path::PathBuf,
    pub condition: std::path::PathBuf,
    pub denoiser: std::path::PathBuf,
    pub info: Info,
}

/// Builds `naive.onnx`, `condition.onnx` and `denoiser.onnx` in `folder` from a
/// combo checkpoint.
pub fn export(checkpoint: &Path, folder: &Path) -> Result<Exported> {
    let (architecture, naive, diff) = read(checkpoint)?;
    let info = diffusion_info(&architecture, &diff)?;
    std::fs::create_dir_all(folder)?;
    let files = Exported {
        naive: folder.join("naive.onnx"),
        condition: folder.join("condition.onnx"),
        denoiser: folder.join("denoiser.onnx"),
        info: info.clone(),
    };
    std::fs::write(&files.naive, naive_graph(&architecture, &naive)?)?;
    std::fs::write(&files.condition, condition_graph(&architecture, &diff)?)?;
    std::fs::write(
        &files.denoiser,
        denoiser_graph(&architecture, &diff, &info)?,
    )?;
    Ok(files)
}

/// The sampler's settings, from a diffusion model's buffers.
fn diffusion_info(a: &Architecture, diff: &Weights) -> Result<Info> {
    let buffer = |key: &str| -> Result<Vec<f32>> { Ok(weight(diff, key)?.data.clone()) };
    let info = Info {
        sample_rate: a.sample_rate,
        block_size: a.block_size,
        units: a.units,
        speakers: a.speakers,
        k_step_max: a.k_step_max,
        spec_min: buffer("decoder.spec_min")?
            .first()
            .copied()
            .unwrap_or(-12.0),
        spec_max: buffer("decoder.spec_max")?.first().copied().unwrap_or(2.0),
        betas: buffer("decoder.betas")?,
        alphas_cumprod: buffer("decoder.alphas_cumprod")?,
    };
    ensure!(
        info.betas.len() == info.alphas_cumprod.len()
            && info.k_step_max <= info.betas.len()
            && info.spec_max > info.spec_min,
        "The diffusion model's noise schedule is inconsistent"
    );
    Ok(info)
}

/// A shallow-diffusion model on its own (`Unit2Mel`, as DDSP-SVC trains it):
/// its configuration, for [`diffusion_graphs`].
pub(crate) struct DiffusionNet {
    pub units: usize,
    pub sample_rate: u32,
    pub block_size: usize,
    pub speakers: usize,
    pub layers: usize,
    pub chans: usize,
    pub hidden: usize,
    /// The deepest diffusion the model takes; 0 for its whole noise schedule.
    pub k_step_max: usize,
}

/// `condition.onnx` and `denoiser.onnx` (the denoiser with the sampler's
/// [`Info`] in its metadata) for a diffusion model's weights.
pub(crate) fn diffusion_graphs(
    net: &DiffusionNet,
    diff: &Weights,
) -> Result<(Vec<u8>, Vec<u8>, Info)> {
    let schedule = weight(diff, "decoder.betas")?.data.len();
    let architecture = Architecture {
        units: net.units,
        sample_rate: net.sample_rate,
        block_size: net.block_size,
        speakers: net.speakers,
        naive_layers: 0,
        naive_chans: 0,
        diff_layers: net.layers,
        diff_chans: net.chans,
        diff_hidden: net.hidden,
        k_step_max: if net.k_step_max == 0 {
            schedule
        } else {
            net.k_step_max
        },
    };
    let info = diffusion_info(&architecture, diff)?;
    let condition = condition_graph(&architecture, diff)?;
    let denoiser = denoiser_graph(&architecture, diff, &info)?;
    Ok((condition, denoiser, info))
}
