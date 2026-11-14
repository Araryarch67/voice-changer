//! A small ONNX graph builder: writes a `ModelProto` (opset 17) from Rust, for
//! networks whose shape depends on a checkpoint's config (Diffusion-SVC), where a
//! fixed template per architecture does not fit. Only what those graphs use is
//! here: float and int64 tensors, named or fixed dimensions, and int, float and
//! ints attributes.

/// One dimension of a graph input or output.
#[derive(Debug, Clone, Copy)]
pub enum Dim {
    Fixed(i64),
    Named(&'static str),
}

/// A node attribute.
#[derive(Debug, Clone)]
pub enum Attr {
    Int(i64),
    Float(f32),
    Ints(Vec<i64>),
}

const FLOAT: i64 = 1;
const INT64: i64 = 7;

#[derive(Default)]
pub struct Graph {
    nodes: Vec<Vec<u8>>,
    initializers: Vec<Vec<u8>>,
    inputs: Vec<Vec<u8>>,
    outputs: Vec<Vec<u8>>,
    counter: usize,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    fn fresh(&mut self, hint: &str) -> String {
        self.counter += 1;
        format!("{hint}_{}", self.counter)
    }

    /// A float graph input.
    pub fn input(&mut self, name: &str, dims: &[Dim]) -> String {
        self.inputs.push(value_info(name, FLOAT, dims));
        name.to_string()
    }

    /// An int64 graph input.
    pub fn input_int64(&mut self, name: &str, dims: &[Dim]) -> String {
        self.inputs.push(value_info(name, INT64, dims));
        name.to_string()
    }

    /// Makes `value` a float graph output called `name`.
    pub fn output(&mut self, name: &str, value: &str, dims: &[Dim]) {
        self.node("Identity", &[value], &[name], &[]);
        self.outputs.push(value_info(name, FLOAT, dims));
    }

    /// A float initializer.
    pub fn weight(&mut self, data: &[f32], dims: &[usize]) -> String {
        debug_assert_eq!(data.len(), dims.iter().product::<usize>());
        let name = self.fresh("w");
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.initializers
            .push(tensor(&name, FLOAT, &to_i64(dims), &raw));
        name
    }

    /// An int64 initializer, for shapes, axes, pads and splits.
    pub fn ints(&mut self, data: &[i64]) -> String {
        let name = self.fresh("i");
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.initializers
            .push(tensor(&name, INT64, &[data.len() as i64], &raw));
        name
    }

    /// A float scalar initializer (rank 0).
    pub fn scalar(&mut self, value: f32) -> String {
        let name = self.fresh("s");
        self.initializers
            .push(tensor(&name, FLOAT, &[], &value.to_le_bytes()));
        name
    }

    fn node(&mut self, op: &str, inputs: &[&str], outputs: &[&str], attrs: &[(&str, Attr)]) {
        let mut node = Vec::new();
        for input in inputs {
            bytes_field(&mut node, 1, input.as_bytes());
        }
        for output in outputs {
            bytes_field(&mut node, 2, output.as_bytes());
        }
        let name = self.fresh(op);
        bytes_field(&mut node, 3, name.as_bytes());
        bytes_field(&mut node, 4, op.as_bytes());
        for (key, value) in attrs {
            bytes_field(&mut node, 5, &attribute(key, value));
        }
        self.nodes.push(node);
    }

    /// A node with one output; returns the output's name.
    pub fn op(&mut self, op: &str, inputs: &[&str], attrs: &[(&str, Attr)]) -> String {
        let output = self.fresh(&op.to_ascii_lowercase());
        self.node(op, inputs, &[output.as_str()], attrs);
        output
    }

    /// A node with `count` outputs.
    pub fn op_n(
        &mut self,
        op: &str,
        inputs: &[&str],
        count: usize,
        attrs: &[(&str, Attr)],
    ) -> Vec<String> {
        let outputs: Vec<String> = (0..count)
            .map(|_| self.fresh(&op.to_ascii_lowercase()))
            .collect();
        let names: Vec<&str> = outputs.iter().map(String::as_str).collect();
        self.node(op, inputs, &names, attrs);
        outputs
    }

    /// The serialized model, with `metadata` as its metadata properties.
    pub fn finish(self, name: &str, metadata: &[(&str, &str)]) -> Vec<u8> {
        let mut graph = Vec::new();
        for node in &self.nodes {
            bytes_field(&mut graph, 1, node);
        }
        bytes_field(&mut graph, 2, name.as_bytes());
        for initializer in &self.initializers {
            bytes_field(&mut graph, 5, initializer);
        }
        for input in &self.inputs {
            bytes_field(&mut graph, 11, input);
        }
        for output in &self.outputs {
            bytes_field(&mut graph, 12, output);
        }
        let mut model = Vec::new();
        varint_field(&mut model, 1, 8); // IR version 8
        bytes_field(&mut model, 2, b"voice-changer-native");
        bytes_field(&mut model, 7, &graph);
        let mut opset = Vec::new();
        bytes_field(&mut opset, 1, b"");
        varint_field(&mut opset, 2, 17);
        bytes_field(&mut model, 8, &opset);
        for (key, value) in metadata {
            let mut entry = Vec::new();
            bytes_field(&mut entry, 1, key.as_bytes());
            bytes_field(&mut entry, 2, value.as_bytes());
            bytes_field(&mut model, 14, &entry);
        }
        model
    }
}

fn to_i64(dims: &[usize]) -> Vec<i64> {
    dims.iter().map(|&d| d as i64).collect()
}

fn tensor(name: &str, data_type: i64, dims: &[i64], raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for &dim in dims {
        varint_field(&mut out, 1, dim as u64);
    }
    varint_field(&mut out, 2, data_type as u64);
    bytes_field(&mut out, 8, name.as_bytes());
    bytes_field(&mut out, 9, raw);
    out
}

fn value_info(name: &str, elem_type: i64, dims: &[Dim]) -> Vec<u8> {
    let mut shape = Vec::new();
    for dim in dims {
        let mut d = Vec::new();
        match dim {
            Dim::Fixed(value) => varint_field(&mut d, 1, *value as u64),
            Dim::Named(param) => bytes_field(&mut d, 2, param.as_bytes()),
        }
        bytes_field(&mut shape, 1, &d);
    }
    let mut tensor_type = Vec::new();
    varint_field(&mut tensor_type, 1, elem_type as u64);
    bytes_field(&mut tensor_type, 2, &shape);
    let mut type_proto = Vec::new();
    bytes_field(&mut type_proto, 1, &tensor_type);
    let mut out = Vec::new();
    bytes_field(&mut out, 1, name.as_bytes());
    bytes_field(&mut out, 2, &type_proto);
    out
}

fn attribute(name: &str, value: &Attr) -> Vec<u8> {
    let mut out = Vec::new();
    bytes_field(&mut out, 1, name.as_bytes());
    match value {
        Attr::Float(v) => {
            write_varint(&mut out, (2 << 3) | 5);
            out.extend_from_slice(&v.to_le_bytes());
            varint_field(&mut out, 20, 1);
        }
        Attr::Int(v) => {
            varint_field(&mut out, 3, *v as u64);
            varint_field(&mut out, 20, 2);
        }
        Attr::Ints(values) => {
            for v in values {
                varint_field(&mut out, 8, *v as u64);
            }
            varint_field(&mut out, 20, 7);
        }
    }
    out
}

fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
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

/// Layers built from checkpoint weights. Weights use PyTorch's layouts: Linear
/// `[out, in]`, Conv1d `[out, in / groups, kernel]`.
pub mod layers {
    use super::{Attr, Graph};
    use crate::pth::Tensor;

    /// `x @ weight.T + bias` over the last axis.
    pub fn linear(g: &mut Graph, x: &str, weight: &Tensor, bias: Option<&Tensor>) -> String {
        let (out, inp) = (weight.shape[0], weight.shape[1]);
        let mut transposed = vec![0.0; out * inp];
        for o in 0..out {
            for i in 0..inp {
                transposed[i * out + o] = weight.data[o * inp + i];
            }
        }
        let w = g.weight(&transposed, &[inp, out]);
        let y = g.op("MatMul", &[x, &w], &[]);
        match bias {
            Some(bias) => {
                let b = g.weight(&bias.data, &[out]);
                g.op("Add", &[&y, &b], &[])
            }
            None => y,
        }
    }

    /// Conv1d on `[1, channels, frames]` with symmetric `padding`.
    pub fn conv1d(
        g: &mut Graph,
        x: &str,
        weight: &Tensor,
        bias: Option<&Tensor>,
        padding: i64,
        dilation: i64,
        groups: i64,
    ) -> String {
        let w = g.weight(&weight.data, &weight.shape);
        let mut inputs = vec![x.to_string(), w];
        if let Some(bias) = bias {
            inputs.push(g.weight(&bias.data, &bias.shape));
        }
        let names: Vec<&str> = inputs.iter().map(String::as_str).collect();
        g.op(
            "Conv",
            &names,
            &[
                ("kernel_shape", Attr::Ints(vec![weight.shape[2] as i64])),
                ("pads", Attr::Ints(vec![padding, padding])),
                ("dilations", Attr::Ints(vec![dilation])),
                ("group", Attr::Int(groups)),
            ],
        )
    }

    /// LayerNorm over the last axis (eps 1e-5, as PyTorch's default).
    pub fn layer_norm(g: &mut Graph, x: &str, weight: &Tensor, bias: &Tensor) -> String {
        let w = g.weight(&weight.data, &weight.shape);
        let b = g.weight(&bias.data, &bias.shape);
        g.op(
            "LayerNormalization",
            &[x, &w, &b],
            &[("axis", Attr::Int(-1)), ("epsilon", Attr::Float(1e-5))],
        )
    }

    /// GroupNorm on `[1, channels, frames]` (eps 1e-5), built from basic operators
    /// because GroupNormalization needs opset 18.
    pub fn group_norm(
        g: &mut Graph,
        x: &str,
        groups: i64,
        weight: &Tensor,
        bias: &Tensor,
    ) -> String {
        let channels = weight.shape[0];
        let grouped_shape = g.ints(&[1, groups, -1]);
        let grouped = g.op("Reshape", &[x, &grouped_shape], &[]);
        let axes = Attr::Ints(vec![2]);
        let mean = g.op(
            "ReduceMean",
            &[&grouped],
            &[("axes", axes.clone()), ("keepdims", Attr::Int(1))],
        );
        let centred = g.op("Sub", &[&grouped, &mean], &[]);
        let squared = g.op("Mul", &[&centred, &centred], &[]);
        let variance = g.op(
            "ReduceMean",
            &[&squared],
            &[("axes", axes), ("keepdims", Attr::Int(1))],
        );
        let eps = g.scalar(1e-5);
        let padded = g.op("Add", &[&variance, &eps], &[]);
        let deviation = g.op("Sqrt", &[&padded], &[]);
        let normed = g.op("Div", &[&centred, &deviation], &[]);
        let shape = g.op("Shape", &[x], &[]);
        let back = g.op("Reshape", &[&normed, &shape], &[]);
        let w = g.weight(&weight.data, &[channels, 1]);
        let b = g.weight(&bias.data, &[channels, 1]);
        let scaled = g.op("Mul", &[&back, &w], &[]);
        g.op("Add", &[&scaled, &b], &[])
    }

    pub fn transpose(g: &mut Graph, x: &str, perm: &[i64]) -> String {
        g.op("Transpose", &[x], &[("perm", Attr::Ints(perm.to_vec()))])
    }

    /// Splits `x` along `axis` into parts of `sizes`.
    pub fn split(g: &mut Graph, x: &str, axis: i64, sizes: &[i64]) -> Vec<String> {
        let split = g.ints(sizes);
        g.op_n(
            "Split",
            &[x, &split],
            sizes.len(),
            &[("axis", Attr::Int(axis))],
        )
    }

    pub fn unsqueeze(g: &mut Graph, x: &str, axes: &[i64]) -> String {
        let axes = g.ints(axes);
        g.op("Unsqueeze", &[x, &axes], &[])
    }

    pub fn reshape(g: &mut Graph, x: &str, shape: &[i64]) -> String {
        let shape = g.ints(shape);
        g.op("Reshape", &[x, &shape], &[])
    }

    pub fn scale(g: &mut Graph, x: &str, factor: f32) -> String {
        let f = g.scalar(factor);
        g.op("Mul", &[x, &f], &[])
    }
}
