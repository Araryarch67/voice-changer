//! Reader for PyTorch zip checkpoints (`torch.save` since 1.6) without Python.
//!
//! Only data is reconstructed: the pickle VM understands containers, scalars, strings and
//! `torch._utils._rebuild_tensor_v2`; any other callable becomes an inert [`Value::Object`].
//! No code from the file is executed.
use anyhow::{bail, ensure, Context, Result};
use std::{collections::HashMap, io::Read, path::Path};

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    List(Vec<Value>),
    Tuple(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Tensor(Tensor),
    Global(String, String),
    Object(Box<Value>, Box<Value>),
    Storage(String, String),
}
impl Value {
    /// Value for a string key in a dict.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Self::Dict(items) => items
                .iter()
                .find(|(k, _)| matches!(k, Self::Str(s) if s == key))
                .map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Bool(v) => Some(*v as i64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(v) => Some(v),
            _ => None,
        }
    }
    /// Plain data as JSON; tensors and objects are not representable.
    pub fn to_json(&self) -> Option<serde_json::Value> {
        Some(match self {
            Self::None => serde_json::Value::Null,
            Self::Bool(v) => (*v).into(),
            Self::Int(v) => (*v).into(),
            Self::Float(v) => serde_json::Number::from_f64(*v)?.into(),
            Self::Str(v) => v.as_str().into(),
            Self::List(items) | Self::Tuple(items) => items
                .iter()
                .map(Self::to_json)
                .collect::<Option<Vec<_>>>()?
                .into(),
            _ => return None,
        })
    }
}

/// Dense tensor, converted to f32 and laid out contiguously in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub fn load(path: &Path) -> Result<Value> {
    let file = std::fs::File::open(path).with_context(|| format!("Open {}", path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .context("Not a zip checkpoint; legacy (pre-1.6) torch.save files are not supported")?;
    let pickle = (0..archive.len())
        .filter_map(|i| archive.name_for_index(i).map(str::to_owned))
        .find(|name| name == "data.pkl" || name.ends_with("/data.pkl"))
        .context("Checkpoint has no data.pkl")?;
    let prefix = pickle.trim_end_matches("data.pkl").to_owned();
    let mut bytes = Vec::new();
    archive.by_name(&pickle)?.read_to_end(&mut bytes)?;
    let mut storages: HashMap<String, Vec<u8>> = HashMap::new();
    let mut storage = |key: &str| -> Result<Vec<u8>> {
        if let Some(data) = storages.get(key) {
            return Ok(data.clone());
        }
        let mut data = Vec::new();
        archive
            .by_name(&format!("{prefix}data/{key}"))
            .with_context(|| format!("Missing tensor storage {key}"))?
            .read_to_end(&mut data)?;
        storages.insert(key.to_owned(), data.clone());
        Ok(data)
    };
    unpickle(&bytes, &mut storage)
}

fn rebuild_tensor(
    args: &[Value],
    storage: &mut dyn FnMut(&str) -> Result<Vec<u8>>,
) -> Result<Tensor> {
    let [Value::Storage(dtype, key), Value::Int(offset), Value::Tuple(size), Value::Tuple(stride), ..] =
        args
    else {
        bail!("Unexpected _rebuild_tensor_v2 arguments");
    };
    let dims = |items: &[Value]| -> Result<Vec<usize>> {
        items
            .iter()
            .map(|v| {
                v.as_int()
                    .and_then(|v| usize::try_from(v).ok())
                    .context("Invalid tensor dimension")
            })
            .collect()
    };
    let (shape, stride) = (dims(size)?, dims(stride)?);
    ensure!(
        shape.len() == stride.len(),
        "Tensor size/stride rank mismatch"
    );
    let bytes = storage(key)?;
    let (width, convert): (usize, fn(&[u8]) -> f32) = match dtype.as_str() {
        "FloatStorage" => (4, |b| f32::from_le_bytes(b.try_into().unwrap())),
        "HalfStorage" => (2, |b| f16_to_f32(u16::from_le_bytes(b.try_into().unwrap()))),
        "BFloat16Storage" => (2, |b| {
            f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16)
        }),
        "DoubleStorage" => (8, |b| f64::from_le_bytes(b.try_into().unwrap()) as f32),
        other => bail!("Unsupported tensor storage {other}"),
    };
    let count: usize = shape.iter().product();
    let mut data = Vec::with_capacity(count);
    let mut index = vec![0; shape.len()];
    for _ in 0..count {
        let element =
            *offset as usize + index.iter().zip(&stride).map(|(i, s)| i * s).sum::<usize>();
        let bytes = bytes
            .get(element * width..(element + 1) * width)
            .context("Tensor exceeds its storage")?;
        data.push(convert(bytes));
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    Ok(Tensor { shape, data })
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;
    f32::from_bits(match exponent {
        0 if mantissa == 0 => sign,
        // Subnormal: normalize into the f32 exponent range.
        0 => {
            let shift = mantissa.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | (((mantissa << shift) & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 112) << 23) | (mantissa << 13),
    })
}

struct Input<'a> {
    data: &'a [u8],
    pos: usize,
}
impl<'a> Input<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let rest = self.data.get(self.pos..).context("Truncated pickle")?;
        let bytes = rest.get(..len).context("Truncated pickle")?;
        self.pos += len;
        Ok(bytes)
    }
    /// Text up to the next newline, which is consumed.
    fn line(&mut self) -> Result<&'a [u8]> {
        let rest = self.data.get(self.pos..).context("Truncated pickle")?;
        let end = rest
            .iter()
            .position(|b| *b == b'\n')
            .context("Truncated pickle line")?;
        self.pos += end + 1;
        Ok(&rest[..end])
    }
}

fn unpickle(data: &[u8], storage: &mut dyn FnMut(&str) -> Result<Vec<u8>>) -> Result<Value> {
    let mut input = Input { data, pos: 0 };
    let mut stack: Vec<Value> = Vec::new();
    let mut marks: Vec<usize> = Vec::new();
    let mut memo: HashMap<u32, Value> = HashMap::new();
    macro_rules! pop {
        () => {
            stack.pop().context("Pickle stack underflow")?
        };
    }
    macro_rules! pop_mark {
        () => {{
            let mark = marks.pop().context("Pickle mark underflow")?;
            stack.split_off(mark)
        }};
    }
    let u32_at = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());
    let string = |b: &[u8]| String::from_utf8(b.to_vec()).context("Invalid UTF-8 in pickle");
    loop {
        let op = input.take(1)?[0];
        match op {
            0x80 => {
                input.take(1)?;
            }
            0x95 => {
                input.take(8)?;
            }
            b'.' => return stack.pop().context("Empty pickle"),
            b'(' => marks.push(stack.len()),
            b'}' => stack.push(Value::Dict(Vec::new())),
            b']' => stack.push(Value::List(Vec::new())),
            b')' => stack.push(Value::Tuple(Vec::new())),
            b't' => {
                let items = pop_mark!();
                stack.push(Value::Tuple(items));
            }
            0x85..=0x87 => {
                let at = stack
                    .len()
                    .checked_sub((op - 0x84) as usize)
                    .context("Pickle stack underflow")?;
                let items = stack.split_off(at);
                stack.push(Value::Tuple(items));
            }
            b'N' => stack.push(Value::None),
            0x88 => stack.push(Value::Bool(true)),
            0x89 => stack.push(Value::Bool(false)),
            b'J' => stack.push(Value::Int(
                i32::from_le_bytes(input.take(4)?.try_into().unwrap()) as i64,
            )),
            b'K' => stack.push(Value::Int(input.take(1)?[0] as i64)),
            b'M' => stack.push(Value::Int(
                u16::from_le_bytes(input.take(2)?.try_into().unwrap()) as i64,
            )),
            0x8a => {
                let len = input.take(1)?[0] as usize;
                ensure!(len <= 8, "Pickle integer too large");
                let bytes = input.take(len)?;
                let mut value = [if bytes.last().is_some_and(|b| b & 0x80 != 0) {
                    0xff
                } else {
                    0
                }; 8];
                value[..len].copy_from_slice(bytes);
                stack.push(Value::Int(i64::from_le_bytes(value)));
            }
            b'G' => stack.push(Value::Float(f64::from_be_bytes(
                input.take(8)?.try_into().unwrap(),
            ))),
            b'X' => {
                let len = u32_at(input.take(4)?) as usize;
                stack.push(Value::Str(string(input.take(len)?)?));
            }
            0x8c => {
                let len = input.take(1)?[0] as usize;
                stack.push(Value::Str(string(input.take(len)?)?));
            }
            b'T' => {
                let len = u32_at(input.take(4)?) as usize;
                stack.push(Value::Str(string(input.take(len)?)?));
            }
            b'U' => {
                let len = input.take(1)?[0] as usize;
                stack.push(Value::Str(string(input.take(len)?)?));
            }
            b'B' => {
                let len = u32_at(input.take(4)?) as usize;
                stack.push(Value::Bytes(input.take(len)?.to_vec()));
            }
            b'C' => {
                let len = input.take(1)?[0] as usize;
                stack.push(Value::Bytes(input.take(len)?.to_vec()));
            }
            b'c' => {
                let module = string(input.line()?)?;
                let name = string(input.line()?)?;
                stack.push(Value::Global(module, name));
            }
            0x93 => {
                let name = pop!();
                let module = pop!();
                let (Value::Str(module), Value::Str(name)) = (module, name) else {
                    bail!("STACK_GLOBAL expects strings");
                };
                stack.push(Value::Global(module, name));
            }
            b'q' => {
                let key = input.take(1)?[0] as u32;
                memo.insert(key, stack.last().context("Empty stack")?.clone());
            }
            b'r' => {
                let key = u32_at(input.take(4)?);
                memo.insert(key, stack.last().context("Empty stack")?.clone());
            }
            0x94 => {
                let key = memo.len() as u32;
                memo.insert(key, stack.last().context("Empty stack")?.clone());
            }
            b'h' => {
                let key = input.take(1)?[0] as u32;
                stack.push(memo.get(&key).context("Missing memo entry")?.clone());
            }
            b'j' => {
                let key = u32_at(input.take(4)?);
                stack.push(memo.get(&key).context("Missing memo entry")?.clone());
            }
            b'a' => {
                let value = pop!();
                let Some(Value::List(items)) = stack.last_mut() else {
                    bail!("APPEND to non-list")
                };
                items.push(value);
            }
            b'e' => {
                let values = pop_mark!();
                let Some(Value::List(items)) = stack.last_mut() else {
                    bail!("APPENDS to non-list")
                };
                items.extend(values);
            }
            b's' => {
                let value = pop!();
                let key = pop!();
                let Some(Value::Dict(items)) = stack.last_mut() else {
                    bail!("SETITEM on non-dict")
                };
                items.push((key, value));
            }
            b'u' => {
                let values = pop_mark!();
                let Some(Value::Dict(items)) = stack.last_mut() else {
                    bail!("SETITEMS on non-dict")
                };
                let mut values = values.into_iter();
                while let (Some(key), Some(value)) = (values.next(), values.next()) {
                    items.push((key, value));
                }
            }
            // State for OrderedDict/objects (e.g. state-dict _metadata) carries no tensor data.
            b'b' => {
                pop!();
            }
            b'0' => {
                pop!();
            }
            b'1' => {
                drop(pop_mark!());
            }
            b'2' => stack.push(stack.last().context("Empty stack")?.clone()),
            b'Q' => {
                let Value::Tuple(id) = pop!() else {
                    bail!("Unexpected persistent id")
                };
                let [Value::Str(kind), Value::Global(_, dtype), Value::Str(key), ..] =
                    id.as_slice()
                else {
                    bail!("Unexpected persistent id");
                };
                ensure!(kind == "storage", "Unsupported persistent id {kind}");
                stack.push(Value::Storage(dtype.clone(), key.clone()));
            }
            b'R' | 0x81 => {
                let args = pop!();
                let callable = pop!();
                let value = match (&callable, &args) {
                    (Value::Global(m, n), Value::Tuple(args))
                        if m == "torch._utils" && n == "_rebuild_tensor_v2" =>
                    {
                        Value::Tensor(rebuild_tensor(args, storage)?)
                    }
                    (Value::Global(m, n), _) if m == "collections" && n == "OrderedDict" => {
                        Value::Dict(Vec::new())
                    }
                    _ => Value::Object(Box::new(callable), Box::new(args)),
                };
                stack.push(value);
            }
            other => bail!("Unsupported pickle opcode 0x{other:02x}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::f16_to_f32;

    #[test]
    fn half_conversion_covers_normal_subnormal_and_special_values() {
        for (bits, value) in [
            (0x3c00, 1.0),
            (0xc000, -2.0),
            (0x3555, 0.333_251_95),
            (0x0001, 5.960_464_5e-8),
            (0x03ff, 6.097_555e-5),
            (0x7bff, 65504.0),
            (0x8000, -0.0),
        ] {
            assert_eq!(f16_to_f32(bits), value, "{bits:#06x}");
        }
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
    }
}
