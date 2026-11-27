//! The handful of onnx.proto fields the exported graphs use, read straight off the protobuf wire format.

use crate::tensor::{Data, Tensor};

pub struct Model {
    pub nodes: Vec<Node>,
    pub initializers: Vec<(String, Tensor)>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

pub struct Node {
    pub op: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: Vec<(String, Attr)>,
}

pub enum Attr {
    Int(i64),
    Float(f32),
    Ints(Vec<i64>),
    Str(String),
    Tensor(Tensor),
}

impl Node {
    fn attr(&self, name: &str) -> Option<&Attr> {
        self.attrs.iter().find(|(n, _)| n == name).map(|(_, a)| a)
    }
    pub fn int(&self, name: &str, default: i64) -> i64 {
        match self.attr(name) {
            Some(Attr::Int(i)) => *i,
            None => default,
            _ => panic!("{} attribute {name} is not an int", self.op),
        }
    }
    pub fn float(&self, name: &str, default: f32) -> f32 {
        match self.attr(name) {
            Some(Attr::Float(f)) => *f,
            None => default,
            _ => panic!("{} attribute {name} is not a float", self.op),
        }
    }
    pub fn ints(&self, name: &str) -> Option<&[i64]> {
        match self.attr(name) {
            Some(Attr::Ints(v)) => Some(v),
            None => None,
            _ => panic!("{} attribute {name} is not ints", self.op),
        }
    }
    pub fn str(&self, name: &str, default: &'static str) -> &str {
        match self.attr(name) {
            Some(Attr::Str(s)) => s,
            None => default,
            _ => panic!("{} attribute {name} is not a string", self.op),
        }
    }
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        match self.attr(name) {
            Some(Attr::Tensor(t)) => Some(t),
            _ => None,
        }
    }
}

enum Wire<'a> {
    Varint(u64),
    Fixed(u64),
    Bytes(&'a [u8]),
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn varint(&mut self) -> u64 {
        let mut value = 0;
        for shift in (0..64).step_by(7) {
            let byte = self.0[0];
            self.0 = &self.0[1..];
            value |= ((byte & 0x7f) as u64) << shift;
            if byte < 0x80 {
                break;
            }
        }
        value
    }
    fn take(&mut self, n: usize) -> &'a [u8] {
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        head
    }
}

impl<'a> Iterator for Reader<'a> {
    type Item = (u64, Wire<'a>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.0.is_empty() {
            return None;
        }
        let key = self.varint();
        let value = match key & 7 {
            0 => Wire::Varint(self.varint()),
            1 => Wire::Fixed(u64::from_le_bytes(self.take(8).try_into().unwrap())),
            2 => {
                let n = self.varint() as usize;
                Wire::Bytes(self.take(n))
            }
            5 => Wire::Fixed(u32::from_le_bytes(self.take(4).try_into().unwrap()) as u64),
            kind => panic!("protobuf wire type {kind}"),
        };
        Some((key >> 3, value))
    }
}

fn string(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).expect("utf-8 name")
}

/// Repeated int64 fields arrive packed or one varint per entry.
fn push_ints(out: &mut Vec<i64>, value: Wire) {
    match value {
        Wire::Varint(v) => out.push(v as i64),
        Wire::Bytes(b) => {
            let mut r = Reader(b);
            while !r.0.is_empty() {
                out.push(r.varint() as i64);
            }
        }
        Wire::Fixed(_) => panic!("fixed-width int"),
    }
}

fn tensor(bytes: &[u8]) -> Result<(String, Tensor), String> {
    let (mut name, mut dims, mut kind, mut raw) = (String::new(), vec![], 0, None);
    for (field, value) in Reader(bytes) {
        match (field, value) {
            (1, v) => push_ints(&mut dims, v),
            (2, Wire::Varint(v)) => kind = v,
            (8, Wire::Bytes(b)) => name = string(b),
            (9, Wire::Bytes(b)) => raw = Some(b),
            (4 | 5 | 7 | 10, _) => {
                return Err(format!("tensor {name} stores typed (non-raw) data"));
            }
            (13 | 14, _) => return Err(format!("tensor {name} uses external data")),
            _ => {}
        }
    }
    let raw = raw.unwrap_or_default();
    let shape: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
    fn words<const N: usize>(raw: &[u8]) -> impl Iterator<Item = [u8; N]> + '_ {
        raw.chunks_exact(N).map(|c| c.try_into().unwrap())
    }
    // int8 and float16 initializers only feed DequantizeLinear and Cast, which fold at load, so decode them to f32.
    let data = match kind {
        1 => Data::from(words(raw).map(f32::from_le_bytes).collect::<Vec<_>>()),
        3 => Data::from(raw.iter().map(|&b| b as i8 as f32).collect::<Vec<_>>()),
        6 => Data::from(
            words(raw)
                .map(|w| i32::from_le_bytes(w) as i64)
                .collect::<Vec<_>>(),
        ),
        7 => Data::from(words(raw).map(i64::from_le_bytes).collect::<Vec<_>>()),
        9 => Data::from(raw.iter().map(|&b| b != 0).collect::<Vec<_>>()),
        10 => Data::from(
            words(raw)
                .map(|w| f16(u16::from_le_bytes(w)))
                .collect::<Vec<_>>(),
        ),
        11 => Data::from(words(raw).map(f64::from_le_bytes).collect::<Vec<_>>()),
        _ => return Err(format!("tensor {name} has unsupported data type {kind}")),
    };
    let t = Tensor { shape, data };
    if t.data.len() != t.len() {
        return Err(format!(
            "tensor {name} has {} values for shape {:?}",
            t.data.len(),
            t.shape
        ));
    }
    Ok((name, t))
}

fn f16(h: u16) -> f32 {
    let (sign, exp, frac) = (
        (h as u32 & 0x8000) << 16,
        (h >> 10) & 0x1f,
        h as u32 & 0x3ff,
    );
    let bits = match exp {
        0 if frac == 0 => sign,
        0 => {
            // Subnormal: normalize into an f32 exponent.
            let shift = frac.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((frac << shift) & 0x3ff) << 13
        }
        31 => sign | 0x7f80_0000 | frac << 13,
        _ => sign | ((exp as u32 + 112) << 23) | frac << 13,
    };
    f32::from_bits(bits)
}

fn attribute(bytes: &[u8]) -> Result<(String, Attr), String> {
    let (mut name, mut kind, mut ints) = (String::new(), 0, vec![]);
    let (mut int, mut float, mut text, mut t) = (0, 0.0, String::new(), None);
    for (field, value) in Reader(bytes) {
        match (field, value) {
            (1, Wire::Bytes(b)) => name = string(b),
            (2, Wire::Fixed(v)) => float = f32::from_bits(v as u32),
            (3, Wire::Varint(v)) => int = v as i64,
            (4, Wire::Bytes(b)) => text = string(b),
            (5, Wire::Bytes(b)) => t = Some(tensor(b)?.1),
            (8, v) => push_ints(&mut ints, v),
            (20, Wire::Varint(v)) => kind = v,
            _ => {}
        }
    }
    let attr = match kind {
        1 => Attr::Float(float),
        2 => Attr::Int(int),
        3 => Attr::Str(text),
        4 => Attr::Tensor(t.unwrap()),
        7 => Attr::Ints(ints),
        _ => return Err(format!("attribute {name} has unsupported type {kind}")),
    };
    Ok((name, attr))
}

fn node(bytes: &[u8]) -> Result<Node, String> {
    let mut node = Node {
        op: String::new(),
        inputs: vec![],
        outputs: vec![],
        attrs: vec![],
    };
    for (field, value) in Reader(bytes) {
        match (field, value) {
            (1, Wire::Bytes(b)) => node.inputs.push(string(b)),
            (2, Wire::Bytes(b)) => node.outputs.push(string(b)),
            (4, Wire::Bytes(b)) => node.op = string(b),
            (5, Wire::Bytes(b)) => node.attrs.push(attribute(b)?),
            (7, Wire::Bytes(b)) if !b.is_empty() => {
                return Err(format!("operator domain {}", string(b)));
            }
            _ => {}
        }
    }
    Ok(node)
}

fn value_name(bytes: &[u8]) -> String {
    Reader(bytes)
        .find_map(|(f, v)| match (f, v) {
            (1, Wire::Bytes(b)) => Some(string(b)),
            _ => None,
        })
        .unwrap()
}

pub fn parse(bytes: &[u8]) -> Result<Model, String> {
    let graph = Reader(bytes)
        .find_map(|(f, v)| match (f, v) {
            (7, Wire::Bytes(b)) => Some(b),
            _ => None,
        })
        .ok_or("no graph in model")?;
    let mut model = Model {
        nodes: vec![],
        initializers: vec![],
        inputs: vec![],
        outputs: vec![],
    };
    for (field, value) in Reader(graph) {
        let Wire::Bytes(b) = value else { continue };
        match field {
            1 => model.nodes.push(node(b)?),
            5 => model.initializers.push(tensor(b)?),
            11 => model.inputs.push(value_name(b)),
            12 => model.outputs.push(value_name(b)),
            _ => {}
        }
    }
    Ok(model)
}

#[cfg(test)]
mod tests {
    #[test]
    fn f16_decodes() {
        for (h, f) in [
            (0x3c00, 1.0),
            (0xc000, -2.0),
            (0x0001, 5.960_464_5e-8),
            (0x03ff, 6.097_555e-5),
            (0x7bff, 65504.0),
        ] {
            assert_eq!(super::f16(h), f, "{h:#x}");
        }
        assert_eq!(super::f16(0x7c00), f32::INFINITY);
    }
}
