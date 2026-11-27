//! Kernels by ONNX op name. `prepare` reads a node's attributes (and its constant operands) once at load and
//! returns the closure that runs it.

mod elementwise;
mod nn;
mod shape;

use crate::proto::Node;
use crate::tensor::Tensor;

pub type Kernel = Box<dyn Fn(Args) -> Vec<Tensor> + Send + Sync>;

/// A node's inputs; optional ones that the graph leaves out are `None`.
pub struct Args(pub Vec<Option<Tensor>>);

impl Args {
    pub fn get(&self, i: usize) -> &Tensor {
        self.0[i]
            .as_ref()
            .unwrap_or_else(|| panic!("missing input {i}"))
    }
    pub fn opt(&self, i: usize) -> Option<&Tensor> {
        self.0.get(i).and_then(Option::as_ref)
    }
    pub fn take(&mut self, i: usize) -> Tensor {
        self.0[i]
            .take()
            .unwrap_or_else(|| panic!("missing input {i}"))
    }
    pub fn ints(&self, i: usize) -> Option<Vec<i64>> {
        self.opt(i).map(Tensor::ints)
    }
}

fn one(f: impl Fn(Args) -> Tensor + Send + Sync + 'static) -> Kernel {
    Box::new(move |a| vec![f(a)])
}

/// The kernel for `node`, plus whether it captured (pre-packed) its constant operands after the first, which the
/// graph then stops passing.
pub fn prepare(node: &Node, constant: &[Option<&Tensor>]) -> Result<(Kernel, bool), String> {
    if let Some(k) = nn::prepare(node, constant)? {
        return Ok(k);
    }
    let kernel = elementwise::prepare(node).or_else(|| shape::prepare(node));
    kernel
        .map(|k| (k, false))
        .ok_or_else(|| "unsupported operator".to_string())
}
