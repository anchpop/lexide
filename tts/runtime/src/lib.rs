//! The lexide-speech engine: tokenizer, espeak mapping, a small ONNX interpreter for the exported StyleVoca
//! stages, the stage chain and playback trimming. Load a voice from its `.voca` bytes, then `speak`.
//!
//! The interpreter is generic over the graphs (export_onnx.py stays the source of truth); an operator the
//! kernels don't cover is a load-time error naming it.

mod espeak;
mod frontend;
mod gemm;
mod graph;
mod math;
mod ops;
mod proto;
mod sampler;
mod tensor;
mod voice;
#[cfg(feature = "web")]
mod web;

pub use espeak::{Espeak, Spec as EspeakSpec};
pub use frontend::{Gram, Spec as FrontendSpec, Tokenizer, Tokens};
pub use graph::{Graph, Values};
pub use sampler::normal;
pub use tensor::{Data, Tensor};
pub use voice::{Audio, Options, STAGES, Synthesis, Trace, Voice};
