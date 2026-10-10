import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";
import { decodeHeads } from "./pronunciation-decoder.2c63f52165fa993c.mjs";

// One thread works on static Pages without cross-origin isolation headers.
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
const modelId = "anchpop/lexide-pronunciation-small";
// Run 2 (a5472e4a) plus its int8 export under onnx/.
const revision = "58bd3171a23ec86876dd71ba4c9d9179aab2ce58";
const base = `https://huggingface.co/${modelId}/resolve/${revision}/onnx`;
let session;
let metadata;

self.onmessage = async ({ data: { waveform } }) => {
  try {
    if (!session) {
      self.postMessage({ status: "Loading browser model…" });
      const response = await fetch(`${base}/frame_matrix.json`);
      if (!response.ok) throw new Error(`Could not download the model (${response.status}).`);
      metadata = await response.json();
      if (metadata.sample_rate !== 16000) throw new Error("Invalid model sample rate.");
      session = await ort.InferenceSession.create(`${base}/model.int8.onnx`, { executionProviders: ["wasm"] });
    }
    self.postMessage({ status: "Transcribing locally…" });
    // Scratch training uses do_normalize=False. Waveform and mel normalization
    // are already inside this ONNX graph; do not normalize in JavaScript.
    const outputs = await session.run({
      waveform: new ort.Tensor("float32", waveform, [1, waveform.length]),
      lengths: new ort.Tensor("int64", BigInt64Array.of(BigInt(waveform.length)), [1]),
    });
    try {
      const { frames, result } = await decodeHeads(outputs, metadata);
      const heads = Object.fromEntries(Object.entries(outputs).map(([name, tensor]) => [name, {
        ...metadata.heads[name], shape: tensor.dims.slice(1), dtype: "float32", data: tensor.data,
      }]));
      self.postMessage({ result, output: {
        model_id: modelId, model_revision: revision, model_identity: `${modelId}@${revision}`,
        runtime: "onnxruntime-web@1.30.0/wasm", precision: "int8", decoder_version: "nonblank_v1",
        frame_matrix: { ...metadata, heads }, frames,
      } });
    } finally {
      Object.values(outputs).forEach(tensor => tensor.dispose());
    }
  } catch (error) {
    self.postMessage({ error: error.message });
  }
};
