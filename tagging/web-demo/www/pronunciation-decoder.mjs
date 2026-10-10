// Wire validation, float16 unpacking and nonblank-first CTC live in shared Rust.
// Tests supply the nodejs-target bindings; the browser initializes its web build lazily.
let ready;
let Matrix;
export async function initDecoder(bindings) {
  if (!ready) {
    ready = (async () => {
      const wasm = bindings ?? await import("./pkg/parsley_web_demo.js");
      if (!bindings) await wasm.default();
      Matrix = wasm.PronunciationMatrix;
    })().catch(error => { ready = undefined; throw error; });
  }
  return ready;
}

export async function unpackMatrix(matrix) {
  await initDecoder();
  return new Matrix(JSON.stringify(matrix ?? null));
}

export async function rawMatrix(values, frames, phone) {
  await initDecoder();
  if (phone.value_semantics !== "joint_log_probability") throw new Error("Invalid phone head semantics.");
  return Matrix.from_log_probs(values, frames, JSON.stringify(phone.labels), phone.blank_id);
}

// Caller owns the WASM matrix and must free it when finished.
export function decodePath(matrix, frames) {
  return JSON.parse(matrix.decode(JSON.stringify(frames ?? null)));
}

// Local ONNX tensors retain all heads; only phone + frame stress enter the UI decoder.
export async function decodeHeads(outputs, metadata) {
  matrixMetadata(metadata);
  const count = outputs.phone.dims[1];
  if (outputs.phone.dims.length !== 3 || outputs.phone.dims[0] !== 1
      || outputs.phone.dims[2] !== metadata.heads.phone.labels.length
      || outputs.stress.dims.join() !== [1, count, 3].join()) {
    throw new Error("The model's phoneme and stress frames do not align.");
  }
  const frames = Array.from({ length: count }, (_, frame) => {
    let stress = 0;
    for (let i = 1; i < 3; i++) {
      if (outputs.stress.data[frame * 3 + i] > outputs.stress.data[frame * 3 + stress]) stress = i;
    }
    return { frame, stress };
  });
  const matrix = await rawMatrix(outputs.phone.data, count, metadata.heads.phone);
  try { return { frames, result: decodePath(matrix, frames) }; }
  finally { matrix.free(); }
}

// Typed arrays become JSON arrays; masked scores remain reconstructable, not null.
export function outputJSON(output) {
  return JSON.stringify(output, (_, value) => ArrayBuffer.isView(value) ? Array.from(value)
    : value === -Infinity ? "-Infinity" : value);
}

// Legacy artifacts have no timebase; only that explicit legacy case gets 20 ms.
// New artifacts must declare valid metadata, never silently fall back.
export function matrixMetadata(payload) {
  if (!payload || typeof payload !== "object") throw new Error("Missing frame matrix metadata.");
  if (!Object.hasOwn(payload, "schema_version")) {
    return { frameSeconds: 0.02, blankId: payload.blank_id };
  }
  if (payload.schema_version !== 1 || typeof payload.frame_rate_ms !== "number"
      || !Number.isFinite(payload.frame_rate_ms) || payload.frame_rate_ms <= 0) {
    throw new Error("Invalid frame matrix version or timebase.");
  }
  const blankId = payload.heads?.phone?.blank_id;
  if (!Number.isInteger(blankId) || blankId < 0) throw new Error("Missing phone head blank_id.");
  return { frameSeconds: payload.frame_rate_ms / 1000, blankId };
}
