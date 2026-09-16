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

// Caller owns the WASM matrix and must free it when finished.
export function decodePath(matrix, frames) {
  return JSON.parse(matrix.decode(JSON.stringify(frames ?? null)));
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
