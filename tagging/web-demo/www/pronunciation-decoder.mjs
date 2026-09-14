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
