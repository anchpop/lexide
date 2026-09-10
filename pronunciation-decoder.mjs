// Production frame_matrix contract: row-major float16 log probabilities,
// zlib-compressed, with vocabulary IDs and the CTC blank explicitly supplied.
export function float16(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 31;
  const fraction = bits & 1023;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * (exponent ? 2 ** (exponent - 15) * (1 + fraction / 1024) : 2 ** -14 * fraction / 1024);
}

export async function unpackMatrix(matrix) {
  const invalid = () => { throw new Error("The model returned an invalid frame matrix."); };
  if (!matrix || matrix.dtype !== "float16" || matrix.encoding !== "zlib+base64"
      || !Array.isArray(matrix.shape) || matrix.shape.length !== 2) invalid();
  const [rows, columns] = matrix.shape;
  if (!Number.isInteger(rows) || rows < 1 || rows > 2000
      || !Number.isInteger(columns) || columns < 2 || columns > 4096
      || !Array.isArray(matrix.vocab) || matrix.vocab.length !== columns
      || matrix.vocab.some((token) => typeof token !== "string")
      || !Number.isInteger(matrix.blank_id) || matrix.blank_id < 0 || matrix.blank_id >= columns
      || typeof matrix.data !== "string") invalid();
  if (typeof DecompressionStream === "undefined") {
    throw new Error("This browser cannot unpack the model output. Please use a current browser.");
  }
  const compressed = Uint8Array.from(atob(matrix.data), (char) => char.charCodeAt(0));
  const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("deflate"));
  const bytes = await new Response(stream).arrayBuffer();
  if (bytes.byteLength !== rows * columns * 2) invalid();
  const view = new DataView(bytes);
  const values = new Float32Array(rows * columns);
  for (let i = 0; i < values.length; i++) {
    const value = float16(view.getUint16(i * 2, true));
    if (Number.isNaN(value) || value > 0) invalid();
    values[i] = value;
  }
  return { rows, columns, values, vocab: matrix.vocab, blankId: matrix.blank_id };
}

// Best frame path (greedy CTC), not the sequence with maximum summed CTC mass.
// Stress is a separate head: it must never split a phoneme's CTC run.
export function decodePath(matrix, frames) {
  const { rows, columns, values, vocab, blankId } = matrix;
  if (!Array.isArray(frames) || frames.length !== rows || frames.some((frame, i) =>
    frame.frame !== i || ![0, 1, 2].includes(frame.stress))) {
    throw new Error("The model's phoneme and stress frames do not align.");
  }
  const path = [];
  const phones = [];
  let run = null;
  function finish() {
    if (!run) return;
    const count = run.endFrame - run.startFrame;
    const probabilities = Array.from(run.sums, (sum, id) => ({ phoneme: vocab[id], probability: sum / count, id }));
    probabilities.sort((a, b) => b.probability - a.probability);
    phones.push({
      id: run.id, phoneme: vocab[run.id], startFrame: run.startFrame, endFrame: run.endFrame,
      confidence: run.sums[run.id] / count, top_k: probabilities.slice(0, 3),
    });
    run = null;
  }
  for (let frame = 0; frame < rows; frame++) {
    const offset = frame * columns;
    let id = 0;
    for (let j = 1; j < columns; j++) if (values[offset + j] > values[offset + id]) id = j;
    if (values[offset + id] === -Infinity) throw new Error("The model returned an empty probability frame.");
    const blank = id === blankId;
    path.push({ ...frames[frame], id, phoneme: blank ? "CTC blank" : vocab[id], blank,
      probability: Math.exp(values[offset + id]) });
    if (run && (blank || id !== run.id)) finish();
    if (blank || /^<.*>$/.test(vocab[id])) continue;
    if (!run) run = { id, startFrame: frame, endFrame: frame, sums: new Float64Array(columns) };
    run.endFrame = frame + 1;
    for (let j = 0; j < columns; j++) run.sums[j] += Math.exp(values[offset + j]);
  }
  finish();
  return { path, phones };
}
