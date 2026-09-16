import test, { after } from "node:test";
import assert from "node:assert/strict";
import { deflateSync } from "node:zlib";
import wasm from "../target/node-pkg/parsley_web_demo.js";
import { initDecoder, unpackMatrix, decodePath } from "../www/pronunciation-decoder.mjs";

await initDecoder(wasm);
const allocated = [];
after(() => allocated.forEach(matrix => matrix.free()));
async function unpack(payload) {
  const matrix = await unpackMatrix(payload);
  allocated.push(matrix);
  return matrix;
}

function wire(bits, vocab = ["<pad>", "a", "b"], blank_id = 0) {
  const data = Buffer.alloc(bits.length * 2);
  bits.forEach((value, i) => data.writeUInt16LE(value, i * 2));
  return { shape: [bits.length / vocab.length, vocab.length], dtype: "float16",
    encoding: "zlib+base64", blank_id, vocab, data: deflateSync(data).toString("base64") };
}
function fixture(ids) {
  return wire(ids.flatMap(winner => [0, 1, 2].map(id => id === winner ? 0 : 0xfc00)));
}
const frames = count => Array.from({length: count}, (_, frame) => ({frame, stress: frame % 3}));

// Fixed IEEE-f16 representations of log probabilities, not a second decoder.
const LOG = { p04: 0xbb55, p024: 0xbdb5, p018: 0xbedc, half: 0xb98c, aboveHalf: 0xb98b, p06: 0xb816 };

test("shared WASM CTC collapses runs, keeps blank-separated repeats and frame stress", async () => {
  const matrix = await unpack(fixture([0, 1, 1, 0, 1, 2, 2, 0]));
  const {phones, path} = decodePath(matrix, frames(8));
  assert.deepEqual(phones.map(p => [p.phoneme, p.startFrame, p.endFrame]), [["a", 1, 3], ["a", 4, 5], ["b", 5, 7]]);
  assert.equal(phones[0].confidence, 1);
  assert.equal(phones[0].top_k[0].phoneme, "a");
  assert.equal(path[0].blank, true);
  assert.equal(path[1].stress, 1);
  assert.equal(path[2].stress, 2);
});

test("0.6 nonblank mass emits conditional 0.4 winner despite 0.4 blank winning joint argmax", async () => {
  const matrix = await unpack(wire([LOG.p04, LOG.p024, LOG.p018, LOG.p018], ["<pad>", "a", "b", "c"]));
  const {phones, path} = decodePath(matrix, frames(1));
  assert.equal(path[0].id, 1);
  assert.ok(Math.abs(path[0].probability - 0.24) < 0.001);
  assert.equal(phones.length, 1);
  assert.ok(Math.abs(phones[0].confidence - 0.4) < 0.001);
  assert.deepEqual(phones[0].top_k.map(p => p.phoneme), ["a", "b", "c"]);
  assert.ok(Math.abs(phones[0].top_k.reduce((sum, p) => sum + p.probability, 0) - 1) < 1e-10);
});

test("float16 half-probability rounding follows shared raw-log threshold without adjustment", async () => {
  // log(.5) rounds to -0.693359375, slightly BELOW ln(.5): this wire value emits.
  // The next f16 toward zero (-0.69287109375) is above the threshold and blanks.
  const matrix = await unpack(wire([LOG.half, LOG.p024, LOG.p024, LOG.aboveHalf, LOG.p024, LOG.p024]));
  assert.deepEqual(decodePath(matrix, frames(2)).path.map(p => p.blank), [false, true]);
});

test("specials and masked -infinity never emit or appear in alternatives, even above every phone", async () => {
  const vocab = ["a", "<s>", "|", "", "b", "<pad>", "c"];
  const matrix = await unpack(wire([LOG.p024, 0, 0, 0, LOG.p018, LOG.p04, 0xfc00], vocab, 5));
  const result = decodePath(matrix, frames(1));
  assert.equal(result.path[0].id, 0);
  assert.deepEqual(result.phones[0].top_k.map(p => p.phoneme), ["a", "b"]);
  assert.ok(Math.abs(result.phones[0].confidence - 0.24 / 0.42) < 0.001);
});

test("conditional confidence averages over the run, not blank gaps or stress boundaries", async () => {
  const matrix = await unpack(wire([
    LOG.p04, LOG.p024, LOG.p018,
    LOG.p04, 0, 0xfc00,
    0, 0xfc00, 0xfc00,
  ]));
  const result = decodePath(matrix, frames(3));
  assert.equal(result.phones.length, 1);
  assert.equal(result.phones[0].endFrame, 2);
  assert.ok(Math.abs(result.phones[0].confidence - (0.24 / 0.42 + 1) / 2) < 0.001);
});

test("very negative finite phone scores normalize stably", async () => {
  const matrix = await unpack(wire([LOG.p04, 0xfbf0, 0xfc00]));
  const result = decodePath(matrix, frames(1));
  assert.equal(result.phones[0].confidence, 1);
});

test("all blanks produce an empty transcription with an inspectable path", async () => {
  const result = decodePath(await unpack(fixture([0, 0])), frames(2));
  assert.deepEqual(result.phones, []);
  assert.equal(result.path.length, 2);
});

test("empty probability rows and speech without eligible phones are rejected", async () => {
  await assert.rejects(unpack(wire([0xfc00, 0xfc00, 0xfc00])), /invalid frame matrix/);
  await assert.rejects(unpack(wire([LOG.p04, 0, 0xfc00], ["<pad>", "<s>", "a"])), /invalid frame matrix/);
});

test("rejects malformed matrices through actual Rust wire validation", async () => {
  const valid = fixture([1]);
  for (const change of [
    {shape: [2, 3]}, {shape: [1]}, {shape: []}, {shape: [1, 3, 1]}, {shape: [0, 3]},
    {shape: [2001, 3]}, {shape: [1, 4097]}, {shape: [1.5, 3]},
    {blank_id: 9}, {blank_id: -1}, {dtype: "float32"}, {encoding: "gzip+base64"},
    {vocab: ["<pad>", "a"]}, {vocab: ["<pad>", 3, "b"]}, {data: "not base64!"},
    {data: Buffer.from("not zlib").toString("base64")},
    {data: deflateSync(Buffer.alloc(100)).toString("base64")},
  ]) await assert.rejects(unpack({...valid, ...change}), /invalid frame matrix/);
  for (const invalid of [null, {}, wire([0, 0x7e00, 0]), wire([0, 0x7c00, 0]), wire([0, 0x3c00, 0])]) {
    await assert.rejects(unpack(invalid), /invalid frame matrix/);
  }
});

test("float16 masked infinities and negative subnormals survive wire unpacking", async () => {
  const result = decodePath(await unpack(wire([0xfc00, 0x8001, 0xfc00])), frames(1));
  assert.equal(result.path[0].id, 1);
  assert.ok(Math.abs(result.path[0].probability - Math.exp(-(2 ** -24))) < 1e-7);
});

test("rejects misaligned or invalid stress records without splitting phone runs", async () => {
  const matrix = await unpack(fixture([1]));
  for (const invalid of [[], null, {}, [null], [{frame: 1, stress: 0}], [{frame: 0, stress: 3}], [{frame: 0, stress: 1.5}]]) {
    assert.throws(() => decodePath(matrix, invalid), /do not align/);
  }
});

test("0.4 nonblank mass emits blank regardless of phone argmax", async () => {
  const result = decodePath(await unpack(wire([LOG.p06, 0xbf55, 0xc03e, 0xc03e], ["<pad>", "a", "b", "c"])), frames(1));
  assert.equal(result.path[0].blank, true);
  assert.deepEqual(result.phones, []);
});

function versioned(legacy, frame_rate_ms = 20) {
  const t = legacy.shape[0];
  const probability = (shape, labels, bits, value_semantics = "probability") => ({
    shape, labels, dtype: "float16", encoding: "zlib+base64", value_semantics,
    data: wire(bits).data,
  });
  return {schema_version: 1, producer: {model_id: "test", model_revision: "test", deploy_marker: "test", decoder_version: "nonblank_v1"},
    trained_against_g2p: null, frame_rate_ms, sample_rate: 16000, heads: {
      phone: {shape: legacy.shape, labels: legacy.vocab, blank_id: legacy.blank_id, dtype: legacy.dtype,
        encoding: legacy.encoding, data: legacy.data, value_semantics: "joint_log_probability"},
      nonblank: probability([t], ["nonblank"], Array(t).fill(0x3c00), "sigmoid_probability"),
      stress: probability([t, 3], ["none", "primary", "secondary"], Array(t).fill([0x3c00, 0, 0]).flat()),
    }};
}

import { matrixMetadata } from "../www/pronunciation-decoder.mjs";

test("actual WASM accepts both schemas and exposes declared 40 ms timing to the UI", async () => {
  const legacy = fixture([0, 1, 1, 0, 2]);
  const modern = versioned(legacy, 40);
  const old = decodePath(await unpack(legacy), frames(5));
  const current = decodePath(await unpack(modern), frames(5));
  assert.deepEqual(current, old);
  assert.deepEqual(matrixMetadata(legacy), {frameSeconds: .02, blankId: 0});
  assert.deepEqual(matrixMetadata(modern), {frameSeconds: .04, blankId: 0});
  assert.equal(current.phones[0].startFrame * matrixMetadata(modern).frameSeconds, .04);
  assert.equal(current.phones[0].endFrame * matrixMetadata(modern).frameSeconds, .12);
  for (const change of [{schema_version: 2}, {schema_version: null}, {frame_rate_ms: undefined}, {frame_rate_ms: 0}, {frame_rate_ms: "40"}]) {
    assert.throws(() => matrixMetadata({...modern, ...change}), /Invalid/);
    await assert.rejects(unpack({...modern, ...change}), /invalid frame matrix/);
  }
});

// Optional live artifacts from the authorized eval, through the actual WASM path.
import { readFileSync, readdirSync } from "node:fs";
if (process.env.FRAME_MATRIX_ARTIFACTS) {
  for (const name of readdirSync(process.env.FRAME_MATRIX_ARTIFACTS).filter(name => name.endsWith("-score.json"))) {
    test(`live new and legacy artifact through WASM: ${name}`, async () => {
      const artifact = JSON.parse(readFileSync(`${process.env.FRAME_MATRIX_ARTIFACTS}/${name}`));
      const modern = artifact.response.frame_matrix;
      const phone = modern.heads.phone;
      const legacy = {shape: phone.shape, vocab: phone.labels, blank_id: phone.blank_id,
        dtype: phone.dtype, encoding: phone.encoding, data: phone.data};
      const current = decodePath(await unpack(modern), frames(phone.shape[0]));
      const old = decodePath(await unpack(legacy), frames(phone.shape[0]));
      assert.deepEqual(current, old);
      assert.equal(matrixMetadata(modern).frameSeconds, .02);
    });
  }
}
