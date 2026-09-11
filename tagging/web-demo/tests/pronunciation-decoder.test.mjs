import test from "node:test";
import assert from "node:assert/strict";
import { deflateSync } from "node:zlib";
import { float16, unpackMatrix, decodePath } from "../www/pronunciation-decoder.mjs";

function fixture(ids) {
  const data = Buffer.alloc(ids.length * 3 * 2);
  for (let frame = 0; frame < ids.length; frame++) {
    for (let id = 0; id < 3; id++) data.writeUInt16LE(id === ids[frame] ? 0 : 0xfc00, (frame * 3 + id) * 2);
  }
  return { shape: [ids.length, 3], dtype: "float16", encoding: "zlib+base64", blank_id: 0,
    vocab: ["<pad>", "a", "b"], data: deflateSync(data).toString("base64") };
}

test("float16 handles normals, subnormals, sign and masked negative infinity", () => {
  assert.equal(float16(0x3c00), 1);
  assert.equal(float16(0xbc00), -1);
  assert.equal(float16(1), 2 ** -24);
  assert.equal(float16(0xfc00), -Infinity);
  assert.ok(Number.isNaN(float16(0x7e00)));
});

test("greedy CTC collapses runs, preserves repeats separated by blanks, and ignores stress changes", async () => {
  const matrix = await unpackMatrix(fixture([0, 1, 1, 0, 1, 2, 2, 0]));
  const frames = Array.from({length: 8}, (_, frame) => ({frame, stress: frame % 3}));
  const {phones, path} = decodePath(matrix, frames);
  assert.deepEqual(phones.map(p => [p.phoneme, p.startFrame, p.endFrame]), [["a", 1, 3], ["a", 4, 5], ["b", 5, 7]]);
  assert.equal(phones[0].confidence, 1);
  assert.equal(phones[0].top_k[0].phoneme, "a");
  assert.equal(path[0].blank, true);
  assert.equal(path[1].stress, 1); // The original head output remains available.
  assert.equal(path[2].stress, 2);
});

test("all blanks produce an empty transcription with an inspectable frame path", async () => {
  const matrix = await unpackMatrix(fixture([0, 0]));
  const result = decodePath(matrix, [{frame: 0, stress: 0}, {frame: 1, stress: 1}]);
  assert.deepEqual(result.phones, []);
  assert.equal(result.path.length, 2);
});

test("rejects malformed matrices and misaligned frame data", async () => {
  await assert.rejects(unpackMatrix({...fixture([1]), shape: [2, 3]}), /invalid frame matrix/);
  await assert.rejects(unpackMatrix({...fixture([1]), blank_id: 9}), /invalid frame matrix/);
  const matrix = await unpackMatrix(fixture([1]));
  assert.throws(() => decodePath(matrix, []), /do not align/);
});
