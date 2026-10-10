import test from "node:test";
import assert from "node:assert/strict";
import { LocalTranscriber } from "../www/pronunciation-inference.mjs";

function fixture() {
  const workers = [];
  const runner = new LocalTranscriber(() => {
    const worker = { postMessage(data, transfer) { this.data = structuredClone(data, { transfer }); },
      terminate() { this.terminated = true; } };
    workers.push(worker);
    return worker;
  }, { modelURL: "model.onnx", metadataURL: "metadata.json" });
  return { runner, workers };
}

test("lazy worker retains audio for retry, reuses a successful session, and forwards progress", async () => {
  const {runner, workers} = fixture();
  assert.equal(workers.length, 0);
  const samples = Float32Array.of(.1, -.2);
  const statuses = [];
  const pending = runner.run(samples, status => statuses.push(status));
  assert.deepEqual(workers[0].data.waveform, samples);
  assert.equal(samples.length, 2);
  workers[0].onmessage({data: {status: "Loading browser model…"}});
  workers[0].onmessage({data: {result: {phones: []}}});
  assert.deepEqual(await pending, {result: {phones: []}});
  assert.deepEqual(statuses, ["Loading browser model…"]);
  const next = runner.run(samples, () => {});
  assert.equal(workers.length, 1);
  workers[0].onmessage({data: {result: {phones: []}}});
  await next;
});

test("Cancel terminates inference; retry gets a fresh worker and intact samples", async () => {
  const {runner, workers} = fixture();
  const samples = Float32Array.of(.1);
  const first = runner.run(samples, () => {});
  assert.throws(() => runner.run(samples, () => {}), /already running/);
  runner.cancel();
  await assert.rejects(first, {name: "AbortError"});
  assert.equal(workers[0].terminated, true);
  const next = runner.run(samples, () => {});
  assert.equal(workers.length, 2);
  assert.equal(workers[1].data.waveform[0], samples[0]);
  workers[0].onerror({message: "late failure"});
  workers[0].onmessage({data: {error: "late result"}});
  assert.notEqual(workers[1].terminated, true);
  workers[1].onmessage({data: {result: {phones: []}}});
  await next;
});

test("load and inference failures discard the worker so retry can recover", async () => {
  for (const crash of [false, true]) {
    const {runner, workers} = fixture();
    const pending = runner.run([0], () => {});
    if (crash) workers[0].onerror({message: "load failed"});
    else workers[0].onmessage({data: {error: "load failed"}});
    await assert.rejects(pending, /load failed/);
    assert.equal(workers[0].terminated, true);
    const next = runner.run([0], () => {});
    assert.equal(workers.length, 2);
    workers[1].onmessage({data: {result: {phones: []}}});
    await next;
  }
});
