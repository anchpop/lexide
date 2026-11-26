// PyTorch CPU's MT19937. Delivery, pitch jitter and the NSF source use independent seeds.
function normalSampler(seed) {
  const state = new Uint32Array(624);
  state[0] = seed;
  for (let i = 1; i < 624; ++i) {
    const prev = state[i - 1];
    state[i] = Math.imul(1812433253, prev ^ (prev >>> 30)) + i;
  }
  let at = 624;
  function next() {
    if (at === 624) {
      for (let j = 0; j < 624; ++j) {
        const bits = (state[j] & 0x80000000) | (state[(j + 1) % 624] & 0x7fffffff);
        state[j] = state[(j + 397) % 624] ^ (bits >>> 1) ^ ((bits & 1) ? 0x9908b0df : 0);
      }
      at = 0;
    }
    let y = state[at++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }
  const uniform = () => (next() & 0xffffff) / 16777216;
  const uniformDouble = () => ((next() & 0x1fffff) * 4294967296 + next()) / 9007199254740992;
  return function normal(count) {
    const values = new Float32Array(count);
    if (count < 16) {
      for (let i = 0; i < count; i += 2) {
        const theta = 2 * Math.PI * uniformDouble();
        const radius = Math.sqrt(-2 * Math.log1p(-uniformDouble()));
        values[i] = radius * Math.cos(theta);
        if (i + 1 < count) values[i + 1] = radius * Math.sin(theta);
      }
      return values;
    }
    function transform(i) {
      for (let j = i; j < i + 8; ++j) {
        const r = Math.fround(Math.sqrt(-2 * Math.log(1 - values[j])));
        const theta = Math.fround(Math.fround(2 * Math.PI) * values[j + 8]);
        values[j] = r * Math.cos(theta);
        values[j + 8] = r * Math.sin(theta);
      }
    }
    for (let i = 0; i < count; ++i) values[i] = uniform();
    for (let i = 0; i <= count - 16; i += 16) transform(i);
    if (count % 16) {
      for (let i = count - 16; i < count; ++i) values[i] = uniform();
      transform(count - 16);
    }
    return values;
  };
}

function pitchOffsets(frames, shift, jitter, seed) {
  if (!(jitter > 0)) return Float32Array.of(shift);
  const normal = normalSampler(seed + 7919);
  const fast = normal(frames);
  const drift = normal(frames + 16);
  const slow = new Float32Array(frames);
  for (let i = 0; i < frames; ++i) {
    let sum = 0;
    for (let j = 0; j < 17; ++j) sum = Math.fround(sum + drift[i + j]);
    slow[i] = sum / 17;
  }
  const mean = slow.reduce((a, b) => a + b, 0) / frames;
  const variance = slow.reduce((a, b) => a + (b - mean) ** 2, 0) / (frames - 1);
  const scale = Math.max(Math.sqrt(variance), 1e-6);
  return fast.map((v, i) => shift + (2 * slow[i] / scale + v) * jitter * Math.LN2 / (12 * Math.sqrt(5)));
}

// ORT Web 1.31.1 on NVIDIA Ampere miscomputes the first strided noise
// projection on GPU. Keep just that convolution and the predictors on WASM.
export function renderOnWebGPU(stage) {
  const gpu = {name: "webgpu", forceCpuNodeNames: ["/noise_convs.0/Conv"]};
  return {executionProviders: stage === "render" ? [gpu, "wasm"] : ["wasm"]};
}

// Supply onnxruntime-web, a URL ending in '/', and session options or a function of stage name.
// Inputs come from the checkpoint's tokenizer. Each lexical row follows model.json's column order.
export async function loadVoice(ort, base, options = {}) {
  const response = await fetch(new URL('model.json', base));
  if (!response.ok) throw new Error('Failed to fetch voice manifest: ' + response.status);
  const config = await response.json();
  const sessions = {};
  const release = () => Promise.all(Object.values(sessions).map(s => s.release()));
  try {
    for (const stage of ['encode', 'predict', 'melody', 'blend', 'render']) {
      const sessionOptions = typeof options === 'function' ? options(stage) : options;
      sessions[stage] = await ort.InferenceSession.create(new URL(stage + '.onnx', base).href, sessionOptions);
    }
  } catch (error) {
    await release();
    throw error;
  }
  const run = (stage, inputs) => sessions[stage].run(Object.fromEntries(
    sessions[stage].inputNames.map(name => [name, inputs[name]])));
  const ints = (array, dims) => new ort.Tensor('int64', BigInt64Array.from(array, BigInt), dims);
  const floats = (array, dims) => new ort.Tensor('float32', Float32Array.from(array), dims);
  const scalar = x => floats([x], []);

  async function predict(common, draw) {
    const prediction = await run('predict', {...common, draw});
    const durations = prediction.durations.data;
    const frames = Number(durations.reduce((a, b) => a + b, 0n));
    if (frames < 1 || frames > config.maxFrames) throw new RangeError('Invalid predicted frame count: ' + frames);
    const indices = new BigInt64Array(frames);
    let end = 0;
    durations.forEach((n, i) => {
      indices.fill(BigInt(i), end, end + Number(n));
      end += Number(n);
    });
    const owner = new ort.Tensor('int64', indices, [1, frames]);
    const reading = {...common, ...prediction, owner};
    return {...reading, ...await run('melody', reading)};
  }

  async function synthesize(ids, lexical, {
    speaker = 0, take = 0, seed = 0, variety = 1, speed = 1, pitchShift = 0, pitchRange = 1, pitchJitter = 0,
    commaPause = 0, stopPause = 0, voicedVowels = true, fixEnding = true,
  } = {}) {
    if (!ids.length || ids.length > config.maxTokens || lexical.length !== ids.length ||
        lexical.some(row => row.length !== config.lexical.length)) throw new RangeError('Invalid token or lexical shape');
    if (!(speed >= 0.25 && speed <= 4 && pitchRange >= 0 && pitchRange <= 3 && Number.isFinite(pitchShift))) {
      throw new RangeError('Invalid speed or pitch controls');
    }
    const common = {
      tokens: ints(ids, [1, ids.length]), lexical: ints(lexical.flat(), [1, ids.length, config.lexical.length]),
      speaker: ints([speaker], [1]), speed: scalar(speed),
      pitch_range: scalar(pitchRange), comma_pause: scalar(commaPause), stop_pause: scalar(stopPause),
      voiced_vowels: new ort.Tensor('bool', Uint8Array.of(+voicedVowels), []),
    };
    Object.assign(common, await run('encode', common));
    const z = normalSampler(take)(common.mean.data.length);
    const draw = floats(z.map((v, i) => common.mean.data[i] + Math.fround(v * Math.fround(variety * common.scale.data[i]))), common.mean.dims);
    const reading = await predict(common, draw);
    let endingMs = 0;
    if (fixEnding && variety !== 0 && reading.rising.data[0]) {
      const start = performance.now();
      const typical = await predict(common, common.mean);
      const {blended} = await run('blend', {...reading, typical: typical.prosody, typical_durations: typical.durations});
      reading.prosody = blended;
      endingMs = performance.now() - start;
    }
    const samples = reading.owner.size * config.hop;
    const noise = new ort.Tensor('float32', normalSampler(seed)(samples), [1, 1, samples]);
    const offsets = pitchOffsets(reading.owner.size, pitchShift, pitchJitter, seed);
    const pitch_shift = new ort.Tensor('float32', offsets, [1, offsets.length]);
    const {wave} = await run('render', {...reading, noise, pitch_shift});
    return {wave: wave.data, durations: reading.durations.data, prosody: reading.prosody.data, endingMs};
  }
  return {config, sessions, synthesize, release};
}
