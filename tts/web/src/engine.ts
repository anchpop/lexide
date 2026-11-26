import * as ort from './ort.wasm.min.mjs'
import {normalSampler} from './sampler.js'
import {tokenizer, type Gram} from './tokenizer.js'
import type {SpeakOptions} from './index.js'

interface Manifest {
  frontend: Parameters<typeof tokenizer>[0]
  lexical: Parameters<typeof tokenizer>[1]
  graphs: Record<string, number>
  speakers: string[]
  sampleRate: number
  hop: number
  maxTokens: number
  maxFrames: number
}

type Tensors = Record<string, ort.Tensor>
const ints = (data: number[], dims: number[]) => new ort.Tensor('int64', BigInt64Array.from(data, BigInt), dims)
const scalar = (x: number) => new ort.Tensor('float32', Float32Array.of(x), [])

function playback(wave: Float32Array, durations: BigInt64Array, firstPhone: number, hop: number) {
  let onset = 0
  for (let i = 0; i < firstPhone; i++) onset += Number(durations[i])
  let previous = false
  for (let frame = 0; frame <= onset && (frame + 1) * hop <= wave.length; frame++) {
    let power = 0
    for (let i = frame * hop; i < (frame + 1) * hop; i++) power += wave[i] ** 2
    const loud = power / hop > 1e-5
    if (previous && loud) { onset = frame - 1; break }
    previous = loud
  }
  const samples = wave.slice(Math.max(0, onset - 4) * hop)
  let peak = 0
  for (const x of samples) peak = Math.max(peak, Math.abs(x))
  if (peak > .99) {
    const gain = .99 / peak
    for (let i = 0; i < samples.length; i++) samples[i] *= gain
  }
  return samples
}

export async function createEngine(bundle: ArrayBuffer) {
  const length = new DataView(bundle).getUint32(0, true)
  const config: Manifest = JSON.parse(new TextDecoder().decode(new Uint8Array(bundle, 4, length)))
  const tokenize = tokenizer(config.frontend, config.lexical)
  const sessions: Record<string, ort.InferenceSession> = {}
  let offset = 4 + length
  for (const [name, size] of Object.entries(config.graphs)) {
    sessions[name] = await ort.InferenceSession.create(new Uint8Array(bundle, offset, size), {executionProviders: ['wasm']})
    offset += size
  }

  function run(name: string, inputs: Tensors): Promise<Tensors> {
    const session = sessions[name]
    return session.run(Object.fromEntries(session.inputNames.map(key => [key, inputs[key]])))
  }
  async function predict(common: Tensors, draw: ort.Tensor) {
    const prediction = await run('predict', {...common, draw})
    const durations = prediction.durations.data as BigInt64Array
    const frames = Number(durations.reduce((a, b) => a + b, 0n))
    if (frames < 1 || frames > config.maxFrames) throw new RangeError('Invalid predicted frame count ' + frames)
    const indices = new BigInt64Array(frames)
    let end = 0
    durations.forEach((n, i) => {
      indices.fill(BigInt(i), end, end + Number(n))
      end += Number(n)
    })
    const owner = new ort.Tensor('int64', indices, [1, frames])
    const reading = {...common, ...prediction, owner}
    return {...reading, ...await run('melody', reading)} as Tensors
  }

  async function speak(grams: readonly Gram[], {speaker = config.speakers[0],
                       take = crypto.getRandomValues(new Uint32Array(1))[0], speed = 1}: SpeakOptions = {}) {
    const {ids, lexical, firstPhone} = tokenize(grams)
    if (firstPhone < 0 || ids.length > config.maxTokens) throw new RangeError('Expected speech within the voice token limit')
    const speakerId = config.speakers.indexOf(speaker)
    if (speakerId < 0 || !(speed >= .25 && speed <= 4) || !Number.isSafeInteger(take)) {
      throw new RangeError('Invalid speaker, speed or take')
    }
    const common: Tensors = {
      tokens: ints(ids, [1, ids.length]), lexical: ints(lexical.flat(), [1, ids.length, config.lexical.length]),
      speaker: ints([speakerId], [1]), speed: scalar(speed), pitch_range: scalar(1),
      comma_pause: scalar(.15), stop_pause: scalar(.3), voiced_vowels: new ort.Tensor('bool', Uint8Array.of(1), []),
    }
    Object.assign(common, await run('encode', common))
    const mean = common.mean.data as Float32Array
    const scale = common.scale.data as Float32Array
    const z = normalSampler(take)(mean.length)
    const draw = new ort.Tensor('float32', z.map((v, i) => mean[i] + Math.fround(v * scale[i])), common.mean.dims)
    const reading = await predict(common, draw)
    if (reading.rising.data[0]) {
      const typical = await predict(common, common.mean)
      const {blended} = await run('blend', {...reading, typical: typical.prosody, typical_durations: typical.durations})
      reading.prosody = blended
    }
    const count = reading.owner.size * config.hop
    const noise = new ort.Tensor('float32', normalSampler(0)(count), [1, 1, count])
    const pitch_shift = new ort.Tensor('float32', Float32Array.of(0), [1, 1])
    const {wave} = await run('render', {...reading, noise, pitch_shift})
    const samples = playback(wave.data as Float32Array, reading.durations.data as BigInt64Array, firstPhone, config.hop)
    return {samples, sampleRate: config.sampleRate}
  }
  const warmup = () => speak([{words: ['x'], pos: ['X'], count: 1, phonemes: config.frontend.phones[0]}], {take: 0})
  return {speak, speakers: config.speakers, warmup}
}
