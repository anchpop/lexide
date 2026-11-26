import {env} from './ort.wasm.min.mjs'
import {wasmHash} from './assets.mjs'
import {readAsset} from './cache.js'
import {voiceBase, voices} from './catalog.js'
import {createEngine} from './engine.js'
import type {Progress, Gram, SpeakOptions} from './index.js'

const progress = (progress: Progress) => self.postMessage({progress})
async function load(language: string, threads?: number) {
  const hash = voices.get(language)
  if (!hash) throw new Error('No voice for ' + language)
  const cores = navigator.hardwareConcurrency || 1
  const count = threads ?? Math.min(4, cores)
  if (!Number.isInteger(count) || count < 1) throw new RangeError('Invalid thread count')
  env.wasm.numThreads = crossOriginIsolated ? Math.min(count, cores) : 1
  env.wasm.wasmPaths = {mjs: new URL('./ort-wasm-simd-threaded.mjs', import.meta.url).href}
  const root = await navigator.storage?.getDirectory?.().catch(() => undefined)
  const dir = await root?.getDirectoryHandle('lexide-speech', {create: true})
  const [bundle, wasm] = await Promise.all([
    readAsset(dir, 'voice', hash, new URL(hash + '.voca', voiceBase), progress),
    readAsset(dir, 'wasm', wasmHash, new URL('./ort-wasm-simd-threaded.wasm', import.meta.url), progress),
  ])
  env.wasm.wasmBinary = new Uint8Array(wasm)
  progress({phase: 'sessions'})
  const engine = await createEngine(bundle)
  progress({phase: 'warmup'})
  await engine.warmup()
  if (dir) {
    const keep = new Set([...voices.values()].map(hash => 'voice-' + hash))
    keep.add('wasm-' + wasmHash)
    for await (const name of dir.keys()) {
      if (!keep.has(name)) {
        await navigator.locks.request('lexide-speech-' + name, async () => {
          await dir.removeEntry(name).catch((e: DOMException) => { if (e.name !== 'NotFoundError') throw e })
        })
      }
    }
  }
  return engine
}

type Request = {id: number} & ({language: string, threads?: number} | {grams: Gram[], options?: SpeakOptions})
let engine: Awaited<ReturnType<typeof load>>
let queue = Promise.resolve()
self.onmessage = ({data}: MessageEvent<Request>) => {
  queue = queue.then(async () => {
    try {
      if ('language' in data) {
        engine = await load(data.language, data.threads)
        self.postMessage({id: data.id, value: engine.speakers})
      } else {
        const value = await engine.speak(data.grams, data.options)
        self.postMessage({id: data.id, value}, {transfer: [value.samples.buffer]})
      }
    } catch (e) {
      self.postMessage({id: data.id, error: String(e)})
    }
  })
}
