// CHROMIUM_PATH=/path/to/chromium pnpm test:browser /path/to/<hash>.voca
import assert from 'node:assert/strict'
import {createServer} from 'node:http'
import {readFile} from 'node:fs/promises'
import {basename} from 'node:path'
import {chromium} from 'playwright-core'

const bundlePath = process.argv[2]
if (!bundlePath) throw new Error('Supply the exported English .voca file')
const bundle = await readFile(bundlePath)
const header = bundle.readUInt32LE(0)
const config = JSON.parse(bundle.subarray(4, 4 + header))
const files = new Map([
  ['/voices/' + basename(bundlePath), bundle],
  ['/reference/model.json', Buffer.from(JSON.stringify(config))],
  ['/runner.mjs', await readFile(new URL('../synthesize.mjs', import.meta.url))],
])
let offset = 4 + header
for (const [stage, size] of Object.entries(config.graphs)) {
  files.set('/reference/' + stage + '.onnx', bundle.subarray(offset, offset + size))
  offset += size
}
const requests = []
const server = createServer(async (req, res) => {
  requests.push(req.url)
  res.setHeader('Access-Control-Allow-Origin', '*')
  if (req.url !== '/single') {
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin')
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp')
  }
  if (['/', '/single'].includes(req.url)) {
    res.setHeader('Content-Type', 'text/html')
    return res.end('<!doctype html><title>Voice test</title>')
  }
  let data = files.get(req.url)
  if (/^\/pkg\/[\w.-]+$/.test(req.url)) {
    data = await readFile(new URL('../dist/' + basename(req.url), import.meta.url)).catch(() => undefined)
  }
  if (!data) { res.writeHead(404); return res.end() }
  res.setHeader('Content-Type', req.url.endsWith('js') ? 'text/javascript' : 'application/octet-stream')
  res.setHeader('Content-Length', data.length)
  res.end(data)
})
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
const browser = await chromium.launch({executablePath: process.env.CHROMIUM_PATH, args: ['--disable-dev-shm-usage']})
try {
  const page = await browser.newPage()
  const origin = 'http://127.0.0.1:' + server.address().port
  // Redirect the pinned hosting origin to the identical exported asset on loopback.
  await page.context().route('**/voices/*.voca', route => {
    const url = new URL(route.request().url())
    if (url.origin === origin) return route.continue()
    return route.fulfill({status: 307, headers: {location: origin + url.pathname, 'access-control-allow-origin': '*'}})
  })
  await page.goto(origin)
  const rows = JSON.parse(await readFile(new URL('utterances.json', import.meta.url)))
  async function load() {
    return page.evaluate(async () => {
      const {loadVoice} = await import('/pkg/index.js')
      const start = performance.now()
      const events = []
      window.voice = await loadVoice('eng', {onProgress: e => {
        if (e.elapsedMs !== undefined || !e.asset) events.push({...e, time: performance.now() - start})
      }})
      return {total: performance.now() - start, events}
    })
  }
  const cold = await load()
  assert.equal(cold.events.filter(e => e.phase === 'download').length, 2)
  console.log('cold', JSON.stringify(cold))
  const comparisons = await page.evaluate(async rows => {
    const ort = await import('/pkg/ort.wasm.min.mjs')
    const {loadVoice} = await import('/runner.mjs')
    ort.env.wasm.wasmPaths = location.origin + '/pkg/'
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency)
    const original = await loadVoice(ort, location.origin + '/reference/')
    const results = []
    for (const [i, row] of rows.entries()) {
      const take = 9 * i, speed = 1 + .1 * i
      const {samples, sampleRate} = await voice.speak(row.grams, {take, speed, speaker: voice.speakers[i]})
      const {wave, durations} = await original.synthesize(row.ids, row.lexical,
          {speaker: i, speed, take, seed: 0, commaPause: .15, stopPause: .3})
      // Original phone page's trim and peak normalization, applied to raw runner PCM.
      const hop = original.config.hop
      const phone = Number(durations.slice(0, row.firstPhone).reduce((a, b) => a + b, 0n))
      const loud = f => {
        let s = 0
        for (let i = f * hop; i < (f + 1) * hop; i++) s += wave[i] ** 2
        return 10 * Math.log10(Math.max(s / hop, 1e-12)) > -50
      }
      let first = phone
      for (let f = 0; f + 1 < wave.length / hop; f++) {
        if (loud(f) && loud(f + 1)) { first = Math.min(f, phone); break }
      }
      const expected = wave.slice(Math.max(0, first - 4) * hop)
      let peak = 1e-9
      for (const x of expected) peak = Math.max(peak, Math.abs(x))
      const gain = Math.min(1, .99 / peak)
      for (let j = 0; j < expected.length; j++) expected[j] *= gain
      const bits = new Uint32Array(samples.buffer), target = new Uint32Array(expected.buffer)
      if (sampleRate !== original.config.sampleRate || bits.length !== target.length || bits.some((v, j) => v !== target[j])) {
        throw new Error('PCM mismatch in case ' + i)
      }
      results.push(bits.length)
    }
    await original.release()
    voice.release()
    return results
  }, rows)
  console.log('identical float32 samples per sentence', comparisons)
  await page.evaluate(async () => {
    const cache = await (await navigator.storage.getDirectory()).getDirectoryHandle('lexide-speech')
    await cache.getFileHandle('voice-obsolete', {create: true})
  })
  const start = requests.length
  const cached = await load()
  assert.equal(cached.events.filter(e => e.phase === 'opfs').length, 2)
  assert.ok(!requests.slice(start).some(url => /\.(wasm|voca)$/.test(url)))
  console.log('cached', JSON.stringify(cached))
  assert.equal(await page.evaluate(async () => {
    const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle('lexide-speech')
    const keys = []
    for await (const key of dir.keys()) keys.push(key)
    return keys.includes('voice-obsolete')
  }), false)
  await page.goto(origin + '/single')
  assert.equal(await page.evaluate(() => crossOriginIsolated), false)
  await load()
  assert.ok(await page.evaluate(async grams => (await voice.speak(grams)).samples.length > 0, rows[0].grams))
} finally {
  await browser.close()
  server.close()
}
