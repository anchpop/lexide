// CHROMIUM_PATH=/path/to/chromium pnpm test:browser /path/to/yap.voca [/path/to/espeak.voca]
// Serves the built package with the given voices standing in for the catalog's, then checks loading, OPFS caching,
// speaking with every option, reproducibility, and (with an espeak voice) text through ../../web-espeak/dist.
import assert from 'node:assert/strict'
import {createServer} from 'node:http'
import {readFile} from 'node:fs/promises'
import {basename} from 'node:path'
import {chromium} from 'playwright-core'

const [yapPath, espeakPath] = process.argv.slice(2)
if (!yapPath) throw new Error('Supply an English yap .voca file')
const hashOf = path => basename(path, '.voca')
const files = new Map([[yapPath, espeakPath].filter(Boolean).map(p => ['/pkg/voices/' + basename(p), p])].flat())
const dist = new URL('../dist/', import.meta.url)
// The catalog pins hashes; point it at the voices under test.
const catalog = (await readFile(new URL('catalog.js', dist), 'utf8'))
  .replace(/(\['eng\/yap', ')\w+/, '$1' + hashOf(yapPath))
  .replace(/(\['eng\/espeak', ')\w+/, '$1' + (espeakPath ? hashOf(espeakPath) : 'none'))
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
  let data
  if (files.has(req.url)) data = await readFile(files.get(req.url))
  else if (req.url === '/pkg/catalog.js') data = Buffer.from(catalog)
  else if (/^\/pkg\/[\w.-]+$/.test(req.url)) data = await readFile(new URL(basename(req.url), dist)).catch(() => undefined)
  else if (/^\/espeak\/[\w.-]+$/.test(req.url)) {
    data = await readFile(new URL('../../web-espeak/dist/' + basename(req.url), import.meta.url)).catch(() => undefined)
  }
  if (!data) { res.writeHead(404); return res.end() }
  res.setHeader('Content-Type', req.url.endsWith('js') ? 'text/javascript'
    : req.url.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream')
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
  const spoken = await page.evaluate(async rows => {
    const options = [{}, {pitchShift: .1, pitchJitter: 1.5, seed: 3, variety: .7},
      {pitchRange: 1.6, commaPause: .3, stopPause: .5, voicedVowels: false}, {variety: 0, fixEnding: false}]
    const results = []
    for (const [i, row] of rows.entries()) {
      for (const extra of options) {
        const request = {take: 9 * i, speed: 1 + .1 * i, speaker: voice.speakers[i], ...extra}
        const [a, b] = [await voice.speak(row.grams, request), await voice.speak(row.grams, request)]
        if (a.sampleRate !== 24000 || !a.samples.length || a.samples.some(x => !(Math.abs(x) <= 1))) {
          throw new Error('Bad audio for case ' + i)
        }
        if (a.samples.length !== b.samples.length || a.samples.some((x, j) => x !== b.samples[j])) {
          throw new Error('Same take and seed differ in case ' + i)
        }
        results.push(a.samples.length)
      }
    }
    const errors = []
    for (const bad of [{speed: 9}, {speaker: 'nobody'}, {take: 1.5}, {pitch: 1}]) {
      await voice.speak(rows[0].grams, bad).then(() => errors.push('accepted ' + JSON.stringify(bad)), () => {})
    }
    await voice.speak([{words: ['.'], pos: ['PUNCT'], count: 0, phonemes: ''}])
      .then(() => errors.push('accepted silence'), e => e.message.includes('speech') || errors.push(e.message))
    if (errors.length) throw new Error(errors.join('; '))
    return results
  }, rows)
  console.log('reproducible samples per request', spoken)
  if (espeakPath) {
    const lengths = await page.evaluate(async () => {
      const {loadVoice} = await import('/pkg/index.js')
      const {loadEspeak} = await import('/espeak/index.mjs')
      const voice = await loadVoice('eng', await loadEspeak('en-us'))
      const lengths = []
      for (const text of ['Wait, are you sure?', '‘Don’t’ – d’Artagnan… (hello)!', 'It was only a line.']) {
        lengths.push((await voice.speak(text, {take: 0})).samples.length)
      }
      voice.release()
      return lengths
    })
    assert.ok(lengths.every(n => n > 0))
    console.log('espeak samples per sentence', lengths)
  }
  await page.evaluate(async () => {
    voice.release()
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
