import {loadVoice} from './lexide-speech/index.js'
import {loadEspeak} from './espeak-ng/index.mjs'

const el = id => document.getElementById(id)
const field = el('text')
const speak = el('speak')
const initialText = field.defaultValue
let voice, initialAudio, context, source, activeButton
let generation = 0

function report(error) {
  el('error').textContent = error.message || String(error)
}

async function read(url) {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Could not load ${url} (HTTP ${response.status})`)
  return response
}

// Unlock synchronously during every tap, including on an iPhone in silent mode.
function unlock() {
  if (navigator.audioSession) navigator.audioSession.type = 'playback'
  context ??= new AudioContext()
  const ready = context.resume()
  const silence = context.createBufferSource()
  silence.buffer = context.createBuffer(1, 1, context.sampleRate)
  silence.connect(context.destination)
  silence.start()
  return ready
}

function finish() {
  if (activeButton) activeButton.textContent = 'Play'
  activeButton = undefined
  if (source) {
    source.onended = null
    source.stop()
    source.disconnect()
    source = undefined
  }
}

async function listen(button, makeBuffer) {
  const id = ++generation
  try {
    const ready = unlock()
    const stopping = button === activeButton
    finish()
    el('error').textContent = ''
    if (stopping) { await ready; return }
    activeButton = button
    button.textContent = 'Loading…'
    if (context.state !== 'running') el('error').textContent = 'Audio ' + context.state
    await ready
    if (id !== generation) return
    if (context.state !== 'running') throw new Error('Audio ' + context.state)
    el('error').textContent = ''
    const buffer = await makeBuffer()
    if (id !== generation) return
    source = context.createBufferSource()
    source.buffer = buffer
    source.connect(context.destination)
    source.onended = () => {
      if (id === generation) {
        source.disconnect()
        source = undefined
        finish()
      }
    }
    source.start()
    button.textContent = 'Stop'
  } catch (error) {
    if (id === generation) { finish(); report(error) }
  }
}

async function synthesize(text) {
  try { return await voice.speak(text) }
  catch (error) {
    if (!/exceeds the voice token limit|Invalid predicted frame count/.test(error.message)) throw error
  }
  const sentences = [...text.matchAll(/(?<=[.!?])\s+(?=\S)/gu)]
  const spaces = sentences.length ? sentences : [...text.matchAll(/\s+(?=\S)/gu)]
  if (!spaces.length) throw new Error('This word exceeds the voice token limit.')
  const middle = text.length / 2
  const cut = spaces.reduce((a,b) => Math.abs(a.index-middle) <= Math.abs(b.index-middle) ? a : b).index
  const a = await synthesize(text.slice(0, cut))
  const b = await synthesize(text.slice(cut).trimStart())
  const samples = new Float32Array(a.samples.length + b.samples.length)
  samples.set(a.samples)
  samples.set(b.samples, a.samples.length)
  return {samples, sampleRate: a.sampleRate}
}

el('demo').onsubmit = event => {
  event.preventDefault()
  const text = field.value.trim()
  listen(speak, async () => {
    // Until the voice is ready, the default sentence plays a clip Pizza rendered ahead of time.
    if (text === initialText && !initialAudio) return context.decodeAudioData(await (await read('./clips/default.mp3')).arrayBuffer())
    await voiceReady
    const {samples, sampleRate} = text === initialText ? initialAudio : await synthesize(text)
    const buffer = context.createBuffer(1, samples.length, sampleRate)
    buffer.copyToChannel(samples, 0)
    return buffer
  })
}

async function prepare() {
  voice = await loadVoice('eng', await loadEspeak('en-us'))
  initialAudio = await synthesize(initialText)
}
speak.textContent = 'Play'
speak.disabled = false

async function chart() {
  const models = await (await read('./speeds.json')).json()
  const max = Math.max(...models.map(model => model.speed))
  for (const {name, id, speed} of models) {
    const label = document.createElement(id === 'pizza' ? 'strong' : 'span')
    label.textContent = name
    const button = document.createElement('button')
    button.textContent = 'Play'
    button.setAttribute('aria-label', name + ' audio sample')
    button.onclick = () => listen(button, async () => {
      const response = await read('./clips/' + id + '.mp3')
      return context.decodeAudioData(await response.arrayBuffer())
    })
    const track = document.createElement('div')
    track.className = 'track'
    const bar = document.createElement('span')
    bar.className = id === 'pizza' ? 'bar pizza' : 'bar'
    bar.style.width = (speed / max * 100) + '%'
    const value = document.createElement('span')
    value.className = 'value'
    value.textContent = speed.toFixed(1) + '×'
    track.append(bar, value)
    el('chart').append(label, button, track)
  }
}

async function languages() {
  for (const {name, available} of await (await read('./languages.json')).json()) {
    const entry = document.createElement('li')
    entry.textContent = name + (available ? ' (available)' : ' (in preparation)')
    el('languages').append(entry)
  }
}

const voiceReady = prepare()
voiceReady.catch(report)
chart().catch(report)
languages().catch(report)
