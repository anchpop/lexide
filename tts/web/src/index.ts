/** One word or phrase as yap annotates it: words, one POS tag each, corpus count, and space-separated phones. */
export interface Gram {
  words: readonly string[]
  pos: readonly string[]
  count: number
  phonemes: string
}
/** Runs espeak-ng (the voice's `-v` voice, `-q --ipa --sep=_`) on one stretch of text and returns stdout. */
export interface Espeak {
  phonemize(text: string): string | Promise<string>
}

export interface Progress {
  phase: 'download' | 'opfs' | 'sessions' | 'warmup'
  asset?: 'voice' | 'wasm'
  loaded?: number
  elapsedMs?: number
  total?: number
}
export interface LoadOptions {
  onProgress?: (event: Progress) => void
}
export interface SpeakOptions {
  speaker?: string
  /** Seed of the delivery; unset draws a new one each call. */
  take?: number
  /** Seed of the source noise and pitch jitter (0). */
  seed?: number
  /** Spread of the delivery around the typical reading (1; 0 is the typical reading). */
  variety?: number
  /** 0.25 to 4 (1). */
  speed?: number
  /** Pitch offset in the model's log-frequency units (0). */
  pitchShift?: number
  /** 0 to 3 (1). */
  pitchRange?: number
  /** Random pitch wander (0, off). */
  pitchJitter?: number
  /** Extra pause in seconds after commas (0.15) and sentence stops (0.3). */
  commaPause?: number
  stopPause?: number
  voicedVowels?: boolean
  /** Pull a rising (question) ending back toward the typical reading's (true). */
  fixEnding?: boolean
}
export interface Audio {
  samples: Float32Array
  sampleRate: number
}
export interface Voice<Input = readonly Gram[]> {
  speakers: readonly string[]
  speak(input: Input, options?: SpeakOptions): Promise<Audio>
  release(): void
}

export function loadVoice(language: string, source: Espeak, options?: LoadOptions): Promise<Voice<string>>
export function loadVoice(language: string, source: 'yap', options?: LoadOptions): Promise<Voice>
export function loadVoice(language: string, options?: LoadOptions): Promise<Voice>
export async function loadVoice(language: string, source: 'yap' | Espeak | LoadOptions = 'yap',
                                options: LoadOptions = {}): Promise<Voice | Voice<string>> {
  let espeak: Espeak | undefined
  if (typeof source === 'object') {
    if ('phonemize' in source) espeak = source
    else options = source // Original loadVoice(language, options) loads yap.
  } else if (source !== 'yap') throw new TypeError('Expected yap or an espeak phonemizer')
  const {onProgress} = options
  void navigator.storage?.persist?.().catch(() => {})
  const worker = new Worker(new URL('./worker.js', import.meta.url), {type: 'module'})
  const pending = new Map<number, {resolve: (value: any) => void, reject: (error: Error) => void}>()
  let next = 0, closed = false
  function close(error: Error) {
    closed = true
    worker.terminate()
    for (const call of pending.values()) call.reject(error)
    pending.clear()
  }
  worker.onerror = event => close(new Error(event.message || 'Voice worker failed'))
  worker.onmessageerror = () => close(new Error('Unreadable worker reply'))
  worker.onmessage = ({data}) => {
    if (data.progress) { onProgress?.(data.progress); return }
    const call = pending.get(data.id)
    pending.delete(data.id)
    if (data.error) call?.reject(new Error(data.error))
    else call?.resolve(data.value)
  }
  function post<T>(message: object): Promise<T> {
    if (closed) return Promise.reject(new Error('Voice released'))
    const id = ++next
    return new Promise((resolve, reject) => {
      pending.set(id, {resolve, reject})
      worker.postMessage({...message, id})
    })
  }
  let sending = Promise.resolve()
  function request<T>(message: object | (() => Promise<object>)): Promise<T> {
    // A caller's async phonemizer can finish out of order. Post in speak order.
    const result = sending.then(async () => post<T>(typeof message === 'function' ? await message() : message))
    sending = result.then(() => {}, () => {})
    return result
  }
  try {
    const info = await request<{speakers: string[], espeak?: string}>({language: language + (espeak ? '/espeak' : '/yap')})
    const release = () => close(new Error('Voice released'))
    const speak = (grams: readonly Gram[], options?: SpeakOptions) => request<Audio>({grams, options})
    if (espeak) {
      if (!info.espeak) throw new Error('Missing espeak configuration')
      const speakText = (text: string, options?: SpeakOptions) => request<Audio>(async () => {
        // The engine splits off punctuation; the caller's espeak reads each stretch between.
        const ipa = []
        for (const chunk of await post<string[]>({split: text})) ipa.push(await espeak.phonemize(chunk))
        return {text, ipa, options}
      })
      return {speakers: info.speakers, release, speak: speakText}
    }
    return {speakers: info.speakers, release, speak}
  } catch (error) {
    close(new Error('Voice load failed'))
    throw error
  }
}
