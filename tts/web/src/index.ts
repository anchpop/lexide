import type {Gram} from './tokenizer.js'
export type {Gram}

export interface Progress {
  phase: 'download' | 'opfs' | 'sessions' | 'warmup'
  asset?: 'voice' | 'wasm'
  loaded?: number
  elapsedMs?: number
  total?: number
}
export interface LoadOptions {
  threads?: number
  onProgress?: (event: Progress) => void
}
export interface SpeakOptions {
  speaker?: string
  take?: number
  speed?: number
}
export interface Audio {
  samples: Float32Array
  sampleRate: number
}
export interface Voice {
  speakers: readonly string[]
  speak(grams: readonly Gram[], options?: SpeakOptions): Promise<Audio>
  release(): void
}

export async function loadVoice(language: string, {threads, onProgress}: LoadOptions = {}): Promise<Voice> {
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
  function request<T>(message: object): Promise<T> {
    if (closed) return Promise.reject(new Error('Voice released'))
    const id = ++next
    return new Promise((resolve, reject) => {
      worker.postMessage({...message, id})
      pending.set(id, {resolve, reject})
    })
  }
  try {
    const speakers = await request<string[]>({language, threads})
    return {speakers, speak: (grams, options) => request<Audio>({grams, options}),
            release: () => close(new Error('Voice released'))}
  } catch (error) {
    close(new Error('Voice load failed'))
    throw error
  }
}
