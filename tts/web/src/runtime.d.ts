// The engine's wasm-bindgen glue (tts/runtime/src/web.rs), which build.mjs writes to dist/runtime.js.
export default function init(options: {module_or_path: BufferSource}): Promise<unknown>
export class Voice {
  constructor(bundle: Uint8Array)
  readonly speakers: string[]
  readonly sampleRate: number
  /** The espeak voice name, for voices that read espeak's IPA. */
  readonly espeak: string | undefined
  speak(grams: string, options: string): Float32Array
  chunks(text: string): string[]
  speakText(text: string, ipa: string[], options: string): Float32Array
  warmup(): void
}
