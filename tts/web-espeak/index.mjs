// SPDX-License-Identifier: GPL-3.0-or-later
import createModule from './espeak.mjs'

const voices = new Set(['en-us', 'it', 'fr-fr', 'de', 'es', 'es-419', 'pt-br', 'ru'])

export async function loadEspeak(voice = 'en-us') {
  if (!voices.has(voice)) throw new RangeError(`Unsupported espeak voice: ${voice}`)
  let fetchError
  const mod = await createModule({
    async loadDictionary(language) {
      const name = language + '_dict'
      try {
        const response = await fetch(new URL(name, import.meta.url))
        if (!response.ok) throw new Error(`Loading ${name}: HTTP ${response.status}`)
        mod.FS.writeFile('/espeak-ng-data/' + name, new Uint8Array(await response.arrayBuffer()))
        return 1
      } catch (error) {
        fetchError = error
        return 0
      }
    },
  })

  const status = await mod.ccall('initialize', 'number', ['string'], [voice], {async: true})
  if (fetchError) throw fetchError
  if (status) throw new Error(`Initializing espeak ${voice}: ${status}`)

  async function translate(text) {
    if (text.includes('\0')) throw new TypeError('espeak text contains NUL')
    fetchError = undefined
    // Text can be larger than the ccall string argument's temporary stack.
    const input = mod.stringToNewUTF8(text)
    let output = 0
    try {
      output = await mod.ccall('phonemize', 'number', ['number'], [input], {async: true})
      if (fetchError) throw fetchError
      if (!output) throw new Error('espeak phonemization failed')
      return mod.UTF8ToString(output)
    } finally {
      mod._free(output)
      mod._free(input)
    }
  }

  // A suspended dictionary lookup still owns espeak's per-instance globals.
  let ready = Promise.resolve()
  return {
    phonemize(text) {
      const result = ready.then(() => translate(text))
      ready = result.catch(() => {})
      return result
    },
  }
}
