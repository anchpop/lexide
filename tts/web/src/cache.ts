import type {Progress} from './index.js'

async function digest(data: ArrayBuffer) {
  const bytes = new Uint8Array(await crypto.subtle.digest('SHA-256', data))
  return Array.from(bytes, x => x.toString(16).padStart(2, '0')).join('')
}

export async function readAsset(dir: FileSystemDirectoryHandle | undefined, asset: 'voice' | 'wasm',
                                hash: string, url: URL, progress: (event: Progress) => void) {
  const key = asset + '-' + hash
  async function load() {
    const started = performance.now()
    if (dir) {
      try {
        const file = await (await dir.getFileHandle(key)).getFile()
        const data = await file.arrayBuffer()
        if (await digest(data) === hash) {
          progress({phase: 'opfs', asset, loaded: file.size, total: file.size, elapsedMs: performance.now() - started})
          return data
        }
      } catch (error) {
        if (!(error instanceof DOMException && error.name === 'NotFoundError')) throw error
      }
    }
    const response = await fetch(url)
    if (!response.ok || !response.body) throw new Error(`Cannot download ${asset}: HTTP ${response.status}`)
    const total = Number(response.headers.get('Content-Length')) || undefined
    progress({phase: 'download', asset, loaded: 0, total})
    const reader = response.body.getReader()
    const chunks: Uint8Array<ArrayBuffer>[] = []
    let loaded = 0, reported = 0
    for (;;) {
      const {done, value} = await reader.read()
      if (done) break
      chunks.push(value)
      loaded += value.length
      if (loaded - reported >= 262144) {
        progress({phase: 'download', asset, loaded, total})
        reported = loaded
      }
    }
    const data = await new Blob(chunks).arrayBuffer()
    if (await digest(data) !== hash) throw new Error(asset + ' checksum mismatch')
    if (dir) {
      const handle = await dir.getFileHandle(key, {create: true})
      const file = await handle.createSyncAccessHandle()
      try {
        if (file.write(data, {at: 0}) !== data.byteLength) throw new Error('Incomplete OPFS write')
        file.truncate(data.byteLength)
        file.flush()
      } catch (error) {
        if (!(error instanceof DOMException && error.name === 'QuotaExceededError')) throw error
      } finally {
        file.close()
      }
    }
    progress({phase: 'download', asset, loaded, total: loaded, elapsedMs: performance.now() - started})
    return data
  }
  return dir ? navigator.locks.request('lexide-speech-' + key, load) : load()
}
