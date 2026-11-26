import {copyFile, readFile, writeFile} from 'node:fs/promises'
import {createHash} from 'node:crypto'

const ort = 'node_modules/onnxruntime-web/'
const wasm = 'ort-wasm-simd-threaded.wasm'
for (const file of ['ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs', wasm]) {
  await copyFile(ort + 'dist/' + file, 'dist/' + file)
}
const hash = createHash('sha256').update(await readFile('dist/' + wasm)).digest('hex')
await writeFile('dist/assets.mjs', `export const wasmHash = '${hash}'\n`)

