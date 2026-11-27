// Builds the Rust engine (tts/runtime) into dist/ with wasm-pack; run tsc after. Needs cargo and wasm-pack.
import {execFileSync} from 'node:child_process'
import {readFile, rm, writeFile} from 'node:fs/promises'
import {createHash} from 'node:crypto'

execFileSync('wasm-pack', ['build', '../runtime', '--release', '--target', 'web', '--no-pack', '--no-typescript',
  '--out-dir', '../web/dist', '--out-name', 'runtime', '--', '--features', 'web'], {stdio: 'inherit'})
await rm('dist/.gitignore')
const hash = createHash('sha256').update(await readFile('dist/runtime_bg.wasm')).digest('hex')
await writeFile('dist/assets.mjs', `export const wasmHash = '${hash}'\n`)
