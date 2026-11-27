import init, { Voice as Engine } from './runtime.js';
import { wasmHash } from './assets.mjs';
import { readAsset } from './cache.js';
import { voiceBase, voices } from './catalog.js';
const progress = (progress) => self.postMessage({ progress });
async function load(language) {
    const hash = voices.get(language);
    if (!hash)
        throw new Error('No voice for ' + language);
    const root = await navigator.storage?.getDirectory?.().catch(() => undefined);
    const dir = await root?.getDirectoryHandle('lexide-speech', { create: true });
    const [bundle, wasm] = await Promise.all([
        readAsset(dir, 'voice', hash, new URL(hash + '.voca', voiceBase), progress),
        readAsset(dir, 'wasm', wasmHash, new URL('./runtime_bg.wasm', import.meta.url), progress),
    ]);
    await init({ module_or_path: wasm });
    progress({ phase: 'sessions' });
    const engine = new Engine(new Uint8Array(bundle));
    progress({ phase: 'warmup' });
    engine.warmup();
    if (dir) {
        const keep = new Set([...voices.values()].map(hash => 'voice-' + hash));
        keep.add('wasm-' + wasmHash);
        for await (const name of dir.keys()) {
            if (!keep.has(name)) {
                await navigator.locks.request('lexide-speech-' + name, async () => {
                    await dir.removeEntry(name).catch((e) => { if (e.name !== 'NotFoundError')
                        throw e; });
                });
            }
        }
    }
    return engine;
}
// The engine takes seeds as u32; an unset take draws a fresh delivery.
function settings({ take = crypto.getRandomValues(new Uint32Array(1))[0], seed = 0, ...rest } = {}) {
    if (!Number.isSafeInteger(take) || !Number.isSafeInteger(seed))
        throw new RangeError('Invalid take or seed');
    return JSON.stringify({ ...rest, take: take >>> 0, seed: seed >>> 0 });
}
let engine;
let queue = Promise.resolve();
self.onmessage = ({ data }) => {
    queue = queue.then(async () => {
        try {
            if ('language' in data) {
                engine = await load(data.language);
                self.postMessage({ id: data.id, value: { speakers: engine.speakers, espeak: engine.espeak } });
            }
            else if ('split' in data) {
                self.postMessage({ id: data.id, value: engine.chunks(data.split) });
            }
            else {
                const samples = 'grams' in data ? engine.speak(JSON.stringify(data.grams), settings(data.options))
                    : engine.speakText(data.text, data.ipa, settings(data.options));
                self.postMessage({ id: data.id, value: { samples, sampleRate: engine.sampleRate } }, { transfer: [samples.buffer] });
            }
        }
        catch (e) {
            self.postMessage({ id: data.id, error: e instanceof Error ? e.message : String(e) });
        }
    });
};
