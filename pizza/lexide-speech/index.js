export async function loadVoice(language, source = 'yap', options = {}) {
    let espeak;
    if (typeof source === 'object') {
        if ('phonemize' in source)
            espeak = source;
        else
            options = source; // Original loadVoice(language, options) loads yap.
    }
    else if (source !== 'yap')
        throw new TypeError('Expected yap or an espeak phonemizer');
    const { onProgress } = options;
    void navigator.storage?.persist?.().catch(() => { });
    const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    const pending = new Map();
    let next = 0, closed = false;
    function close(error) {
        closed = true;
        worker.terminate();
        for (const call of pending.values())
            call.reject(error);
        pending.clear();
    }
    worker.onerror = event => close(new Error(event.message || 'Voice worker failed'));
    worker.onmessageerror = () => close(new Error('Unreadable worker reply'));
    worker.onmessage = ({ data }) => {
        if (data.progress) {
            onProgress?.(data.progress);
            return;
        }
        const call = pending.get(data.id);
        pending.delete(data.id);
        if (data.error)
            call?.reject(new Error(data.error));
        else
            call?.resolve(data.value);
    };
    function post(message) {
        if (closed)
            return Promise.reject(new Error('Voice released'));
        const id = ++next;
        return new Promise((resolve, reject) => {
            pending.set(id, { resolve, reject });
            worker.postMessage({ ...message, id });
        });
    }
    let sending = Promise.resolve();
    function request(message) {
        // A caller's async phonemizer can finish out of order. Post in speak order.
        const result = sending.then(async () => post(typeof message === 'function' ? await message() : message));
        sending = result.then(() => { }, () => { });
        return result;
    }
    try {
        const info = await request({ language: language + (espeak ? '/espeak' : '/yap') });
        const release = () => close(new Error('Voice released'));
        const speak = (grams, options) => request({ grams, options });
        if (espeak) {
            if (!info.espeak)
                throw new Error('Missing espeak configuration');
            const speakText = (text, options) => request(async () => {
                // The engine splits off punctuation; the caller's espeak reads each stretch between.
                const ipa = [];
                for (const chunk of await post({ split: text }))
                    ipa.push(await espeak.phonemize(chunk));
                return { text, ipa, options };
            });
            return { speakers: info.speakers, release, speak: speakText };
        }
        return { speakers: info.speakers, release, speak };
    }
    catch (error) {
        close(new Error('Voice load failed'));
        throw error;
    }
}
