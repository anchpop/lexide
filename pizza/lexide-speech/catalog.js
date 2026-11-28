// Voices are served from voices/ next to the package until a public host is chosen.
export const voiceBase = new URL('./voices/', import.meta.url);
export const voices = new Map([
    ['eng/yap', '9982783a8fa66d0aae6ac700e08e06598959b1e78b016bd6e617f129c45e47d6'],
    ['eng/espeak', 'fab088305f746fa433f418b6a02f4d52cacddd16fe6333ba7d90543614364089'],
    ['fra/espeak', '32998b591d4a21bc8068b66bdf43f7fda0f177e1fa502a62890685be5145aedc'],
    ['deu/espeak', '3471c51c8c4564e4e2570327d4b7e0a9f2250b4be3f9d137c511c7b834c63fdd'],
    ['ita/espeak', 'b2ee6cefc0387ea1ca80b07af015f9a6ca9d882e170de9f8995083d5a23bdb87'],
    ['spa-es/espeak', 'b7c9932d727714df818d67e20ecf37b662277c5464a7d4111e1d58fd8065c79f'],
    ['spa/espeak', '2cd5130b0222edefa1e321de443a5b19a398fbf2e5f3fdece507b07b83e51aec'],
    ['por/espeak', '228c5674c615718912a2c8804608d068282f4360bc4c699f43f428e5b3ace954'],
    ['rus/espeak', '5eb0a8fdea089a654e09f7f6ceb437d7aa14459900dc2c26e5d323856bee905a'],
]);
