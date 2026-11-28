// Voices are served from voices/ next to the package until a public host is chosen.
export const voiceBase = new URL('./voices/', import.meta.url);
export const voices = new Map([
    ['eng/yap', '9982783a8fa66d0aae6ac700e08e06598959b1e78b016bd6e617f129c45e47d6'],
    ['eng/espeak', 'fab088305f746fa433f418b6a02f4d52cacddd16fe6333ba7d90543614364089'],
]);
