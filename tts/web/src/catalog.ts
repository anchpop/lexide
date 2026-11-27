// Voices are served from voices/ next to the package until a public host is chosen.
export const voiceBase = new URL('./voices/', import.meta.url)
export const voices = new Map([
  ['eng/yap', '0d1d58a8236e0a436059d797d29deab0a442452026d56fad486cd516bc9cbd08'],
  ['eng/espeak', 'a4089dd8d3321da8902a64f57b4d8d7d90bb3461f96af4e54ce5480d2de0ceb6'],
])
