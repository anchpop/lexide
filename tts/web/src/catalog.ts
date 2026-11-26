// Where the voices are served: the page's own site until a public host is chosen.
export const voiceBase = new URL('/voices/', location.href)
export const voices = new Map([
  ['eng', '0e7344547ef8da7491ce49b50375fb931a3303667210ecd6a5e42b016866c936'],
])
