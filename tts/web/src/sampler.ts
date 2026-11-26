export function normalSampler(seed: number) {
  const state = new Uint32Array(624);
  state[0] = seed;
  for (let i = 1; i < 624; ++i) {
    const prev = state[i - 1];
    state[i] = Math.imul(1812433253, prev ^ (prev >>> 30)) + i;
  }
  let at = 624;
  function next() {
    if (at === 624) {
      for (let j = 0; j < 624; ++j) {
        const bits = (state[j] & 0x80000000) | (state[(j + 1) % 624] & 0x7fffffff);
        state[j] = state[(j + 397) % 624] ^ (bits >>> 1) ^ ((bits & 1) ? 0x9908b0df : 0);
      }
      at = 0;
    }
    let y = state[at++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }
  const uniform = () => (next() & 0xffffff) / 16777216;
  const uniformDouble = () => ((next() & 0x1fffff) * 4294967296 + next()) / 9007199254740992;
  return function normal(count: number) {
    const values = new Float32Array(count);
    if (count < 16) {
      for (let i = 0; i < count; i += 2) {
        const theta = 2 * Math.PI * uniformDouble();
        const radius = Math.sqrt(-2 * Math.log1p(-uniformDouble()));
        values[i] = radius * Math.cos(theta);
        if (i + 1 < count) values[i + 1] = radius * Math.sin(theta);
      }
      return values;
    }
    function transform(i: number) {
      for (let j = i; j < i + 8; ++j) {
        const r = Math.fround(Math.sqrt(-2 * Math.log(1 - values[j])));
        const theta = Math.fround(Math.fround(2 * Math.PI) * values[j + 8]);
        values[j] = r * Math.cos(theta);
        values[j + 8] = r * Math.sin(theta);
      }
    }
    for (let i = 0; i < count; ++i) values[i] = uniform();
    for (let i = 0; i <= count - 16; i += 16) transform(i);
    if (count % 16) {
      for (let i = count - 16; i < count; ++i) values[i] = uniform();
      transform(count - 16);
    }
    return values;
  };
}
