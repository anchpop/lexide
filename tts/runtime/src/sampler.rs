//! PyTorch CPU's MT19937 normal sampler, bit-compatible with `torch.randn` (and the web engine's copy), so a take
//! reproduces the same delivery everywhere.

struct Mt {
    state: [u32; 624],
    at: usize,
}

impl Mt {
    fn new(seed: u32) -> Mt {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            let prev = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        Mt { state, at: 624 }
    }

    fn next(&mut self) -> u32 {
        if self.at == 624 {
            for j in 0..624 {
                let bits =
                    (self.state[j] & 0x8000_0000) | (self.state[(j + 1) % 624] & 0x7fff_ffff);
                self.state[j] = self.state[(j + 397) % 624]
                    ^ (bits >> 1)
                    ^ if bits & 1 != 0 { 0x9908_b0df } else { 0 };
            }
            self.at = 0;
        }
        let mut y = self.state[self.at];
        self.at += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    fn uniform(&mut self) -> f32 {
        (self.next() & 0xff_ffff) as f32 / 16_777_216.0
    }

    fn uniform_double(&mut self) -> f64 {
        let high = (self.next() & 0x1f_ffff) as f64;
        (high * 4_294_967_296.0 + self.next() as f64) / 9_007_199_254_740_992.0
    }
}

/// A stream of standard normal draws; successive `draw`s continue the stream, as successive `torch.randn`
/// calls do.
pub struct Normal(Mt);

/// `count` standard normal draws from `seed`.
pub fn normal(seed: u32, count: usize) -> Vec<f32> {
    Normal::new(seed).draw(count)
}

/// Per-frame pitch offsets in octaves-of-ln2 units: the constant shift plus, with jitter, fast noise and a slow
/// 17-frame drift normalized to unit spread (synthesize.mjs `pitchOffsets`). Without jitter, one value broadcasts.
pub fn pitch_offsets(frames: usize, shift: f32, jitter: f32, seed: u32) -> Vec<f32> {
    if jitter.is_nan() || jitter <= 0.0 {
        return vec![shift];
    }
    let mut normal = Normal::new(seed.wrapping_add(7919));
    let (fast, drift) = (normal.draw(frames), normal.draw(frames + 16));
    let slow: Vec<f32> = (0..frames)
        .map(|i| (drift[i..i + 17].iter().fold(0f32, |s, &d| s + d) as f64 / 17.0) as f32)
        .collect();
    let mean = slow.iter().map(|&s| s as f64).sum::<f64>() / frames as f64;
    let variance =
        slow.iter().map(|&s| (s as f64 - mean).powi(2)).sum::<f64>() / (frames as f64 - 1.0);
    let scale = variance.sqrt().max(1e-6);
    let unit = jitter as f64 * std::f64::consts::LN_2 / (12.0 * 5f64.sqrt());
    fast.iter()
        .zip(&slow)
        .map(|(&v, &s)| (shift as f64 + (2.0 * s as f64 / scale + v as f64) * unit) as f32)
        .collect()
}

impl Normal {
    pub fn new(seed: u32) -> Normal {
        Normal(Mt::new(seed))
    }

    pub fn draw(&mut self, count: usize) -> Vec<f32> {
        let mt = &mut self.0;
        let mut v = vec![0f32; count];
        if count < 16 {
            for i in (0..count).step_by(2) {
                let theta = 2.0 * std::f64::consts::PI * mt.uniform_double();
                let radius = (-2.0 * (-mt.uniform_double()).ln_1p()).sqrt();
                v[i] = (radius * theta.cos()) as f32;
                if i + 1 < count {
                    v[i + 1] = (radius * theta.sin()) as f32;
                }
            }
            return v;
        }
        // Box-Muller over blocks of 16 uniforms: the first eight give radii, the last eight angles.
        let transform = |v: &mut [f32]| {
            for j in 0..8 {
                let r = (-2.0 * (1.0 - v[j] as f64).ln()).sqrt() as f32 as f64;
                let theta = (std::f32::consts::TAU * v[j + 8]) as f64;
                v[j] = (r * theta.cos()) as f32;
                v[j + 8] = (r * theta.sin()) as f32;
            }
        };
        v.iter_mut().for_each(|x| *x = mt.uniform());
        for block in v.chunks_exact_mut(16) {
            transform(block);
        }
        if !count.is_multiple_of(16) {
            let tail = &mut v[count - 16..];
            tail.iter_mut().for_each(|x| *x = mt.uniform());
            transform(tail);
        }
        v
    }
}

#[cfg(test)]
mod tests {
    /// Reference values from the web engine's sampler (which matches torch.randn).
    #[test]
    #[allow(clippy::excessive_precision)]
    fn matches_web_sampler() {
        assert_eq!(
            super::normal(0, 4),
            [
                1.5409960746765137,
                -0.293428897857666,
                -2.1787893772125244,
                0.5684312582015991
            ]
        );
        let big = super::normal(0, 17);
        assert_eq!(
            [big[0], big[7], big[16]],
            [-1.1258398294448853, 1.85300612449646, 0.9318265914916992]
        );
        let long = super::normal(12345, 1000);
        assert_eq!(
            [long[0], long[500], long[999]],
            [-1.479807734489441, -0.082284115254879, -0.3743748962879181]
        );
    }
}
