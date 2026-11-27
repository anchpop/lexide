//! Transcendentals over slices, four lanes at a time, for the activations on long renderer tensors. LLVM won't
//! vectorize the scalar forms on wasm (and SSE2 lacks a vector round), so they're written on `F4`, which is
//! simd128 on wasm and a plain array (autovectorized) elsewhere. Each stays within a few ulp of libm over the
//! ranges the graphs produce; trig hands huge arguments to libm.

use std::f32::consts::{FRAC_2_PI, LOG2_E};

#[cfg(target_arch = "wasm32")]
mod lanes {
    use std::arch::wasm32::*;

    #[derive(Clone, Copy)]
    pub struct F4(v128);

    impl F4 {
        #[inline(always)]
        pub fn splat(x: f32) -> F4 {
            F4(f32x4_splat(x))
        }
        #[inline(always)]
        pub fn load(s: &[f32; 4]) -> F4 {
            // SAFETY: four floats, no alignment needed.
            F4(unsafe { v128_load(s.as_ptr().cast()) })
        }
        #[inline(always)]
        pub fn store(self, s: &mut [f32; 4]) {
            // SAFETY: as above.
            unsafe { v128_store(s.as_mut_ptr().cast(), self.0) }
        }
        #[inline(always)]
        pub fn add(self, o: F4) -> F4 {
            F4(f32x4_add(self.0, o.0))
        }
        #[inline(always)]
        pub fn sub(self, o: F4) -> F4 {
            F4(f32x4_sub(self.0, o.0))
        }
        #[inline(always)]
        pub fn mul(self, o: F4) -> F4 {
            F4(f32x4_mul(self.0, o.0))
        }
        #[inline(always)]
        pub fn div(self, o: F4) -> F4 {
            F4(f32x4_div(self.0, o.0))
        }
        /// Clamp that keeps NaN (pmin/pmax return the first operand when unordered).
        #[inline(always)]
        pub fn clamp(self, lo: f32, hi: f32) -> F4 {
            F4(f32x4_pmin(
                f32x4_pmax(self.0, f32x4_splat(lo)),
                f32x4_splat(hi),
            ))
        }
        /// Integer ops on the bit patterns.
        #[inline(always)]
        pub fn iadd(self, o: F4) -> F4 {
            F4(i32x4_add(self.0, o.0))
        }
        #[inline(always)]
        pub fn isub(self, o: F4) -> F4 {
            F4(i32x4_sub(self.0, o.0))
        }
        #[inline(always)]
        pub fn and(self, o: F4) -> F4 {
            F4(v128_and(self.0, o.0))
        }
        #[inline(always)]
        pub fn xor(self, o: F4) -> F4 {
            F4(v128_xor(self.0, o.0))
        }
        #[inline(always)]
        pub fn shl(self, n: u32) -> F4 {
            F4(i32x4_shl(self.0, n))
        }
        #[inline(always)]
        pub fn sar(self, n: u32) -> F4 {
            F4(i32x4_shr(self.0, n))
        }
        #[inline(always)]
        pub fn int(i: i32) -> F4 {
            F4(i32x4_splat(i))
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod lanes {
    #[derive(Clone, Copy)]
    pub struct F4([f32; 4]);

    fn f2(a: F4, b: F4, g: impl Fn(f32, f32) -> f32) -> F4 {
        F4(std::array::from_fn(|i| g(a.0[i], b.0[i])))
    }
    fn i2(a: F4, b: F4, g: impl Fn(i32, i32) -> i32) -> F4 {
        f2(a, b, |x, y| {
            f32::from_bits(g(x.to_bits() as i32, y.to_bits() as i32) as u32)
        })
    }

    impl F4 {
        #[inline(always)]
        pub fn splat(x: f32) -> F4 {
            F4([x; 4])
        }
        #[inline(always)]
        pub fn load(s: &[f32; 4]) -> F4 {
            F4(*s)
        }
        #[inline(always)]
        pub fn store(self, s: &mut [f32; 4]) {
            *s = self.0
        }
        #[inline(always)]
        pub fn add(self, o: F4) -> F4 {
            f2(self, o, |a, b| a + b)
        }
        #[inline(always)]
        pub fn sub(self, o: F4) -> F4 {
            f2(self, o, |a, b| a - b)
        }
        #[inline(always)]
        pub fn mul(self, o: F4) -> F4 {
            f2(self, o, |a, b| a * b)
        }
        #[inline(always)]
        pub fn div(self, o: F4) -> F4 {
            f2(self, o, |a, b| a / b)
        }
        #[inline(always)]
        pub fn clamp(self, lo: f32, hi: f32) -> F4 {
            F4(self.0.map(|x| {
                let x = if lo > x { lo } else { x };
                if x > hi { hi } else { x }
            }))
        }
        #[inline(always)]
        pub fn iadd(self, o: F4) -> F4 {
            i2(self, o, i32::wrapping_add)
        }
        #[inline(always)]
        pub fn isub(self, o: F4) -> F4 {
            i2(self, o, i32::wrapping_sub)
        }
        #[inline(always)]
        pub fn and(self, o: F4) -> F4 {
            i2(self, o, |a, b| a & b)
        }
        #[inline(always)]
        pub fn xor(self, o: F4) -> F4 {
            i2(self, o, |a, b| a ^ b)
        }
        #[inline(always)]
        pub fn shl(self, n: u32) -> F4 {
            i2(self, self, |a, _| a << n)
        }
        #[inline(always)]
        pub fn sar(self, n: u32) -> F4 {
            i2(self, self, |a, _| a >> n)
        }
        #[inline(always)]
        pub fn int(i: i32) -> F4 {
            F4([f32::from_bits(i as u32); 4])
        }
    }
}

use lanes::F4;

fn s(x: f32) -> F4 {
    F4::splat(x)
}

/// Applies `f` to every element, four at a time (the tail padded with zeros).
#[inline(always)]
fn each(x: &mut [f32], f: impl Fn(F4) -> F4) {
    let mut chunks = x.chunks_exact_mut(4);
    for c in &mut chunks {
        let c: &mut [f32; 4] = c.try_into().unwrap();
        f(F4::load(c)).store(c);
    }
    let tail = chunks.into_remainder();
    if !tail.is_empty() {
        let mut pad = [0f32; 4];
        pad[..tail.len()].copy_from_slice(tail);
        f(F4::load(&pad)).store(&mut pad);
        tail.copy_from_slice(&pad[..tail.len()]);
    }
}

/// Adding 1.5 * 2^23 rounds to the nearest integer (ties to even) and leaves it in the low mantissa bits.
const ROUND: f32 = 12_582_912.0;

/// exp(x), saturating to inf above the f32 range and flushing to the smallest normal below it.
#[inline(always)]
fn exp4(x: F4) -> F4 {
    let x = x.clamp(-87.3, 89.0);
    let t = x.mul(s(LOG2_E)).add(s(ROUND));
    let n = t.sub(s(ROUND));
    // Cody-Waite split of ln 2, then a degree-6 polynomial on |r| <= ln2/2.
    let r = x.sub(n.mul(s(0.693_145_75))).sub(n.mul(s(1.428_606_8e-6)));
    let mut p = s(0.001_394_4);
    for c in [0.008_333_45, 0.041_666_42, 0.166_666_67, 0.5, 1.0, 1.0] {
        p = p.mul(r).add(s(c));
    }
    // 2^k in two factors, so k = 128 still scales a p below 1 to a finite result (and anything larger to inf).
    let k = t.isub(F4::int(ROUND.to_bits() as i32));
    let half = k.sar(1);
    let scale = |k: F4| k.iadd(F4::int(127)).shl(23);
    p.mul(scale(half)).mul(scale(k.isub(half)))
}

/// sin(x + shift * pi/2) for |x| up to `TRIG_LIMIT`.
#[inline(always)]
fn sin4(x: F4, shift: i32) -> F4 {
    let t = x.mul(s(FRAC_2_PI)).add(s(ROUND));
    let j = t.sub(s(ROUND));
    let r = x
        .sub(j.mul(s(1.570_312_5)))
        .sub(j.mul(s(4.837_513e-4)))
        .sub(j.mul(s(7.549_789_4e-8)));
    let q = t.iadd(F4::int(shift));
    let r2 = r.mul(r);
    let sp = s(-1.666_665_5e-1).add(r2.mul(s(8.332_161e-3).add(r2.mul(s(-1.951_529_6e-4)))));
    let sin = r.add(r.mul(r2).mul(sp));
    let cp = s(4.166_664_6e-2).add(r2.mul(s(-1.388_731_6e-3).add(r2.mul(s(2.443_315_7e-5)))));
    let cos = s(1.0).sub(s(0.5).mul(r2)).add(r2.mul(r2).mul(cp));
    // Odd quadrants take the cosine (mask all ones when q is even), quadrants 2 and 3 flip the sign.
    let even = q.and(F4::int(1)).isub(F4::int(1));
    let v = cos.xor(sin.xor(cos).and(even));
    v.xor(q.and(F4::int(2)).shl(30))
}

pub fn exp(x: &mut [f32]) {
    each(x, exp4)
}

pub fn sigmoid(x: &mut [f32]) {
    each(x, |v| s(1.0).div(s(1.0).add(exp4(s(0.0).sub(v)))))
}

/// tanh(|x|) = (1 - e) / (1 + e) with e = exp(-2|x|), sign restored; a series near zero, where that loses
/// relative accuracy.
pub fn tanh(x: &mut [f32]) {
    each(x, |v| {
        let sign = v.and(F4::int(i32::MIN));
        let a = v.xor(sign);
        let e = exp4(s(-2.0).mul(a));
        let t = s(1.0).sub(e).div(s(1.0).add(e)).xor(sign);
        let x2 = v.mul(v);
        let small = v.mul(s(1.0).add(
            x2.mul(s(-0.333_333_34).add(x2.mul(s(0.133_333_33).add(x2.mul(s(-0.053_968_25)))))),
        ));
        // |x| < 0.0625: all ones from the sign bit of x^2 - 0.0625^2.
        let near = x2.sub(s(0.003_906_25)).sar(31);
        t.xor(t.xor(small).and(near))
    })
}

/// Rational approximation on [-4, 4] (beyond, erf rounds to ±1), as in Eigen and XNNPACK.
pub fn erf(x: &mut [f32]) {
    each(x, |v| {
        let v = v.clamp(-4.0, 4.0);
        let x2 = v.mul(v);
        let mut p = s(-2.726_142_3e-10);
        for c in [
            2.770_681_4e-8,
            -2.101_024e-6,
            -5.692_506_4e-5,
            -7.349_906_3e-4,
            -2.954_600_1e-3,
            -1.609_603_3e-2,
        ] {
            p = p.mul(x2).add(s(c));
        }
        let mut q = s(-1.456_607_2e-5);
        for c in [
            -2.133_740_6e-4,
            -1.682_827e-3,
            -7.373_329_2e-3,
            -1.426_474e-2,
        ] {
            q = q.mul(x2).add(s(c));
        }
        v.mul(p).div(q)
    })
}

/// Beyond this the three-part pi/2 reduction loses accuracy; libm handles those (rare) inputs.
const TRIG_LIMIT: f32 = 8192.0;

pub fn sin(x: &mut [f32]) {
    trig(x, 0, libm::sinf)
}

pub fn cos(x: &mut [f32]) {
    trig(x, 1, libm::cosf)
}

fn trig(x: &mut [f32], shift: i32, exact: fn(f32) -> f32) {
    let beyond = |v: f32| v.is_nan() || v.abs() > TRIG_LIMIT;
    let big = x.iter().any(|&v| beyond(v));
    let original = if big { x.to_vec() } else { vec![] };
    each(x, |v| sin4(v, shift));
    for (v, &o) in x.iter_mut().zip(&original) {
        if beyond(o) {
            *v = exact(o);
        }
    }
}

/// Sum with eight independent lanes so it vectorizes.
pub fn sum(x: &[f32]) -> f32 {
    let mut lanes = [0.0; 8];
    let chunks = x.chunks_exact(8);
    let tail: f32 = chunks.remainder().iter().sum();
    for c in chunks {
        for i in 0..8 {
            lanes[i] += c[i];
        }
    }
    lanes.iter().sum::<f32>() + tail
}

#[cfg(test)]
mod tests {
    fn ulps(a: f32, b: f32) -> f32 {
        (a - b).abs() / (b.abs().max(f32::MIN_POSITIVE) * f32::EPSILON)
    }

    fn apply(f: fn(&mut [f32]), xs: &[f32]) -> Vec<f32> {
        let mut v = xs.to_vec();
        f(&mut v);
        v
    }

    #[test]
    fn exp_matches_libm() {
        // Down to -87, where results are still normal floats.
        let xs: Vec<f32> = (-19300..19700).map(|i| i as f32 * 0.0045).collect();
        let worst = apply(super::exp, &xs)
            .iter()
            .zip(&xs)
            .map(|(&y, &x)| ulps(y, libm::expf(x)))
            .fold(0f32, f32::max);
        assert!(worst < 3.0, "{worst} ulp");
        let edge = apply(super::exp, &[-200.0, 100.0, f32::NAN]);
        assert!(edge[0] < 1.3e-38 && edge[1] == f32::INFINITY && edge[2].is_nan());
    }

    #[test]
    fn erf_matches_libm() {
        let xs: Vec<f32> = (-60000..60000).map(|i| i as f32 * 1e-4).collect();
        for (y, x) in apply(super::erf, &xs).iter().zip(&xs) {
            assert!(ulps(*y, libm::erff(*x)) < 4.0 || *x == 0.0, "erf {x}");
        }
    }

    #[test]
    fn trig_matches_libm() {
        let xs: Vec<f32> = (-100000..100000)
            .map(|i| i as f32 * 0.0813)
            .chain([1e5, -3e6, f32::NAN])
            .collect();
        let (s, c) = (apply(super::sin, &xs), apply(super::cos, &xs));
        for (i, &x) in xs.iter().enumerate().take(xs.len() - 1) {
            let tol = 2e-7 * x.abs().max(1.0).log2().max(1.0);
            assert!(
                (s[i] - libm::sinf(x)).abs() < tol && (c[i] - libm::cosf(x)).abs() < tol,
                "{x}"
            );
        }
        assert!(s.last().unwrap().is_nan());
    }

    #[test]
    fn tanh_sigmoid() {
        let xs: Vec<f32> = (-4000..4000)
            .map(|i| i as f32 * 0.005)
            .chain([1e-30, -50.0, 50.0])
            .collect();
        let (t, g) = (apply(super::tanh, &xs), apply(super::sigmoid, &xs));
        for (i, &x) in xs.iter().enumerate() {
            assert!(
                (t[i] - libm::tanhf(x)).abs() < 2e-7
                    && (ulps(t[i], libm::tanhf(x)) < 4.0 || x.abs() > 9.0),
                "tanh {x}"
            );
            assert!(
                (g[i] - 1.0 / (1.0 + libm::expf(-x))).abs() < 2e-7,
                "sigmoid {x}"
            );
        }
    }
}
