//! C += A·B with A packed in MR-row panels and B in NR-column panels, the layout the microkernel streams.
//! Plain Rust on fixed-size arrays: LLVM turns the microkernel into simd128 (wasm), NEON or SSE.

pub const MR: usize = 4;
pub const NR: usize = 8;
/// Blocks of B are KC×NC (256 KiB, in L2) while each A panel's KC slice (4 KiB) sits in L1.
const KC: usize = 256;
const NC: usize = 256;

/// A as `[ceil(m/MR)][k][MR]`, zero-padded.
pub fn pack_a(m: usize, k: usize, at: impl Fn(usize, usize) -> f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(m.div_ceil(MR) * k * MR);
    for p in (0..m).step_by(MR) {
        for c in 0..k {
            out.extend((p..p + MR).map(|r| if r < m { at(r, c) } else { 0.0 }));
        }
    }
    out
}

/// A whole k×n B as `[ceil(n/NR)][k][NR]`, zero-padded (for constant weights).
pub fn pack_b(k: usize, n: usize, at: impl Fn(usize, usize) -> f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(n.div_ceil(NR) * k * NR);
    for q in (0..n).step_by(NR) {
        for r in 0..k {
            out.extend((q..q + NR).map(|c| if c < n { at(r, c) } else { 0.0 }));
        }
    }
    out
}

/// Row r of B as a shifted slice: `B[r][j] = src[j + offset]`, zero where that falls outside `src`. Convolutions
/// (stride 1, any dilation and padding) and plain row-major matrices all read this way.
pub type Row<'a> = &'a dyn Fn(usize) -> (&'a [f32], isize);

/// Rows `k0..k0+k`, columns `j0..j0+n` of B in panel layout, writing every element (zeros included).
fn pack_rows(k0: usize, k: usize, j0: usize, n: usize, out: &mut Vec<f32>, row: Row) {
    out.resize(n.div_ceil(NR) * k * NR, 0.0);
    for r in 0..k {
        let (src, offset) = row(k0 + r);
        for q in 0..n.div_ceil(NR) {
            let dst = &mut out[(q * k + r) * NR..][..NR];
            let (start, cols) = (offset + (j0 + q * NR) as isize, NR.min(n - q * NR));
            if start >= 0 && start as usize + NR <= src.len() && cols == NR {
                dst.copy_from_slice(&src[start as usize..][..NR]);
            } else {
                for (c, d) in dst.iter_mut().enumerate() {
                    let i = start + c as isize;
                    *d = if c < cols && (0..src.len() as isize).contains(&i) {
                        src[i as usize]
                    } else {
                        0.0
                    };
                }
            }
        }
    }
}

pub enum B<'a> {
    /// The whole matrix from `pack_b` (constant weights).
    Packed(&'a [f32]),
    /// Packed block by block as needed (activations).
    Rows(Row<'a>),
}

/// `c[r * ldc + j] += sum_k A[r][k] * B[k][j]` for an m×n block of C.
#[inline(never)]
pub fn gemm(m: usize, n: usize, k: usize, a: &[f32], b: B, c: &mut [f32], ldc: usize) {
    let mut buf = vec![];
    for j0 in (0..n).step_by(NC) {
        let cols = NC.min(n - j0);
        for k0 in (0..k).step_by(KC) {
            let kc = KC.min(k - k0);
            // Where panel q's rows k0..k0+kc start, and the distance between panels.
            let (block, start, stride) = match b {
                B::Packed(all) => (&all[j0 * k..], k0 * NR, k * NR),
                B::Rows(row) => {
                    pack_rows(k0, kc, j0, cols, &mut buf, row);
                    (&buf[..], 0, kc * NR)
                }
            };
            for p in 0..m.div_ceil(MR) {
                let a = &a[(p * k + k0) * MR..][..kc * MR];
                let rows = MR.min(m - p * MR);
                for q in 0..cols.div_ceil(NR) {
                    let mut acc = [[0f32; NR]; MR];
                    micro(a, &block[q * stride + start..][..kc * NR], &mut acc);
                    let j = j0 + q * NR;
                    for (r, acc) in acc.iter().enumerate().take(rows) {
                        let row = &mut c[(p * MR + r) * ldc + j..];
                        for (x, &y) in row.iter_mut().zip(&acc[..NR.min(n - j)]) {
                            *x += y;
                        }
                    }
                }
            }
        }
    }
}

/// LLVM leaves the generic kernel scalar on wasm (its SLP cost model rejects the splats), so spell it out.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn micro(a: &[f32], b: &[f32], acc: &mut [[f32; NR]; MR]) {
    use std::arch::wasm32::*;
    // SAFETY: a valid f32 reference.
    let v128_load32_splat_ref = |x: &f32| unsafe { v128_load32_splat((x as *const f32).cast()) };
    let mut c = [[f32x4_splat(0.0); 2]; MR];
    for (a, b) in a.chunks_exact(MR).zip(b.chunks_exact(NR)) {
        // SAFETY: chunks_exact guarantees NR = 8 floats; v128_load has no alignment requirement.
        let (b0, b1) = unsafe {
            (
                v128_load(b.as_ptr().cast()),
                v128_load(b.as_ptr().add(4).cast()),
            )
        };
        for (a, c) in a.iter().zip(&mut c) {
            // A load-splat, so engines can emit a single broadcast load.
            let x = v128_load32_splat_ref(a);
            c[0] = f32x4_add(c[0], f32x4_mul(x, b0));
            c[1] = f32x4_add(c[1], f32x4_mul(x, b1));
        }
    }
    for (acc, c) in acc.iter_mut().zip(c) {
        // SAFETY: each row of acc is NR = 8 floats.
        unsafe {
            v128_store(acc.as_mut_ptr().cast(), c[0]);
            v128_store(acc.as_mut_ptr().add(4).cast(), c[1]);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
fn micro(a: &[f32], b: &[f32], acc: &mut [[f32; NR]; MR]) {
    for (a, b) in a.chunks_exact(MR).zip(b.chunks_exact(NR)) {
        let a: &[f32; MR] = a.try_into().unwrap();
        let b: &[f32; NR] = b.try_into().unwrap();
        for r in 0..MR {
            for c in 0..NR {
                // Fused where the target has it in hardware (NEON on iOS); a libm call otherwise.
                #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
                {
                    acc[r][c] = a[r].mul_add(b[c], acc[r][c]);
                }
                #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
                {
                    acc[r][c] += a[r] * b[c];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_naive() {
        for (m, n, k) in [
            (1, 1, 1),
            (5, 17, 3),
            (9, 300, 70),
            (4, 8, 5000),
            (13, 600, 300),
        ] {
            let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.5).collect();
            let packed = pack_a(m, k, |r, c| a[r * k + c]);
            let row = |r: usize| (&b[r * n..][..n], 0);
            let whole = pack_b(k, n, |r, c| b[r * n + c]);
            for source in [B::Rows(&row), B::Packed(&whole)] {
                let mut c = vec![1.0; m * n];
                gemm(m, n, k, &packed, source, &mut c, n);
                for r in 0..m {
                    for j in 0..n {
                        let want: f32 =
                            1.0 + (0..k).map(|i| a[r * k + i] * b[i * n + j]).sum::<f32>();
                        assert!(
                            (c[r * n + j] - want).abs() <= 1e-3 * want.abs().max(1.0),
                            "{m}x{n}x{k} at {r},{j}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn speed() {
        for (m, n, k) in [
            (448, 521, 1542),
            (36, 8337, 396),
            (64, 8337, 66),
            (521, 576, 192),
        ] {
            let a = pack_a(m, k, |r, c| (r + c) as f32 * 1e-3);
            let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 1e-6).collect();
            let row = |r: usize| (&b[r * n..][..n], 0);
            let mut c = vec![0f32; m * n];
            let t = std::time::Instant::now();
            for _ in 0..5 {
                gemm(m, n, k, &a, B::Rows(&row), &mut c, n);
            }
            let s = t.elapsed().as_secs_f64() / 5.0;
            eprintln!(
                "{m}x{n}x{k}: {:.2} ms, {:.1} GFLOP/s",
                s * 1e3,
                2.0 * (m * n * k) as f64 / s / 1e9
            );
        }
    }
}
