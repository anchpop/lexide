#![allow(clippy::too_many_arguments)]
//! The compute-heavy ops. Weights arrive constant (folded out of DequantizeLinear/Cast at load) and are packed
//! once for the GEMM microkernel.

use super::{Args, Kernel, one};
use crate::gemm::{B, gemm, pack_a, pack_b};
use crate::math;
use crate::proto::Node;
use crate::tensor::{Data, Elem, Tensor, axis};

type Prepared = Result<Option<(Kernel, bool)>, String>;

fn constant<'a>(c: &[Option<&'a Tensor>], i: usize, what: &str) -> Result<&'a Tensor, String> {
    c.get(i)
        .copied()
        .flatten()
        .ok_or_else(|| format!("{what} must be constant"))
}

fn one_dim(node: &Node, name: &str, default: i64) -> Result<usize, String> {
    match node.ints(name) {
        None => Ok(default as usize),
        Some([x]) => Ok(*x as usize),
        Some(v) => Err(format!("{name} {v:?}: only 1-D is supported")),
    }
}

fn pads(node: &Node) -> Result<(usize, usize), String> {
    match node.ints("pads") {
        None => Ok((0, 0)),
        Some(&[a, b]) => Ok((a as usize, b as usize)),
        Some(v) => Err(format!("pads {v:?}: only 1-D is supported")),
    }
}

/// Winograd F(4, 3): four outputs of a 3-tap filter from six inputs with six multiplies instead of twelve.
/// y = AT [(G g) * (BT d)] (Lavin & Gray), batched over channels as six [cout x cin] x [cin x tiles] GEMMs.
const WINOGRAD_BT: [[f32; 6]; 6] = [
    [4., 0., -5., 0., 1., 0.],
    [0., -4., -4., 1., 1., 0.],
    [0., 4., -4., -1., 1., 0.],
    [0., -2., -1., 2., 1., 0.],
    [0., 2., -1., -2., 1., 0.],
    [0., 4., 0., -5., 0., 1.],
];
const WINOGRAD_G: [[f32; 3]; 6] = [
    [1. / 4., 0., 0.],
    [-1. / 6., -1. / 6., -1. / 6.],
    [-1. / 6., 1. / 6., -1. / 6.],
    [1. / 24., 1. / 12., 1. / 6.],
    [1. / 24., -1. / 12., 1. / 6.],
    [0., 0., 1.],
];
const WINOGRAD_AT: [[f32; 6]; 4] = [
    [1., 1., 1., 1., 1., 0.],
    [0., 1., -1., 2., -2., 0.],
    [0., 1., 1., 4., 4., 0.],
    [0., 1., -1., 8., -8., 1.],
];

enum Weights {
    /// Raw `[c, k]` taps.
    Depthwise(Vec<f32>),
    /// Per group, `pack_a` of the `[cout/groups, cin_g * k]` weights.
    Im2col(Vec<Vec<f32>>),
    /// Per transform point, `pack_a` of the `[cout, cin]` transformed weights.
    Winograd(Vec<Vec<f32>>),
}

struct Conv {
    groups: usize,
    cout: usize,
    cin_g: usize,
    k: usize,
    stride: usize,
    dilation: usize,
    pads: (usize, usize),
    weights: Weights,
    bias: Option<Vec<f32>>,
}

impl Conv {
    fn run(&self, x: &Tensor) -> Tensor {
        let &[n, cin, t] = &x.shape[..] else {
            panic!("Conv input {:?}", x.shape)
        };
        assert_eq!(cin, self.cin_g * self.groups, "Conv channels");
        let (k, s, d, (p0, p1)) = (self.k, self.stride, self.dilation, self.pads);
        let to = (t + p0 + p1 - d * (k - 1) - 1) / s + 1;
        let mut out = Vec::with_capacity(n * self.cout * to);
        for _ in 0..n {
            for c in 0..self.cout {
                out.extend(std::iter::repeat_n(
                    self.bias.as_ref().map_or(0.0, |b| b[c]),
                    to,
                ));
            }
        }
        let cout_g = self.cout / self.groups;
        for b in 0..n {
            let xb = &x.f32()[b * cin * t..][..cin * t];
            let yb = &mut out[b * self.cout * to..][..self.cout * to];
            match &self.weights {
                Weights::Depthwise(w) => depthwise(xb, t, w, k, s, d, p0, yb, to),
                Weights::Winograd(u) => winograd(xb, cin, t, u, self.cout, p0, yb, to),
                Weights::Im2col(w) => {
                    for g in 0..self.groups {
                        let xg = &xb[g * self.cin_g * t..][..self.cin_g * t];
                        let c = &mut yb[g * cout_g * to..];
                        // Row (i, kk) of the im2col matrix is input row i shifted by kk * d - p0; strided convs copy
                        // it out.
                        let shift = |r: usize| ((r % k) * d) as isize - p0 as isize;
                        let cols: Vec<f32> = if s == 1 {
                            vec![]
                        } else {
                            (0..self.cin_g * k)
                                .flat_map(|r| (0..to).map(move |j| (r, j)))
                                .map(|(r, j)| {
                                    let p = (j * s) as isize + shift(r);
                                    if (0..t as isize).contains(&p) {
                                        xg[(r / k) * t + p as usize]
                                    } else {
                                        0.0
                                    }
                                })
                                .collect()
                        };
                        let row = |r: usize| {
                            if s == 1 {
                                (&xg[(r / k) * t..][..t], shift(r))
                            } else {
                                (&cols[r * to..][..to], 0)
                            }
                        };
                        gemm(cout_g, to, self.cin_g * k, &w[g], B::Rows(&row), c, to);
                    }
                }
            }
        }
        Tensor::new(vec![n, self.cout, to], out)
    }
}

/// One tap at a time, a shifted axpy along time.
fn depthwise(
    x: &[f32],
    t: usize,
    w: &[f32],
    k: usize,
    s: usize,
    d: usize,
    p0: usize,
    y: &mut [f32],
    to: usize,
) {
    for (c, (src, dst)) in x.chunks_exact(t).zip(y.chunks_exact_mut(to)).enumerate() {
        for kk in 0..k {
            let wk = w[c * k + kk];
            let shift = (kk * d) as isize - p0 as isize;
            // Outputs j whose tap j * s + shift lands inside the input.
            let lo = ((-shift).max(0) as usize).div_ceil(s).min(to);
            let hi = ((t as isize - shift).max(0) as usize)
                .div_ceil(s)
                .clamp(lo, to);
            if s == 1 {
                let src = &src[(lo as isize + shift) as usize..];
                dst[lo..hi]
                    .iter_mut()
                    .zip(src)
                    .for_each(|(y, &x)| *y += wk * x);
            } else {
                for j in lo..hi {
                    dst[j] += wk * src[(j as isize * s as isize + shift) as usize];
                }
            }
        }
    }
}

fn winograd(
    x: &[f32],
    cin: usize,
    t: usize,
    u: &[Vec<f32>],
    cout: usize,
    p0: usize,
    y: &mut [f32],
    to: usize,
) {
    let tiles = to.div_ceil(4);
    // v[xi][i][tile] = (BT d)[xi] for the six (zero-padded) inputs under each tile.
    let mut v = vec![0f32; 6 * cin * tiles];
    for (i, src) in x.chunks_exact(t).enumerate() {
        for tile in 0..tiles {
            let at = |e: usize| {
                let p = (tile * 4 + e) as isize - p0 as isize;
                if (0..t as isize).contains(&p) {
                    src[p as usize]
                } else {
                    0.0
                }
            };
            let dv: [f32; 6] = std::array::from_fn(at);
            for (xi, bt) in WINOGRAD_BT.iter().enumerate() {
                v[(xi * cin + i) * tiles + tile] = bt.iter().zip(&dv).map(|(b, d)| b * d).sum();
            }
        }
    }
    let mut m = vec![0f32; 6 * cout * tiles];
    for xi in 0..6 {
        let vx = &v[xi * cin * tiles..][..cin * tiles];
        let row = |i: usize| (&vx[i * tiles..][..tiles], 0);
        gemm(
            cout,
            tiles,
            cin,
            &u[xi],
            B::Rows(&row),
            &mut m[xi * cout * tiles..][..cout * tiles],
            tiles,
        );
    }
    for (o, dst) in y.chunks_exact_mut(to).enumerate() {
        for (j, at) in WINOGRAD_AT.iter().enumerate() {
            for tile in 0..tiles {
                if tile * 4 + j < to {
                    dst[tile * 4 + j] += (0..6)
                        .map(|xi| at[xi] * m[(xi * cout + o) * tiles + tile])
                        .sum::<f32>();
                }
            }
        }
    }
}

fn conv(node: &Node, c: &[Option<&Tensor>]) -> Prepared {
    let w = constant(c, 1, "Conv weights")?;
    let &[cout, cin_g, k] = &w.shape[..] else {
        return Err(format!("weights {:?}: only 1-D is supported", w.shape));
    };
    let (groups, stride, dilation) = (
        node.int("group", 1) as usize,
        one_dim(node, "strides", 1)?,
        one_dim(node, "dilations", 1)?,
    );
    let wv = w.f32();
    let cout_g = cout / groups;
    let weights = if cin_g == 1 && cout_g == 1 {
        Weights::Depthwise(wv.to_vec())
    } else if (k, stride, dilation, groups) == (3, 1, 1, 1) && cin_g >= 64 {
        // Pays off once the GEMMs are wide enough to hide the transforms.
        Weights::Winograd(
            WINOGRAD_G
                .iter()
                .map(|gx| {
                    pack_a(cout, cin_g, |o, i| {
                        (0..3).map(|kk| gx[kk] * wv[(o * cin_g + i) * 3 + kk]).sum()
                    })
                })
                .collect(),
        )
    } else {
        Weights::Im2col(
            (0..groups)
                .map(|g| {
                    pack_a(cout_g, cin_g * k, |r, col| {
                        wv[(g * cout_g + r) * cin_g * k + col]
                    })
                })
                .collect(),
        )
    };
    let conv = Conv {
        groups,
        cout,
        cin_g,
        k,
        stride,
        dilation,
        weights,
        pads: pads(node)?,
        bias: c
            .get(2)
            .map(|_| constant(c, 2, "Conv bias").map(|b| b.f32().to_vec()))
            .transpose()?,
    };
    if node.str("auto_pad", "NOTSET") != "NOTSET" {
        return Err("auto_pad".into());
    }
    Ok(Some((one(move |a| conv.run(a.get(0))), true)))
}

fn conv_transpose(node: &Node, c: &[Option<&Tensor>]) -> Prepared {
    let w = constant(c, 1, "ConvTranspose weights")?;
    let &[cin, cout, k] = &w.shape[..] else {
        return Err(format!("weights {:?}: only 1-D is supported", w.shape));
    };
    if node.int("group", 1) != 1
        || one_dim(node, "dilations", 1)? != 1
        || node.ints("output_padding").is_some()
    {
        return Err("only ungrouped, undilated ConvTranspose is supported".into());
    }
    let (stride, (p0, p1)) = (one_dim(node, "strides", 1)?, pads(node)?);
    let wv = w.f32();
    // frames[t][(o, kk)] = sum_i x[i][t] W[i][o][kk] (W is already that [cin, cout * k] matrix), then each frame
    // overlap-adds into y[o][t * stride + kk - p0].
    let packed = pack_b(cin, cout * k, |i, j| wv[i * cout * k + j]);
    let bias = c
        .get(2)
        .map(|_| constant(c, 2, "ConvTranspose bias").map(|b| b.f32().to_vec()))
        .transpose()?;
    Ok(Some((
        one(move |args| {
            let x = args.get(0);
            let &[n, ci, t] = &x.shape[..] else {
                panic!("ConvTranspose input {:?}", x.shape)
            };
            assert_eq!(ci, cin);
            let len = (t - 1) * stride + k - p0 - p1;
            let mut out = vec![0f32; n * cout * len];
            let mut frames = vec![0f32; t * cout * k];
            for b in 0..n {
                let xb = &x.f32()[b * cin * t..][..cin * t];
                frames.fill(0.0);
                gemm(
                    t,
                    cout * k,
                    cin,
                    &pack_a(t, cin, |r, i| xb[i * t + r]),
                    B::Packed(&packed),
                    &mut frames,
                    cout * k,
                );
                for o in 0..cout {
                    let y = &mut out[(b * cout + o) * len..][..len];
                    y.fill(bias.as_ref().map_or(0.0, |v| v[o]));
                    for (i, frame) in frames.chunks_exact(cout * k).enumerate() {
                        // The part of this frame's k taps that lands inside the cropped output.
                        let start = (i * stride) as isize - p0 as isize;
                        let lo = (-start).clamp(0, k as isize) as usize;
                        let hi = (len as isize - start).clamp(lo as isize, k as isize) as usize;
                        let dst = &mut y[(start + lo as isize) as usize..][..hi - lo];
                        dst.iter_mut()
                            .zip(&frame[o * k + lo..o * k + hi])
                            .for_each(|(y, v)| *y += v);
                    }
                }
            }
            Tensor::new(vec![n, cout, len], out)
        }),
        true,
    )))
}

/// MatMul over trailing matrices; the right operand is either constant (prepacked) or a batch matching the left.
fn matmul(c: &[Option<&Tensor>]) -> Prepared {
    let packed = match c.get(1).copied().flatten() {
        Some(w) => {
            let &[k, n] = &w.shape[..] else {
                return Err(format!("constant MatMul operand {:?}", w.shape));
            };
            let wv = w.f32();
            Some((k, n, pack_b(k, n, |r, c| wv[r * n + c])))
        }
        None => None,
    };
    Ok(Some((
        one(move |args| {
            let x = args.get(0);
            let (&k, lead) = x.shape.split_last().unwrap();
            let m = *lead.last().unwrap();
            let batch: usize = lead[..lead.len() - 1].iter().product();
            let xv = x.f32();
            let n = match &packed {
                Some((kw, n, _)) => {
                    assert_eq!(*kw, k, "MatMul inner dimension");
                    *n
                }
                None => *args.get(1).shape.last().unwrap(),
            };
            let mut out = vec![0f32; batch * m * n];
            for b in 0..batch {
                let a = pack_a(m, k, |r, i| xv[(b * m + r) * k + i]);
                let c = &mut out[b * m * n..][..m * n];
                match &packed {
                    Some((_, _, w)) => gemm(m, n, k, &a, B::Packed(w), c, n),
                    None => {
                        let y = args.get(1);
                        assert!(
                            y.shape.len() == x.shape.len()
                                && y.shape[..lead.len() - 1] == lead[..lead.len() - 1],
                            "MatMul {:?} x {:?}",
                            x.shape,
                            y.shape
                        );
                        let yb = &y.f32()[b * k * n..][..k * n];
                        let row = |r: usize| (&yb[r * n..][..n], 0);
                        gemm(m, n, k, &a, B::Rows(&row), c, n);
                    }
                }
            }
            let mut shape = x.shape.clone();
            *shape.last_mut().unwrap() = n;
            Tensor::new(shape, out)
        }),
        true,
    )))
}

fn gemm_op(node: &Node, c: &[Option<&Tensor>]) -> Prepared {
    if node.int("transA", 0) != 0
        || node.float("alpha", 1.0) != 1.0
        || node.float("beta", 1.0) != 1.0
    {
        return Err("only transB, unit alpha and beta are supported".into());
    }
    let w = constant(c, 1, "Gemm B")?;
    let trans = node.int("transB", 0) != 0;
    let &[r0, r1] = &w.shape[..] else {
        return Err(format!("Gemm B {:?}", w.shape));
    };
    let (k, n) = if trans { (r1, r0) } else { (r0, r1) };
    let wv = w.f32();
    let packed = pack_b(
        k,
        n,
        |r, j| if trans { wv[j * k + r] } else { wv[r * n + j] },
    );
    let bias = c
        .get(2)
        .map(|_| constant(c, 2, "Gemm C").cloned())
        .transpose()?;
    Ok(Some((
        one(move |args| {
            let x = args.get(0);
            let &[m, kx] = &x.shape[..] else {
                panic!("Gemm A {:?}", x.shape)
            };
            assert_eq!(kx, k);
            let mut out = match &bias {
                Some(b) => {
                    let bv = b.f32();
                    assert!(bv.len() == n || bv.len() == m * n, "Gemm C {:?}", b.shape);
                    (0..m * n).map(|i| bv[i % bv.len()]).collect()
                }
                None => vec![0f32; m * n],
            };
            let xv = x.f32();
            gemm(
                m,
                n,
                k,
                &pack_a(m, k, |r, i| xv[r * k + i]),
                B::Packed(&packed),
                &mut out,
                n,
            );
            Tensor::new(vec![m, n], out)
        }),
        true,
    )))
}

struct Direction {
    /// W^T packed as GEMM B: `[input, 4 * hidden]`.
    w: Vec<f32>,
    /// R^T as `[hidden][4 * hidden]` rows for the per-step axpy.
    r: Vec<f32>,
    bias: Vec<f32>,
}

fn lstm(node: &Node, c: &[Option<&Tensor>]) -> Prepared {
    let h = node.int("hidden_size", 0) as usize;
    let dirs = match node.str("direction", "forward") {
        "forward" => 1,
        "bidirectional" => 2,
        d => return Err(format!("direction {d}")),
    };
    if node.ints("activations").is_some()
        || node.float("clip", 0.0) != 0.0
        || node.int("input_forget", 0) != 0
        || node.int("layout", 0) != 0
        || c.get(4).copied().flatten().is_some()
    {
        return Err("only the default LSTM cell is supported".into());
    }
    let (w, r) = (constant(c, 1, "LSTM W")?, constant(c, 2, "LSTM R")?);
    let input = w.shape[2];
    let g = 4 * h;
    let bias = c.get(3).copied().flatten();
    let directions: Vec<Direction> = (0..dirs)
        .map(|d| {
            let (wv, rv) = (
                &w.f32()[d * g * input..][..g * input],
                &r.f32()[d * g * h..][..g * h],
            );
            Direction {
                w: pack_b(input, g, |i, j| wv[j * input + i]),
                r: (0..h)
                    .flat_map(|hh| (0..g).map(move |j| rv[j * h + hh]))
                    .collect(),
                bias: match bias {
                    Some(b) => (0..g)
                        .map(|j| b.f32()[d * 2 * g + j] + b.f32()[d * 2 * g + g + j])
                        .collect(),
                    None => vec![0.0; g],
                },
            }
        })
        .collect();
    // Constant initial states are captured along with the weights.
    let start = [5, 6].map(|i| c.get(i).copied().flatten().cloned());
    Ok(Some((
        Box::new(move |a: Args| {
            let x = a.get(0);
            let &[t, batch, inp] = &x.shape[..] else {
                panic!("LSTM input {:?}", x.shape)
            };
            assert_eq!(inp, input);
            let initial = |i: usize| {
                a.opt(i)
                    .or(start[i - 5].as_ref())
                    .map_or_else(|| vec![0.0; dirs * batch * h], |t| t.f32().to_vec())
            };
            let (mut hs, mut cs) = (initial(5), initial(6));
            let mut y = vec![0f32; t * dirs * batch * h];
            let xa = pack_a(t * batch, input, |row, i| x.f32()[row * input + i]);
            for (d, dir) in directions.iter().enumerate() {
                let mut gates: Vec<f32> = (0..t * batch)
                    .flat_map(|_| dir.bias.iter().copied())
                    .collect();
                gemm(t * batch, g, input, &xa, B::Packed(&dir.w), &mut gates, g);
                for step in 0..t {
                    let s = if d == 0 { step } else { t - 1 - step };
                    for b in 0..batch {
                        let gt = &mut gates[(s * batch + b) * g..][..g];
                        let (hb, cb) = (
                            &mut hs[(d * batch + b) * h..][..h],
                            &mut cs[(d * batch + b) * h..][..h],
                        );
                        for (hh, &hv) in hb.iter().enumerate() {
                            for (x, &rv) in gt.iter_mut().zip(&dir.r[hh * g..][..g]) {
                                *x += hv * rv;
                            }
                        }
                        // ONNX gate order: input, output, forget, cell.
                        math::sigmoid(&mut gt[..3 * h]);
                        math::tanh(&mut gt[3 * h..]);
                        let (gi, rest) = gt.split_at(h);
                        let (go, rest) = rest.split_at(h);
                        let (gf, gc) = rest.split_at(h);
                        for j in 0..h {
                            cb[j] = gf[j] * cb[j] + gi[j] * gc[j];
                        }
                        let out = &mut y[((s * dirs + d) * batch + b) * h..][..h];
                        out.copy_from_slice(cb);
                        math::tanh(out);
                        for j in 0..h {
                            out[j] *= go[j];
                        }
                        hb.copy_from_slice(out);
                    }
                }
            }
            vec![
                Tensor::new(vec![t, dirs, batch, h], y),
                Tensor::new(vec![dirs, batch, h], hs),
                Tensor::new(vec![dirs, batch, h], cs),
            ]
        }),
        true,
    )))
}

fn mean_var(x: &[f32]) -> (f32, f32) {
    let mean = math::sum(x) / x.len() as f32;
    let mut lanes = [0f32; 8];
    let chunks = x.chunks_exact(8);
    let tail: f32 = chunks
        .remainder()
        .iter()
        .map(|v| (v - mean) * (v - mean))
        .sum();
    for c in chunks {
        for i in 0..8 {
            lanes[i] += (c[i] - mean) * (c[i] - mean);
        }
    }
    (mean, (lanes.iter().sum::<f32>() + tail) / x.len() as f32)
}

fn layer_norm(a: &Args, eps: f32) -> Tensor {
    let (x, scale, bias) = (a.get(0), a.get(1).f32(), a.opt(2).map(Tensor::f32));
    let d = *x.shape.last().unwrap();
    let mut out = Vec::with_capacity(x.len());
    for row in x.f32().chunks_exact(d) {
        let (mean, var) = mean_var(row);
        let inv = 1.0 / (var + eps).sqrt();
        out.extend(
            row.iter()
                .enumerate()
                .map(|(i, &v)| (v - mean) * inv * scale[i] + bias.map_or(0.0, |b| b[i])),
        );
    }
    Tensor::new(x.shape.clone(), out)
}

fn instance_norm(a: &Args, eps: f32) -> Tensor {
    let (x, scale, bias) = (a.get(0), a.get(1).f32(), a.get(2).f32());
    let (c, t) = (x.shape[1], x.shape[2..].iter().product::<usize>());
    let mut out = Vec::with_capacity(x.len());
    for (i, row) in x.f32().chunks_exact(t).enumerate() {
        let (mean, var) = mean_var(row);
        let mul = scale[i % c] / (var + eps).sqrt();
        out.extend(row.iter().map(|&v| (v - mean) * mul + bias[i % c]));
    }
    Tensor::new(x.shape.clone(), out)
}

fn softmax(x: &Tensor, a: i64) -> Tensor {
    let a = axis(a, x.shape.len());
    let (dim, inner) = (x.shape[a], x.shape[a + 1..].iter().product::<usize>());
    let mut v = x.f32().to_vec();
    let mut col = vec![0f32; dim];
    for block in v.chunks_exact_mut(dim * inner) {
        for i in 0..inner {
            for (j, c) in col.iter_mut().enumerate() {
                *c = block[j * inner + i];
            }
            let max = col.iter().fold(f32::NEG_INFINITY, |m, &c| m.max(c));
            col.iter_mut().for_each(|c| *c -= max);
            math::exp(&mut col);
            let inv = 1.0 / math::sum(&col);
            for (j, c) in col.iter().enumerate() {
                block[j * inner + i] = c * inv;
            }
        }
    }
    Tensor::new(x.shape.clone(), v)
}

fn reduce<T: Elem>(
    x: &Tensor,
    axes: Option<Vec<i64>>,
    keep: bool,
    noop: bool,
    init: T,
    f: impl Fn(T, T) -> T,
) -> Tensor {
    let rank = x.shape.len();
    let axes: Vec<usize> = match axes {
        Some(a) if !a.is_empty() => a.iter().map(|&a| axis(a, rank)).collect(),
        _ if noop => return x.clone(),
        _ => (0..rank).collect(),
    };
    let (mut shape, mut v) = (x.shape.clone(), x.view::<T>().to_vec());
    for &a in &axes {
        let (dim, inner) = (shape[a], shape[a + 1..].iter().product::<usize>());
        let mut out = Vec::with_capacity(v.len() / dim.max(1));
        for block in v.chunks_exact(dim * inner) {
            // Row by row, so the inner loop runs over contiguous memory.
            let row = out.len();
            out.resize(row + inner, init);
            for chunk in block.chunks_exact(inner) {
                out[row..]
                    .iter_mut()
                    .zip(chunk)
                    .for_each(|(acc, &x)| *acc = f(*acc, x));
            }
        }
        shape[a] = 1;
        v = out;
    }
    if !keep {
        shape = (0..rank)
            .filter(|i| !axes.contains(i))
            .map(|i| shape[i])
            .collect();
    }
    Tensor::new(shape, v)
}

fn resize(node: &Node) -> Result<Kernel, String> {
    let (mode, coords) = (
        node.str("mode", "nearest").to_string(),
        node.str("coordinate_transformation_mode", "half_pixel")
            .to_string(),
    );
    let nearest = node.str("nearest_mode", "round_prefer_floor").to_string();
    match (mode.as_str(), coords.as_str(), nearest.as_str()) {
        ("nearest", "asymmetric", "floor") | ("linear", "half_pixel", _) => {}
        m => return Err(format!("Resize mode {m:?}")),
    }
    Ok(one(move |a| {
        let x = a.get(0);
        assert!(a.opt(3).is_none(), "Resize by sizes");
        let scales = a.get(2).f32();
        let (&scale, rest) = scales.split_last().unwrap();
        assert!(
            rest.iter().all(|&s| s == 1.0),
            "Resize scales {scales:?}: only the last axis may change"
        );
        let n = *x.shape.last().unwrap();
        let len = (n as f32 * scale).floor() as usize;
        let mut shape = x.shape.clone();
        *shape.last_mut().unwrap() = len;
        let mut out = Vec::with_capacity(x.len() / n * len);
        // Coordinates in f32, the same arithmetic as onnxruntime's CPU kernel.
        if mode == "nearest" {
            let src: Vec<usize> = (0..len)
                .map(|o| ((o as f32 / scale).floor() as usize).min(n - 1))
                .collect();
            for row in x.f32().chunks_exact(n) {
                out.extend(src.iter().map(|&i| row[i]));
            }
        } else {
            let taps: Vec<(usize, usize, f32, f32)> = (0..len)
                .map(|o| {
                    let p = ((o as f32 + 0.5) / scale - 0.5).clamp(0.0, (n - 1) as f32);
                    let i1 = (p as usize).min(n - 1);
                    let i2 = (i1 + 1).min(n - 1);
                    let (d1, d2) = if i1 == i2 {
                        (0.5, 0.5)
                    } else {
                        ((p - i1 as f32).abs(), (p - i2 as f32).abs())
                    };
                    (i1, i2, d2, d1)
                })
                .collect();
            for row in x.f32().chunks_exact(n) {
                out.extend(
                    taps.iter()
                        .map(|&(i1, i2, w1, w2)| w1 * row[i1] + w2 * row[i2]),
                );
            }
        }
        Tensor::new(shape, out)
    }))
}

fn average_pool(node: &Node) -> Result<Kernel, String> {
    let (k, s) = (
        one_dim(node, "kernel_shape", 1)?,
        one_dim(node, "strides", 1)?,
    );
    if pads(node)? != (0, 0) || node.int("ceil_mode", 0) != 0 {
        return Err("only unpadded AveragePool is supported".into());
    }
    Ok(one(move |a| {
        let x = a.get(0);
        let t = *x.shape.last().unwrap();
        let len = (t - k) / s + 1;
        let mut shape = x.shape.clone();
        *shape.last_mut().unwrap() = len;
        let mut out = Vec::with_capacity(x.len() / t * len);
        for row in x.f32().chunks_exact(t) {
            out.extend((0..len).map(|o| row[o * s..o * s + k].iter().sum::<f32>() / k as f32));
        }
        Tensor::new(shape, out)
    }))
}

pub fn prepare(node: &Node, c: &[Option<&Tensor>]) -> Prepared {
    let plain = |k: Kernel| Ok(Some((k, false)));
    let keep = node.int("keepdims", 1) != 0;
    let noop = node.int("noop_with_empty_axes", 0) != 0;
    match node.op.as_str() {
        "Conv" => conv(node, c),
        "ConvTranspose" => conv_transpose(node, c),
        "MatMul" => matmul(c),
        "Gemm" => gemm_op(node, c),
        "LSTM" => lstm(node, c),
        "LayerNormalization" => {
            if node.int("axis", -1) != -1 || node.outputs.len() > 1 {
                return Err(
                    "only last-axis LayerNormalization with one output is supported".into(),
                );
            }
            let eps = node.float("epsilon", 1e-5);
            plain(one(move |a| layer_norm(&a, eps)))
        }
        "InstanceNormalization" => {
            let eps = node.float("epsilon", 1e-5);
            plain(one(move |a| instance_norm(&a, eps)))
        }
        "Softmax" => {
            let ax = node.int("axis", -1);
            plain(one(move |a| softmax(a.get(0), ax)))
        }
        "ReduceSum" => plain(one(move |a| match &a.get(0).data {
            Data::F32(_) => reduce::<f32>(a.get(0), a.ints(1), keep, noop, 0.0, |x, y| x + y),
            Data::I64(_) => reduce::<i64>(a.get(0), a.ints(1), keep, noop, 0, |x, y| x + y),
            d => panic!("ReduceSum on {}", d.kind()),
        })),
        "ReduceMax" => plain(one(move |a| match &a.get(0).data {
            Data::F32(_) => {
                reduce::<f32>(a.get(0), a.ints(1), keep, noop, f32::NEG_INFINITY, f32::max)
            }
            Data::I64(_) => reduce::<i64>(a.get(0), a.ints(1), keep, noop, i64::MIN, i64::max),
            d => panic!("ReduceMax on {}", d.kind()),
        })),
        "Resize" => plain(resize(node)?),
        "AveragePool" => plain(average_pool(node)?),
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::Attr;

    fn node(op: &str, attrs: Vec<(&str, Attr)>) -> Node {
        Node {
            op: op.into(),
            inputs: vec![],
            outputs: vec!["y".into()],
            attrs: attrs.into_iter().map(|(n, a)| (n.into(), a)).collect(),
        }
    }

    fn run(op: &str, attrs: Vec<(&str, Attr)>, inputs: Vec<Option<Tensor>>) -> Vec<Tensor> {
        let n = node(op, attrs);
        let consts: Vec<Option<&Tensor>> = inputs
            .iter()
            .enumerate()
            .map(|(i, t)| if i == 0 { None } else { t.as_ref() })
            .collect();
        let (k, _) = prepare(&n, &consts).unwrap().unwrap();
        k(Args(inputs))
    }

    fn naive_conv(
        x: &[f32],
        cin: usize,
        t: usize,
        w: &[f32],
        cout: usize,
        k: usize,
        s: usize,
        d: usize,
        p: usize,
        groups: usize,
    ) -> Vec<f32> {
        let to = (t + 2 * p - d * (k - 1) - 1) / s + 1;
        let cin_g = cin / groups;
        let mut y = vec![0f32; cout * to];
        for o in 0..cout {
            let g = o / (cout / groups);
            for j in 0..to {
                for i in 0..cin_g {
                    for kk in 0..k {
                        let pos = (j * s + kk * d) as isize - p as isize;
                        if (0..t as isize).contains(&pos) {
                            y[o * to + j] +=
                                w[(o * cin_g + i) * k + kk] * x[(g * cin_g + i) * t + pos as usize];
                        }
                    }
                }
            }
        }
        y
    }

    #[test]
    fn conv_matches_naive() {
        for (cin, cout, t, k, s, d, p, groups) in [
            (3, 5, 40, 3, 1, 1, 1, 1),
            (6, 9, 101, 7, 1, 3, 9, 1),
            (4, 4, 50, 5, 1, 2, 4, 4),
            (2, 3, 64, 16, 4, 1, 8, 1),
            (8, 8, 30, 3, 2, 1, 1, 8),
            (64, 7, 37, 3, 1, 1, 1, 1),
            (70, 9, 9, 3, 1, 1, 0, 1),
            (64, 5, 3, 3, 1, 1, 2, 1),
        ] {
            let x: Vec<f32> = (0..cin * t)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
                .collect();
            let w: Vec<f32> = (0..cout * cin / groups * k)
                .map(|i| ((i * 13 % 17) as f32 - 8.0) / 9.0)
                .collect();
            let ints = |v: Vec<i64>| Attr::Ints(v);
            let attrs = vec![
                ("strides", ints(vec![s as i64])),
                ("dilations", ints(vec![d as i64])),
                ("pads", ints(vec![p as i64; 2])),
                ("group", Attr::Int(groups as i64)),
            ];
            let y = run(
                "Conv",
                attrs,
                vec![
                    Some(Tensor::new(vec![1, cin, t], x.clone())),
                    Some(Tensor::new(vec![cout, cin / groups, k], w.clone())),
                ],
            );
            let want = naive_conv(&x, cin, t, &w, cout, k, s, d, p, groups);
            assert_eq!(y[0].shape, [1, cout, want.len() / cout]);
            for (a, b) in y[0].f32().iter().zip(&want) {
                assert!((a - b).abs() < 1e-4, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn conv_transpose_overlap_adds() {
        // One input channel, kernel [1, 2, 3], stride 2: each input spreads its kernel two samples apart.
        let y = run(
            "ConvTranspose",
            vec![
                ("strides", Attr::Ints(vec![2])),
                ("pads", Attr::Ints(vec![1, 1])),
            ],
            vec![
                Some(Tensor::new(vec![1, 1, 3], vec![1f32, 10., 100.])),
                Some(Tensor::new(vec![1, 1, 3], vec![1f32, 2., 3.])),
            ],
        );
        // Full output [1, 2, 3+10, 20, 30+100, 200, 300] with one sample cropped from each end.
        assert_eq!(y[0].f32(), [2., 13., 20., 130., 200.]);
    }

    #[test]
    fn lstm_matches_reference_cell() {
        // One step, hidden 1, input 1: gates iofc from W = [1, 2, 3, 4], R = 0, bias 0.
        let x = Tensor::new(vec![1, 1, 1], vec![0.5f32]);
        let w = Tensor::new(vec![1, 4, 1], vec![1f32, 2., 3., 4.]);
        let r = Tensor::new(vec![1, 4, 1], vec![0f32; 4]);
        let c0 = Tensor::new(vec![1, 1, 1], vec![1f32]);
        let y = run(
            "LSTM",
            vec![("hidden_size", Attr::Int(1))],
            vec![Some(x), Some(w), Some(r), None, None, None, Some(c0)],
        );
        let s = |v: f32| 1.0 / (1.0 + (-v).exp());
        let c = s(1.5) * 1.0 + s(0.5) * 2f32.tanh();
        let h = s(1.0) * c.tanh();
        assert!((y[2].f32()[0] - c).abs() < 1e-6 && (y[0].f32()[0] - h).abs() < 1e-6);
    }

    #[test]
    fn resize_linear_half_pixel() {
        let attrs = vec![
            ("mode", Attr::Str("linear".into())),
            (
                "coordinate_transformation_mode",
                Attr::Str("half_pixel".into()),
            ),
        ];
        let y = run(
            "Resize",
            attrs,
            vec![
                Some(Tensor::new(vec![1, 1, 2], vec![0f32, 4.])),
                None,
                Some(Tensor::new(vec![3], vec![1f32, 1., 2.])),
            ],
        );
        assert_eq!(y[0].f32(), [0., 1., 3., 4.]);
        let attrs = vec![
            (
                "coordinate_transformation_mode",
                Attr::Str("asymmetric".into()),
            ),
            ("nearest_mode", Attr::Str("floor".into())),
        ];
        let y = run(
            "Resize",
            attrs,
            vec![
                Some(Tensor::new(vec![1, 2], vec![1f32, 2.])),
                None,
                Some(Tensor::new(vec![2], vec![1f32, 1.5])),
            ],
        );
        assert_eq!(y[0].f32(), [1., 1., 2.]);
    }
}
