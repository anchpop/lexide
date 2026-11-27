//! Data movement. Most ops are a strided view of their input, copied out by `strided`.

use super::{Args, Kernel, one};
use crate::proto::Node;
use crate::tensor::{Data, Elem, Tensor, advance, axis, broadcast_shape, each, strides};

fn gather_strided<T: Copy>(v: &[T], shape: &[usize], st: &[isize], offset: isize) -> Vec<T> {
    let mut out = Vec::with_capacity(shape.iter().product());
    let Some((&len, outer)) = shape.split_last() else {
        out.push(v[offset as usize]);
        return out;
    };
    if out.capacity() == 0 {
        return out;
    }
    let step = st[outer.len()];
    let walk: Vec<[isize; 1]> = st[..outer.len()].iter().map(|&s| [s]).collect();
    let (mut index, mut base) = (vec![0; outer.len()], [offset]);
    loop {
        let b = base[0];
        if step == 1 {
            out.extend_from_slice(&v[b as usize..b as usize + len]);
        } else {
            out.extend((0..len as isize).map(|i| v[(b + i * step) as usize]));
        }
        if !advance(&mut index, outer, &walk, &mut base) {
            return out;
        }
    }
}

/// The view of `t` with the given shape, element strides and starting offset, copied out.
fn strided(t: &Tensor, shape: Vec<usize>, st: &[isize], offset: isize) -> Tensor {
    each!(&t.data, v => Tensor::new(shape.clone(), gather_strided(v, &shape, st, offset)))
}

fn signed(s: &[usize]) -> Vec<isize> {
    strides(s).into_iter().map(|x| x as isize).collect()
}

fn transpose(t: &Tensor, perm: &[usize]) -> Tensor {
    let st = signed(&t.shape);
    strided(
        t,
        perm.iter().map(|&p| t.shape[p]).collect(),
        &perm.iter().map(|&p| st[p]).collect::<Vec<_>>(),
        0,
    )
}

fn expand(t: &Tensor, shape: &[usize]) -> Tensor {
    let out = broadcast_shape(&[&t.shape, shape]);
    let st = crate::tensor::broadcast_strides(&t.shape, &out);
    strided(
        t,
        out,
        &st.into_iter().map(|x| x as isize).collect::<Vec<_>>(),
        0,
    )
}

fn tile(t: &Tensor, repeats: &[i64]) -> Tensor {
    // Each dimension d becomes (repeat, d) with the repeat read at stride 0, then the pairs merge.
    let st = signed(&t.shape);
    let (mut shape, mut view) = (vec![], vec![]);
    for (i, &d) in t.shape.iter().enumerate() {
        shape.extend([repeats[i] as usize, d]);
        view.extend([0, st[i]]);
    }
    let out = t
        .shape
        .iter()
        .zip(repeats)
        .map(|(&d, &r)| d * r as usize)
        .collect();
    strided(t, shape, &view, 0).reshape(out)
}

fn slice(
    t: &Tensor,
    starts: &[i64],
    ends: &[i64],
    axes: Option<Vec<i64>>,
    steps: Option<Vec<i64>>,
) -> Tensor {
    let rank = t.shape.len();
    let st = signed(&t.shape);
    let (mut shape, mut view, mut offset) = (t.shape.clone(), st.clone(), 0);
    let axes = axes.unwrap_or_else(|| (0..starts.len() as i64).collect());
    for (i, &a) in axes.iter().enumerate() {
        let a = axis(a, rank);
        let (dim, step) = (t.shape[a] as i64, steps.as_ref().map_or(1, |s| s[i]));
        let wrap = |x: i64| if x < 0 { x + dim } else { x };
        let (start, end) = if step > 0 {
            (wrap(starts[i]).clamp(0, dim), wrap(ends[i]).clamp(0, dim))
        } else {
            (
                wrap(starts[i]).clamp(0, dim - 1),
                wrap(ends[i]).clamp(-1, dim - 1),
            )
        };
        let count = ((end - start) as f64 / step as f64).ceil().max(0.0) as usize;
        shape[a] = count;
        view[a] = st[a] * step as isize;
        offset += st[a] * start as isize;
    }
    let offset = if shape.contains(&0) { 0 } else { offset };
    strided(t, shape, &view, offset)
}

fn concat(parts: &[&Tensor], a: i64) -> Tensor {
    let a = axis(a, parts[0].shape.len());
    let outer: usize = parts[0].shape[..a].iter().product();
    let mut shape = parts[0].shape.clone();
    shape[a] = parts.iter().map(|p| p.shape[a]).sum();
    fn join<T: Elem>(parts: &[&Tensor], outer: usize) -> Vec<T> {
        let chunks: Vec<(&[T], usize)> = parts
            .iter()
            .map(|p| (p.view::<T>(), p.len() / outer.max(1)))
            .collect();
        let mut out = Vec::with_capacity(chunks.iter().map(|c| c.0.len()).sum());
        for o in 0..outer {
            for &(v, n) in &chunks {
                out.extend_from_slice(&v[o * n..(o + 1) * n]);
            }
        }
        out
    }
    let data: Data = match &parts[0].data {
        Data::F32(_) => join::<f32>(parts, outer).into(),
        Data::F64(_) => join::<f64>(parts, outer).into(),
        Data::I64(_) => join::<i64>(parts, outer).into(),
        Data::Bool(_) => join::<bool>(parts, outer).into(),
    };
    Tensor::new(shape, data)
}

fn gather(t: &Tensor, idx: &Tensor, a: i64) -> Tensor {
    let a = axis(a, t.shape.len());
    let (outer, dim): (usize, usize) = (t.shape[..a].iter().product(), t.shape[a]);
    let inner: usize = t.shape[a + 1..].iter().product();
    let shape = [&t.shape[..a], &idx.shape[..], &t.shape[a + 1..]].concat();
    let rows: Vec<usize> = idx
        .i64()
        .iter()
        .map(|&i| if i < 0 { i + dim as i64 } else { i } as usize)
        .collect();
    each!(&t.data, v => {
        let mut out = Vec::with_capacity(outer * rows.len() * inner);
        for o in 0..outer {
            for &r in &rows {
                assert!(r < dim, "Gather index {r} out of {dim}");
                out.extend_from_slice(&v[(o * dim + r) * inner..][..inner]);
            }
        }
        Tensor::new(shape.clone(), out)
    })
}

/// Splits shapes around `axis` for the elements ops, which here always agree off the axis.
fn around(data: &[usize], idx: &[usize], a: usize) -> (usize, usize, usize, usize) {
    assert!(
        data[..a] == idx[..a] && data[a + 1..] == idx[a + 1..],
        "element op shapes {data:?} {idx:?}"
    );
    (
        data[..a].iter().product(),
        data[a],
        idx[a],
        data[a + 1..].iter().product(),
    )
}

fn gather_elements(t: &Tensor, idx: &Tensor, a: i64) -> Tensor {
    let a = axis(a, t.shape.len());
    let (outer, dim, n, inner) = around(&t.shape, &idx.shape, a);
    let ix = idx.i64();
    each!(&t.data, v => {
        let mut out = Vec::with_capacity(ix.len());
        for o in 0..outer {
            for i in 0..n {
                for k in 0..inner {
                    let j = ix[(o * n + i) * inner + k];
                    let j = if j < 0 { j + dim as i64 } else { j } as usize;
                    out.push(v[(o * dim + j) * inner + k]);
                }
            }
        }
        Tensor::new(idx.shape.clone(), out)
    })
}

fn scatter_add<T: Elem + std::ops::Add<Output = T>>(
    t: Tensor,
    idx: &Tensor,
    upd: &Tensor,
    a: i64,
    add: bool,
) -> Tensor {
    let a = axis(a, t.shape.len());
    let (outer, dim, n, inner) = around(&t.shape, &idx.shape, a);
    let (ix, u, shape) = (idx.i64(), upd.view::<T>(), t.shape.clone());
    let mut v = t.into_vec::<T>();
    for o in 0..outer {
        for i in 0..n {
            for k in 0..inner {
                let p = (o * n + i) * inner + k;
                let j = if ix[p] < 0 { ix[p] + dim as i64 } else { ix[p] } as usize;
                let dst = &mut v[(o * dim + j) * inner + k];
                *dst = if add { *dst + u[p] } else { u[p] };
            }
        }
    }
    Tensor::new(shape, v)
}

fn pad(t: &Tensor, pads: &[i64], value: Option<&Tensor>, mode: &str) -> Tensor {
    let rank = t.shape.len();
    assert!(pads.iter().all(|&p| p >= 0), "negative pads {pads:?}");
    let shape: Vec<usize> = (0..rank)
        .map(|d| t.shape[d] + (pads[d] + pads[d + rank]) as usize)
        .collect();
    // For each dimension, the source coordinate of every output coordinate (None reads the constant).
    let maps: Vec<Vec<Option<usize>>> = (0..rank)
        .map(|d| {
            let n = t.shape[d] as i64;
            (0..shape[d] as i64)
                .map(|o| {
                    let i = o - pads[d];
                    match mode {
                        _ if (0..n).contains(&i) => Some(i as usize),
                        "constant" => None,
                        "edge" => Some(i.clamp(0, n - 1) as usize),
                        "reflect" => {
                            let period = 2 * (n - 1).max(1);
                            let m = i.rem_euclid(period);
                            Some(if m < n { m } else { period - m } as usize)
                        }
                        _ => panic!("Pad mode {mode}"),
                    }
                })
                .collect()
        })
        .collect();
    let st = strides(&t.shape);
    each!(&t.data, v => {
        let fill = value.map_or(Default::default(), |c| Elem::get(&c.data).unwrap()[0]);
        let n: usize = shape.iter().product();
        let (mut out, mut index) = (Vec::with_capacity(n), vec![0; rank]);
        for _ in 0..n {
            let src: Option<usize> = (0..rank).try_fold(0, |acc, d| maps[d][index[d]].map(|i| acc + i * st[d]));
            out.push(src.map_or(fill, |s| v[s]));
            for d in (0..rank).rev() {
                index[d] += 1;
                if index[d] < shape[d] {
                    break;
                }
                index[d] = 0;
            }
        }
        Tensor::new(shape.clone(), out)
    })
}

fn cumsum<T: Elem + std::ops::Add<Output = T>>(t: Tensor, a: usize) -> Tensor {
    let (shape, dim, inner) = (
        t.shape.clone(),
        t.shape[a],
        t.shape[a + 1..].iter().product::<usize>(),
    );
    let mut v = t.into_vec::<T>();
    for block in v.chunks_mut(dim * inner) {
        for i in inner..block.len() {
            block[i] = block[i - inner] + block[i];
        }
    }
    Tensor::new(shape, v)
}

fn range<T: Elem + std::ops::Add<Output = T> + std::ops::Mul<Output = T>>(
    a: &Args,
    count: impl Fn(T, T, T) -> f64,
    from: impl Fn(usize) -> T,
) -> Tensor {
    let [s, l, d] = [0, 1, 2].map(|i| a.get(i).view::<T>()[0]);
    let n = count(s, l, d).ceil().max(0.0) as usize;
    Tensor::new(vec![n], (0..n).map(|i| s + from(i) * d).collect::<Vec<T>>())
}

fn unsqueeze(t: Tensor, axes: &[i64]) -> Tensor {
    let rank = t.shape.len() + axes.len();
    let axes: Vec<usize> = axes.iter().map(|&a| axis(a, rank)).collect();
    let mut dims = t.shape.iter();
    let shape = (0..rank)
        .map(|i| {
            if axes.contains(&i) {
                1
            } else {
                *dims.next().unwrap()
            }
        })
        .collect();
    t.reshape(shape)
}

fn squeeze(t: Tensor, axes: Option<Vec<i64>>) -> Tensor {
    let rank = t.shape.len();
    let axes: Vec<usize> = match axes {
        Some(a) => a.iter().map(|&a| axis(a, rank)).collect(),
        None => (0..rank).filter(|&i| t.shape[i] == 1).collect(),
    };
    let shape = (0..rank)
        .filter(|i| !axes.contains(i))
        .map(|i| t.shape[i])
        .collect();
    t.reshape(shape)
}

fn reshape(t: Tensor, spec: &[i64]) -> Tensor {
    let mut shape: Vec<usize> = spec
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            if d == 0 {
                t.shape[i]
            } else {
                d.max(1) as usize
            }
        })
        .collect();
    if let Some(i) = spec.iter().position(|&d| d == -1) {
        let known: usize = shape
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, d)| d)
            .product();
        shape[i] = t.len() / known;
    }
    t.reshape(shape)
}

pub fn prepare(node: &Node) -> Option<Kernel> {
    let ax = node.int("axis", 0);
    Some(match node.op.as_str() {
        "Constant" => {
            let t = node
                .tensor("value")
                .expect("Constant without a tensor value")
                .clone();
            one(move |_| t.clone())
        }
        "Shape" => {
            assert!(
                node.int("start", 0) == 0 && node.int("end", i64::MAX) == i64::MAX,
                "Shape with start/end"
            );
            one(|a| {
                let s = &a.get(0).shape;
                Tensor::new(
                    vec![s.len()],
                    s.iter().map(|&d| d as i64).collect::<Vec<_>>(),
                )
            })
        }
        "ConstantOfShape" => {
            let value = node
                .tensor("value")
                .cloned()
                .unwrap_or_else(|| Tensor::new(vec![1], vec![0f32]));
            one(move |a| {
                let shape: Vec<usize> = a.get(0).i64().iter().map(|&d| d as usize).collect();
                let n = shape.iter().product();
                each!(&value.data, v => Tensor::new(shape, vec![v[0]; n]))
            })
        }
        "Reshape" => {
            assert_eq!(node.int("allowzero", 0), 0, "Reshape allowzero");
            one(|mut a| {
                let spec = a.ints(1).unwrap();
                reshape(a.take(0), &spec)
            })
        }
        "Flatten" => one(move |mut a| {
            let t = a.take(0);
            let k = axis(ax, t.shape.len() + 1);
            let rows = t.shape[..k].iter().product();
            let n = t.len();
            t.reshape(vec![rows, n / rows.max(1)])
        }),
        "Squeeze" => one(|mut a| {
            let axes = a.ints(1);
            squeeze(a.take(0), axes)
        }),
        "Unsqueeze" => one(|mut a| {
            let axes = a.ints(1).unwrap();
            unsqueeze(a.take(0), &axes)
        }),
        "Transpose" => {
            let perm: Option<Vec<usize>> = node
                .ints("perm")
                .map(|p| p.iter().map(|&x| x as usize).collect());
            one(move |a| {
                let t = a.get(0);
                let perm = perm
                    .clone()
                    .unwrap_or_else(|| (0..t.shape.len()).rev().collect());
                transpose(t, &perm)
            })
        }
        "Expand" => one(|a| {
            let shape: Vec<usize> = a.get(1).i64().iter().map(|&d| d as usize).collect();
            expand(a.get(0), &shape)
        }),
        "Tile" => one(|a| tile(a.get(0), a.get(1).i64())),
        "Slice" => one(|a| {
            slice(
                a.get(0),
                a.get(1).i64(),
                a.get(2).i64(),
                a.ints(3),
                a.ints(4),
            )
        }),
        "Concat" => one(move |a| concat(&a.0.iter().flatten().collect::<Vec<_>>(), ax)),
        "Split" => {
            let parts = node.outputs.len();
            Box::new(move |a| {
                let t = a.get(0);
                let k = axis(ax, t.shape.len());
                let sizes = a
                    .ints(1)
                    .unwrap_or_else(|| vec![(t.shape[k] / parts) as i64; parts]);
                let mut start = 0;
                sizes
                    .iter()
                    .map(|&n| {
                        start += n;
                        slice(t, &[start - n], &[start], Some(vec![k as i64]), None)
                    })
                    .collect()
            })
        }
        "Gather" => one(move |a| gather(a.get(0), a.get(1), ax)),
        "GatherElements" => one(move |a| gather_elements(a.get(0), a.get(1), ax)),
        "ScatterElements" => {
            let add = match node.str("reduction", "none") {
                "add" => true,
                "none" => false,
                r => panic!("ScatterElements reduction {r}"),
            };
            one(move |mut a| {
                let t = a.take(0);
                match &t.data {
                    Data::F32(_) => scatter_add::<f32>(t, a.get(1), a.get(2), ax, add),
                    Data::I64(_) => scatter_add::<i64>(t, a.get(1), a.get(2), ax, add),
                    d => panic!("ScatterElements on {}", d.kind()),
                }
            })
        }
        "Pad" => {
            let mode = node.str("mode", "constant").to_string();
            one(move |a| {
                assert!(a.opt(3).is_none(), "Pad axes");
                pad(a.get(0), a.get(1).i64(), a.opt(2), &mode)
            })
        }
        "CumSum" => {
            assert!(
                node.int("exclusive", 0) == 0 && node.int("reverse", 0) == 0,
                "CumSum exclusive/reverse"
            );
            one(|mut a| {
                let t = a.take(0);
                let k = axis(a.get(1).i64()[0], t.shape.len());
                match &t.data {
                    Data::F32(_) => cumsum::<f32>(t, k),
                    Data::F64(_) => cumsum::<f64>(t, k),
                    Data::I64(_) => cumsum::<i64>(t, k),
                    d => panic!("CumSum on {}", d.kind()),
                }
            })
        }
        "Range" => one(|a| match &a.get(0).data {
            // ONNX sizes the range as ceil((limit - start) / delta) in double after subtracting in T.
            Data::F32(_) => range::<f32>(&a, |s, l, d| (l - s) as f64 / d as f64, |i| i as f32),
            Data::I64(_) => range::<i64>(&a, |s, l, d| (l - s) as f64 / d as f64, |i| i as i64),
            d => panic!("Range on {}", d.kind()),
        }),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iota(shape: &[usize]) -> Tensor {
        Tensor::new(
            shape.to_vec(),
            (0..shape.iter().product::<usize>() as i64).collect::<Vec<_>>(),
        )
    }

    #[test]
    fn slices_follow_onnx_clamping() {
        let t = iota(&[10]);
        assert_eq!(
            slice(&t, &[2], &[i64::MAX], None, None).i64(),
            [2, 3, 4, 5, 6, 7, 8, 9]
        );
        assert_eq!(slice(&t, &[-3], &[-1], None, None).i64(), [7, 8]);
        assert_eq!(
            slice(&t, &[i64::MAX], &[i64::MIN], None, Some(vec![-3])).i64(),
            [9, 6, 3, 0]
        );
        assert_eq!(slice(&t, &[5], &[2], None, None).len(), 0);
        let m = iota(&[3, 4]);
        assert_eq!(
            slice(&m, &[1], &[3], Some(vec![-1]), None).i64(),
            [1, 2, 5, 6, 9, 10]
        );
    }

    #[test]
    fn pads_reflect_and_edge() {
        let t = Tensor::new(vec![1, 4], vec![1f32, 2., 3., 4.]);
        assert_eq!(
            pad(&t, &[0, 2, 0, 3], None, "reflect").f32(),
            [3., 2., 1., 2., 3., 4., 3., 2., 1.]
        );
        assert_eq!(
            pad(&t, &[0, 1, 0, 2], None, "edge").f32(),
            [1., 1., 2., 3., 4., 4., 4.]
        );
        let v = Tensor::scalar(9f32);
        assert_eq!(
            pad(&t, &[1, 0, 0, 1], Some(&v), "constant").f32(),
            [9., 9., 9., 9., 9., 1., 2., 3., 4., 9.]
        );
    }

    #[test]
    fn tile_transpose_gather() {
        let t = iota(&[2, 3]);
        assert_eq!(
            tile(&t, &[2, 2]).i64(),
            [
                0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5
            ]
        );
        assert_eq!(transpose(&t, &[1, 0]).i64(), [0, 3, 1, 4, 2, 5]);
        let idx = Tensor::new(vec![2], vec![-1i64, 0]);
        assert_eq!(gather(&t, &idx, 1).i64(), [2, 0, 5, 3]);
        let e = Tensor::new(vec![2, 2], vec![2i64, 0, 1, 1]);
        assert_eq!(
            gather_elements(&t.clone(), &Tensor::new(vec![2, 2], vec![2i64, 0, 1, 1]), 1).i64(),
            [2, 0, 4, 4]
        );
        let s = scatter_add::<i64>(
            Tensor::new(vec![2, 3], vec![0i64; 6]),
            &e,
            &Tensor::new(vec![2, 2], vec![1i64, 2, 3, 4]),
            1,
            true,
        );
        assert_eq!(s.i64(), [2, 0, 1, 0, 7, 0]);
    }
}
