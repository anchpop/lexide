use super::{Kernel, one};
use crate::math;
use crate::proto::Node;
use crate::tensor::{Data, Elem, Tensor, advance, axis, broadcast_shape, broadcast_strides};

/// Walks the broadcast of N shapes as contiguous runs: `f(offsets, len, inner strides)`, each stride 0 or 1.
fn runs<const N: usize>(
    shapes: [&[usize]; N],
    mut f: impl FnMut([usize; N], usize, [usize; N]),
) -> Vec<usize> {
    let out = broadcast_shape(&shapes);
    let all: [Vec<usize>; N] = shapes.map(|s| broadcast_strides(s, &out));
    // Merge neighbouring dimensions that every input walks the same way, so most cases become one or two runs.
    let (mut dims, mut st): (Vec<usize>, Vec<[isize; N]>) = (vec![], vec![]);
    for (i, &d) in out.iter().enumerate() {
        let s: [isize; N] = std::array::from_fn(|n| all[n][i] as isize);
        if d == 1 {
            continue;
        }
        match (dims.last_mut(), st.last()) {
            (Some(last), Some(prev)) if (0..N).all(|n| prev[n] == s[n] * d as isize) => {
                *last *= d;
                *st.last_mut().unwrap() = s;
            }
            _ => {
                dims.push(d);
                st.push(s);
            }
        }
    }
    let (len, inner) = match (dims.pop(), st.pop()) {
        (Some(d), Some(s)) => (d, s),
        _ => (1, [0; N]),
    };
    let (mut index, mut offset) = (vec![0; dims.len()], [0; N]);
    loop {
        f(offset.map(|o| o as usize), len, inner.map(|s| s as usize));
        if !advance(&mut index, &dims, &st, &mut offset) {
            return out;
        }
    }
}

pub fn binary<A: Elem, B: Elem, O: Elem>(a: &Tensor, b: &Tensor, f: impl Fn(A, B) -> O) -> Tensor {
    let (x, y) = (a.view::<A>(), b.view::<B>());
    let mut out = Vec::with_capacity(a.len().max(b.len()));
    let shape = runs([&a.shape, &b.shape], |[i, j], n, s| match s {
        [1, 1] => out.extend(x[i..i + n].iter().zip(&y[j..j + n]).map(|(&p, &q)| f(p, q))),
        [1, 0] => out.extend(x[i..i + n].iter().map(|&p| f(p, y[j]))),
        [0, 1] => out.extend(y[j..j + n].iter().map(|&q| f(x[i], q))),
        _ => out.extend(std::iter::repeat_n(f(x[i], y[j]), n)),
    });
    Tensor::new(shape, out)
}

/// `f` over every element, in place when nothing else shares the buffer.
pub fn map<T: Elem>(t: Tensor, f: impl Fn(T) -> T) -> Tensor {
    let shape = t.shape.clone();
    let mut v = t.into_vec::<T>();
    v.iter_mut().for_each(|x| *x = f(*x));
    Tensor::new(shape, v)
}

fn map_slice(t: Tensor, f: impl Fn(&mut [f32])) -> Tensor {
    let shape = t.shape.clone();
    let mut v = t.into_vec::<f32>();
    f(&mut v);
    Tensor::new(shape, v)
}

fn cast(t: &Tensor, to: i64) -> Tensor {
    fn convert<T: Elem>(t: &Tensor, f: impl Fn(f64) -> T, from_bool: impl Fn(bool) -> T) -> Tensor {
        let data: Vec<T> = match &t.data {
            Data::F32(v) => v.iter().map(|&x| f(x as f64)).collect(),
            Data::F64(v) => v.iter().map(|&x| f(x)).collect(),
            Data::I64(v) => v.iter().map(|&x| f(x as f64)).collect(),
            Data::Bool(v) => v.iter().map(|&x| from_bool(x)).collect(),
        };
        Tensor::new(t.shape.clone(), data)
    }
    match (to, &t.data) {
        (1, Data::F32(_)) | (11, Data::F64(_)) | (7, Data::I64(_)) | (9, Data::Bool(_)) => {
            t.clone()
        }
        (1, _) => convert(t, |x| x as f32, |b| b as u8 as f32),
        (11, _) => convert(t, |x| x, |b| b as u8 as f64),
        // Exact for the i64 -> i64 path above; through f64 only floats reach here.
        (7, _) => convert(t, |x| x as i64, |b| b as i64),
        (9, _) => convert(t, |x| x != 0.0, |b| b),
        _ => panic!("Cast to type {to}"),
    }
}

macro_rules! numeric {
    ($a:expr, $b:expr, $f:expr) => {
        match (&$a.data, &$b.data) {
            (Data::F32(_), _) => binary::<f32, f32, f32>($a, $b, $f),
            (Data::F64(_), _) => binary::<f64, f64, f64>($a, $b, $f),
            (Data::I64(_), _) => binary::<i64, i64, i64>($a, $b, $f),
            (d, _) => panic!("arithmetic on {}", d.kind()),
        }
    };
}

macro_rules! compare {
    ($a:expr, $b:expr, $f:expr) => {
        match (&$a.data, &$b.data) {
            (Data::F32(_), _) => binary::<f32, f32, bool>($a, $b, $f),
            (Data::F64(_), _) => binary::<f64, f64, bool>($a, $b, $f),
            (Data::I64(_), _) => binary::<i64, i64, bool>($a, $b, $f),
            (Data::Bool(_), _) => panic!("ordering on bool"),
        }
    };
}

macro_rules! float {
    ($t:expr, $x:ident => $e:expr) => {{
        let t = $t;
        match &t.data {
            Data::F32(_) => map::<f32>(t, |$x| $e),
            Data::F64(_) => map::<f64>(t, |$x| $e),
            d => panic!("float op on {}", d.kind()),
        }
    }};
}

fn select<T: Elem>(c: &Tensor, x: &Tensor, y: &Tensor) -> Tensor {
    let (c_, xs, ys) = (c.bool(), x.view::<T>(), y.view::<T>());
    let mut out = Vec::with_capacity(x.len().max(y.len()));
    let shape = runs(
        [&c.shape, &x.shape, &y.shape],
        |[i, j, k], n, [si, sj, sk]| {
            out.extend((0..n).map(|t| {
                if c_[i + t * si] {
                    xs[j + t * sj]
                } else {
                    ys[k + t * sk]
                }
            }))
        },
    );
    Tensor::new(shape, out)
}

fn clip<T: Elem>(t: Tensor, lo: Option<&Tensor>, hi: Option<&Tensor>) -> Tensor {
    let lo = lo.map(|t| t.view::<T>()[0]);
    let hi = hi.map(|t| t.view::<T>()[0]);
    map::<T>(t, |x| {
        let x = match lo {
            Some(l) if x < l => l,
            _ => x,
        };
        match hi {
            Some(h) if x > h => h,
            _ => x,
        }
    })
}

/// `(x - zero) * scale`, per channel along `axis` or per tensor.
fn dequantize(x: &Tensor, scale: &Tensor, zero: Option<&Tensor>, axis_: i64) -> Tensor {
    let per_channel = |t: &Tensor| {
        if t.len() == 1 {
            return t.clone().reshape(vec![]);
        }
        let a = axis(axis_, x.shape.len());
        let mut shape = vec![1; x.shape.len()];
        shape[a] = t.len();
        t.clone().reshape(shape)
    };
    let shifted = match zero {
        Some(z) => binary::<f32, f32, f32>(x, &per_channel(z), |a, b| a - b),
        None => x.clone(),
    };
    binary::<f32, f32, f32>(&shifted, &per_channel(scale), |a, b| a * b)
}

pub fn prepare(node: &Node) -> Option<Kernel> {
    Some(match node.op.as_str() {
        "Add" => one(|a| numeric!(a.get(0), a.get(1), |x, y| x + y)),
        "Sub" => one(|a| numeric!(a.get(0), a.get(1), |x, y| x - y)),
        "Mul" => one(|a| numeric!(a.get(0), a.get(1), |x, y| x * y)),
        "Div" => one(|a| numeric!(a.get(0), a.get(1), |x, y| x / y)),
        "Min" => one(|a| {
            assert_eq!(a.0.len(), 2, "Min of {} inputs", a.0.len());
            numeric!(a.get(0), a.get(1), |x, y| if y < x { y } else { x })
        }),
        "Equal" => one(|a| match &a.get(0).data {
            Data::Bool(_) => binary::<bool, bool, bool>(a.get(0), a.get(1), |x, y| x == y),
            _ => compare!(a.get(0), a.get(1), |x, y| x == y),
        }),
        "Greater" => one(|a| compare!(a.get(0), a.get(1), |x, y| x > y)),
        "GreaterOrEqual" => one(|a| compare!(a.get(0), a.get(1), |x, y| x >= y)),
        "Less" => one(|a| compare!(a.get(0), a.get(1), |x, y| x < y)),
        "LessOrEqual" => one(|a| compare!(a.get(0), a.get(1), |x, y| x <= y)),
        "And" => one(|a| binary::<bool, bool, bool>(a.get(0), a.get(1), |x, y| x & y)),
        "Or" => one(|a| binary::<bool, bool, bool>(a.get(0), a.get(1), |x, y| x | y)),
        "Not" => one(|mut a| map::<bool>(a.take(0), |x| !x)),
        "IsNaN" => one(|a| {
            Tensor::new(
                a.get(0).shape.clone(),
                a.get(0)
                    .f32()
                    .iter()
                    .map(|x| x.is_nan())
                    .collect::<Vec<_>>(),
            )
        }),
        "Where" => one(|a| select_typed(a.get(0), a.get(1), a.get(2))),
        "Identity" => one(|mut a| a.take(0)),
        "Cast" => {
            let to = node.int("to", 0);
            one(move |a| cast(a.get(0), to))
        }
        "Clip" => one(|mut a| {
            let t = a.take(0);
            match &t.data {
                Data::F32(_) => clip::<f32>(t, a.opt(1), a.opt(2)),
                Data::I64(_) => clip::<i64>(t, a.opt(1), a.opt(2)),
                d => panic!("Clip on {}", d.kind()),
            }
        }),
        "DequantizeLinear" => {
            let axis_ = node.int("axis", 1);
            one(move |a| dequantize(a.get(0), a.get(1), a.opt(2), axis_))
        }
        "LeakyRelu" => {
            let alpha = node.float("alpha", 0.01);
            one(move |mut a| map::<f32>(a.take(0), |x| if x < 0.0 { alpha * x } else { x }))
        }
        "Sin" => one(|mut a| map_slice(a.take(0), math::sin)),
        "Cos" => one(|mut a| map_slice(a.take(0), math::cos)),
        "Exp" => one(|mut a| map_slice(a.take(0), math::exp)),
        "Sigmoid" => one(|mut a| map_slice(a.take(0), math::sigmoid)),
        "Tanh" => one(|mut a| map_slice(a.take(0), math::tanh)),
        "Erf" => one(|mut a| map_slice(a.take(0), math::erf)),
        "Log" => one(|mut a| float!(a.take(0), x => x.ln())),
        "Sqrt" => one(|mut a| float!(a.take(0), x => x.sqrt())),
        "Reciprocal" => one(|mut a| float!(a.take(0), x => x.recip())),
        "Floor" => one(|mut a| float!(a.take(0), x => x.floor())),
        "Ceil" => one(|mut a| float!(a.take(0), x => x.ceil())),
        "Round" => one(|mut a| float!(a.take(0), x => x.round_ties_even())),
        _ => return None,
    })
}

fn select_typed(c: &Tensor, x: &Tensor, y: &Tensor) -> Tensor {
    match &x.data {
        Data::F32(_) => select::<f32>(c, x, y),
        Data::F64(_) => select::<f64>(c, x, y),
        Data::I64(_) => select::<i64>(c, x, y),
        Data::Bool(_) => select::<bool>(c, x, y),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcasting() {
        let a = Tensor::new(vec![2, 1, 3], (0..6).map(|x| x as f32).collect::<Vec<_>>());
        let b = Tensor::new(
            vec![4, 1],
            (0..4).map(|x| 10.0 * x as f32).collect::<Vec<_>>(),
        );
        let c = binary::<f32, f32, f32>(&a, &b, |x, y| x + y);
        assert_eq!(c.shape, [2, 4, 3]);
        let want: Vec<f32> = (0..2)
            .flat_map(|i| {
                (0..4).flat_map(move |j| (0..3).map(move |k| (3 * i + k) as f32 + 10.0 * j as f32))
            })
            .collect();
        assert_eq!(c.f32(), want);
        let s = binary::<f32, f32, f32>(&Tensor::scalar(2f32), &Tensor::scalar(3f32), |x, y| x * y);
        assert_eq!((s.shape.len(), s.f32()), (0, &[6.0][..]));
    }

    #[test]
    fn round_and_cast() {
        let t = Tensor::new(vec![6], vec![0.5f32, 1.5, 2.5, -0.5, -1.5, 2.6]);
        let r = float!(t.clone(), x => x.round_ties_even());
        assert_eq!(r.f32(), [0.0, 2.0, 2.0, -0.0, -2.0, 3.0]);
        assert_eq!(cast(&t, 7).i64(), [0, 1, 2, 0, -1, 2]);
        assert_eq!(cast(&t, 9).bool(), [true; 6]);
    }
}
