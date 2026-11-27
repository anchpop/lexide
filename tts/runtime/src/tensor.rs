use std::sync::Arc;

/// Element storage. Buffers are shared so reshapes and graph-value copies are free; a kernel that holds the only
/// reference may reuse the buffer in place.
#[derive(Clone, Debug)]
pub enum Data {
    F32(Arc<Vec<f32>>),
    F64(Arc<Vec<f64>>),
    I64(Arc<Vec<i64>>),
    Bool(Arc<Vec<bool>>),
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Data,
}

/// Runs `$body` with `$v` bound to the typed buffer, whatever the element type.
macro_rules! each {
    ($data:expr, $v:ident => $body:expr) => {
        match $data {
            $crate::tensor::Data::F32($v) => $body,
            $crate::tensor::Data::F64($v) => $body,
            $crate::tensor::Data::I64($v) => $body,
            $crate::tensor::Data::Bool($v) => $body,
        }
    };
}
pub(crate) use each;

pub trait Elem: Copy + PartialEq + PartialOrd + Default + Send + Sync + 'static {
    fn wrap(v: Vec<Self>) -> Data;
    fn get(d: &Data) -> Option<&Arc<Vec<Self>>>;
}

macro_rules! elem {
    ($t:ty, $variant:ident) => {
        impl Elem for $t {
            fn wrap(v: Vec<Self>) -> Data {
                Data::$variant(Arc::new(v))
            }
            fn get(d: &Data) -> Option<&Arc<Vec<Self>>> {
                if let Data::$variant(v) = d {
                    Some(v)
                } else {
                    None
                }
            }
        }
    };
}
impl<T: Elem> From<Vec<T>> for Data {
    fn from(v: Vec<T>) -> Data {
        T::wrap(v)
    }
}

elem!(f32, F32);
elem!(f64, F64);
elem!(i64, I64);
elem!(bool, Bool);

impl Data {
    pub fn len(&self) -> usize {
        each!(self, v => v.len())
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn kind(&self) -> &'static str {
        match self {
            Data::F32(_) => "f32",
            Data::F64(_) => "f64",
            Data::I64(_) => "i64",
            Data::Bool(_) => "bool",
        }
    }
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: impl Into<Data>) -> Tensor {
        let t = Tensor {
            shape,
            data: data.into(),
        };
        assert_eq!(t.data.len(), t.len(), "shape {:?}", t.shape);
        t
    }
    pub fn scalar(x: impl Elem) -> Tensor {
        Tensor {
            shape: vec![],
            data: Elem::wrap(vec![x]),
        }
    }
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn view<T: Elem>(&self) -> &[T] {
        T::get(&self.data).unwrap_or_else(|| {
            panic!(
                "expected {}, got {}",
                std::any::type_name::<T>(),
                self.data.kind()
            )
        })
    }
    pub fn f32(&self) -> &[f32] {
        self.view()
    }
    pub fn i64(&self) -> &[i64] {
        self.view()
    }
    pub fn bool(&self) -> &[bool] {
        self.view()
    }
    /// The buffer itself, copied only if something else still shares it.
    pub fn into_vec<T: Elem>(self) -> Vec<T> {
        let v = T::get(&self.data)
            .unwrap_or_else(|| {
                panic!(
                    "expected {}, got {}",
                    std::any::type_name::<T>(),
                    self.data.kind()
                )
            })
            .clone();
        drop(self);
        Arc::unwrap_or_clone(v)
    }
    pub fn reshape(self, shape: Vec<usize>) -> Tensor {
        assert_eq!(
            shape.iter().product::<usize>(),
            self.len(),
            "reshape {:?} to {shape:?}",
            self.shape
        );
        Tensor {
            shape,
            data: self.data,
        }
    }
    /// Integer contents, for shape-like inputs (axes, pads, sizes).
    pub fn ints(&self) -> Vec<i64> {
        self.i64().to_vec()
    }
}

/// Wraps a possibly negative axis into `0..rank`.
pub fn axis(a: i64, rank: usize) -> usize {
    let r = rank as i64;
    assert!((-r..r).contains(&a), "axis {a} out of rank {rank}");
    a.rem_euclid(r) as usize
}

/// Steps an odometer over `dims`, keeping each `offset[n]` at `sum(index[d] * strides[d][n])`. Returns false once
/// it wraps around.
pub fn advance<const N: usize>(
    index: &mut [usize],
    dims: &[usize],
    strides: &[[isize; N]],
    offset: &mut [isize; N],
) -> bool {
    for d in (0..index.len()).rev() {
        index[d] += 1;
        for n in 0..N {
            offset[n] += strides[d][n];
        }
        if index[d] < dims[d] {
            return true;
        }
        for n in 0..N {
            offset[n] -= strides[d][n] * dims[d] as isize;
        }
        index[d] = 0;
    }
    false
}

/// Row-major strides.
pub fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// Numpy-style broadcast of several shapes.
pub fn broadcast_shape(shapes: &[&[usize]]) -> Vec<usize> {
    let rank = shapes.iter().map(|s| s.len()).max().unwrap();
    (0..rank)
        .map(|i| {
            shapes.iter().fold(1, |d, s| {
                let n = if i + s.len() >= rank {
                    s[i + s.len() - rank]
                } else {
                    1
                };
                match (d, n) {
                    (1, n) => n,
                    (d, 1) => d,
                    (d, n) => {
                        assert_eq!(d, n, "cannot broadcast {shapes:?}");
                        d
                    }
                }
            })
        })
        .collect()
}

/// Strides that read `shape` as if broadcast to `out` (zero along broadcast dimensions).
pub fn broadcast_strides(shape: &[usize], out: &[usize]) -> Vec<usize> {
    let s = strides(shape);
    let pad = out.len() - shape.len();
    (0..out.len())
        .map(|i| {
            if i < pad || shape[i - pad] == 1 {
                0
            } else {
                s[i - pad]
            }
        })
        .collect()
}
