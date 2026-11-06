//! Backend-independent, fixed-shape tensor contracts. This module owns metadata,
//! not values, parameter identity, device allocation, or implicit dtype casts.
use std::{collections::HashSet, error::Error, fmt};

pub const MAX_RANK: usize = 8;
pub const MAX_EXTENT: usize = 16_384;
pub const MAX_ELEMENTS: usize = 16_777_216;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    Rank,
    Extent,
    Overflow,
    ElementLimit,
    Shape,
    DType,
    Axis,
    Permutation,
    Layout,
    Bounds,
    Materialization,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorError {
    pub kind: ErrorKind,
    message: String,
}
impl TensorError {
    fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}
impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}
impl Error for TensorError {}
pub type Result<T> = std::result::Result<T, TensorError>;

/// Admission limits are explicit at each boundary. In particular, legacy graph
/// metadata admits scalars/empty axes; native registers do not. A shared type
/// does not silently broaden either execution contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShapeLimits {
    pub min_rank: usize,
    pub max_rank: usize,
    pub max_extent: usize,
    pub max_elements: usize,
    pub allow_zero: bool,
}
impl ShapeLimits {
    pub const REGISTER: Self = Self {
        min_rank: 1, max_rank: MAX_RANK, max_extent: MAX_EXTENT,
        max_elements: MAX_ELEMENTS, allow_zero: false,
    };
    pub const PARAMETER: Self = Self {
        max_extent: usize::MAX, ..Self::REGISTER
    };
    pub const VECTOR: Self = Self {
        max_rank: 1, max_elements: MAX_EXTENT, ..Self::REGISTER
    };
    /// Compatibility profile for artifact metadata, not permission to execute
    /// an arbitrary shape on a native backend.
    pub const ARTIFACT: Self = Self {
        min_rank: 0, max_rank: usize::MAX, max_extent: usize::MAX,
        max_elements: usize::MAX, allow_zero: true,
    };
}

/// Validate before allocating a value buffer. All cardinality arithmetic is
/// checked. A scalar has one element; an admitted zero-extent shape has zero.
pub fn validate_shape(shape: &[usize], limits: ShapeLimits) -> Result<usize> {
    if shape.len() < limits.min_rank || shape.len() > limits.max_rank {
        return Err(TensorError::new(ErrorKind::Rank, format!(
            "tensor rank {} is outside {}..={}", shape.len(), limits.min_rank, limits.max_rank,
        )));
    }
    for (axis, &extent) in shape.iter().enumerate() {
        if (extent == 0 && !limits.allow_zero) || extent > limits.max_extent {
            return Err(TensorError::new(ErrorKind::Extent, format!(
                "tensor axis {axis} has invalid extent {extent}",
            )));
        }
    }
    // Validate all axes first, and avoid order-dependent overflow on empty
    // metadata such as [usize::MAX, 2, 0]. Such a tensor has no elements.
    let elements = if shape.contains(&0) {
        0
    } else {
        shape.iter().try_fold(1usize, |n, &extent| {
            n.checked_mul(extent).ok_or_else(|| {
                TensorError::new(ErrorKind::Overflow, "tensor element count overflow")
            })
        })?
    };
    if elements > limits.max_elements {
        return Err(TensorError::new(ErrorKind::ElementLimit, format!(
            "tensor has {elements} elements; limit is {}", limits.max_elements,
        )));
    }
    Ok(elements)
}

pub fn validate_axes(axes: &[Option<String>], rank: usize) -> Result<()> {
    if axes.len() != rank {
        return Err(TensorError::new(ErrorKind::Axis, "axis labels must match tensor rank"));
    }
    let mut seen = HashSet::new();
    for name in axes.iter().flatten() {
        let identifier = !name.is_empty() && name.bytes().enumerate().all(|(i, byte)| {
            byte == b'_' || byte.is_ascii_alphabetic() || (i > 0 && byte.is_ascii_digit())
        });
        if !identifier || !seen.insert(name.as_str()) {
            return Err(TensorError::new(ErrorKind::Axis, format!(
                "invalid or duplicate tensor axis {name:?}",
            )));
        }
    }
    Ok(())
}

pub fn validate_permutation(axes: &[usize], rank: usize) -> Result<()> {
    if axes.len() != rank {
        return Err(TensorError::new(ErrorKind::Permutation, "permutation rank mismatch"));
    }
    let mut seen = HashSet::with_capacity(rank);
    for &axis in axes {
        if axis >= rank || !seen.insert(axis) {
            return Err(TensorError::new(ErrorKind::Permutation,
                "permutation axes must form a complete bijection"));
        }
    }
    Ok(())
}

/// Logical element metadata. Representing a dtype here does not add a kernel,
/// conversion, packed quantization format, or native execution support for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType { F32, F16, Bf16, I8, U8, I32, I64, Bool }
impl DType {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "f32" => Ok(Self::F32), "f16" => Ok(Self::F16),
            "bf16" => Ok(Self::Bf16), "i8" => Ok(Self::I8),
            "u8" => Ok(Self::U8), "i32" => Ok(Self::I32),
            "i64" => Ok(Self::I64), "bool" => Ok(Self::Bool),
            _ => Err(TensorError::new(ErrorKind::DType, format!("unknown tensor dtype {name:?}"))),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32", Self::F16 => "f16", Self::Bf16 => "bf16",
            Self::I8 => "i8", Self::U8 => "u8", Self::I32 => "i32",
            Self::I64 => "i64", Self::Bool => "bool",
        }
    }
}

/// Semantic type: fixed extents and positional optional axis names. Layout and
/// storage ownership are separate. Fields are private to keep invariants valid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorType {
    dtype: DType,
    shape: Vec<usize>,
    axes: Vec<Option<String>>,
}
impl TensorType {
    pub fn new(dtype: DType, shape: Vec<usize>, axes: Vec<Option<String>>, limits: ShapeLimits) -> Result<Self> {
        validate_shape(&shape, limits)?;
        validate_axes(&axes, shape.len())?;
        Ok(Self { dtype, shape, axes })
    }
    pub fn unnamed(dtype: DType, shape: Vec<usize>, limits: ShapeLimits) -> Result<Self> {
        let axes = vec![None; shape.len()];
        Self::new(dtype, shape, axes, limits)
    }
    pub fn dtype(&self) -> DType { self.dtype }
    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn axes(&self) -> &[Option<String>] { &self.axes }
    pub fn rank(&self) -> usize { self.shape.len() }
    pub fn elements(&self) -> usize {
        if self.shape.contains(&0) { 0 } else { self.shape.iter().product() }
    }
    pub fn axis(&self, name: &str) -> Result<usize> {
        self.axes.iter().position(|axis| axis.as_deref() == Some(name))
            .ok_or_else(|| TensorError::new(ErrorKind::Axis, format!("unknown tensor axis {name:?}")))
    }
    pub fn compatible(&self, other: &Self) -> bool {
        self.dtype == other.dtype && self.shape == other.shape && self.axes.iter().zip(&other.axes)
            .all(|(a, b)| a.is_none() || b.is_none() || a == b)
    }
    /// Exact positional matching. Scalars, size-one axes and names do not
    /// trigger implicit expansion or reordering.
    pub fn elementwise(&self, other: &Self, limits: ShapeLimits) -> Result<Self> {
        self.same_dtype(other)?;
        if !self.compatible(other) {
            return Err(TensorError::new(ErrorKind::Shape, "elementwise shape/axis mismatch; broadcasting is not implicit"));
        }
        Self::new(self.dtype, self.shape.clone(), self.axes.iter().zip(&other.axes)
            .map(|(a, b)| a.clone().or_else(|| b.clone())).collect(), limits)
    }
    pub fn reshape(&self, shape: Vec<usize>, axes: Vec<Option<String>>, limits: ShapeLimits) -> Result<Self> {
        let output = Self::new(self.dtype, shape, axes, limits)?;
        if output.elements() != self.elements() {
            return Err(TensorError::new(ErrorKind::Shape, "reshape element count mismatch"));
        }
        Ok(output)
    }
    pub fn permute(&self, order: &[usize], limits: ShapeLimits) -> Result<Self> {
        validate_permutation(order, self.rank())?;
        Self::new(self.dtype, order.iter().map(|&a| self.shape[a]).collect(),
            order.iter().map(|&a| self.axes[a].clone()).collect(), limits)
    }
    pub fn transpose(&self, limits: ShapeLimits) -> Result<Self> {
        if self.rank() != 2 {
            return Err(TensorError::new(ErrorKind::Rank, "transpose requires rank 2"));
        }
        self.permute(&[1, 0], limits)
    }
    pub fn matmul(&self, right: &Self, limits: ShapeLimits) -> Result<Self> {
        self.product_type(right)?;
        if self.rank() != 2 || right.rank() != 2 {
            return Err(TensorError::new(ErrorKind::Rank, "matmul requires two rank-2 tensors"));
        }
        self.match_axis(1, right, 0)?;
        Self::new(self.dtype, vec![self.shape[0], right.shape[1]],
            vec![self.axes[0].clone(), right.axes[1].clone()], limits)
    }
    pub fn batch_matmul(&self, right: &Self, limits: ShapeLimits) -> Result<Self> {
        self.product_type(right)?;
        if self.rank() != 3 || right.rank() != 3 {
            return Err(TensorError::new(ErrorKind::Rank, "batch_matmul requires two rank-3 tensors"));
        }
        self.match_axis(0, right, 0)?;
        self.match_axis(2, right, 1)?;
        Self::new(self.dtype, vec![self.shape[0], self.shape[1], right.shape[2]],
            vec![self.axes[0].clone().or_else(|| right.axes[0].clone()),
                self.axes[1].clone(), right.axes[2].clone()], limits)
    }
    pub fn validate_normalization(&self, gamma: &Self, beta: &Self, axis: usize) -> Result<()> {
        self.same_dtype(gamma)?;
        self.same_dtype(beta)?;
        let width = *self.shape.get(axis)
            .ok_or_else(|| TensorError::new(ErrorKind::Axis, "normalization axis out of bounds"))?;
        if gamma.shape != [width] || beta.shape != [width] {
            return Err(TensorError::new(ErrorKind::Shape, "normalization affine arguments must be vectors matching the axis width"));
        }
        self.match_axis(axis, gamma, 0)?;
        self.match_axis(axis, beta, 0)
    }
    fn same_dtype(&self, other: &Self) -> Result<()> {
        if self.dtype != other.dtype {
            return Err(TensorError::new(ErrorKind::DType, "tensor dtype mismatch; casts must be explicit"));
        }
        Ok(())
    }
    fn product_type(&self, other: &Self) -> Result<()> {
        self.same_dtype(other)?;
        if self.dtype == DType::Bool {
            return Err(TensorError::new(ErrorKind::DType, "boolean matrix products are not defined"));
        }
        Ok(())
    }
    fn match_axis(&self, a: usize, other: &Self, b: usize) -> Result<()> {
        if self.shape[a] != other.shape[b] {
            return Err(TensorError::new(ErrorKind::Shape, format!("tensor axis extents disagree: {} vs {}", self.shape[a], other.shape[b])));
        }
        if let (Some(left), Some(right)) = (&self.axes[a], &other.axes[b]) {
            if left != right {
                return Err(TensorError::new(ErrorKind::Axis, format!("tensor axis names disagree: {left} vs {right}")));
            }
        }
        Ok(())
    }
}

/// Validate an element-stride view without allocating metadata or value storage.
pub fn validate_layout(shape: &[usize], strides: &[usize], offset: usize,
    storage_elements: usize, limits: ShapeLimits) -> Result<usize>
{
    let elements = validate_shape(shape, limits)?;
    if shape.len() != strides.len() {
        return Err(TensorError::new(ErrorKind::Layout, "stride count must equal rank"));
    }
    if elements == 0 {
        if offset > storage_elements {
            return Err(TensorError::new(ErrorKind::Bounds, "empty view offset exceeds storage"));
        }
    } else {
        let end = shape.iter().zip(strides).try_fold(offset, |end, (&extent, &stride)| {
            (extent - 1).checked_mul(stride).and_then(|span| end.checked_add(span))
                .ok_or_else(|| TensorError::new(ErrorKind::Overflow, "view address overflow"))
        })?;
        if end >= storage_elements {
            return Err(TensorError::new(ErrorKind::Bounds, "view exceeds backing storage"));
        }
    }
    Ok(elements)
}

/// Extent-one axes do not affect the addressed order. Overflow is rejected even
/// when this helper is called with metadata that has not yet been admitted.
pub fn is_contiguous_layout(shape: &[usize], strides: &[usize]) -> bool {
    if shape.len() != strides.len() { return false; }
    if shape.contains(&0) { return true; }
    let mut expected = 1usize;
    for (&extent, &stride) in shape.iter().zip(strides).rev() {
        if extent > 1 && stride != expected { return false; }
        let Some(next) = expected.checked_mul(extent) else { return false; };
        expected = next;
    }
    true
}

/// Bounds-checked element-stride layout. A layout is only an index map into a
/// caller-owned storage; it never assigns a Weight identity or allocates values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    storage_elements: usize,
    elements: usize,
}
impl Layout {
    pub fn new(shape: Vec<usize>, strides: Vec<usize>, offset: usize, storage_elements: usize, limits: ShapeLimits) -> Result<Self> {
        let elements = validate_layout(&shape, &strides, offset, storage_elements, limits)?;
        Ok(Self { shape, strides, offset, storage_elements, elements })
    }
    pub fn contiguous(shape: Vec<usize>, limits: ShapeLimits) -> Result<Self> {
        let elements = validate_shape(&shape, limits)?;
        let strides = Self::row_major_strides(&shape)?;
        Self::new(shape, strides, 0, elements, limits)
    }
    fn row_major_strides(shape: &[usize]) -> Result<Vec<usize>> {
        let mut strides = vec![1; shape.len()];
        let mut stride = 1usize;
        for axis in (0..shape.len()).rev() {
            strides[axis] = stride;
            stride = stride.checked_mul(shape[axis])
                .ok_or_else(|| TensorError::new(ErrorKind::Overflow, "contiguous stride overflow"))?;
        }
        Ok(strides)
    }
    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn strides(&self) -> &[usize] { &self.strides }
    pub fn offset(&self) -> usize { self.offset }
    pub fn is_contiguous(&self) -> bool {
        is_contiguous_layout(&self.shape, &self.strides)
    }
    /// Sufficient, conservative proof of no self-overlap. Zero-stride or other
    /// overlapping read views must not be used for in-place updates.
    pub fn is_non_overlapping(&self) -> bool {
        if self.elements == 0 { return true; }
        let mut axes: Vec<_> = self.shape.iter().copied().zip(self.strides.iter().copied())
            .filter(|(extent, _)| *extent > 1).collect();
        axes.sort_unstable_by_key(|(_, stride)| *stride);
        let mut span = 1usize;
        for (extent, stride) in axes {
            if stride < span { return false; }
            let Some(next) = (extent - 1).checked_mul(stride).and_then(|n| span.checked_add(n)) else {
                return false;
            };
            span = next;
        }
        true
    }
    pub fn permute(&self, order: &[usize], limits: ShapeLimits) -> Result<Self> {
        validate_permutation(order, self.shape.len())?;
        Self::new(order.iter().map(|&axis| self.shape[axis]).collect(),
            order.iter().map(|&axis| self.strides[axis]).collect(),
            self.offset, self.storage_elements, limits)
    }
    /// Refuse a semantic reorder disguised as a zero-copy reshape. A caller
    /// may explicitly materialize logical row-major values and retry.
    pub fn reshape(&self, shape: Vec<usize>, limits: ShapeLimits) -> Result<Self> {
        if validate_shape(&shape, limits)? != self.elements {
            return Err(TensorError::new(ErrorKind::Shape, "view reshape element count mismatch"));
        }
        if !self.is_contiguous() {
            return Err(TensorError::new(ErrorKind::Materialization, "noncontiguous reshape requires materialization"));
        }
        let strides = Self::row_major_strides(&shape)?;
        Self::new(shape, strides, self.offset, self.storage_elements, limits)
    }
    pub fn logical_offset(&self, mut index: usize) -> Result<usize> {
        if index >= self.elements {
            return Err(TensorError::new(ErrorKind::Bounds, "logical tensor index out of bounds"));
        }
        let mut offset = self.offset;
        for (&extent, &stride) in self.shape.iter().zip(&self.strides).rev() {
            offset += (index % extent) * stride;
            index /= extent;
        }
        Ok(offset)
    }
}
