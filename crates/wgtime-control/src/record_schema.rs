//! Bounded record-batch schemas, independent of parser and runtime values.
//! The runtime adapter must visit values in source order and charge that visit
//! before constructing these descriptors; handles are deliberately unsupported.
use std::collections::BTreeSet;

pub const MAX_ROWS: usize = 100_000;
pub const MAX_FIELDS: usize = 256;
pub const MAX_DEPTH: usize = 64;
pub const MAX_NODES: usize = 1_000_000;

pub const MAX_TEXT_BYTES: usize = 1_000_000;

/// Borrowed adapter; resource variants return None and are rejected before copy.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Number,
    Bool,
    Text,
    Null,
    List,
    Record,
}
pub trait RecordValue: Sized {
    fn kind(&self) -> Option<Kind>;
    fn finite(&self) -> bool {
        true
    }
    fn text_bytes(&self) -> usize {
        0
    }
    fn list(&self) -> Option<&[Self]> {
        None
    }
    fn fields(&self) -> Option<Box<dyn Iterator<Item = (&str, &Self)> + '_>> {
        None
    }
}
impl RecordValue for serde_json::Value {
    fn kind(&self) -> Option<Kind> {
        Some(match self {
            Self::Number(_) => Kind::Number,
            Self::Bool(_) => Kind::Bool,
            Self::String(_) => Kind::Text,
            Self::Null => Kind::Null,
            Self::Array(_) => Kind::List,
            Self::Object(_) => Kind::Record,
        })
    }
    fn finite(&self) -> bool {
        self.as_f64().is_none_or(f64::is_finite)
    }
    fn text_bytes(&self) -> usize {
        self.as_str().map_or(0, str::len)
    }
    fn list(&self) -> Option<&[Self]> {
        self.as_array().map(Vec::as_slice)
    }
    fn fields(&self) -> Option<Box<dyn Iterator<Item = (&str, &Self)> + '_>> {
        self.as_object()
            .map(|v| Box::new(v.iter().map(|(k, v)| (k.as_str(), v))) as _)
    }
}
/// Charges before allocating a descriptor, including all work before failure.
/// Returns shape and text bytes; the value itself is never cloned.
pub fn inspect<V: RecordValue>(
    value: &V,
    depth: usize,
    charge: &mut impl FnMut() -> anyhow::Result<()>,
    nodes: &mut usize,
) -> anyhow::Result<(Shape, usize)> {
    use anyhow::{ensure, Context};
    ensure!(depth <= MAX_DEPTH, "record value nesting limit exceeded");
    *nodes += 1;
    ensure!(*nodes <= MAX_NODES, "record value node limit exceeded");
    charge()?;
    Ok(
        match value
            .kind()
            .context("record fields cannot contain resource handles or record batches")?
        {
            Kind::Number => {
                ensure!(value.finite(), "record numbers must be finite");
                (Shape::Number, 0)
            }
            Kind::Bool => (Shape::Bool, 0),
            Kind::Null => (Shape::Null, 0),
            Kind::Text => {
                let bytes = value.text_bytes();
                ensure!(bytes <= MAX_TEXT_BYTES, "record text too large");
                (Shape::Text, bytes)
            }
            Kind::List => {
                let values = value.list().context("invalid record list adapter")?;
                let mut element = None;
                let mut bytes = 0usize;
                for value in values {
                    let (candidate, size) = inspect(value, depth + 1, charge, nodes)?;
                    ensure!(
                        element.as_ref().is_none_or(|s| s == &candidate),
                        "record lists require homogeneous fixed shapes"
                    );
                    element = Some(candidate);
                    bytes = bytes.checked_add(size).context("record text overflow")?;
                    ensure!(
                        bytes <= MAX_TEXT_BYTES,
                        "record batch text byte limit exceeded"
                    );
                }
                (
                    Shape::List {
                        len: values.len(),
                        element: element.map(Box::new),
                    },
                    bytes,
                )
            }
            Kind::Record => {
                let mut fields = Vec::new();
                let mut bytes = 0usize;
                for (name, value) in value.fields().context("invalid record object adapter")? {
                    let (shape, size) = inspect(value, depth + 1, charge, nodes)?;
                    bytes = bytes
                        .checked_add(name.len())
                        .and_then(|n| n.checked_add(size))
                        .context("record text overflow")?;
                    ensure!(
                        bytes <= MAX_TEXT_BYTES,
                        "record batch text byte limit exceeded"
                    );
                    fields.push((name.to_owned(), shape));
                }
                (Shape::Record(fields), bytes)
            }
        },
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Shape {
    Number,
    Bool,
    Text,
    Null,
    /// A homogeneous fixed-width list. None is only valid for an empty list.
    List {
        len: usize,
        element: Option<Box<Shape>>,
    },
    /// Fields are canonicalized by name, independent of evaluation order.
    Record(Vec<(String, Shape)>),
}

impl Shape {
    /// Counts actual represented values, including containers, with checked
    /// multiplication for repeated list elements. This is also the copy cost.
    pub fn nodes(&self) -> Result<usize, String> {
        self.nodes_at(0)
    }

    fn nodes_at(&self, depth: usize) -> Result<usize, String> {
        if depth > MAX_DEPTH {
            return Err("record value nesting limit exceeded".into());
        }
        let nodes = match self {
            Self::List {
                len: 0,
                element: None,
            } => 1,
            Self::List {
                len,
                element: Some(element),
            } if *len > 0 => element
                .nodes_at(depth + 1)?
                .checked_mul(*len)
                .and_then(|n| n.checked_add(1))
                .ok_or("record value size overflow")?,
            Self::List { .. } => return Err("invalid record list shape".into()),
            Self::Record(fields) => {
                validate_fields(fields)?;
                let mut total = 1usize;
                for (_, shape) in fields {
                    total = total
                        .checked_add(shape.nodes_at(depth + 1)?)
                        .ok_or("record value size overflow")?;
                }
                total
            }
            _ => 1,
        };
        if nodes > MAX_NODES {
            return Err("record value node limit exceeded".into());
        }
        Ok(nodes)
    }
}

fn validate_fields(fields: &[(String, Shape)]) -> Result<(), String> {
    if fields.is_empty() || fields.len() > MAX_FIELDS {
        return Err(format!("record requires 1..{MAX_FIELDS} fields"));
    }
    let mut names = BTreeSet::new();
    for (name, _) in fields {
        if name.is_empty() || !names.insert(name) {
            return Err("record field names must be nonempty and unique".into());
        }
    }
    Ok(())
}

/// No row values are retained here. A runtime batch owns rows separately and
/// commits its row only after accept succeeds. Failed checks never change state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BatchSchema {
    fields: Option<Vec<(String, Shape)>>,
    rows: usize,
    nodes: usize,
}

impl BatchSchema {
    pub fn fields(&self) -> Option<&[(String, Shape)]> {
        self.fields.as_deref()
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn nodes(&self) -> usize {
        self.nodes
    }

    /// Field expressions have already been evaluated left to right. Sorting
    /// here affects validation only; it must never reorder expression effects.
    /// Returns the row's node count so the caller can charge work explicitly.
    pub fn accept(&mut self, mut fields: Vec<(String, Shape)>) -> Result<usize, String> {
        validate_fields(&fields)?;
        fields.sort_by(|a, b| a.0.cmp(&b.0));
        let cost = Shape::Record(fields.clone()).nodes()?;
        if self.fields.as_ref().is_some_and(|schema| schema != &fields) {
            return Err("record batch fields, value kinds and list shapes must match".into());
        }
        let rows = self
            .rows
            .checked_add(1)
            .ok_or("record batch row overflow")?;
        let nodes = self
            .nodes
            .checked_add(cost)
            .ok_or("record batch size overflow")?;
        if rows > MAX_ROWS || nodes > MAX_NODES {
            return Err("record batch storage limit exceeded".into());
        }
        if self.fields.is_none() {
            self.fields = Some(fields);
        }
        self.rows = rows;
        self.nodes = nodes;
        Ok(cost)
    }
}

/// Produces aligned row ranges without materializing or copying the batch.
/// The caller evaluates and validates the numeric size once, then visits each
/// range in order. A short final range is retained; an empty batch is empty.
pub fn chunks(
    len: usize,
    size: usize,
) -> Result<impl Iterator<Item = std::ops::Range<usize>>, String> {
    if size == 0 {
        return Err("batch size must be a positive integer".into());
    }
    let mut start = 0usize;
    Ok(std::iter::from_fn(move || {
        if start >= len {
            return None;
        }
        let end = start.saturating_add(size).min(len);
        let range = start..end;
        start = end;
        Some(range)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn vector(len: usize) -> Shape {
        Shape::List {
            len,
            element: (len > 0).then(|| Box::new(Shape::Number)),
        }
    }
    fn row() -> Vec<(String, Shape)> {
        vec![
            ("observation".into(), vector(2)),
            ("action".into(), Shape::Number),
        ]
    }

    #[test]
    fn schema_matches_fields_by_name_and_preserves_shapes() {
        let mut batch = BatchSchema::default();
        assert_eq!(batch.fields(), None);
        assert_eq!(batch.accept(row()).unwrap(), 5);
        let mut reordered = row();
        reordered.reverse();
        assert_eq!(batch.accept(reordered).unwrap(), 5);
        assert_eq!(batch.rows(), 2);
        assert_eq!(batch.nodes(), 10);
        assert_eq!(batch.fields().unwrap()[0].0, "action");
    }

    #[test]
    fn invalid_rows_leave_schema_and_counters_unchanged() {
        let mut batch = BatchSchema::default();
        batch.accept(row()).unwrap();
        let before = batch.clone();
        for fields in [
            vec![],
            vec![("action".into(), Shape::Number)],
            vec![
                ("action".into(), Shape::Number),
                ("action".into(), Shape::Bool),
            ],
            vec![
                ("observation".into(), vector(3)),
                ("action".into(), Shape::Number),
            ],
            vec![
                ("observation".into(), vector(2)),
                ("action".into(), Shape::Bool),
            ],
        ] {
            assert!(batch.accept(fields).is_err());
            assert_eq!(batch, before);
        }
    }

    #[test]
    fn storage_bounds_and_overflow_are_checked_before_commit() {
        let mut batch = BatchSchema::default();
        assert!(batch
            .accept(vec![("x".into(), vector(usize::MAX))])
            .is_err());
        assert_eq!(batch, BatchSchema::default());
        let mut full = BatchSchema {
            fields: None,
            rows: MAX_ROWS,
            nodes: 0,
        };
        let before = full.clone();
        assert!(full.accept(row()).is_err());
        assert_eq!(full, before);
        let mut full = BatchSchema {
            fields: None,
            rows: 0,
            nodes: MAX_NODES - 4,
        };
        let before = full.clone();
        assert!(full.accept(row()).is_err());
        assert_eq!(full, before);
    }

    #[test]
    fn aligned_chunks_keep_tail_order_and_empty_batches() {
        assert_eq!(
            chunks(48, 16).unwrap().collect::<Vec<_>>(),
            vec![0..16, 16..32, 32..48]
        );
        assert_eq!(
            chunks(35, 16).unwrap().collect::<Vec<_>>(),
            vec![0..16, 16..32, 32..35]
        );
        assert_eq!(
            chunks(3, usize::MAX).unwrap().collect::<Vec<_>>(),
            vec![0..3]
        );
        assert_eq!(chunks(0, 16).unwrap().count(), 0);
        assert!(chunks(0, 0).is_err());
        assert_eq!(
            chunks(usize::MAX, usize::MAX - 1)
                .unwrap()
                .collect::<Vec<_>>(),
            vec![0..usize::MAX - 1, usize::MAX - 1..usize::MAX]
        );
    }

    #[test]
    fn nested_shapes_empty_lists_and_reserved_postfix_names() {
        let fields = vec![
            ("mean".into(), vector(0)),
            ("probabilities".into(), Shape::Text),
            (
                "nested".into(),
                Shape::Record(vec![("ok".into(), Shape::Bool)]),
            ),
            ("optional".into(), Shape::Null),
        ];
        let mut batch = BatchSchema::default();
        assert_eq!(batch.accept(fields).unwrap(), 6);
        assert!(Shape::List {
            len: 0,
            element: Some(Box::new(Shape::Number))
        }
        .nodes()
        .is_err());
        let mut deep = Shape::Number;
        for _ in 0..=MAX_DEPTH {
            deep = Shape::List {
                len: 1,
                element: Some(Box::new(deep)),
            };
        }
        assert!(deep.nodes().is_err());
    }
}
