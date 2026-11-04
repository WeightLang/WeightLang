//! Record batches retain a schema distinct from ordinary indexed lists.
use super::{record_schema::{BatchSchema, Shape, MAX_DEPTH, MAX_NODES}, Value};
use anyhow::{bail, ensure, Result};
use std::collections::BTreeMap;

pub const MAX_TEXT_BYTES: usize = 1_000_000;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RecordBatch {
    pub schema: BatchSchema,
    pub rows: Vec<Value>,
    bytes: usize,
}

fn shape_with(value: &Value, depth: usize, charge: &mut impl FnMut() -> Result<()>, nodes: &mut usize) -> Result<Shape> {
    ensure!(depth <= MAX_DEPTH, "record value nesting limit exceeded");
    *nodes += 1;
    ensure!(*nodes <= MAX_NODES, "record value node limit exceeded");
    charge()?;
    Ok(match value {
        Value::Number(n) => { ensure!(n.is_finite(), "record numbers must be finite"); Shape::Number }
        Value::Bool(_) => Shape::Bool,
        Value::Text(s) => { ensure!(s.len() <= MAX_TEXT_BYTES, "record text too large"); Shape::Text }
        Value::Null => Shape::Null,
        Value::List(values) => {
            let mut element = None;
            for value in values {
                let candidate = shape_with(value, depth + 1, charge, nodes)?;
                ensure!(element.as_ref().is_none_or(|s| s == &candidate), "record lists require homogeneous fixed shapes");
                element = Some(candidate);
            }
            Shape::List { len: values.len(), element: element.map(Box::new) }
        }
        Value::Object(fields) => Shape::Record(fields.iter().map(|(name,value)|
            Ok((name.clone(), shape_with(value, depth + 1, charge, nodes)?))).collect::<Result<_>>()?),
        _ => bail!("record fields cannot contain resource handles or record batches"),
    })
}

fn shape(value:&Value,depth:usize,remaining:&mut u64,nodes:&mut usize)->Result<Shape>{
    shape_with(value,depth,&mut || { *remaining=remaining.checked_sub(1).ok_or_else(||anyhow::anyhow!("script operation budget exceeded"))?;Ok(()) },nodes)
}

/// Validate before cloning a record into storage. Each visited node is charged
/// directly to the caller's ledger, including work before a validation failure.
pub(super) fn validate_record_fields(args:&[Value],mut charge:impl FnMut()->Result<()>)->Result<()> {
    ensure!(args.len()%2==0,"invalid record fields");
    charge()?; // The record root is charged identically to state validation.
    let mut nodes=1;
    let mut bytes=5usize; // state validation historically includes the wrapper name.
    let mut names=std::collections::BTreeSet::new();
    for pair in args.chunks_exact(2) {
        let name=pair[0].text()?;
        ensure!(names.insert(name),"duplicate record field");
        shape_with(&pair[1],2,&mut charge,&mut nodes)?;
        bytes=bytes.checked_add(name.len()).and_then(|n|n.checked_add(data_bytes(&pair[1]))).ok_or_else(||anyhow::anyhow!("record text size overflow"))?;
        ensure!(bytes<=MAX_TEXT_BYTES,"record batch text byte limit exceeded");
    }
    Ok(())
}

fn data_bytes(value: &Value) -> usize {
    match value {
        Value::Text(text) => text.len(),
        Value::List(values) => values.iter().map(data_bytes).sum(),
        Value::Object(fields) => fields.iter().map(|(name,value)| name.len() + data_bytes(value)).sum(),
        _ => 0,
    }
}

impl RecordBatch {
    pub fn push(&mut self, fields: Vec<(String, Value)>, remaining: &mut u64) -> Result<()> {
        let mut names = BTreeMap::new();
        let mut shapes = Vec::new();
        let mut nodes = 0;
        for (name, value) in fields {
            shapes.push((name.clone(), shape(&value, 1, remaining, &mut nodes)?));
            ensure!(names.insert(name, value).is_none(), "duplicate record field");
        }
        let bytes = names.iter().try_fold(self.bytes, |total,(name,value)|
            total.checked_add(name.len()).and_then(|n|n.checked_add(data_bytes(value))))
            .ok_or_else(|| anyhow::anyhow!("record batch text size overflow"))?;
        ensure!(bytes <= MAX_TEXT_BYTES, "record batch text byte limit exceeded");
        self.schema.accept(shapes).map_err(anyhow::Error::msg)?;
        self.bytes = bytes;
        self.rows.push(Value::Object(names));
        Ok(())
    }

    pub fn project(&self, field: &str, remaining: &mut u64) -> Result<Value> {
        if let Some(fields) = self.schema.fields() {
            ensure!(fields.iter().any(|(name,_)| name == field), "missing record field: {field}");
        }
        let mut result = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            let Value::Object(fields) = row else { unreachable!() };
            let value = fields.get(field).expect("validated batch schema");
            shape(value, 0, remaining, &mut 0)?;
            result.push(value.clone());
        }
        Ok(Value::List(result))
    }

    pub fn chunk(&self, range: std::ops::Range<usize>, remaining: &mut u64) -> Result<Self> {
        let mut batch = Self::default();
        for row in &self.rows[range] {
            let Value::Object(fields) = row else { unreachable!() };
            batch.push(fields.iter().map(|(k,v)| (k.clone(),v.clone())).collect(), remaining)?;
        }
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn record_callbacks_charge_failures_and_share_atomic_budget() {
        use std::sync::{Arc,atomic::{AtomicU64,Ordering}};
        let args=vec![Value::Text("x".into()),Value::List(vec![Value::Number(1.),Value::Number(2.)])];
        let remaining=Arc::new(AtomicU64::new(4));
        let charge=|budget:Arc<AtomicU64>|move||->Result<()>{budget.try_update(Ordering::AcqRel,Ordering::Acquire,|n|n.checked_sub(1)).map_err(|_|anyhow::anyhow!("exhausted"))?;Ok(())};
        validate_record_fields(&args,charge(remaining.clone())).unwrap();
        assert_eq!(remaining.load(Ordering::Acquire),0);
        assert!(validate_record_fields(&args,charge(remaining.clone())).is_err());
        let invalid=vec![Value::Text("x".into()),Value::List(vec![Value::Number(1.),Value::Bool(true)])];
        let mut charged=0;
        assert!(validate_record_fields(&invalid,||{charged+=1;Ok(())}).is_err());
        assert_eq!(charged,4);
        let remaining=Arc::new(AtomicU64::new(7));
        let handles=(0..2).map(|_|{let args=args.clone();let budget=remaining.clone();std::thread::spawn(move||validate_record_fields(&args,charge(budget)))}).collect::<Vec<_>>();
        let successes=handles.into_iter().map(|h|h.join().unwrap()).filter(Result::is_ok).count();
        assert_eq!(successes,1);assert_eq!(remaining.load(Ordering::Acquire),0);
    }
    #[test]
    fn utf8_text_limit_counts_nested_names_and_is_atomic() {
        let mut budget=10_000_000;
        let mut batch=RecordBatch::default();
        let nested=Value::Object(BTreeMap::from([("s".into(),Value::Text("é".repeat((MAX_TEXT_BYTES-2)/2)))]));
        batch.push(vec![("x".into(),nested)], &mut budget).unwrap();
        assert_eq!(batch.bytes,MAX_TEXT_BYTES);
        let before=batch.clone();
        assert!(batch.push(vec![("x".into(),Value::Object(BTreeMap::from([("s".into(),Value::Text("".into()))])))],&mut budget).is_err());
        assert_eq!(batch,before);
        let mut over=RecordBatch::default();
        assert!(over.push(vec![("x".into(),Value::Text("a".repeat(MAX_TEXT_BYTES)))],&mut budget).is_err());
        assert_eq!(over,RecordBatch::default());
    }
    #[test]
    fn schema_failure_and_budget_failure_never_append_partial_row() {
        let mut budget=100;
        let mut batch=RecordBatch::default();
        batch.push(vec![("x".into(),Value::Number(1.))],&mut budget).unwrap();
        let before=batch.clone();
        assert!(batch.push(vec![("x".into(),Value::Bool(true))],&mut budget).is_err());
        assert_eq!(batch,before);
        assert!(batch.push(vec![("x".into(),Value::Number(2.))],&mut 0).is_err());
        assert_eq!(batch,before);
    }
}
