use crate::{record_schema::{self,BatchSchema},Runtime,Value};
use anyhow::{ensure,Context,Result};
use serde_json::Value as Json;
#[derive(Clone,Debug,Default)]
pub struct RecordBatch {pub(crate) schema:BatchSchema,pub(crate) rows:Vec<Json>,bytes:usize}
impl RecordBatch {pub fn rows_len(&self)->usize{self.rows.len()}}
impl Runtime {
    pub fn yield_record(&mut self,env:&mut crate::Env,fields:Vec<(String,Value)>)->Result<()> {
        let mut shapes=Vec::new();let mut bytes=0usize;let mut nodes=0;
        for(name,value)in &fields{
            let(shape,size)=record_schema::inspect(value.data()?,0,&mut||self.tick(),&mut nodes)?;
            bytes=bytes.checked_add(name.len()).and_then(|n|n.checked_add(size)).context("record size overflow")?;
            shapes.push((name.clone(),shape));
        }
        let Value::RecordBatch(batch)=env.get("__collect").context("yield requires collect")? else {anyhow::bail!("invalid collector")};
        let mut candidate=batch.schema.clone();candidate.accept(shapes).map_err(anyhow::Error::msg)?;
        ensure!(batch.bytes.checked_add(bytes).is_some_and(|n|n<=record_schema::MAX_TEXT_BYTES),"record text limit exceeded");
        let row=Json::Object(fields.into_iter().map(|(name,value)|Ok((name,value.data()?.clone()))).collect::<Result<_>>()?);
        let Value::RecordBatch(batch)=env.get_mut("__collect").unwrap() else {unreachable!()};
        batch.schema=candidate;batch.rows.push(row);batch.bytes+=bytes;Ok(())
    }
    pub fn record_chunk(&mut self,batch:&RecordBatch,range:std::ops::Range<usize>)->Result<Value>{
        let mut result=RecordBatch::default();
        for row in &batch.rows[range]{
            let mut nodes=0;let(shape,bytes)=record_schema::inspect(row,0,&mut||self.tick(),&mut nodes)?;
            let record_schema::Shape::Record(fields)=shape else {unreachable!()};
            result.schema.accept(fields).map_err(anyhow::Error::msg)?;result.bytes+=bytes;result.rows.push(row.clone());
        }Ok(Value::RecordBatch(result))
    }
}
pub fn index(value:Value,index:Value)->Result<Value>{match value{Value::RecordBatch(batch)=>Ok(Value::Data(batch.rows.get(index.integer()?).context("record index out of bounds")?.clone())),Value::Data(value)=>{let item=if let Some(key)=index.data()?.as_str(){value.get(key)}else{value.get(index.integer()?)}.context("index out of bounds")?;Ok(Value::Data(item.clone()))},_=>anyhow::bail!("index requires list, record or record batch")}}
pub fn field(value:&Value,name:&str)->Result<Value>{match value{Value::RecordBatch(batch)=>Ok(Value::Data(Json::Array(batch.rows.iter().map(|row|row.get(name).cloned().context("missing record field")).collect::<Result<_>>()?))),_=>Ok(Value::Data(value.data()?.get(name).context("missing record field")?.clone()))}}
