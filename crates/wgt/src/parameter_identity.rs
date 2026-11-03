use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::collections::{BTreeMap,BTreeSet};

pub fn validate(interface: &Value, shapes: &BTreeMap<String,Vec<u64>>) -> Result<Vec<(String,String)>> {
    let ids = interface["parameter_ids"].as_object().context("missing parameter_ids")?;
    let shares = interface["parameter_shares"].as_object().context("missing parameter_shares")?;
    let mut owners = BTreeSet::new();
    for (name,path) in ids {
        ensure!(crate::manifest::identifier(name), "invalid parameter identity");
        let path = path.as_str().context("invalid parameter path")?;
        ensure!(shapes.contains_key(path) && owners.insert(path), "unknown or duplicate parameter owner");
    }
    let mut pairs = Vec::new();
    for (destination,identity) in shares {
        let identity = identity.as_str().context("invalid share identity")?;
        let owner = ids.get(identity).and_then(Value::as_str).context("unknown shared identity")?;
        ensure!(!owners.contains(destination.as_str()), "shared destination declares an identity");
        ensure!(shapes.get(destination).is_some() && shapes.get(destination) == shapes.get(owner), "shared shape mismatch");
        pairs.push((destination.clone(),owner.to_owned()));
    }
    Ok(pairs)
}

pub fn adapters(interface: &Value, layers: &Value, shapes: &mut BTreeMap<String,Vec<u64>>) -> Result<()> {
    let Some(specs) = interface.get("adapters") else { return Ok(()); };
    let specs = specs.as_object().context("invalid adapters")?;
    let mut targets = BTreeSet::new();
    for (name,spec) in specs {
        ensure!(crate::manifest::identifier(name), "invalid adapter name");
        let fields = spec.as_object().context("invalid adapter")?;
        let method = match spec.get("method") { Some(v) => v.as_str().context("invalid adapter method")?, None => "lora" };
        ensure!(["lora", "dora", "pissa", "qlora"].contains(&method), "unknown adapter method");
        ensure!(fields.len() == if fields.contains_key("method") { 5 } else { 4 }
            && fields.keys().all(|k| ["target", "rank", "alpha", "freeze", "method"].contains(&k.as_str())), "invalid adapter fields");
        ensure!(method == "lora" || spec["freeze"].as_u64() == Some(1), "adapter requires freeze=1");
        let target = spec["target"].as_str().context("missing adapter target")?;
        ensure!(targets.insert(target), "duplicate adapter target");
        let mut scope = layers;
        let mut found = &Value::Null;
        for part in target.split('.') {
            found = scope.as_array().context("invalid target scope")?.iter().find(|l| l["name"].as_str() == Some(part)).context("unknown adapter target")?;
            scope = &found["layers"];
        }
        ensure!(found["op"] == "linear" && found["args"]["tie_embedding"].as_u64().unwrap_or(0) == 0, "LoRA requires untied linear");
        let weight = format!("{target}.weight");
        let shape = shapes.get(&weight).context("missing target weight")?;
        let rank = spec["rank"].as_u64().context("invalid adapter rank")?;
        ensure!(shape.len() == 2 && rank > 0 && rank <= shape[0].min(shape[1]), "invalid adapter rank");
        ensure!(spec["alpha"].as_f64().is_some_and(|x| x.is_finite() && x > 0.0), "invalid alpha");
        ensure!(matches!(spec["freeze"].as_u64(),Some(0|1)), "invalid adapter freeze");
        for (destination,owner) in validate(interface,shapes)? {
            ensure!(destination != weight && owner != weight && destination != format!("{target}.bias") && owner != format!("{target}.bias"), "adapter shared base unsupported");
        }
        let (output,input) = (shape[0],shape[1]);
        shapes.insert(format!("{target}.lora_a"),vec![rank,input]);
        shapes.insert(format!("{target}.lora_b"),vec![output,rank]);
        if method == "dora" {
            shapes.insert(format!("{target}.magnitude"),vec![output]);
        }
    }
    Ok(())
}
