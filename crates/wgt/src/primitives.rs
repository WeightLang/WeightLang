//! Native operator signature catalog and shape inference.
use crate::contracts::TensorType;
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::sync::OnceLock;

fn specs() -> &'static Value {
    static SPECS: OnceLock<Value> = OnceLock::new();
    SPECS.get_or_init(|| serde_json::from_str(include_str!("../../../language/primitive_specs.json")).expect("checked operator catalog"))
}
pub fn contains(op: &str) -> bool { specs().get(op).is_some() }
pub fn infer(op: &str, args: &Value, operands: &[TensorType]) -> Result<TensorType> {
    let spec = &specs()[op];
    ensure!(operands.len() as u64 == spec["arity"].as_u64().context("missing arity")?, "wrong primitive input count");
    let args = args.as_object().context("primitive arguments must be an object")?;
    let defaults = spec["defaults"].as_object().unwrap();
    let required = spec["required"].as_array().unwrap();
    ensure!(args.keys().all(|k| defaults.contains_key(k) || required.iter().any(|v| v.as_str() == Some(k))), "unknown primitive argument");
    ensure!(required.iter().all(|k| args.contains_key(k.as_str().unwrap())), "missing primitive argument");
    for (key, value) in args {
        if spec["types"][key] == "float" {
            ensure!(value.as_f64().is_some_and(f64::is_finite), "primitive argument must be a finite number");
        } else {
            ensure!(value.as_i64().is_some(), "primitive argument must be an integer");
        }
    }
    let arg = |key: &str| -> Result<i64> { args.get(key).or_else(|| defaults.get(key)).and_then(Value::as_i64).context("missing integer argument") };
    let x = &operands[0];
    let mut out = x.clone();
    let rank = x.shape.len();
    let axis = |key: &str, insert: bool| -> Result<usize> {
        let n = rank as i64 + i64::from(insert);
        let a = arg(key)?;
        ensure!(n > 0 && -n <= a && a < n, "axis outside sample rank");
        Ok(a.rem_euclid(n) as usize)
    };
    match op {
        "square" | "rsqrt" | "exp" | "log" | "sin" | "cos" | "softplus" | "shift" | "cumsum" => {
            ensure!(x.dtype == "f32", "tensor math expects f32");
            if op == "cumsum" { axis("axis", false)?; }
        }
        "split_axis" => {
            let a = axis("axis", false)?;
            let parts = arg("parts")?;
            ensure!(parts > 0 && x.shape[a] % parts as u64 == 0, "parts must be positive and divide the selected axis");
            out.shape[a] = parts as u64;
            out.shape.insert(a + 1, x.shape[a] / parts as u64);
        }
        "merge_axes" => {
            let a = axis("axis", false)?;
            ensure!(a + 1 < rank, "merge requires two adjacent sample axes");
            out.shape[a] = x.shape[a].checked_mul(x.shape[a + 1]).context("merged dimension overflow")?;
            out.shape.remove(a + 1);
        }
        "repeat_interleave" => {
            let a = axis("axis", false)?;
            let repeats = arg("repeats")?;
            ensure!(repeats > 0, "repeats must be positive");
            out.shape[a] = x.shape[a].checked_mul(repeats as u64).context("repeated dimension overflow")?;
        }
        "resize2d" | "pad2d" | "avgpool2d" => {
            ensure!(x.dtype == "f32" && rank == 3 && x.shape.iter().all(|v| *v > 0), "spatial operator expects nonempty f32[C,H,W]");
            if op == "resize2d" {
                let (h,w,mode,align) = (arg("height")?,arg("width")?,arg("mode")?,arg("align_corners")?);
                ensure!(h > 0 && w > 0 && (0..=1).contains(&mode) && (0..=1).contains(&align) && (mode == 1 || align == 0), "invalid resize2d configuration");
                out.shape[1] = h as u64; out.shape[2] = w as u64;
            } else if op == "pad2d" {
                let (l,r,t,b,mode) = (arg("left")?,arg("right")?,arg("top")?,arg("bottom")?,arg("mode")?);
                ensure!([l,r,t,b].iter().all(|v| *v >= 0) && (0..=2).contains(&mode), "invalid pad2d configuration");
                ensure!(mode != 1 || ((l.max(r) as u64) < x.shape[2] && (t.max(b) as u64) < x.shape[1]), "reflection padding exceeds input");
                out.shape[1] = out.shape[1].checked_add(t as u64).and_then(|v| v.checked_add(b as u64)).context("padding overflow")?;
                out.shape[2] = out.shape[2].checked_add(l as u64).and_then(|v| v.checked_add(r as u64)).context("padding overflow")?;
            } else {
                let (k,s,p,include) = (arg("kernel")?,arg("stride")?,arg("padding")?,arg("count_include_pad")?);
                ensure!(k > 0 && s > 0 && p >= 0 && p <= k/2 && (0..=1).contains(&include), "invalid avgpool2d configuration");
                for i in 1..3 {
                    let padded = x.shape[i].checked_add(2*p as u64).context("pooling overflow")?;
                    ensure!(padded >= k as u64, "pooling produces empty output");
                    out.shape[i] = (padded-k as u64)/s as u64+1;
                }
            }
        }
        "matmul" => {
            let y = &operands[1];
            ensure!(x.dtype == "f32" && y.dtype == "f32" && rank >= 2 && y.shape.len() == rank, "matmul requires same-rank f32 matrices");
            ensure!(x.shape[..rank-2] == y.shape[..rank-2] && x.shape[rank-1] == y.shape[rank-2], "matmul contraction/prefix mismatch");
            out.shape[rank-1] = y.shape[rank-1];
        }
        "broadcast" => {
            let y = &operands[1];
            ensure!(rank <= y.shape.len(), "broadcast source rank exceeds target");
            ensure!(x.shape.iter().rev().zip(y.shape.iter().rev()).all(|(a,b)| a == b || *a == 1), "invalid broadcast expansion");
            out.shape = y.shape.clone();
        }
        "cat" | "stack" => {
            let y = &operands[1];
            ensure!(x.dtype == y.dtype && rank == y.shape.len(), "cat/stack rank/dtype mismatch");
            let a = axis("axis", op == "stack")?;
            if op == "stack" {
                ensure!(x.shape == y.shape, "stack shape mismatch");
                out.shape.insert(a,2);
            } else {
                ensure!(x.shape.iter().zip(&y.shape).enumerate().all(|(i,(x,y))| i == a || x == y), "cat shape mismatch");
                out.shape[a] = out.shape[a].checked_add(y.shape[a]).context("cat size overflow")?;
            }
        }
        "slice" => {
            let a = axis("axis",false)?;
            let (start,stop,step) = (arg("start")?,arg("stop")?,arg("step")?);
            ensure!(0 <= start && start <= stop && stop as u64 <= x.shape[a] && step > 0, "invalid slice bounds/step");
            out.shape[a] = ((stop-start) / step + i64::from((stop-start) % step != 0)) as u64;
        }
        "transpose_axes" => { out.shape.swap(axis("axis1",false)?,axis("axis2",false)?); }
        "reduce_sum" | "reduce_mean" | "reduce_max" => {
            ensure!(x.dtype == "f32", "reduction expects f32");
            let a = axis("axis",false)?;
            let keep = arg("keepdim")?;
            ensure!(keep == 0 || keep == 1, "keepdim must be 0 or 1");
            ensure!(op == "reduce_sum" || x.shape[a] > 0, "mean/max require nonempty reduction axis");
            if keep == 1 { out.shape[a] = 1; } else { out.shape.remove(a); }
        }
        "gather" | "scatter_add" => {
            let selected = axis("axis",false)?;
            let y = &operands[1];
            ensure!(y.dtype == "i64" && y.shape.len() == rank, "indices require i64 and matching rank");
            ensure!(y.shape.iter().zip(&x.shape).enumerate().all(|(i,(a,b))| i == selected || a <= b), "index dimensions exceed input");
            if op == "gather" { out.shape = y.shape.clone(); }
            else { ensure!(x.dtype == "f32" && operands[2].dtype == "f32" && operands[2].shape == y.shape, "scatter_add source mismatch"); }
        }
        "eq" | "lt" => {
            ensure!(x.dtype == operands[1].dtype && x.shape == operands[1].shape, "comparison requires matching shape/dtype");
            ensure!(op != "lt" || x.dtype != "bool", "lt expects numeric inputs");
            out.dtype = "bool".into();
        }
        "masked_softmax" => {
            axis("axis",false)?;
            ensure!(x.dtype == "f32" && operands[1].dtype == "bool" && x.shape == operands[1].shape, "masked_softmax mask mismatch");
        }
        _ => anyhow::bail!("unknown tensor primitive"),
    }
    out.axes = vec![None; out.shape.len()];
    Ok(out)
}
