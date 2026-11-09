//! Small, reproducible bridge for the Rust/Python snapshot acceptance test.
//! Run with `--create OUTPUT` or `INPUT OUTPUT`; all outputs must be new paths.
use std::collections::BTreeMap;
use wgt::{
    Result,
    weights::{Tensor, WeightSet},
};

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    anyhow::ensure!(
        args.len() == 2,
        "usage: weights_roundtrip (--create|INPUT) OUTPUT"
    );
    let weights = if args[0] == "--create" {
        WeightSet::new(BTreeMap::from([
            (
                "matrix".into(),
                Tensor::new(vec![2, 2], vec![1., 2., 3., 4.])?,
            ),
            ("bias".into(), Tensor::new(vec![2], vec![5., 6.])?),
        ]))?
    } else {
        WeightSet::open(&args[0], None)?
    };
    weights.save(&args[1])?;
    println!("{}", weights.revision());
    Ok(())
}
