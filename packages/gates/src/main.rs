use anyhow::{bail, Context, Result};
use std::{env, path::{Path, PathBuf}};
use wgt::{include_model, Client, InstanceOptions, Loss, ModelDefinition, Optimizer, Targets, TrainOptions};

const ROWS: [[f32; 2]; 4] = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]];

fn targets(name: &str) -> Result<Vec<Vec<f32>>> {
    let outputs: [f32; 4] = match name {
        "xor" => [0., 1., 1., 0.],
        "and" => [0., 0., 0., 1.],
        "or" => [0., 1., 1., 1.],
        "nand" => [1., 1., 1., 0.],
        "nor" => [1., 0., 0., 0.],
        "xnor" => [1., 0., 0., 1.],
        _ => bail!("unknown gate {name}; expected xor, and, or, nand, nor, or xnor"),
    };
    Ok(outputs.into_iter().map(|value| vec![value]).collect())
}

fn train(name: &str, definition: &ModelDefinition, output: Option<&Path>) -> Result<()> {
    anyhow::ensure!(definition.name() == format!("{name}_mlp"),
        "model definition {} does not match selected gate {name}", definition.name());
    let targets = targets(name)?;
    let rows: Vec<Vec<f32>> = ROWS.into_iter().map(Vec::from).collect();
    let client = Client::connect()?;
    // 显式初始化创建服务端模型状态；定义解析本身不创建实例或权重。
    let model = client.initialize(
        definition,
        InstanceOptions { memory_bytes: 64 * 1024 * 1024, optimizer: Some(Optimizer::Adam) },
    )?;

    // Rust 控制 1500 次全批次更新；实际模型计算由唯一 wgtime 服务执行。
    let mut loss = f32::NAN;
    for _ in 0..1500 {
        loss = model.train(
            &rows,
            Targets::Values(&targets),
            TrainOptions { rate: 0.05, loss: Loss::Mse, clip: None },
        )?;
    }

    let mut correct = 0;
    for (input, target) in ROWS.iter().zip(&targets) {
        let prediction = model.forward(input)?[0] >= 0.5;
        if prediction == (target[0] >= 0.5) { correct += 1; }
    }
    anyhow::ensure!(correct == 4, "{name} truth table failed: {correct}/4");
    anyhow::ensure!(model.backend() == "native-cpu", "gate requires native compiled execution");
    let stats = client.stats()?;
    let execution = &stats["execution"][0];
    anyhow::ensure!(execution["native_tape_forwards"].as_u64() == Some(6000), "gate training did not execute every sample natively");
    println!("{name}: backend={} native_tape_forwards={} truth_table_correct={correct}/4 final_mse={loss}", model.backend(), execution["native_tape_forwards"]);
    if let Some(path) = output {
        // 文件发布仅由调用者显式请求；save_weights 默认拒绝覆盖。
        model.save_weights(path)?;
    }
    model.close()
}

fn main() -> Result<()> {
    // 宏编译期验证并嵌入源码；构造 ModelDefinition 仍是运行期纯解析。
    let gates = [
        ("xor", include_model!("xor/model.wgt")),
        ("and", include_model!("and/model.wgt")),
        ("or", include_model!("or/model.wgt")),
        ("nand", include_model!("nand/model.wgt")),
        ("nor", include_model!("nor/model.wgt")),
        ("xnor", include_model!("xnor/model.wgt")),
    ];
    let mut args = env::args().skip(1);
    let selected = args.next().unwrap_or_else(|| "all".into());
    let mut runtime_definition: Option<ModelDefinition> = None;
    let mut output: Option<PathBuf> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-file" => {
                let path = args.next().context("--model-file requires a WGT path")?;
                runtime_definition = Some(ModelDefinition::from_file(&path).with_context(|| format!("loading model definition {path}"))?);
            }
            "--output" => output = Some(PathBuf::from(args.next().context("--output requires a file path")?)),
            _ => bail!("usage: gates [all|xor|and|or|nand|nor|xnor] [--model-file model.wgt] [--output weights.safetensors]"),
        }
    }
    let selected_gates: Vec<_> = if selected == "all" {
        gates.iter().collect()
    } else {
        vec![gates.iter().find(|(name, _)| *name == selected)
            .with_context(|| format!("unknown gate {selected}; expected all, xor, and, or, nand, nor, or xnor"))?]
    };
    anyhow::ensure!(runtime_definition.is_none() || selected_gates.len() == 1,
        "--model-file can only be used with one selected gate");
    anyhow::ensure!(output.is_none() || selected_gates.len() == 1,
        "--output can only be used with one selected gate");
    for (name, embedded) in selected_gates {
        let definition = runtime_definition.as_ref().unwrap_or(embedded);
        train(name, definition, output.as_deref())?;
    }
    Ok(())
}
