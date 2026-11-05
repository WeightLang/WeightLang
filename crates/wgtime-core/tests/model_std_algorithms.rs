//! Explicit std reuse through a closed ModelDefinition, with independent f64 oracles.
use std::collections::BTreeMap;
use wgt_language::model::{ModelDefinition, ModelPlan};
use wgtime_core::adaptive::AdaptiveSession;

const MEMORY: usize = 2 * 1024 * 1024;
const WORK: u64 = 100_000;
type Weights = BTreeMap<String, Vec<f64>>;
type Oracle = fn(&[f64], &Weights) -> Vec<f64>;

const ATTENTION: &str = r#"wgt 0.5
model ExplicitAttention
import attention from "std/attention"
input x: f32[4]
output y: f32[4]
# 单个输入向量显式恢复为 [查询位置=2,点积通道=2]；没有隐含 batch。
# 键为 [键位置=3,通道=2]，值为 [键位置=3,值通道=2]。
# 初始化值不是已训练权重；解析定义本身不分配运行时参数。
parameter keys: f32[3,2] = [0.3,-0.2,0.6,0.4,-0.5,0.7]
parameter values: f32[3,2] = [0.8,-0.3,-0.4,0.9,0.2,0.5]
fn forward(x) {
    # 调用方提供字面量 1/sqrt(2)，库不推断缩放或依赖标量广播。
    let q = scale(reshape(x,[2,2]),0.7071067811865476);
    return reshape(attention.dot_product_attention(q,reshape(keys,[3,2]),reshape(values,[3,2])),[4]);
}
"#;

const RELAX: &str = r#"wgt 0.5
model ExplicitRelax
import numerics from "std/numerics"
input x: f32[3]
output y: f32[3]
# 同一特征轴逐元素更新；步长包含负数和大于 1 的值，不能替换成 gate。
# 两次调用共享同一 gain 参数，检验共享路径的梯度累加。
parameter observed: f32[3] = [0.7,-0.4,0.6]
parameter gain: f32[3] = [-0.2,0.35,1.2]
fn forward(x) {
    let first = numerics.relax(x,observed,gain);
    return numerics.relax(first,observed,gain);
}
"#;

fn session(source: &str, compiled: bool) -> AdaptiveSession {
    let definition = ModelDefinition::parse(source).expect("explicit std import must resolve");
    assert!(
        !definition
            .source()
            .lines()
            .any(|line| line.trim_start().starts_with("import "))
    );
    // A serialized definition is closed: a second parse needs no import lookup.
    let embedded = serde_json::to_string(&definition).unwrap();
    let restored: ModelDefinition = serde_json::from_str(&embedded).unwrap();
    assert_eq!(restored.source(), definition.source());
    let ModelPlan::Forward(program) = restored.instantiate_plan().unwrap() else {
        panic!("test needs a forward model");
    };
    let mut session = AdaptiveSession::prepare(program, MEMORY).unwrap();
    if compiled {
        session.enable_compiled().unwrap();
    }
    assert_eq!(session.is_compiled(), compiled);
    assert_eq!(
        session.execution_backend(),
        if compiled {
            "native-cpu"
        } else {
            "reference-interpreter"
        }
    );
    session
}

fn attention_oracle(x: &[f64], weights: &Weights) -> Vec<f64> {
    let keys = &weights["keys"];
    let values = &weights["values"];
    let mut output = vec![0.0; 4];
    // Independent row-major scalar loops, with stable normalization over Tk.
    for query in 0..2 {
        let mut scores = [0.0; 3];
        for key in 0..3 {
            for channel in 0..2 {
                scores[key] += x[query * 2 + channel]
                    * std::f64::consts::FRAC_1_SQRT_2
                    * keys[key * 2 + channel];
            }
        }
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denominator: f64 = scores.iter().map(|s| (s - maximum).exp()).sum();
        for key in 0..3 {
            let probability = (scores[key] - maximum).exp() / denominator;
            for channel in 0..2 {
                output[query * 2 + channel] += probability * values[key * 2 + channel];
            }
        }
    }
    output
}

fn relax_oracle(x: &[f64], weights: &Weights) -> Vec<f64> {
    x.iter()
        .enumerate()
        .map(|(i, &estimate)| {
            let gain = weights["gain"][i];
            let observation = weights["observed"][i];
            let first = estimate + gain * (observation - estimate);
            first + gain * (observation - first)
        })
        .collect()
}

fn close(actual: &[f32], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (f64::from(actual) - expected).abs() < tolerance,
            "element {index}: {actual} vs {expected}"
        );
    }
}

fn dot(output: Vec<f64>, seed: &[f32]) -> f64 {
    output
        .iter()
        .zip(seed)
        .map(|(value, &seed)| value * f64::from(seed))
        .sum()
}

fn check_vjp(source: &str, input: &[f32], names: &[&str], oracle: Oracle) {
    let input64: Vec<_> = input.iter().map(|&value| f64::from(value)).collect();
    for compiled in [false, true] {
        let mut model = session(source, compiled);
        let weights: Weights = names
            .iter()
            .map(|&name| {
                (
                    name.to_owned(),
                    model
                        .parameter(name)
                        .unwrap()
                        .iter()
                        .map(|&value| f64::from(value))
                        .collect(),
                )
            })
            .collect();
        let expected = oracle(&input64, &weights);
        close(&model.evaluate(input, WORK).unwrap(), &expected, 2e-6);
        let seed: Vec<_> = (0..expected.len())
            .map(|i| if i % 2 == 0 { 0.7 } else { -0.4 })
            .collect();
        let mut pass = model.forward(input, WORK, MEMORY).unwrap();
        pass.backward(&seed, &BTreeMap::new()).unwrap();
        close(pass.output(), &expected, 2e-6);
        // Central differences use only the scalar oracle, never runtime tensor ops.
        let epsilon = 1e-5;
        for &name in names {
            for index in 0..weights[name].len() {
                let mut plus = weights.clone();
                let mut minus = weights.clone();
                plus.get_mut(name).unwrap()[index] += epsilon;
                minus.get_mut(name).unwrap()[index] -= epsilon;
                let numerical = (dot(oracle(&input64, &plus), &seed)
                    - dot(oracle(&input64, &minus), &seed))
                    / (2.0 * epsilon);
                let actual = pass.parameter_gradient(name).unwrap()[index];
                assert!(
                    (f64::from(actual) - numerical).abs() < 3e-5,
                    "compiled={compiled}, {name}[{index}]: {actual} vs {numerical}"
                );
            }
        }
        for index in 0..input64.len() {
            let mut plus = input64.clone();
            let mut minus = input64.clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numerical = (dot(oracle(&plus, &weights), &seed)
                - dot(oracle(&minus, &weights), &seed))
                / (2.0 * epsilon);
            let actual = pass.input_gradient().unwrap()[index];
            assert!(
                (f64::from(actual) - numerical).abs() < 3e-5,
                "compiled={compiled}, input[{index}]: {actual} vs {numerical}"
            );
        }
        assert_eq!(
            model.compiled_forward_counts(),
            if compiled { (2, 1) } else { (0, 0) }
        );
    }
}

#[test]
fn explicit_attention_forward_and_native_gradients() {
    check_vjp(
        ATTENTION,
        &[0.4, -0.6, 0.8, 0.1],
        &["keys", "values"],
        attention_oracle,
    );
}

#[test]
fn explicit_relax_shared_gain_forward_and_native_gradients() {
    check_vjp(
        RELAX,
        &[0.2, -0.3, 0.9],
        &["observed", "gain"],
        relax_oracle,
    );
}

#[test]
fn relax_supports_explicit_scalar_and_matching_matrix_shapes() {
    for (width, shape, input, observation, gain, expected) in [
        (1, "[1]", vec![2.0], "[6]", "[1.5]", vec![8.0]),
        (
            4,
            "[2,2]",
            vec![1.0, 2.0, 3.0, 4.0],
            "[3,4,5,6]",
            "[0,0.5,1,1.5]",
            vec![1.0, 3.0, 5.0, 7.0],
        ),
    ] {
        let source = format!(
            r#"wgt 0.5
model MatchingRelax
import numerics from "std/numerics"
input x: f32[{width}]
output y: f32[{width}]
# 同形标量或矩阵；reshape 显式保存逐轴对应关系，不广播。
parameter observed: f32{shape} = {observation}
parameter gain: f32{shape} = {gain}
fn forward(x) {{
    return reshape(numerics.relax(reshape(x,{shape}),reshape(observed,{shape}),reshape(gain,{shape})),[{width}]);
}}
"#
        );
        for compiled in [false, true] {
            close(
                &session(&source, compiled).evaluate(&input, WORK).unwrap(),
                &expected,
                1e-6,
            );
        }
    }
}

#[test]
fn relax_rejects_scalar_gain_broadcast() {
    let source = RELAX.replace("gain: f32[3] = [-0.2,0.35,1.2]", "gain: f32[1] = [0.5]");
    assert!(
        ModelDefinition::parse(&source).is_err(),
        "scalar gain must not broadcast to three features"
    );
}

#[test]
fn algorithms_require_explicit_imports() {
    for (source, declaration, qualified, bare) in [
        (
            ATTENTION,
            "import attention from \"std/attention\"\n",
            "attention.dot_product_attention",
            "dot_product_attention",
        ),
        (
            RELAX,
            "import numerics from \"std/numerics\"\n",
            "numerics.relax",
            "relax",
        ),
    ] {
        let no_import = source.replace(declaration, "");
        assert!(
            ModelDefinition::parse(&no_import).is_err(),
            "qualified algorithm must need import"
        );
        assert!(
            ModelDefinition::parse(&no_import.replace(qualified, bare)).is_err(),
            "algorithm must not become a builtin or preimported preset"
        );
        assert!(
            ModelDefinition::parse(&source.replace(qualified, bare)).is_err(),
            "explicit import exposes the qualified name only"
        );
    }
}
