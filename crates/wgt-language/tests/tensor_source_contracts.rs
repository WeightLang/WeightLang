use wgt_language::adaptive::{self, Instruction};

fn model(input: usize, output: usize, body: &str) -> String {
    format!("wgt 0.5\nmodel TensorContracts\ninput x: f32[{input}]\noutput y: f32[{output}]\nfn forward(x) {{\n{body}\n}}\n")
}

#[test]
fn matrix_forms_share_named_axis_rules_and_preserve_result_axes() {
    for product in ["left @ right", "matmul(left,right)"] {
        let source = model(12, 4, &format!(r#"
            let left: f32[rows=2, inner=3] = x[0:6];
            let right: f32[inner=3, cols=2] = x[6:12];
            let product = {product};
            let activated = relu(product);
            let copy = activated;
            return softmax(copy,"cols") as [4];
        "#));
        let program = adaptive::parse(&source).unwrap();
        assert!(program.instructions.iter().any(|instruction|
            matches!(instruction, Instruction::SoftmaxAxis { axis: 1, .. })));
        assert!(adaptive::parse(&source.replace("f32[inner=3, cols=2]", "f32[context=3, cols=2]")).is_err());
        assert!(adaptive::parse(&source.replace("f32[inner=3, cols=2]", "f32[inner=3, rows=2]")).is_err());
    }
}

#[test]
fn batch_product_keeps_a_batch_name_from_either_operand() {
    let source = model(24, 8, r#"
        let left: f32[2, rows=2, inner=3] = x[0:12];
        let right: f32[batch=2, inner=3, cols=2] = x[12:24];
        let product = batch_matmul(left,right);
        return permute(product,[cols,batch,rows]) as [8];
    "#);
    let program = adaptive::parse(&source).unwrap();
    assert!(program.instructions.iter().any(|instruction|
        matches!(instruction, Instruction::Permute { axes, .. } if axes == &[2, 0, 1])));
    adaptive::parse(&source.replace("f32[2, rows=2, inner=3]", "f32[batch=2, rows=2, inner=3]")
        .replace("f32[batch=2, inner=3, cols=2]", "f32[2, inner=3, cols=2]")).unwrap();
}

#[test]
fn elementwise_results_refine_and_preserve_names_through_unary_operations() {
    for expression in ["left + right", "add(left,right)", "left * right", "mul(left,right)", "left - right", "minimum(left,right)"] {
        let source = model(12, 6, &format!(r#"
            let left: f32[rows=2, 3] = x[0:6];
            let right: f32[2, features=3] = x[6:12];
            let combined = {expression};
            let changed = -tanh(combined);
            let copy = changed;
            return softmax(copy,"features") as [6];
        "#));
        adaptive::parse(&source).unwrap();
        assert!(adaptive::parse(&source.replace("f32[2, features=3]", "f32[other_rows=2, features=3]")).is_err());
        assert!(adaptive::parse(&source.replace("f32[2, features=3]", "f32[2, rows=3]")).is_err());
    }
}

#[test]
fn layer_norm_checks_affine_axis_names_and_preserves_the_input_axes() {
    let source = model(6, 6, r#"
        let input: f32[rows=2, features=3] = x;
        let gamma: f32[features=3] = [1,1,1];
        let beta: f32[features=3] = [0,0,0];
        let normalized = layer_norm(input,gamma,beta,"features",0.0001);
        return softmax(normalized,"features") as [6];
    "#);
    adaptive::parse(&source).unwrap();
    for operand in ["gamma", "beta"] {
        assert!(adaptive::parse(&source.replace(&format!("{operand}: f32[features=3]"), &format!("{operand}: f32[channels=3]"))).is_err());
    }
    adaptive::parse(&source.replace("f32[features=3]", "f32[3]")).unwrap();
}

#[test]
fn matrix_operator_does_not_resolve_a_submodel_named_matmul() {
    let source = model(6, 4, r#"
        let left = x as [2,3];
        let right = x as [3,2];
        return (left @ right) as [4];
    "#).replace("model TensorContracts", r#"
        model Block {
            input x: f32[1]
            output y: f32[1]
            fn forward(x) { return x; }
        }
        model TensorContracts
        matmul: Block;
    "#);
    adaptive::parse(&source).unwrap();
}

#[test]
fn repeated_bindings_and_branches_keep_one_register_tensor_type() {
    for binding in [
        "let selected = left; selected = right;",
        "let selected = if positive(x[0:1]) { left } else { right };",
    ] {
        let source = model(12, 6, &format!(r#"
            let left: f32[rows=2, features=3] = x[0:6];
            let right: f32[rows=2, channels=3] = x[6:12];
            {binding}
            return selected as [6];
        "#));
        assert!(adaptive::parse(&source).is_err());
        let compatible = source.replace("f32[rows=2, channels=3]", "f32[2,3]")
            .replace("selected as [6]", "softmax(selected,\"features\") as [6]");
        adaptive::parse(&compatible).unwrap();
    }
}

#[test]
fn gate_and_route_merge_named_probability_vectors_but_keep_scalar_gates() {
    for expression in ["gate(probability,value)", "route(probability,value,other)"] {
        let source = model(6, 6, &format!(r#"
            let probability: f32[rows=2, features=3] = [0.2,0.3,0.4,0.5,0.6,0.7];
            let value: f32[2,3] = x;
            let other: f32[2,3] = x;
            let result = {expression};
            return softmax(result,"features") as [6];
        "#));
        adaptive::parse(&source).unwrap();
        let conflicting = source.replace("value: f32[2,3]", "value: f32[rows=2, channels=3]")
            .replace("softmax(result,\"features\")", "result");
        assert!(adaptive::parse(&conflicting).is_err());
        let scalar = source.replace(
            "probability: f32[rows=2, features=3] = [0.2,0.3,0.4,0.5,0.6,0.7]",
            "probability = [0.5]",
        ).replace("f32[2,3]", "f32[rows=2, features=3]");
        adaptive::parse(&scalar).unwrap();
        assert!(adaptive::parse(&scalar.replace("probability = [0.5]", "probability: f32[1,1] = [0.5]")).is_err());
    }
}
