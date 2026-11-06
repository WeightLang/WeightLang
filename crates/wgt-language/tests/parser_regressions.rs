use wgt_language::script::parse::{parse_functions, Expr, Stmt};

fn returned(source: &str) -> Expr {
    let program = parse_functions(source).expect("source should parse");
    let function = program.get("value").expect("value function");
    let statement = function.body.last().expect("return statement");
    match &statement.kind {
        Stmt::Return(expression) => expression.clone(),
        other => panic!("expected return, got {other:?}"),
    }
}

#[test]
fn version_header_accepts_surrounding_whitespace() {
    parse_functions("  wgt 0.5  \nfn value() { return 1; }").unwrap();
}

#[test]
fn exponentiation_is_right_associative() {
    let expression = returned("wgt 0.5\nfn value() { return 2 ** 3 ** 2; }");
    let Expr::Binary(op, left, right) = expression else {
        panic!("expected exponentiation");
    };
    assert_eq!(op, "**");
    assert!(matches!(*left, Expr::Number(2.0)));
    let Expr::Binary(right_op, middle, last) = *right else {
        panic!("right operand must contain the second exponentiation");
    };
    assert_eq!(right_op, "**");
    assert!(matches!(*middle, Expr::Number(3.0)));
    assert!(matches!(*last, Expr::Number(2.0)));
}

#[test]
fn exponentiation_binds_tighter_than_unary_minus() {
    let expression = returned("wgt 0.5\nfn value() { return -2 ** 2; }");
    let Expr::Unary(op, inner) = expression else {
        panic!("expected unary minus");
    };
    assert_eq!(op, "-");
    assert!(matches!(*inner, Expr::Binary(ref op, _, _) if op == "**"));
}

#[test]
fn lexer_enforces_token_limit_during_scanning() {
    let source = format!("wgt 0.5\n{}", "+".repeat(100_001));
    let error = parse_functions(&source).expect_err("token limit must reject oversized input");
    assert!(error.to_string().contains("script token limit exceeded"));
}

#[test]
fn lexer_enforces_source_byte_limit() {
    let source = format!("wgt 0.5\n#{}", "x".repeat(4 * 1024 * 1024));
    let error = parse_functions(&source).expect_err("source byte limit must reject oversized input");
    assert!(error.to_string().contains("script source exceeds 4 MiB limit"));
}

#[test]
fn source_split_enforces_source_byte_limit_before_scanning() {
    let source = format!("wgt 0.5\n#{}", "x".repeat(4 * 1024 * 1024));
    let error = match wgt_language::source::split(&source) {
        Ok(_) => panic!("source splitting must reject oversized input"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("WGT source exceeds 4 MiB limit"));
}

#[test]
fn matrix_product_uses_native_at_operator() {
    let expression = returned(
        "wgt 0.5\nfn value(q,k) { return q @ k; }"
    );
    assert!(matches!(
        expression,
        Expr::Binary(ref op, ref left, ref right)
            if op == "@"
                && matches!(&**left, Expr::Name(name) if name == "q")
                && matches!(&**right, Expr::Name(name) if name == "k")
    ));
}

#[test]
fn transpose_uses_native_t_postfix() {
    let expression = returned(
        "wgt 0.5\nfn value(k) { return k.T; }"
    );
    assert!(matches!(
        expression,
        Expr::BuiltinCall(ref op, ref args)
            if op == "transpose"
                && matches!(args.as_slice(), [Expr::Name(name)] if name == "k")
    ));
}

#[test]
fn matrix_product_precedes_addition() {
    let expression = returned(
        "wgt 0.5\nfn value(a,b,c) { return a + b @ c; }"
    );
    assert!(matches!(
        expression,
        Expr::Binary(ref op, _, ref right)
            if op == "+"
                && matches!(&**right, Expr::Binary(inner, _, _) if inner == "@")
    ));
}

#[test]
fn matrix_syntax_lowers_to_native_tensor_ir() {
    let source = r#"wgt 0.5
model MatrixSyntax
input x: f32[6]
output y: f32[1]
parameter projection: f32[3,2] = [1,0,0,1,1,1]
fn forward(x) {
    let tokens = reshape(x,[2,3]);
    let projection_matrix = reshape(projection,[3,2]);
    let q = tokens @ projection_matrix;
    let k = q.T;
    let scores = q @ k;
    return scores.mean;
}
"#;
    wgt_language::adaptive::parse(source)
        .expect("native @ and .T syntax should lower to matrix IR");
}

#[test]
fn matrix_operator_is_allowed_in_pure_tensor_helpers() {
    let source = r#"wgt 0.5
model MatrixHelper
input x: f32[6]
output y: f32[1]
parameter projection: f32[3,2] = [1,0,0,1,1,1]
fn project(a,b) {
    return a @ b;
}
fn forward(x) {
    let tokens = reshape(x,[2,3]);
    let projection_matrix = reshape(projection,[3,2]);
    let q = project(tokens,projection_matrix);
    return (q @ q.T).mean;
}
"#;
    wgt_language::model::ModelDefinition::parse(source)
        .expect("pure tensor helpers should accept native matrix products");
}

#[test]
fn local_tensor_annotations_validate_shape_and_named_axes() {
    let source = r#"wgt 0.5
model NamedAxes
input x: f32[6]
output y: f32[1]
fn forward(x) {
    let tokens: f32[sequence=2, hidden=3] = x;
    let projection: f32[hidden=3, channel=2] = reshape(x,[3,2]);
    let q = tokens @ projection;
    return q.mean;
}
"#;
    wgt_language::adaptive::parse(source)
        .expect("annotated shape and named matrix axes should type-check");
}

#[test]
fn matrix_products_reject_mismatched_named_contracting_axes() {
    let source = r#"wgt 0.5
model BadNamedAxes
input x: f32[6]
output y: f32[1]
fn forward(x) {
    let left: f32[sequence=2, hidden=3] = reshape(x,[2,3]);
    let right: f32[context=3, features=2] = reshape(x,[3,2]);
    let invalid = left @ right;
    return invalid.mean;
}
"#;
    let error = wgt_language::adaptive::parse(source)
        .expect_err("different named contracting axes must be rejected");
    assert!(format!("{error:#}").contains("matrix product axis mismatch"));
}

#[test]
fn named_softmax_axis_resolves_from_tensor_annotation() {
    let source = r#"wgt 0.5
model NamedSoftmaxAxis
input x: f32[6]
output y: f32[1]
fn forward(x) {
    let scores: f32[queries=2, keys=3] = reshape(x,[2,3]);
    let probabilities = softmax(scores,"keys");
    return probabilities.mean;
}
"#;
    wgt_language::adaptive::parse(source)
        .expect("softmax should accept an annotated named axis");
}

#[test]
fn named_axis_permute_lowers_to_geometry_ir() {
    let source = r#"wgt 0.5
model NamedPermute
input x: f32[24]
output y: f32[1]
fn forward(x) {
    let tensor: f32[sequence=2, heads=3, dim=4] = x;
    let reordered = permute(tensor,[dim,sequence,heads]);
    return reordered.mean;
}
"#;
    let program = wgt_language::adaptive::parse(source)
        .expect("named-axis permutation should lower");
    assert!(program.version >= 3);
    assert!(program.instructions.iter().any(|instruction|
        matches!(instruction, wgt_language::adaptive::Instruction::Permute { axes, .. }
            if axes == &[2,0,1])));
}

#[test]
fn permute_rejects_duplicate_axes() {
    let source = r#"wgt 0.5
model DuplicatePermute
input x: f32[24]
output y: f32[1]
fn forward(x) {
    let tensor: f32[a=2, b=3, c=4] = reshape(x,[2,3,4]);
    let invalid = permute(tensor,[a,a,c]);
    return invalid.mean;
}
"#;
    let error = wgt_language::adaptive::parse(source)
        .expect_err("permutation axes must be unique");
    assert!(format!("{error:#}").contains("permute axes must form a bijection"));
}

#[test]
fn batch_matmul_checks_batch_and_contracting_axes() {
    let source = r#"wgt 0.5
model BatchMatmul
input x: f32[24]
output y: f32[1]
fn forward(x) {
    let left: f32[batch=2, rows=2, inner=3] = x[0:12];
    let right: f32[batch=2, inner=3, cols=2] = x[12:24];
    let product = batch_matmul(left,right);
    return product.mean;
}
"#;
    let program = wgt_language::adaptive::parse(source)
        .expect("rank-3 batched matrix multiplication should lower");
    assert!(program.instructions.iter().any(|instruction|
        matches!(instruction, wgt_language::adaptive::Instruction::BatchMatmul { .. })));
}

#[test]
fn batch_matmul_rejects_batch_or_contracting_axis_mismatch() {
    let source = r#"wgt 0.5
model BadBatchMatmul
input x: f32[24]
output y: f32[1]
fn forward(x) {
    let left: f32[batch=2, rows=2, inner=3] = x[0:12];
    let right: f32[other_batch=2, inner=3, cols=2] = x[12:24];
    let product = batch_matmul(left,right);
    return product.mean;
}
"#;
    let error = wgt_language::adaptive::parse(source)
        .expect_err("different named batch axes must be rejected");
    assert!(format!("{error:#}").contains("batch_matmul batch axis mismatch"));
}

#[test]
fn layer_norm_accepts_named_axis_and_preserves_shape() {
    let source = r#"wgt 0.5
model NamedLayerNorm
input x: f32[6]
output y: f32[6]
parameter gamma: f32[3] = [1,1,1]
parameter beta: f32[3] = [0,0,0]
fn forward(x) {
    let tokens: f32[sequence=2, hidden=3] = x;
    let normalized = layer_norm(tokens,gamma,beta,"hidden",0.0001);
    let flat: f32[6] = normalized;
    return flat;
}
"#;
    let program = wgt_language::adaptive::parse(source)
        .expect("LayerNorm should lower with named axis");
    assert!(program.instructions.iter().any(|instruction|
        matches!(instruction, wgt_language::adaptive::Instruction::LayerNorm { axis: 1, .. })));
}

#[test]
fn pure_helpers_compose_batched_geometry_and_layer_norm() {
    let source = r#"wgt 0.5
model GeometryHelper
input x: f32[8]
output y: f32[2]
parameter gamma: f32[2] = [1,1]
parameter beta: f32[2] = [0,0]
fn transform(left,right,gamma,beta) {
    let product = batch_matmul(left,right);
    let reordered = permute(product,[1,0,2]);
    return layer_norm(reordered,gamma,beta,1,0.0001);
}
fn forward(x) {
    # Two independent batches each contract two features into one value.
    let left: f32[batch=2, row=1, feature=2] = x[0:4];
    let right: f32[batch=2, feature=2, column=1] = x[4:8];
    return reshape(transform(left,right,gamma,beta),[2]);
}
"#;
    let definition = wgt_language::model::ModelDefinition::parse(source).unwrap();
    let wgt_language::model::ModelPlan::Forward(program) = definition.instantiate_plan().unwrap() else {
        panic!("expected tensor forward");
    };
    use wgt_language::adaptive::Instruction;
    assert!(program.instructions.iter().any(|i| matches!(i, Instruction::BatchMatmul { .. })));
    assert!(program.instructions.iter().any(|i| matches!(i, Instruction::Permute { axes, .. } if axes == &[1,0,2])));
    assert!(program.instructions.iter().any(|i| matches!(i, Instruction::LayerNorm { axis: 1, .. })));
}

#[test]
fn pure_helpers_lower_pooling_and_softmax_overloads() {
    let source = r#"wgt 0.5
model PoolHelper
input x: f32[4]
output y: f32[1]
fn pool(image) {
    let maximum = max_pool2d(image,[2,2],[1,1]);
    let average = avg_pool2d(image,[2,2],[1,1]);
    return maximum + average;
}
fn normalize(x) {
    return softmax(softmax(x),0);
}
fn forward(x) {
    # Pool one channel over its entire 2 x 2 spatial extent.
    let image: f32[channel=1, height=2, width=2] = x;
    return normalize(reshape(pool(image),[1]));
}
"#;
    let definition = wgt_language::model::ModelDefinition::parse(source).unwrap();
    let wgt_language::model::ModelPlan::Forward(program) = definition.instantiate_plan().unwrap() else {
        panic!("expected tensor forward");
    };
    use wgt_language::adaptive::Instruction;
    assert!(program.instructions.iter().any(|i| matches!(i, Instruction::MaxPool2d { .. })));
    assert!(program.instructions.iter().any(|i| matches!(i, Instruction::AvgPool2d { .. })));
}

#[test]
fn pure_helper_contracts_reject_wrong_arity_and_stateful_calls() {
    for call in [
        "permute(x)",
        "batch_matmul(x)",
        "layer_norm(x,x,x,0)",
        "max_pool2d(x,[2,2])",
        "avg_pool2d(x,[2,2],[1,1],0)",
        "softmax(x,0,0)",
    ] {
        let source = format!("wgt 0.5\noperators Contract\nfn helper(x) {{ return {call}; }}");
        let error = wgt_language::script::operators::library(&source).unwrap_err();
        assert!(error.to_string().contains("tensor primitive arity mismatch"), "{call}: {error:#}");
    }
    for call in ["read(x)", "linear(x,x)", "memory_read(x,x)", "write(x,x)"] {
        let source = format!("wgt 0.5\noperators Contract\nfn helper(x) {{ return {call}; }}");
        let error = wgt_language::script::operators::library(&source).unwrap_err();
        assert!(error.to_string().contains("not a pure operator"), "{call}: {error:#}");
    }
}
