use wgt_language::{
    adaptive,
    script::parse::{Expr, Stmt, parse_functions},
};

fn expression(body: &str) -> Expr {
    let functions = parse_functions(&format!("wgt 0.5\nfn value(x,a,b) {{ return {body}; }}"))
        .expect("expression should parse");
    let Stmt::Return(value) = &functions["value"].body[0].kind else {
        panic!("expected return");
    };
    value.clone()
}

fn intrinsic(value: &Expr, name: &str) -> Vec<Expr> {
    let Expr::BuiltinCall(operation, args) = value else {
        panic!("expected syntax-generated {name}, got {value:?}");
    };
    assert_eq!(operation, name);
    args.clone()
}

fn model(body: &str) -> String {
    format!(
        "wgt 0.5\nmodel ShapeSyntax\ninput x:f32[4]\noutput y:f32[4]\nfn forward(x) {{ {body} }}"
    )
}

#[test]
fn shape_postfix_precedes_unary_and_matrix_operators() {
    let Expr::Binary(op, left, right) = expression("-a as [2,2] @ b as [2,1]") else {
        panic!("matrix product should be outermost");
    };
    assert_eq!(op, "@");
    let Expr::Unary(op, operand) = *left else {
        panic!("left matrix should be negated");
    };
    assert_eq!(op, "-");
    assert!(matches!(&intrinsic(&operand, "reshape")[0], Expr::Name(n) if n == "a"));
    assert!(matches!(&intrinsic(&right, "reshape")[0], Expr::Name(n) if n == "b"));
    let args = intrinsic(&expression("(a + b) as [2,2]"), "reshape");
    assert!(matches!(&args[0], Expr::Binary(op, _, _) if op == "+"));
}

#[test]
fn shape_postfix_chains_with_transpose_slices_and_reduction() {
    let mean = intrinsic(&expression("x[0:4] as [2,2].T as [4][1:3].mean"), "mean");
    let Expr::Slice(flat, _, _) = &mean[0] else {
        panic!("slice must apply to flattened transpose");
    };
    let flat = intrinsic(flat, "reshape");
    let transposed = intrinsic(&flat[0], "transpose");
    let matrix = intrinsic(&transposed[0], "reshape");
    assert!(matches!(&matrix[0], Expr::Slice(_, _, _)));
    assert!(matches!(&matrix[1], Expr::List(dims) if dims.len() == 2));
}

#[test]
fn as_remains_a_contextual_keyword() {
    let functions = parse_functions(
        "wgt 0.5\nfn as(as) { return as; }\nfn value(x) { return as(x as [2,2]); }",
    )
    .unwrap();
    assert_eq!(functions["as"].args, ["as"]);
    let Stmt::Return(Expr::Call(name, args)) = &functions["value"].body[0].kind else {
        panic!("ordinary function as must remain callable");
    };
    assert_eq!(name, "as");
    intrinsic(&args[0], "reshape");
}

#[test]
fn shape_postfix_requires_bracket_delimiters() {
    for value in ["x as shape", "x as 4", "x as [2,2", "x as [2,2,]"] {
        assert!(
            parse_functions(&format!("wgt 0.5\nfn value(x) {{ return {value}; }}")).is_err(),
            "accepted malformed shape expression: {value}"
        );
    }
}

#[test]
fn shape_syntax_and_reshape_produce_identical_tensor_ir() {
    let source = model("let a = x as [2,2]; let b = x as [2,2].T; return (a @ b) as [4];");
    let legacy =
        model("let a = reshape(x,[2,2]); let b = reshape(x,[2,2]).T; return reshape(a @ b,[4]);");
    let current = adaptive::parse(&source).unwrap();
    let previous = adaptive::parse(&legacy).unwrap();
    assert_eq!(
        serde_json::to_value(current).unwrap(),
        serde_json::to_value(previous).unwrap()
    );
}

#[test]
fn shape_postfix_in_pure_helpers_preserves_the_reshape_contract() {
    let source =
        model("return transform(x) as [4];") + "\nfn transform(x) { return x as [2,2].T; }";
    let legacy = source
        .replace("transform(x) as [4]", "reshape(transform(x),[4])")
        .replace("x as [2,2]", "reshape(x,[2,2])");
    assert_eq!(
        serde_json::to_value(adaptive::parse(&source).unwrap()).unwrap(),
        serde_json::to_value(adaptive::parse(&legacy).unwrap()).unwrap()
    );
}

#[test]
fn shape_postfix_retains_dimension_and_element_count_checks() {
    for shape in [
        "[]",
        "[0,4]",
        "[-1,4]",
        "[1.5,2]",
        "[3,2]",
        "[16385]",
        "[1,1,1,1,1,1,1,1,4]",
        "[n,2]",
        "[2*1,2]",
        "[\"2\",2]",
    ] {
        let cast = adaptive::parse(&model(&format!("return (x as {shape}) as [4];")))
            .expect_err("invalid postfix shape must be rejected");
        let function = adaptive::parse(&model(&format!("return reshape(reshape(x,{shape}),[4]);")))
            .expect_err("invalid function shape must be rejected");
        assert_eq!(format!("{cast:#}"), format!("{function:#}"), "{shape}");
    }
}

#[test]
fn explicit_shape_postfix_clears_axis_names() {
    let body = "let named: f32[rows=2,cols=2] = x;\n\
                let anonymous = named as [2,2];\n\
                return softmax(anonymous,\"cols\") as [4];";
    let source = model(body);
    assert!(adaptive::parse(&source).is_err());
    adaptive::parse(&source.replace("softmax(anonymous,\"cols\")", "softmax(anonymous,1)"))
        .expect("numeric axis remains valid after explicit reshape");
}
