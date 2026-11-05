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
        "wgt 0.5\\nfn value(q,k) { return q @ k; }"
    );
    assert!(matches!(
        expression,
        Expr::Binary(ref op, ref left, ref right)
            if op == "@"
                && matches!(**left, Expr::Name(ref name) if name == "q")
                && matches!(**right, Expr::Name(ref name) if name == "k")
    ));
}

#[test]
fn transpose_uses_native_t_postfix() {
    let expression = returned(
        "wgt 0.5\\nfn value(k) { return k.T; }"
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
        "wgt 0.5\\nfn value(a,b,c) { return a + b @ c; }"
    );
    assert!(matches!(
        expression,
        Expr::Binary(ref op, _, ref right)
            if op == "+"
                && matches!(**right, Expr::Binary(ref inner, _, _) if inner == "@")
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
    let q = tokens @ projection;
    let k = q.T;
    let scores = q @ k;
    return scores.mean;
}
"#;
    wgt_language::adaptive::parse(source)
        .expect("native @ and .T syntax should lower to matrix IR");
}
