//! Compile-time checked model definitions; these macros never contact wgtime.
//!
//! `model!(r#"wgt 0.5\n..."#)` accepts a Rust string literal, not Rust tokens
//! masquerading as WGT. Raw literals preserve WGT newlines, comments, strings
//! and punctuation exactly. Normal literals use Rust's usual escape decoding.
//! `include_model!("models/model.wgt")` resolves against the consuming Cargo
//! package's manifest directory and emits `include_str!` for rustc tracking.
//! Both validate at compile time and parse the embedded source again when the
//! Rust expression runs. They do not embed precompiled IR or promise zero parse
//! overhead. Standard pure tensor imports are embedded; file imports recursively
//! embed the dependency closure and track every file, never looking up runtime sources.
//! Both return `::wgt::ModelDefinition`; when the facade dependency is renamed,
//! use `model!(crate = renamed_wgt; r#"..."#)` (likewise for `include_model!`).

use proc_macro::TokenStream;
use proc_macro2::TokenStream as Tokens;
use quote::quote;
use std::path::Path;
use syn::{LitStr, Token, parse::Parse, parse::ParseStream, parse_macro_input};

struct Input {
    facade: syn::Path,
    literal: LitStr,
}

impl Parse for Input {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let facade = if input.peek(Token![crate]) {
            input.parse::<Token![crate]>()?;
            input.parse::<Token![=]>()?;
            let path = input.parse()?;
            input.parse::<Token![;]>()?;
            path
        } else {
            syn::parse_quote!(::wgt)
        };
        let literal = input.parse()?;
        if !input.is_empty() {
            return Err(input
                .error("expected one Rust string literal containing WGT source or a file path"));
        }
        Ok(Self { facade, literal })
    }
}

fn definition(facade: &syn::Path, source: Tokens) -> Tokens {
    quote! {
        #facade::ModelDefinition::parse(#source)
            .expect("compile-time validated WGT model definition")
    }
}

fn inline(input: Input) -> syn::Result<Tokens> {
    let parsed =
        wgt_language::model::ModelDefinition::parse(&input.literal.value()).map_err(|error| {
            syn::Error::new(
                input.literal.span(),
                format!("invalid WGT model definition: {error:#}"),
            )
        })?;
    let literal = LitStr::new(parsed.source(), input.literal.span());
    Ok(definition(&input.facade, quote!(#literal)))
}

fn file(input: Input, manifest: &Path) -> syn::Result<Tokens> {
    let requested = input.literal.value();
    let path = Path::new(&requested);
    if path.is_absolute() || requested.is_empty() {
        return Err(syn::Error::new(
            input.literal.span(),
            "expected a nonempty path relative to CARGO_MANIFEST_DIR",
        ));
    }
    let resolved = manifest.join(path);
    let (parsed, dependencies) = wgt_language::model::ModelDefinition::from_file_with_dependencies(
        &resolved,
    )
    .map_err(|error| {
        syn::Error::new(
            input.literal.span(),
            format!(
                "invalid WGT model definition {}: {error:#}",
                resolved.display()
            ),
        )
    })?;
    let source = LitStr::new(parsed.source(), input.literal.span());
    let tracked = dependencies
        .iter()
        .map(|path| {
            path.to_str()
                .map(|path| LitStr::new(path, input.literal.span()))
                .ok_or_else(|| {
                    syn::Error::new(
                        input.literal.span(),
                        "WGT model dependency path must be valid UTF-8",
                    )
                })
        })
        .collect::<syn::Result<Vec<_>>>()?;
    let value = definition(&input.facade, quote!(#source));
    Ok(quote!({ #(const _: &str = ::core::include_str!(#tracked);)* #value }))
}

/// Embed and validate a complete WGT model source string without initialization.
#[proc_macro]
pub fn model(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as Input);
    inline(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Embed a manifest-relative WGT file and track it as a compiler dependency.
#[proc_macro]
pub fn include_model(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as Input);
    let result = std::env::var_os("CARGO_MANIFEST_DIR")
        .ok_or_else(|| {
            syn::Error::new(
                input.literal.span(),
                "CARGO_MANIFEST_DIR is required by include_model!",
            )
        })
        .and_then(|manifest| file(input, Path::new(&manifest)));
    result.unwrap_or_else(syn::Error::into_compile_error).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = include_str!("../tests/fixtures/model.wgt");

    #[test]
    fn raw_source_preserves_comments_strings_and_newlines() {
        let literal = LitStr::new(SOURCE, proc_macro2::Span::call_site());
        let input: Input = syn::parse2(quote!(crate = facade_alias; #literal)).unwrap();
        let output = inline(input).unwrap();
        let expected: syn::Expr = syn::parse2(output).unwrap();
        let syn::Expr::MethodCall(call) = expected else {
            panic!("expected checked definition")
        };
        let syn::Expr::Call(parse) = *call.receiver else {
            panic!("expected parse call")
        };
        let syn::Expr::Lit(lit) = &parse.args[0] else {
            panic!("expected literal")
        };
        let syn::Lit::Str(source) = &lit.lit else {
            panic!("expected source string")
        };
        assert_eq!(source.value(), SOURCE);
    }

    #[test]
    fn rejects_tokens_extra_arguments_and_invalid_model() {
        assert!(syn::parse2::<Input>(quote!(wgt 0.5 model Probe)).is_err());
        assert!(syn::parse2::<Input>(quote!("source", "extra")).is_err());
        let input: Input = syn::parse2(quote!("wgt 0.5\nfn main(){return 1;}")).unwrap();
        assert!(inline(input).is_err());
        let imported = SOURCE.replacen(
            "wgt 0.5",
            "wgt 0.5\nimport helper from \"./missing.wgt\"",
            1,
        );
        let literal = LitStr::new(&imported, proc_macro2::Span::call_site());
        let input: Input = syn::parse2(quote!(#literal)).unwrap();
        assert!(inline(input).unwrap_err().to_string().contains("imports"));
    }

    #[test]
    fn manifest_relative_file_emits_tracked_include_and_alias() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let input: Input =
            syn::parse2(quote!(crate = renamed_wgt; "tests/fixtures/model.wgt")).unwrap();
        let output = file(input, manifest).unwrap().to_string();
        assert!(output.contains("renamed_wgt :: ModelDefinition :: parse"));
        assert!(output.contains("include_str"));
        assert!(output.contains(manifest.to_str().unwrap()));
    }

    #[test]
    fn rejects_missing_and_absolute_files() {
        for name in ["", "/absolute/model.wgt", "missing-model.wgt"] {
            let input: Input = syn::parse2(quote!(#name)).unwrap();
            assert!(file(input, Path::new(env!("CARGO_MANIFEST_DIR"))).is_err());
        }
    }
}
