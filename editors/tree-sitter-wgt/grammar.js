// An error-tolerant editor grammar, not a replacement for WeightLang validation.
// Newlines terminate statements except inside parenthesized/bracketed expressions.
// Indentation is deliberately not semantic here: declarations remain flat in the
// tree, so editing an unfinished block does not hide later model declarations.
module.exports = grammar({
  name: 'wgt',
  extras: $ => [/[ \t\r]+/, $.comment],
  word: $ => $.identifier,
  rules: {
    source_file: $ => seq(repeat(choice($._newline, seq($._statement, $._newline))), optional($._statement)),
    _newline: _ => /\n/,
    comment: _ => /#[^\n]*/,
    _statement: $ => choice($.model_declaration, $.block_declaration, $.layer_declaration,
      $.port_declaration, $.constant_declaration, $.dimension_declaration,
      $.directive, $.data_row),
    model_declaration: $ => seq('model', field('name', $.identifier), repeat($._item)),
    block_declaration: $ => seq('block', field('name', $.identifier), $.arguments, ':'),
    layer_declaration: $ => seq('layer', field('name', $.identifier), ':', repeat1($._item)),
    port_declaration: $ => seq(choice('input', 'output'), field('name', $.identifier), ':', repeat1($._item)),
    constant_declaration: $ => seq('const', field('name', $.identifier), '=', repeat1($._item)),
    dimension_declaration: $ => seq('dim', field('name', $.identifier), repeat($._item)),
    directive: $ => seq(field('keyword', choice('wgt', 'require', 'use', 'import', 'inherit',
      'freeze', 'unfreeze', 'data', 'init', 'train')), repeat1($._item)),
    data_row: $ => seq($.array, '->', $.array),
    _item: $ => choice($.call, $.parameter, $.qualified_name, $.number, $.string,
      $.array, $.arguments, $.operator, $.punctuation, $.clause),
    clause: _ => choice('from', 'as', 'extends'),
    call: $ => prec(2, seq(field('function', $.qualified_name), $.arguments)),
    qualified_name: $ => seq($.identifier, repeat(seq('.', $.identifier))),
    parameter: $ => prec(1, seq(field('name', $.identifier), '=')),
    arguments: $ => seq('(', repeat(choice($._item, $._newline)), ')'),
    array: $ => seq('[', repeat(choice($._item, $._newline)), ']'),
    identifier: _ => /[A-Za-z_][A-Za-z0-9_]*/,
    number: _ => /(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?/,
    string: _ => /"[^"\n]*"/,
    operator: _ => choice('->', '==', '!=', '<=', '>=', '//', '=', '+', '-', '*', '/', '%', '<', '>'),
    punctuation: _ => choice(',', ':'),
  },
});
