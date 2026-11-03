"""Human-readable sample shapes; the optional batch axis stays outside the DSL."""

from .symbolic import display


def format_shape(spec):
    axes = spec.get("axes", [None] * len(spec["shape"]))
    dimensions = [f"{axis}={size}" if axis else str(size) for axis, size in zip(axes, spec["shape"])]
    return f"{spec['dtype']}[{', '.join(dimensions)}]"


def shape_rows(program):
    """Include expanded block paths so an error can be located in a nested model."""
    if program.dimensions:
        yield "bounded dimensions: " + ", ".join(f"{name}={b['min']}..{b['max']}" for name, b in program.dimensions.items())
        for op, left, right in program.constraints:
            yield f"require {display(left)} {op} {display(right)}"
        yield "layer shapes use the first legal dimension assignment; port expressions define the runtime contract"
    for port in program.inputs or [program.input]:
        yield f"input {port.name}: " + format_shape({
            "dtype": port.dtype, "shape": tuple(display(e) for e in port.shape_expr) if port.shape_expr else port.shape,
            "axes": port.axes or [None] * len(port.shape),
        })

    def visit(layers, prefix=""):
        for layer in layers:
            name = prefix + layer.name
            if layer.input_spec is not None:
                yield f"{name}: {format_shape(layer.input_spec)} -> {format_shape(layer.output_spec)}"
            yield from visit(layer.children, name + ".")

    yield from visit(program.layers)
    for port in program.outputs or [program.output]:
        ref = program.returns.get(port.name, program.layers[-1].name)
        yield f"output {port.name} from {ref}: " + format_shape({
            "dtype": port.dtype, "shape": tuple(display(e) for e in port.shape_expr) if port.shape_expr else port.shape,
            "axes": port.axes or [None] * len(port.shape),
        })
