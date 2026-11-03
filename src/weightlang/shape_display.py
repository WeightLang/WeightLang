"""Human-readable sample shapes; the optional batch axis stays outside the DSL."""


def format_shape(spec):
    axes = spec.get("axes", [None] * len(spec["shape"]))
    dimensions = [f"{axis}={size}" if axis else str(size) for axis, size in zip(axes, spec["shape"])]
    return f"{spec['dtype']}[{', '.join(dimensions)}]"


def shape_rows(program):
    """Include expanded block paths so an error can be located in a nested model."""
    yield f"input {program.input.name}: " + format_shape({
        "dtype": program.input.dtype, "shape": program.input.shape,
        "axes": program.input.axes or [None] * len(program.input.shape),
    })

    def visit(layers, prefix=""):
        for layer in layers:
            name = prefix + layer.name
            if layer.input_spec is not None:
                yield f"{name}: {format_shape(layer.input_spec)} -> {format_shape(layer.output_spec)}"
            yield from visit(layer.children, name + ".")

    yield from visit(program.layers)
