"""Typed model configuration and compiler-owned source materialization."""
import math


def validate(name, kind, options, value):
    types = {'int':int,'float':float,'bool':bool}
    if kind not in types or 'default' not in options or set(options)-{'default','min','max'}:
        raise ValueError('config requires int/float/bool, default and optional min/max')
    if kind == 'bool':
        if set(options) != {'default'} or options['default'] not in ('true','false') or type(value) is not bool:
            raise ValueError(f'config {name}: bool uses default=true/false and no numeric bounds')
        return value
    if type(value) not in ((int,float) if kind == 'float' else (int,)) or not math.isfinite(value):
        raise ValueError(f'config {name}: expected finite {kind}')
    convert = types[kind]
    value = convert(value)
    bounds = {k:convert(v) for k,v in options.items() if k in {'min','max'}}
    if any(not math.isfinite(v) for v in bounds.values()) or bounds.get('min',-math.inf) > bounds.get('max',math.inf):
        raise ValueError(f'config {name}: invalid bounds')
    if not bounds.get('min',-math.inf) <= value <= bounds.get('max',math.inf):
        raise ValueError(f'config {name}: outside declared bounds')
    return value


def frozen_source(source, schema, values):
    lines = source.splitlines()
    for name,spec in schema.items():
        value = values[name]
        if type(value) is bool:
            value = int(value)
        lines[spec['line']-1] = f'const {name} = {value!r}'
        for line in range(spec['line'],spec['end_line']):
            lines[line] = ''
    return '\n'.join(lines)+'\n'


def model_config(source):
    from .syntax import parse_file
    program = parse_file(source,check_training=False)
    return {'schema':program.config_schema,'values':program.config_values}


def package_model(source, output, *, config=None, force=False):
    """Resolve/expand source dependencies and package one frozen architecture.

    No initialization, weights, data callbacks or training run. For packages with
    explicit weight assets or custom initialization use build_bundle instead.
    """
    import tempfile
    from pathlib import Path
    from .syntax import parse_file, WgtError
    from .ports import interface
    from .decompile import source_from_manifest
    from .bundle import build_bundle
    program = parse_file(source,config=config,check_training=False)
    if program.initializer or program.imports or program.inherit or program.derivation or program.freezing:
        raise WgtError('package_model packages initialized architecture only; use build_bundle for imports/custom initialization/freezing')
    ports = {role:{'name':p.name,'dtype':p.dtype,'shape':list(p.shape),**({'axes':list(p.axes)} if p.axes else {})}
             for role,p in (('input',program.input),('output',program.output))}
    ports.update(interface(program) or {})
    text = source_from_manifest(dict(model=program.model,interface=ports,layers=[layer.to_dict() for layer in program.layers]))
    # Renderer has one terminal seed declaration; retain caller's initialization.
    lines = text.splitlines()
    assert lines[-1] == 'init seed=0'
    lines[-1] = f'init seed={program.seed}'
    cache = Path(source).resolve().parent/'.cache/weightlang'
    cache.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='configured-package-',dir=cache) as tmp:
        stage = Path(tmp)
        (stage/'model.wgt').write_text('\n'.join(lines)+'\n')
        return build_bundle(stage,output,entry='model.wgt',force=force)
