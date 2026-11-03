"""Executable synthetic WGT tasks, split by program family before training.

No downloaded instruction corpus and no claims of human-labelled project data.
Code tasks explicitly ask for a completion, not an entire generated project.
"""
import hashlib
import math
import re

import torch
from weightlang.artifact import build_module
from weightlang.syntax import parse, WgtError


def program(body, input_shape, output_shape):
    return (f'wgt 0.3\nmodel task\ninput x: f32[{input_shape}]\n'
            f'output y: f32[{output_shape}]\n{body}\ninit seed=42\n')


def execute(source):
    """Bounded pure static subset for these tasks; no imports, data or callbacks."""
    if len(source)>6000 or not source.startswith('wgt 0.3\n'):
        raise ValueError('expected bounded static WGT 0.3 source')
    for line in source.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        if not re.match(r'^(wgt 0\.3|model \w+|input \w+:|output \w+:|layer \w+:|init seed=\d+)\s*',line):
            raise ValueError('statement outside evaluation subset')
        if any(word in line for word in ('python=', 'file=', 'entrypoint=', 'checkpoint', 'backend')):
            raise ValueError('external execution is not an evaluation operation')
    p=parse(source)
    if len(p.layers)>16 or len(p.inputs)>1 or len(p.outputs)>1 or p.initializer or p.uses or p.imports:
        raise ValueError('program outside task limits')
    if any(math.prod(port.shape)>4096 for port in (p.input,p.output)):
        raise ValueError('tensor exceeds task budget')
    allowed={'linear','relu','sigmoid','gelu','silu','tanh','rearrange','reduce','add','mul','concat','repeat','recur','seq'}
    def walk(layers):
        for layer in layers:
            if layer.op not in allowed:
                raise ValueError(f'operator outside evaluation subset: {layer.op}')
            if any(isinstance(v,(int,float)) and abs(v)>128 for v in layer.args.values()):
                raise ValueError('operator exceeds task budget')
            yield layer
            yield from walk(layer.children)
    list(walk(p.layers))
    layers=[layer.to_dict() for layer in p.layers]
    with torch.device('meta'):
        skeleton=build_module(layers)
    if sum(v.numel() for v in skeleton.parameters())>100000:
        raise ValueError('parameter budget exceeded')
    # A private seed makes numerical comparisons independent of model training RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(p.seed)
        model=build_module(layers).eval()
        x=torch.randn(2,*p.input.shape)
        with torch.no_grad():
            y=model(x)
    if tuple(y.shape)!=(2,*p.output.shape) or not torch.isfinite(y).all():
        raise ValueError('runtime output contract failed')
    return p,model,x,y


def task(family,kind,prompt,answer,source=None,**extra):
    row=dict(family=family,kind=kind,prompt=prompt,answer=answer,**extra)
    if source:
        execute(source)
        row['reference_source']=source
        row['reference_sha256']=hashlib.sha256(source.encode()).hexdigest()
    row['id']=hashlib.sha256((family+'\n'+prompt).encode()).hexdigest()[:16]
    return row


def dataset():
    data={'train':[],'validation':[],'test':[],'general':[]}
    # Whole templates/families are disjoint; dimensions also vary, including unseen widths.
    for i,(a,b,t) in enumerate([(2,3,2),(3,6,4),(4,2,3),(5,7,2),(6,4,5),(7,3,4),(8,5,3),(9,6,2)]):
        op=['relu','sigmoid','gelu','silu'][i%4]
        src=program(f'layer h: linear({b}) -> {op}',f'{t},{a}',f'{t},{b}')
        data['train'].append(task('pointwise','shape',f'WGT 0.3 input f32[{t},{a}], linear({b}) -> {op}. Give only the output type.',f'f32[{t},{b}]',src))
        src=program(f'layer a: linear({a})\nlayer b: linear({b}) from input\nlayer c: concat(a,b)',str(a),str(a+b))
        data['train'].append(task('concat','shape',f'WGT branches a: f32[{a}], b: f32[{b}]. concat(a,b) output type only.',f'f32[{a+b}]',src))
        src=program(f'layer h: repeat(3, linear({a}))',str(a),str(a))
        data['train'].append(task('repeat','parameters',f'WGT input f32[{a}]; repeat(3, linear({a})), default bias. Number of trainable scalar parameters? Integer only.',str(3*a*(a+1)),src))
        src=program(f'layer h: linear({b})\nlayer y: mean(axis=time)',f'time={t},feature={a}',f'feature={b}')
        data['train'].append(task('mean','shape',f'WGT input f32[time={t},feature={a}], linear({b}), then mean(axis=time). Output type with axis name only.',f'f32[feature={b}]',src))
        answer=f'layer h: linear({b}) -> {op}'
        src=program(answer,str(a),str(b))
        data['train'].append(task('linear_completion','code',f'Complete WGT 0.3 model task: input x f32[{a}], output y f32[{b}]. Add a layer named h: linear to {b} then {op}. Return only the layer declaration.',answer,src,
            input_shape=str(a),output_shape=str(b),expected_ops=['linear',op]))
        src=program(f'layer h: linear({b})',str(a),str(b))
        data['train'].append(task('output_repair','repair',f'Fix only the output declaration: input x: f32[{a}]; layer h: linear({b}); wrong output y: f32[{a+1}]. Return the corrected output declaration.',f'output y: f32[{b}]',src,
            body=f'layer h: linear({b})',input_shape=str(a),output_shape=str(b)))
    for a,b,t in [(3,5,4),(6,9,2),(7,4,3)]:
        src=program(f'layer a: linear({b})\nlayer b: linear({a})',str(b),str(a))
        data['validation'].append(task('chain','shape',f'WGT input f32[{b}]; linear({b}) then linear({a}). Output type only.',f'f32[{a}]',src))
        src=program(f'layer h: recur(4, linear({a}))',str(a),str(a))
        data['validation'].append(task('recur','parameters',f'WGT input f32[{a}]; recur(4, linear({a})), default bias. Number of trainable scalar parameters? Integer only.',str(a*(a+1)),src))
    # Frozen test manifest: withheld templates, never used for gradient/selection.
    for a,b,t in [(5,11,3),(10,7,4)]:
        src=program(f'layer h: linear({b})\nlayer y: transpose(feature,time)',f'time={t},feature={a}',f'feature={b},time={t}')
        data['test'].append(task('transpose','shape',f'WGT input f32[time={t},feature={a}], linear({b}), then transpose(feature,time). Output type with axis names only.',f'f32[feature={b},time={t}]',src))
        src=program(f'layer a: linear({a})\nlayer y: add(input,a)',str(a),str(a))
        data['test'].append(task('residual','parameters',f'WGT input f32[{a}]; a = linear({a}) with default bias; y = add(input,a). Total trainable scalar parameters? Integer only.',str(a*(a+1)),src))
        answer=f'layer h: linear({b})\nlayer y: sum(axis=time)'
        src=program(answer,f'time={t},feature={a}',f'feature={b}')
        data['test'].append(task('sum_completion','code',f'Complete WGT 0.3 model task: input x f32[time={t},feature={a}], output y f32[feature={b}]. First layer h projects to {b}; next layer y sums time. Return only the two layer declarations.',answer,src,
            input_shape=f'time={t},feature={a}',output_shape=f'feature={b}',expected_ops=['linear','reduce']))
        src=program(f'layer h: linear(in={a},out={b})',str(a),str(b))
        answer=f'layer h: linear(in={a},out={b})'
        data['test'].append(task('input_repair','code',f'WGT input x: f32[{a}], output y: f32[{b}]. Fix the incorrect declaration layer h: linear(in={a+2},out={b}). Return only the corrected layer declaration.',answer,src,
            input_shape=str(a),output_shape=str(b),expected_ops=['linear']))
    for prompt,answer in [('Return exactly: READY','READY'),('What is 2 + 3? Integer only.','5'),
        ('Translate dog into Chinese. Translation only.','狗'),('Sort 3, 1, 2 ascending. Commas without spaces.','1,2,3'),
        ('Return only JSON with key count and number 2.','{"count":2}'),('Capital of Japan? City name only.','Tokyo')]:
        data['train'].append(task('general_rehearsal','general',prompt,answer))
    for prompt,answer in [('Return exactly: HELLO','HELLO'),('What is 7 + 8? Answer with the integer only.','15'),
        ('Translate cat into Chinese. Return only the translation.','猫'),('Sort 9, 2, 5 in ascending order. Use commas without spaces.','2,5,9'),
        ('Return only valid JSON with key ok and boolean value true.','{"ok":true}'),('What is the capital of France? City name only.','Paris')]:
        data['general'].append(task('general','general',prompt,answer))
    families=[{r['family'] for r in data[s]} for s in ('train','validation','test')]
    assert not any(families[i]&families[j] for i in range(3) for j in range(i))
    assert len({r['id'] for rows in data.values() for r in rows})==sum(map(len,data.values()))
    return data


def grade(row,text):
    result={'exact':text.strip()==row['answer'].strip(),'parses':False,'runs':False,'task_correct':False}
    if row['kind'] not in ('code','repair'):
        if row['kind']=='general' and row['answer']=='{"ok":true}':
            import json
            try:
                value=json.loads(text)
                result['exact']=isinstance(value,dict) and set(value)=={'ok'} and value['ok'] is True
            except ValueError:
                pass
        result['task_correct']=result['exact']
        if row['kind']=='shape':
            result['task_correct']=re.sub(r'\s+','',text)==re.sub(r'\s+','',row['answer'])
        elif row['kind']=='parameters' and re.fullmatch(r'\d+',text.strip()):
            result['task_correct']=int(text.strip())==int(row['answer'])
        return result
    try:
        # The prompt explicitly requests declarations only. Fenced/prose output is not silently cleaned.
        if row['kind']=='repair':
            source=(f'wgt 0.3\nmodel task\ninput x: f32[{row["input_shape"]}]\n'
                    f'{text.strip()}\n{row["body"]}\ninit seed=42\n')
        else:
            source=program(text.strip(),row['input_shape'],row['output_shape'])
        parse(source);result['parses']=True
        p,m,x,y=execute(source);result['runs']=True
        ref,rm,rx,ry=execute(row['reference_source'])
        # Compare numerical behavior with identical seed, not text spelling or parser success.
        same_contract=(p.input.shape==ref.input.shape and p.output.shape==ref.output.shape
                       and p.input.axes==ref.input.axes and p.output.axes==ref.output.axes)
        torch.testing.assert_close(x,rx,rtol=0,atol=0)
        torch.testing.assert_close(y,ry,rtol=1e-5,atol=1e-6)
        result['task_correct']=same_contract
    except (WgtError,ValueError,RuntimeError,AssertionError,TypeError) as e:
        result['error']=str(e)[:300]
    return result
