"""Train WGT internal recurrent edits through a frozen pretrained Qwen suffix."""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from curriculum import dataset
from train import HERE, REVISION, UPSTREAM_SHA256, StreamedModel, encode, sha, write
from evaluate import verify


def tail_state(tail):
    return {n:p.detach().cpu().clone() for n,p in tail.named_parameters() if p.requires_grad}


def make_tail(stream, start, rank, steps, parent, parent_rank=32):
    with torch.device('meta'):
        tail=stream.classes['EditedTail'](stream.c,start,rank,steps,parent_rank)
    for j,i in enumerate(range(start,len(stream.c['layer_types']))):
        tail.layers[j]=stream.block('Decoder',f'layers.{i}.',stream.c,stream.c['layer_types'][i])
        tail.edits[j]=stream.classes['Refinement'](stream.c['hidden_size'],rank,steps).to(stream.device)
    tail.norm=stream.block('Qwen3_5RMSNorm','norm.',stream.c['hidden_size'],stream.c['rms_norm_eps'])
    tail.surgery=stream.classes['Surgery'](stream.c['hidden_size'],parent_rank).to(stream.device)
    tail.surgery.load_state_dict(parent,strict=True)
    return tail.eval()


def restore_tail(tail, state):
    expected={n for n,p in tail.named_parameters() if p.requires_grad}
    if set(state)!=expected:
        raise ValueError('trainable tail key mismatch')
    with torch.no_grad():
        for n,p in tail.named_parameters():
            if n in state:
                if p.shape!=state[n].shape or not torch.isfinite(state[n]).all():
                    raise ValueError(f'invalid trained tensor: {n}')
                p.copy_(state[n])


def loss(tail, head, hidden, targets, start):
    # Keep the full sequence through attention; slice only before the tied LM head.
    z=tail(hidden.to(head.device).unsqueeze(0))[0,start:]
    logits=torch.nn.functional.linear(z,head).float()
    return torch.nn.functional.cross_entropy(logits,targets.to(head.device))


@torch.no_grad()
def measure(tail,head,hidden,targets,starts,indices):
    total=0.;count=0
    for i in indices:
        total+=loss(tail,head,hidden[i],targets[i],starts[i]).item()*len(targets[i])
        count+=len(targets[i])
    return total/count


def snapshot(root, upstream, source, state, metadata):
    root.mkdir()
    for name in sorted(set(UPSTREAM_SHA256)|{'LICENSE','vocab.json'}):
        if (upstream/name).exists():
            shutil.copy2(upstream/name,root/name)
    shutil.copy2(source,root/'source.wgt')
    save_file(state,str(root/'tail.safetensors'))
    write(root/'manifest.json',dict(metadata,sha256={p.name:sha(p) for p in root.iterdir() if p.is_file()}))


def run(args):
    if args.output.exists() or args.output.is_symlink() or args.steps<1 or args.rank<1 or args.accumulate<1:
        raise ValueError('new output and positive training sizes required')
    parent_manifest=verify(args.parent)
    if parent_manifest['revision']!=REVISION:
        raise ValueError('parent uses a different pretrained revision')
    for name,expected in UPSTREAM_SHA256.items():
        if parent_manifest['sha256'].get(name)!=expected:
            raise ValueError(f'parent is not the fixed upstream: {name}')
    parent_name='selected.safetensors' if 'selected.safetensors' in parent_manifest['sha256'] else 'surgery.safetensors'
    parent=load_file(str(args.parent/parent_name))
    torch.set_num_threads(4);torch.manual_seed(20261103);random.seed(20261103)
    torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True)
    data=dataset()
    write(args.output/'dataset.json',data)
    # Fix acceptance before any new test results, with no threshold adjustment afterwards.
    write(args.output/'acceptance.json',{
        'selection':'minimum validation answer-token NLL; test/general never select weights',
        'thresholds':{'test_task_correct_rate_min':0.75,'test_code_runtime_rate_min':1.0,
                      'general_correct_drop_max':0,'test_tokens':64,'general_tokens':24},
        'scope':'small generated program families; not production/general coding qualification'})
    for name in ('train_tail.py','train.py','evaluate.py','curriculum.py','evaluate_tail.py'):
        if (HERE/name).exists():
            shutil.copy2(HERE/name,args.output/name)
    shutil.copy2(HERE/'model.wgt',args.output/'source.wgt')
    source=args.output/'source.wgt'
    stream=StreamedModel(args.parent,source)
    if not 0<args.start<len(stream.c['layer_types']):
        raise ValueError('invalid suffix start')
    tokenizer=AutoTokenizer.from_pretrained(args.parent,local_files_only=True)
    flat=[];split={}
    for name,rows in data.items():
        split[name]=list(range(len(flat),len(flat)+len(rows)));flat+=rows
    sequences,targets,starts=encode(tokenizer,flat)
    if max(map(len,sequences))>512:
        raise ValueError('task exceeds bounded experiment sequence budget')
    started=time.perf_counter()
    print({'rows':{k:len(v) for k,v in data.items()},'max_sequence':max(map(len,sequences)),'stage':'prefix cache'},flush=True)
    hidden=stream.features(sequences,stop=args.start)
    # Disk-backed cache belongs to this retained experiment; no tmpfs or implicit data access on load.
    save_file({str(i):h for i,h in enumerate(hidden)},str(args.output/'prefix.safetensors'))
    tail=make_tail(stream,args.start,args.rank,args.refinements,parent,parent_manifest['rank'])
    head=stream.tensor(stream.prefix+'embed_tokens.weight')
    frozen={n:p.detach().clone() for n,p in tail.named_parameters() if not p.requires_grad}
    initial_state=tail_state(tail)
    with torch.no_grad():
        probe=hidden[0].to(head.device).unsqueeze(0)
        torch.testing.assert_close(tail(probe),tail(probe,edited=False),rtol=0,atol=0)
    metadata={'format':'weightlang.qwen-tail-experiment','version':1,'model':'Qwen/Qwen3.5-4B','revision':REVISION,
              'parent_sha256':sha(args.parent/parent_name),'start':args.start,'rank':args.rank,
              'refinements':args.refinements,'parent_rank':parent_manifest['rank'],
              'architecture':'QwenText prefix + EditedTail; last two internal recurrent edits + inherited final surgery',
              'origin':'imported+trained','optimizer_resume':False}
    snapshot(args.output/'baseline',args.parent,source,initial_state,metadata)
    # Baseline test measurement is held until after the candidate is fixed, avoiding inspection during tuning.
    initial={s:measure(tail,head,hidden,targets,starts,split[s]) for s in ('train','validation')}
    print({'initial_nll':initial},flush=True)
    opt=torch.optim.AdamW([p for p in tail.parameters() if p.requires_grad],lr=args.lr,weight_decay=0.01)
    history=[];best=initial['validation'];selected=0;best_state=initial_state
    rng=random.Random(20261103);order=[]
    for step in range(1,args.steps+1):
        opt.zero_grad(set_to_none=True);batch_loss=0.
        for _ in range(args.accumulate):
            if not order:
                order=list(split['train']);rng.shuffle(order)
            i=order.pop()
            value=loss(tail,head,hidden[i],targets[i],starts[i])/args.accumulate
            if not torch.isfinite(value):
                raise RuntimeError('nonfinite training loss')
            value.backward();batch_loss+=value.item()
        torch.nn.utils.clip_grad_norm_([p for p in tail.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        opt.step()
        if step%10==0 or step==args.steps:
            val=measure(tail,head,hidden,targets,starts,split['validation'])
            item=dict(step=step,train_batch_nll=batch_loss,validation_nll=val)
            history.append(item);print(item,flush=True)
            if val<best:
                best=val;selected=step;best_state=tail_state(tail)
            write(args.output/'progress.json',dict(step=step,selected_step=selected,history=history))
    last=tail_state(tail)
    save_file(last,str(args.output/'last.safetensors'))
    # Optimizer checkpoint is for last step only; selected inference weights may be earlier.
    torch.save({'optimizer':opt.state_dict(),'step':args.steps,'model':last,
                'rng_state':rng.getstate(),'remaining_order':order},args.output/'optimizer.pt')
    restore_tail(tail,best_state)
    selected_nll={s:measure(tail,head,hidden,targets,starts,split[s]) for s in split}
    with torch.no_grad():
        expected=tail(probe).cpu()
    snapshot(args.output/'candidate',args.parent,source,best_state,dict(metadata,selected_step=selected))
    restore_tail(tail,load_file(str(args.output/'candidate/tail.safetensors')))
    with torch.no_grad():
        torch.testing.assert_close(tail(probe).cpu(),expected,rtol=0,atol=0)
    assert all(torch.equal(p.detach(),frozen[n]) for n,p in tail.named_parameters() if n in frozen)
    restore_tail(tail,initial_state)
    baseline_nll={s:measure(tail,head,hidden,targets,starts,split[s]) for s in split}
    changed=[n for n in best_state if not torch.equal(best_state[n],initial_state[n])]
    report=dict(metadata,steps=args.steps,selected_step=selected,seed=20261103,lr=args.lr,
                accumulate=args.accumulate,history=history,baseline_nll=baseline_nll,selected_nll=selected_nll,
                changed_tensors=changed,trainable_parameters=sum(v.numel() for v in best_state.values()),
                quality_accepted=False,seconds=time.perf_counter()-started,
                checks={'zero_edit_identity':True,'frozen_suffix_bitwise_unchanged':True,'reload_identical':True},
                provenance={'source_sha256':sha(source),'dataset_sha256':sha(args.output/'dataset.json'),
                            'compiler_sha256':sha(Path('src/weightlang/syntax.py')),'torch':torch.__version__},
                limitations=['no full parameter finetuning','synthetic family-split curriculum',
                             'generation acceptance is a separate report','no padding/KV cache/vision'])
    write(args.output/'report.json',report)
    print(json.dumps({k:report[k] for k in ('selected_step','trainable_parameters','baseline_nll','selected_nll','seconds')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=int,default=30)
    p.add_argument('--rank',type=int,default=64)
    p.add_argument('--refinements',type=int,default=2)
    p.add_argument('--steps',type=int,default=80)
    p.add_argument('--accumulate',type=int,default=2)
    p.add_argument('--lr',type=float,default=0.0002)
    run(p.parse_args())
