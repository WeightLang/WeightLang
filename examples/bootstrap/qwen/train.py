"""Pinned pretrained Qwen3.5 text bootstrap with a WGT residual surgery.

Frozen blocks are streamed through CUDA. No large random model is constructed.
Experiment directories are standalone snapshots, not static WGT artifacts.
"""
import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from weightlang.syntax import parse
from weightlang.tensor_program import parse_tensor_program

HERE = Path(__file__).resolve().parent
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
# Hashes of the official files downloaded at the pinned revision above.
UPSTREAM_SHA256 = {'chat_template.jinja': 'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715', 'model.safetensors-00001-of-00002.safetensors': '26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61', 'config.json': 'ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670', 'model.safetensors-00002-of-00002.safetensors': 'cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188', 'model.safetensors.index.json': 'cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d', 'tokenizer.json': '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42', 'tokenizer_config.json': '316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8'}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dataset():
    # 原创合成任务按算子族划分。首轮只证明权重更新链，不代表真实项目能力。
    result = {}
    for split, ops in {'train':['relu','sigmoid','gelu'], 'validation':['silu'], 'test':['tanh']}.items():
        rows=[]
        for op in ops:
            source=f'wgt 0.3\nmodel demo\ninput x: f32[4]\noutput y: f32[8]\nlayer h: linear(8) -> {op}\ninit seed=42\n'
            parse(source)
            rows += [dict(kind='code', family=op, prompt=f'Write WGT 0.3 model demo: input x f32[4], output y f32[8], linear(8) then {op}, init seed=42. Code only.', answer=source),
                     dict(kind='shape',family=op,prompt=f'WGT input f32[4], linear(8) -> {op}. Output type only.',answer='f32[8]')]
        result[split]=rows
    return result


class StreamedModel:
    """Only I/O and device placement live here; every network operation is WGT."""
    def __init__(self, root, source, device='cuda', cache_bytes=0):
        self.root=Path(root)
        self.c=json.loads((self.root/'config.json').read_text())['text_config']
        self.classes=parse_tensor_program(Path(source).read_text()).classes({})
        self.index=json.loads((self.root/'model.safetensors.index.json').read_text())['weight_map']
        self.device=device
        self.cache_bytes=cache_bytes
        self.cached_bytes=0
        self.block_cache={}
        self.prefix='model.language_model.'
        with torch.device('meta'):
            skeleton=self.classes['QwenText'](self.c)
        expected={self.prefix+k for k in skeleton.state_dict() if not k.startswith('surgery.')}
        actual={k for k in self.index if k.startswith(self.prefix)}
        if expected != actual:
            raise ValueError(f'checkpoint text key mismatch: missing={expected-actual}, extra={actual-expected}')

    def tensor(self, key):
        with safe_open(self.root/self.index[key],framework='pt',device='cpu') as f:
            return f.get_tensor(key).to(self.device)

    def block(self, cls, prefix, *args):
        key=(cls,prefix,json.dumps(args,sort_keys=True))
        if key in self.block_cache:
            return self.block_cache[key]
        with torch.device('meta'):
            module=self.classes[cls](*args)
        expected=set(module.state_dict())
        state={key:self.tensor(self.prefix+prefix+key) for key in expected}
        module.load_state_dict(state,strict=True,assign=True)
        module.requires_grad_(False).eval()
        size=sum(p.numel()*p.element_size() for p in module.parameters())
        if self.cached_bytes+size<=self.cache_bytes:
            self.block_cache[key]=module
            self.cached_bytes+=size
        return module

    @torch.no_grad()
    def features(self, sequences, stop=None, embedding_weight=None, batch_size=1):
        if stop is not None and not 0 <= stop <= len(self.c['layer_types']):
            raise ValueError('prefix stop outside decoder')
        embed=embedding_weight if embedding_weight is not None else self.tensor(self.prefix+'embed_tokens.weight')
        if batch_size<1:
            raise ValueError('positive prefix batch size required')
        lengths=[[len(s) for s in sequences[i:i+batch_size]] for i in range(0,len(sequences),batch_size)]
        # Right padding is future context only; causal layers cannot change the kept prefix.
        # Drop every padded position before exposing features; this is not padding-mask support.
        batches=[torch.nn.utils.rnn.pad_sequence(sequences[i:i+batch_size],batch_first=True)
                 for i in range(0,len(sequences),batch_size)]
        xs=[torch.nn.functional.embedding(ids.to(self.device),embed) for ids in batches]
        del embed
        for i,kind in enumerate(self.c['layer_types'][:stop]):
            layer=self.block('Decoder',f'layers.{i}.',self.c,kind)
            xs=[layer(x) for x in xs]
            del layer
            if (i+1)%8==0:
                print(f'frozen WGT layers {i+1}/{len(self.c["layer_types"])}',flush=True)
        if stop is None:
            norm=self.block('Qwen3_5RMSNorm','norm.',self.c['hidden_size'],self.c['rms_norm_eps'])
            xs=[norm(x) for x in xs]
        return [x[j,:length].cpu() for x,lens in zip(xs,lengths) for j,length in enumerate(lens)]


def encode(tokenizer, rows):
    sequences=[]; targets=[]; starts=[]
    for row in rows:
        prefix=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=True,
                                            add_generation_prompt=True,enable_thinking=False,return_dict=False)
        answer=tokenizer.encode(row['answer'],add_special_tokens=False)+[tokenizer.eos_token_id]
        ids=prefix+answer
        sequences.append(torch.tensor(ids[:-1]))
        targets.append(torch.tensor(answer))
        starts.append(len(prefix)-1)
    return sequences,targets,starts


def objective(adapter, x, y, head, backward=False):
    # Full-vocabulary loss in small token chunks avoids [all_tokens,248320] allocation.
    total=0.0
    for start in range(0,len(y),8):
        z=adapter(x[start:start+8].float()).to(head.dtype)
        logits=torch.nn.functional.linear(z,head).float()
        loss=torch.nn.functional.cross_entropy(logits,y[start:start+8],reduction='sum')/len(y)
        if backward:
            loss.backward()
        total+=loss.item()
    return total


def run(args):
    if args.output.exists() or args.output.is_symlink() or args.steps < 1 or args.rank < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('require a new output directory and positive steps')
    for name, expected in UPSTREAM_SHA256.items():
        if sha(args.upstream/name) != expected:
            raise ValueError(f'not the pinned Qwen3.5-4B checkpoint: {name}')
    torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True)
    baseline=args.output/'baseline'; baseline.mkdir()
    # Keep all checkpoint shards (including unused vision tensors) with config/tokenizer.
    # Copies are durable experiment data; never modify the original downloaded snapshot.
    for p in args.upstream.iterdir():
        if p.is_file():
            shutil.copy2(p,baseline/p.name)
    shutil.copy2(HERE/'model.wgt',baseline/'source.wgt')
    shutil.copy2(Path(__file__),args.output/'train.py')
    manifest={'format':'weightlang.qwen-bootstrap','version':1,'origin':'imported',
              'model':'Qwen/Qwen3.5-4B','revision':REVISION,'scope':'text-only, unpadded full sequence; no KV cache',
              'sha256':{p.name:sha(p) for p in baseline.iterdir() if p.is_file()}}
    write(baseline/'manifest.json',manifest)
    data=dataset(); write(args.output/'dataset.json',data)
    write(args.output/'acceptance.json',{'quality_accepted':False,
        'rule':'No quality acceptance from this small synthetic loss experiment. Require held-out generation/runtime and general instruction regression before promotion.',
        'selection':'minimum validation answer-token NLL; test evaluated only after selection'})
    tokenizer=AutoTokenizer.from_pretrained(baseline,local_files_only=True)
    stream=StreamedModel(baseline,baseline/'source.wgt')
    flat=[r for rows in data.values() for r in rows]
    sequences,targets,starts=encode(tokenizer,flat)
    started=time.perf_counter()
    hidden=stream.features(sequences)
    xy={}; offset=0
    for split,rows in data.items():
        count=len(rows)
        xy[split]=(torch.cat([hidden[i][starts[i]:] for i in range(offset,offset+count)]).cuda(),
                   torch.cat(targets[offset:offset+count]).cuda())
        offset+=count
    del hidden
    head=stream.tensor(stream.prefix+'embed_tokens.weight')
    adapter=stream.classes['Surgery'](stream.c['hidden_size'],args.rank).cuda()
    if args.resume is not None:
        adapter.load_state_dict(load_file(str(args.resume)),strict=True)
        manifest['resume_branch_sha256']=sha(args.resume)
        manifest['origin']='imported+trained'
        manifest['resume_policy']='continue branch weights with a fresh AdamW optimizer'
        shutil.copy2(args.resume,args.output/'parent.safetensors')
    save_file({k:v.detach().cpu() for k,v in adapter.state_dict().items()},str(baseline/'surgery.safetensors'))
    manifest['rank']=args.rank
    manifest['sha256']['surgery.safetensors']=sha(baseline/'surgery.safetensors')
    write(baseline/'manifest.json',manifest)
    with torch.no_grad():
        initial={s:objective(adapter,*xy[s],head) for s in ('train','validation','test')}
    print({'baseline_nll':initial},flush=True)
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=args.lr,weight_decay=0.01)
    history=[]; best=initial['validation']; selected=0
    best_state={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
    for step in range(1,args.steps+1):
        optimizer.zero_grad(set_to_none=True)
        train=objective(adapter,*xy['train'],head,backward=True)
        torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            val=objective(adapter,*xy['validation'],head)
        history.append(dict(step=step,train_nll=train,validation_nll=val))
        print(history[-1],flush=True)
        if val<best:
            best=val;selected=step
            best_state={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
    # Always retain the last real iteration, even when validation rejects it.
    candidate=args.output/'candidate'; candidate.mkdir()
    save_file({k:v.detach().cpu() for k,v in adapter.state_dict().items()},str(candidate/'surgery.safetensors'))
    save_file(best_state,str(candidate/'selected.safetensors'))
    torch.save({'optimizer':optimizer.state_dict(),'step':args.steps,'seed':42},args.output/'optimizer.pt')
    adapter.load_state_dict(best_state)
    with torch.no_grad():
        final={s:objective(adapter,*xy[s],head) for s in xy}
    # Candidate bundles the full pinned baseline plus its updated branch, not only a loose adapter.
    for p in baseline.iterdir():
        if p.name not in {'manifest.json','surgery.safetensors'}:
            shutil.copy2(p,candidate/p.name)
    cm=dict(manifest,origin='imported+trained',training_steps=args.steps,
            sha256={p.name:sha(p) for p in candidate.iterdir() if p.is_file()})
    write(candidate/'manifest.json',cm)
    # Reload selected/last adapter and establish that real trainable tensors changed.
    last=load_file(str(candidate/'surgery.safetensors'))
    base=load_file(str(baseline/'surgery.safetensors'))
    changed=[k for k in base if not torch.equal(base[k],last[k])]
    assert changed
    probe=xy['validation'][0][:3].float()
    reference=adapter(probe).detach()
    adapter.load_state_dict(load_file(str(candidate/'selected.safetensors')))
    torch.testing.assert_close(adapter(probe),reference,rtol=0,atol=0)
    report={'model':manifest['model'],'revision':REVISION,'rank':args.rank,'steps':args.steps,
            'lr':args.lr,'seed':42,'resume_branch_sha256':sha(args.resume) if args.resume else None,
            'evaluation_status':'development regression, previously inspected tasks' if args.resume else 'initial held-out synthetic tasks',
            'baseline_nll':initial,'selected_nll':final,
            'selected_step':selected,'history':history,'changed_tensors':changed,
            'trainable_parameters':sum(p.numel() for p in adapter.parameters()),
            'seconds':time.perf_counter()-started,'quality_accepted':False,
            'torch':torch.__version__,'source_sha256':sha(baseline/'source.wgt'),
            'dataset_sha256':sha(args.output/'dataset.json'),
            'checks':{'selected_adapter_reload_identical':True,'pretrained_blocks_frozen':True},
            'limitations':['synthetic answer-token NLL only','generation/runtime acceptance not yet run',
                           'no multimodal inputs, padding, KV cache or Responses backend integration',
                           'candidate/surgery.safetensors is last step; selected.safetensors is validation-selected']}
    write(args.output/'report.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path,help='continue saved residual weights; optimizer starts fresh')
    p.add_argument('--steps',type=int,default=8)
    p.add_argument('--rank',type=int,default=32)
    p.add_argument('--lr',type=float,default=0.0002)
    run(p.parse_args())
