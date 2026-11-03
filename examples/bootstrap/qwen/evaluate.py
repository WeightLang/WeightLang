"""Greedy generation from retained WGT snapshots, with baseline/candidate comparison."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from train import StreamedModel, sha, write


def verify(root):
    manifest=json.loads((root/'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if Path(name).name != name or sha(root/name)!=expected:
            raise ValueError(f'snapshot hash mismatch: {name}')
    return manifest


@torch.no_grad()
def run(args):
    if args.output.exists() or args.output.is_symlink() or args.max_tokens < 1:
        raise ValueError('output already exists')
    torch.set_num_threads(4)
    roots=[args.experiment/'baseline',args.experiment/'candidate']
    manifests=[verify(root) for root in roots]
    # Both branches use the same frozen weights/source; only the residual differs.
    for name,h in manifests[0]['sha256'].items():
        if name != 'surgery.safetensors' and manifests[1]['sha256'].get(name)!=h:
            raise ValueError(f'baseline/candidate base differs: {name}')
    stream=StreamedModel(roots[0],roots[0]/'source.wgt')
    tokenizer=AutoTokenizer.from_pretrained(roots[0],local_files_only=True)
    adapters=[]
    for root,manifest in zip(roots,manifests):
        adapter=stream.classes['Surgery'](stream.c['hidden_size'],manifest['rank']).cuda()
        weight='selected.safetensors' if root.name=='candidate' else 'surgery.safetensors'
        adapter.load_state_dict(load_file(str(root/weight)))
        adapters.append(adapter.eval())
    rows=json.loads((args.experiment/'dataset.json').read_text())['test']
    rows=[r for r in rows if r['kind']=='shape']
    results=[]
    for row in rows:
        prefix=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        sequences=[list(prefix),list(prefix)];done=[False,False];generated=[[],[]]
        for step in range(args.max_tokens):
            features=stream.features([torch.tensor(s) for s in sequences])
            head=stream.tensor(stream.prefix+'embed_tokens.weight')
            for i in range(2):
                if done[i]:
                    continue
                hidden=adapters[i](features[i][-1:].cuda().float()).to(head.dtype)
                token=torch.nn.functional.linear(hidden,head).argmax(-1).item()
                sequences[i].append(token);generated[i].append(token)
                done[i]=token==tokenizer.eos_token_id
            del head
            print({'token_step':step+1,'text':[tokenizer.decode(g,skip_special_tokens=True) for g in generated]},flush=True)
            if all(done):
                break
        results.append(dict(row,outputs={name:{'text':tokenizer.decode(g,skip_special_tokens=True),
            'exact':tokenizer.decode(g,skip_special_tokens=True).strip()==row['answer'].strip(),
            'finished':finished,'tokens':len(g)} for name,g,finished in zip(['baseline','selected'],generated,done)}))
    training_report=json.loads((args.experiment/'report.json').read_text())
    write(args.output,{'scope':'synthetic shape generation only; not code/runtime or general language acceptance',
                       'evaluation_status':training_report.get('evaluation_status','initial held-out synthetic tasks'),
                       'evaluator_sha256':sha(Path(__file__)),
                       'source_sha256':manifests[0]['sha256']['source.wgt'],
                       'revision':manifests[0]['revision'],
                       'max_tokens':args.max_tokens,'results':results})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-tokens',type=int,default=16)
    run(p.parse_args())
