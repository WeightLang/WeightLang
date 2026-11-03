"""Generation/runtime and general-instruction gates for a selected internal edit."""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from curriculum import grade
from evaluate import verify
from train import HERE,StreamedModel,sha,write
from train_tail import make_tail,restore_tail


@torch.no_grad()
def run(args):
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('evaluation output already exists')
    sources=args.output.with_suffix('.sources')
    sources.mkdir()
    for name in ('evaluate_tail.py','train_tail.py','evaluate.py','train.py','curriculum.py'):
        shutil.copy2(HERE/name,sources/name)
    torch.set_num_threads(4)
    roots=[args.experiment/'baseline',args.experiment/'candidate']
    manifests=[verify(p) for p in roots]
    for key in ('format','revision','start','rank','refinements','parent_rank','parent_sha256'):
        if manifests[0][key]!=manifests[1][key]:
            raise ValueError(f'baseline/candidate mismatch: {key}')
    for name,h in manifests[0]['sha256'].items():
        if name!='tail.safetensors' and manifests[1]['sha256'].get(name)!=h:
            raise ValueError(f'frozen source changed: {name}')
    config=manifests[0]
    tokenizer=AutoTokenizer.from_pretrained(roots[0],local_files_only=True)
    stream=StreamedModel(roots[0],roots[0]/'source.wgt',cache_bytes=3*1024**3)
    tails=[]
    for root in roots:
        state=load_file(str(root/'tail.safetensors'))
        parent={k[len('surgery.'):]:v for k,v in state.items() if k.startswith('surgery.')}
        tail=make_tail(stream,config['start'],config['rank'],config['refinements'],parent,config['parent_rank'])
        restore_tail(tail,state);tails.append(tail.eval())
    head=stream.tensor(stream.prefix+'embed_tokens.weight')
    # Check the changed execution schedule on real pretrained weights before any evaluation.
    probes=[torch.tensor([42,17,8]),torch.tensor([42,17,8,9,6,3,5])]
    single=stream.features(probes,stop=config['start'],embedding_weight=head,batch_size=1)
    batched=stream.features(probes,stop=config['start'],embedding_weight=head,batch_size=8)
    schedule_check={'max_abs':max((a.float()-b.float()).abs().max().item() for a,b in zip(single,batched))}
    for a,b in zip(single,batched):
        torch.testing.assert_close(a,b,rtol=0.01,atol=0.1)
    for tail in tails:
        for a,b in zip(single,batched):
            left=torch.nn.functional.linear(tail(a.to(head.device).unsqueeze(0))[0,-1],head).argmax().item()
            right=torch.nn.functional.linear(tail(b.to(head.device).unsqueeze(0))[0,-1],head).argmax().item()
            if left!=right:
                raise ValueError('prefix batching changed probe greedy token')
    schedule_check['greedy_tokens_identical']=True
    schedule_check['passed']=True
    print({'schedule_check':schedule_check},flush=True)
    training_report=json.loads((args.experiment/'report.json').read_text())
    if sha(args.experiment/'dataset.json')!=training_report['provenance']['dataset_sha256']:
        raise ValueError('evaluation dataset differs from frozen training manifest')
    data=json.loads((args.experiment/'dataset.json').read_text())
    thresholds=json.loads((args.experiment/'acceptance.json').read_text())['thresholds']
    pending=[]
    for split in ('test','general'):
        for row in data[split]:
            prefix=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=True,
                add_generation_prompt=True,enable_thinking=False,return_dict=False)
            for branch in (0,1):
                pending.append(dict(row=row,split=split,branch=branch,ids=list(prefix),tokens=[],finished=False,
                                    budget=thresholds['test_tokens' if split=='test' else 'general_tokens']))
    iteration=0
    while True:
        active=[r for r in pending if not r['finished'] and len(r['tokens'])<r['budget']]
        if not active:
            break
        # Share the unchanged prefix computation until branches actually generate different tokens.
        unique={tuple(r['ids']):None for r in active}
        tensors=stream.features([torch.tensor(ids) for ids in unique],stop=config['start'],embedding_weight=head,batch_size=8)
        features=dict(zip(unique,tensors))
        for row in active:
            x=features[tuple(row['ids'])].to(head.device).unsqueeze(0)
            z=tails[row['branch']](x)[0,-1:]
            token=torch.nn.functional.linear(z,head).argmax(-1).item()
            row['ids'].append(token);row['tokens'].append(token)
            row['finished']=token==tokenizer.eos_token_id
        iteration+=1
        print({'decode_step':iteration,'active':len(active)},flush=True)
        # Preserve partial output independently; never claim an interrupted pass completed.
        if iteration%8==0:
            write(args.experiment/'generation-progress.json',{'complete':False,'decode_step':iteration})
    results=[]
    for row in pending:
        text=tokenizer.decode(row['tokens'],skip_special_tokens=True)
        results.append(dict(id=row['row']['id'],family=row['row']['family'],kind=row['row']['kind'],
            split=row['split'],branch=['baseline','candidate'][row['branch']],prompt=row['row']['prompt'],
            expected=row['row']['answer'],generated=text,finished=row['finished'],tokens=len(row['tokens']),
            grade=grade(row['row'],text)))
    metrics={}
    for branch in ('baseline','candidate'):
        test=[r for r in results if r['split']=='test' and r['branch']==branch]
        code=[r for r in test if r['kind'] in ('code','repair')]
        general=[r for r in results if r['split']=='general' and r['branch']==branch]
        metrics[branch]={'test_task_correct':sum(r['grade']['task_correct'] for r in test),'test_total':len(test),
            'test_task_correct_rate':sum(r['grade']['task_correct'] for r in test)/len(test),
            'code_parse_rate':sum(r['grade']['parses'] for r in code)/len(code),
            'code_runtime_rate':sum(r['grade']['runs'] for r in code)/len(code),
            'general_correct':sum(r['grade']['task_correct'] for r in general),'general_total':len(general)}
    accepted=(metrics['candidate']['test_task_correct_rate']>=thresholds['test_task_correct_rate_min']
        and metrics['candidate']['code_runtime_rate']>=thresholds['test_code_runtime_rate_min']
        and metrics['baseline']['general_correct']-metrics['candidate']['general_correct']<=thresholds['general_correct_drop_max'])
    report=dict(complete=True,quality_accepted=accepted,metrics=metrics,results=results,
        execution_schedule_check=schedule_check,
        scope='first evaluation of these held-out synthetic families; six general probes, not comprehensive regression',
        provenance={'evaluator_sha256':sha(Path(__file__)),'grader_sha256':sha(HERE/'curriculum.py'),
                    'dataset_sha256':sha(args.experiment/'dataset.json'),
                    'candidate_manifest_sha256':sha(roots[1]/'manifest.json'),
                    'acceptance_sha256':sha(args.experiment/'acceptance.json'),
                    'evaluation_sources':{p.name:sha(p) for p in sources.iterdir()}})
    write(args.output,report)
    write(args.experiment/'generation-progress.json',dict(complete=True,decode_step=iteration))
    print(json.dumps({'quality_accepted':accepted,'metrics':metrics}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
