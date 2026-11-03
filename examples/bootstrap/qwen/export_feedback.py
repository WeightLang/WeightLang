"""Turn inspected failures into labelled development repair data, never a new test set."""
import argparse
import json
from pathlib import Path

from curriculum import grade
from train import sha


def export(experiment,report_path,output):
    if output.exists() or output.is_symlink():
        raise ValueError('feedback output already exists')
    report=json.loads(report_path.read_text())
    if report.get('complete') is not True:
        raise ValueError('cannot export unfinished evaluation')
    if report['provenance']['dataset_sha256']!=sha(experiment/'dataset.json'):
        raise ValueError('dataset differs from evaluated snapshot')
    if report['provenance']['candidate_manifest_sha256']!=sha(experiment/'candidate/manifest.json'):
        raise ValueError('candidate differs from evaluated snapshot')
    data=json.loads((experiment/'dataset.json').read_text())
    by_id={r['id']:r for rows in data.values() for r in rows}
    rows=[]
    for result in report['results']:
        if result['branch']!='candidate' or result['grade']['task_correct'] or result['split']!='test':
            continue
        original=by_id[result['id']]
        if not grade(original,original['answer'])['task_correct']:
            raise ValueError('reference answer no longer passes the local compiler/runtime')
        diagnostic=result['grade'].get('error','Output does not satisfy the requested task contract.')
        prompt=(original['prompt']+'\nPrevious incorrect attempt:\n'+result['generated']+
                '\nCompiler/evaluator feedback: '+diagnostic+'\nReturn only the corrected answer.')
        rows.append(dict(id=original['id']+'-repair',family=original['family'],kind='repair_feedback',
            prompt=prompt,answer=original['answer'],incorrect_attempt=result['generated'],diagnostic=diagnostic,
            source_report_sha256=sha(report_path),source_task_id=original['id'],source_split='test',
            split='development',eligible_for_blind_evaluation=False,
            reference_source=original.get('reference_source'),reference_sha256=original.get('reference_sha256')))
    with output.open('x') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    return len(rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps({'repair_rows':export(a.experiment,a.report,a.output),
                      'split':'development','eligible_for_blind_evaluation':False}))
