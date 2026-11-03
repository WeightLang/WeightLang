"""Emit checked default-dimension WGT sources and explicit host/parameter maps."""
import json
from pathlib import Path
import shutil
import tempfile

import torch

from project import ROOT, FAMILIES, build


def main():
    torch.set_num_threads(1)
    destination = ROOT / 'models'
    destination.mkdir(exist_ok=False)
    cache = ROOT / '.cache' / 'weightlang'
    cache.mkdir(parents=True, exist_ok=True)
    index = {}
    with tempfile.TemporaryDirectory(prefix='export-', dir=cache) as temp:
        for family in FAMILIES:
            config = {}
            if family == 'any2any_bpe':
                # Complete byte alphabet, no learned merges or training data.
                from tokenizers import Tokenizer, models
                config['semantic_bpe'] = Tokenizer(models.BPE(
                    vocab={chr(256 + i): i for i in range(256)}, merges=[])).to_str()
            _, _, kernels = build(family, Path(temp) / family, config)
            shutil.copytree(kernels.root / 'sources', destination / family)
            index[family] = {'config': config,
                             'components': [{**record, 'source': f'{family}/{record["artifact"].split("/")[-1]}.wgt'}
                                            for record in kernels.records]}
            print(f'{family}: exported {len(kernels.records)} checked sources', flush=True)
    (destination / 'index.json').write_text(json.dumps(index, indent=2) + '\n')


if __name__ == '__main__':
    main()
