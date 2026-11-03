"""Load a complete WeightLang artifact and sample one character at a time."""

import argparse
import json
from pathlib import Path

import torch

from weightlang import load_artifact

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "weights" / "tiny_lm_demo" / "trained"


@torch.inference_mode()
def generate(artifact, tokens, prompt, *, max_new=60, temperature=0.7, seed=42):
    if temperature <= 0 or max_new < 1:
        raise ValueError("temperature and max_new must be positive")
    lookup = {char: i for i, char in enumerate(tokens)}
    unknown = sorted(set(prompt) - set(tokens[3:]))
    if unknown:
        raise ValueError(f"提示包含字表外字符：{''.join(unknown)}")
    ids = [1] + [lookup[char] for char in prompt]
    context = artifact.manifest["interface"]["input"]["shape"][0]
    rng = torch.Generator().manual_seed(seed)
    artifact.module.eval()
    result = prompt
    for _ in range(max_new):
        window = ids[-context:]
        # 右侧补齐固定接口长度；因果掩码保证这些 PAD 不影响当前预测。
        x = torch.tensor(window + [0] * (context - len(window)))
        logits = artifact.predict(x)[len(window) - 1].clone()
        logits[:2] = -torch.inf
        next_id = torch.multinomial((logits / temperature).softmax(-1), 1, generator=rng).item()
        if next_id == 2:
            break
        result += tokens[next_id]
        ids.append(next_id)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="早上，小明")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new", type=int, default=60)
    args = parser.parse_args()
    torch.set_num_threads(4)
    tokens = json.loads((args.model / "tokenizer.json").read_text(encoding="utf-8"))
    artifact = load_artifact(args.model)
    try:
        print(generate(artifact, tokens, args.prompt, temperature=args.temperature,
                       seed=args.seed, max_new=args.max_new))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
