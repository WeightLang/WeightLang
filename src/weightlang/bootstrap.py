"""Small byte-tokenized WGT model backend; no pretrained language capability implied."""

from __future__ import annotations

import codecs
import json
from pathlib import Path

import torch

from .artifact import load_artifact

TOKENIZER = {"format": "weightlang.byte-chat.v1", "vocab_size": 264,
             "pad": 0, "bos": 1, "eos": 2, "byte_offset": 3,
             "roles": {"user": 259, "assistant": 260, "system": 261, "developer": 262},
             "end_message": 263}


def encode(text: str) -> list[int]:
    return [b + 3 for b in text.encode("utf-8")]


def prompt_tokens(messages: list[dict[str, str]]) -> list[int]:
    ids = [1]
    for message in messages:
        ids += [TOKENIZER["roles"][message["role"]]] + encode(message["content"]) + [263]
    return ids + [260]


class ByteModel:
    def __init__(self, path: str | Path):
        path = Path(path)
        if json.loads((path / "tokenizer.json").read_text()) != TOKENIZER:
            raise ValueError("unsupported tokenizer")
        self.artifact = load_artifact(path)
        self.artifact.module.eval()
        self.context = self.artifact.manifest["interface"]["input"]["shape"][0]
        if self.artifact.manifest["interface"]["output"]["shape"] != [self.context, 264]:
            raise ValueError("model output does not match byte tokenizer")

    def prepare(self, messages, max_tokens):
        ids = prompt_tokens(messages)
        if len(ids) + max_tokens > self.context:
            raise ValueError(f"input plus output budget exceeds {self.context} tokens (UTF-8 bytes and markers)")
        return ids

    @torch.inference_mode()
    def generate(self, ids, max_tokens):
        """Yield (decoded text, consumed tokens, EOS). Greedy decoding; each step is counted."""
        ids = list(ids)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        for _ in range(max_tokens):
            x = torch.tensor(ids + [0] * (self.context - len(ids)))
            logits = self.artifact.predict(x)[len(ids) - 1].clone()
            logits[:2] = -torch.inf
            logits[259:] = -torch.inf
            token = logits.argmax().item()
            if token == 2:
                yield decoder.decode(b"", final=True), 1, True
                return
            ids.append(token)
            yield decoder.decode(bytes([token - 3])), 1, False
        tail = decoder.decode(b"", final=True)
        if tail:
            yield tail, 0, False
