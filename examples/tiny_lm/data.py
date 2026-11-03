"""原创合成语料：只测试小语法内的组合学习，不代表自然语言泛化。"""

import random

CONTEXT = 48
PEOPLE = ("小明", "小红", "小林", "小雨", "小白", "小青", "小月", "小星")
PLACES = ("公园", "学校", "书店", "河边", "花园", "山上")
ACTIVITIES = ("看书", "画画", "唱歌", "散步", "聊天", "写诗")
TIMES = ("早上", "中午", "晚上")


def dataset():
    sentences = [
        f"{time}，{person}来到{place}。{person}喜欢{activity}，今天很开心。"
        for time in TIMES for person in PEOPLE for place in PLACES for activity in ACTIVITIES
    ]
    # 字表来自预先定义的合成语法；0/1/2 分别为 PAD/BOS/EOS。
    tokens = ["<pad>", "<bos>", "<eos>"] + sorted(set("".join(sentences)))
    random.Random(42).shuffle(sentences)
    boundary = len(sentences) * 9 // 10
    return sentences[:boundary], sentences[boundary:], tokens


def encode_rows(sentences, tokens):
    import torch

    lookup = {char: i for i, char in enumerate(tokens)}
    rows = []
    for sentence in sentences:
        row = [1] + [lookup[char] for char in sentence] + [2]
        if len(row) > CONTEXT + 1:
            raise ValueError("sentence exceeds the fixed context")
        rows.append(row + [0] * (CONTEXT + 1 - len(row)))
    rows = torch.tensor(rows, dtype=torch.long)
    return rows[:, :-1], rows[:, 1:]
