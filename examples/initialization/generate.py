"""构建时生成完整参数；不参与模型 forward 或训练循环。"""
import torch


def initialize(context):
    result = {}
    for spec in context.tensors:
        if spec.alias_of != spec.name:
            result[spec.name] = result[spec.alias_of]
        elif len(spec.shape) == 2:
            # weight 为 [输出, 输入]；矩形时保留重叠维度的单位映射。
            result[spec.name] = torch.eye(*spec.shape, dtype=spec.dtype)
        else:
            result[spec.name] = torch.zeros(spec.shape, dtype=spec.dtype)
    return result
