"""Small tensor adaptations used by generated PyTorch graphs."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _accumulate(value: torch.Tensor) -> torch.Tensor:
    """Use native f64 accumulation, including on devices without float64."""
    # MPS has no float64 kernels. Transfers are differentiable, so these exact
    # native-precision operations still participate in an ordinary torch graph.
    device = "cpu" if value.device.type == "mps" else value.device
    return value.to(device=device, dtype=torch.float64)


def _restore(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return value.to(device=reference.device, dtype=reference.dtype)


def linear(value: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Adaptive Linear accumulates in storage precision, with bias first.

    A BLAS linear kernel may add bias last or fuse products; that changes even
    finite WGT results under cancellation. Each column remains a torch tensor
    operation, so parameters, batching, and autograd keep their usual behavior.
    """
    result = value.new_zeros((*value.shape[:-1], weight.shape[0]))
    if bias is not None:
        result = result + bias
    for column in range(weight.shape[1]):
        result = result + value[..., column:column + 1] * weight[:, column]
    return result


def _sum_to_shape(value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Sum shared-storage gradients in storage precision and logical order."""
    padded = (1,) * (value.ndim - len(shape)) + tuple(shape)
    reduced = [axis for axis in range(value.ndim) if padded[axis] == 1 and value.shape[axis] != 1]
    kept = [axis for axis in range(value.ndim) if axis not in reduced]
    if not reduced:
        return value.reshape(shape)
    ordered = value.permute(*kept, *reduced).reshape(*(value.shape[axis] for axis in kept), -1)
    total = torch.zeros_like(ordered[..., 0])
    for index in range(ordered.shape[-1]):
        total = total + ordered[..., index]
    return total.reshape(shape)


class _Matmul(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(left, right):
        return _restore(torch.matmul(_accumulate(left), _accumulate(right)), left)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, gradient):
        left, right = ctx.saved_tensors
        # Native matrix VJPs use f32 multiply/add, unlike their f64 forwards.
        # Keeping each output-column/row addition also preserves overflow errors
        # which a double-precision gradient reduction would silently cancel.
        dx = gradient.new_zeros((*gradient.shape[:-2], left.shape[-2], left.shape[-1]))
        dy = gradient.new_zeros((*gradient.shape[:-2], right.shape[-2], right.shape[-1]))
        for column in range(right.shape[-1]):
            dx = dx + gradient[..., :, column:column + 1] * right[..., :, column].unsqueeze(-2)
        for row in range(left.shape[-2]):
            dy = dy + left[..., row, :].unsqueeze(-1) * gradient[..., row, :].unsqueeze(-2)
        return _sum_to_shape(dx, left.shape), _sum_to_shape(dy, right.shape)


def matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.ndim == 1:
        return _Matmul.apply(left.unsqueeze(0), right).squeeze(-2)
    return _Matmul.apply(left, right)


def sum(value: torch.Tensor, dim: tuple[int, ...] | None = None) -> torch.Tensor:
    return _restore(torch.sum(_accumulate(value), dim=dim), value)


def mean(value: torch.Tensor, dim: tuple[int, ...] | None = None) -> torch.Tensor:
    return _restore(torch.mean(_accumulate(value), dim=dim), value)


def _ordered_sum(value: torch.Tensor, dim: int) -> torch.Tensor:
    result = torch.zeros_like(value.select(dim, 0))
    for index in range(value.shape[dim]):
        result = result + value.select(dim, index)
    return result.unsqueeze(dim)


class _Normalized(torch.autograd.Function):
    """Native normalization VJPs consume the rounded, stored output."""

    generate_vmap_rule = True

    @staticmethod
    def forward(value, dim, logarithmic, precise_gradient):
        function = torch.log_softmax if logarithmic else torch.softmax
        return _restore(function(_accumulate(value), dim=dim), value)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, ctx.dim, ctx.logarithmic, ctx.precise_gradient = inputs
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, gradient):
        (stored,) = ctx.saved_tensors
        if ctx.logarithmic:
            result = gradient - stored.exp() * _ordered_sum(gradient, ctx.dim)
        elif ctx.precise_gradient:
            value, upstream = _accumulate(stored), _accumulate(gradient)
            result = _restore(value * (upstream - (value * upstream).sum(dim=ctx.dim, keepdim=True)), stored)
        else:
            result = stored * (gradient - _ordered_sum(stored * gradient, ctx.dim))
        return result, None, None, None


def softmax(value: torch.Tensor, dim: int, precise_gradient: bool = False) -> torch.Tensor:
    return _Normalized.apply(value, dim, False, precise_gradient)


def log_softmax(value: torch.Tensor, dim: int) -> torch.Tensor:
    return _Normalized.apply(value, dim, True, False)


def _normalization(value: torch.Tensor, dim: int, epsilon: float):
    precise = _accumulate(value)
    centered = precise - precise.mean(dim=dim, keepdim=True)
    variance = centered.square().mean(dim=dim, keepdim=True)
    inverse = torch.rsqrt(variance + epsilon)
    return centered * inverse, inverse


class _LayerNorm(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(value, gamma, beta, dim, epsilon):
        normalized, _ = _normalization(value, dim, epsilon)
        return _restore(normalized * _accumulate(gamma) + _accumulate(beta), value)

    @staticmethod
    def setup_context(ctx, inputs, output):
        value, gamma, beta, ctx.dim, ctx.epsilon = inputs
        ctx.save_for_backward(value, gamma)
        ctx.beta_shape = beta.shape

    @staticmethod
    def backward(ctx, gradient):
        value, gamma = ctx.saved_tensors
        normalized, inverse = _normalization(value, ctx.dim, ctx.epsilon)
        upstream = _accumulate(gradient)
        scaled = upstream * _accumulate(gamma)
        dx = inverse * (scaled - scaled.mean(dim=ctx.dim, keepdim=True)
                        - normalized * (scaled * normalized).mean(dim=ctx.dim, keepdim=True))
        dgamma = _sum_to_shape(_restore(upstream * normalized, value), gamma.shape)
        dbeta = _sum_to_shape(gradient, ctx.beta_shape)
        return _sum_to_shape(_restore(dx, value), value.shape), dgamma, dbeta, None, None


def layer_norm(value: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
               dim: int, epsilon: float) -> torch.Tensor:
    """Normalize one model axis; gamma and beta already have broadcast axes."""
    return _LayerNorm.apply(value, gamma, beta, dim, epsilon)


class _Convolution(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(value, kernel, bias, stride, padding, dilation, groups):
        function = (F.conv1d, F.conv2d, F.conv3d)[len(stride) - 1]
        result = function(_accumulate(value), _accumulate(kernel), _accumulate(bias),
                          stride, padding, dilation, groups)
        return _restore(result, value)

    @staticmethod
    def setup_context(ctx, inputs, output):
        value, kernel, bias, ctx.stride, ctx.padding, ctx.dilation, ctx.groups = inputs
        ctx.save_for_backward(value, kernel)
        ctx.bias_shape = bias.shape

    @staticmethod
    def backward(ctx, gradient):
        value, kernel = ctx.saved_tensors
        rank = len(ctx.stride)
        unbatched = value.ndim == rank + 1
        if unbatched:
            value, gradient = value.unsqueeze(0), gradient.unsqueeze(0)
        options = (ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        input_gradient = (torch.nn.grad.conv1d_input, torch.nn.grad.conv2d_input, torch.nn.grad.conv3d_input)[rank - 1]
        weight_gradient = (torch.nn.grad.conv1d_weight, torch.nn.grad.conv2d_weight, torch.nn.grad.conv3d_weight)[rank - 1]
        dx = input_gradient(value.shape, kernel, gradient, *options)
        dk = weight_gradient(value, kernel.shape, gradient, *options)
        bias_shape = (1, kernel.shape[0], *((1,) * rank))
        db = _sum_to_shape(gradient, bias_shape).reshape(ctx.bias_shape)
        return dx.squeeze(0) if unbatched else dx, dk, db, None, None, None, None


def conv(value: torch.Tensor, kernel: torch.Tensor, bias: torch.Tensor,
         stride: tuple[int, ...], padding: tuple[int, ...],
         dilation: tuple[int, ...], groups: int) -> torch.Tensor:
    return _Convolution.apply(value, kernel, bias, stride, padding, dilation, groups)


class _AveragePool(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(value, kernel, stride):
        return _restore(F.avg_pool2d(_accumulate(value), kernel, stride), value)

    @staticmethod
    def setup_context(ctx, inputs, output):
        value, ctx.kernel, ctx.stride = inputs
        ctx.save_for_backward(value)

    @staticmethod
    def backward(ctx, gradient):
        (value,) = ctx.saved_tensors
        result = torch.ops.aten.avg_pool2d_backward.default(gradient, value, ctx.kernel, ctx.stride, [0, 0], False, True, None)
        return result, None, None


def avg_pool2d(value: torch.Tensor, kernel: tuple[int, int], stride: tuple[int, int]) -> torch.Tensor:
    return _AveragePool.apply(value, kernel, stride)


class _MemoryWrite(torch.autograd.Function):
    @staticmethod
    def forward(memory, address, value):
        weights = _accumulate(address).unsqueeze(-1)
        result = (1 - weights) * _accumulate(memory) + weights * _accumulate(value).unsqueeze(0)
        return _restore(result, memory)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, gradient):
        memory, address, value = ctx.saved_tensors
        weights = address.unsqueeze(-1)
        dm = gradient * (1 - weights)
        da = _ordered_sum(gradient * (value.unsqueeze(0) - memory), -1).squeeze(-1)
        dv = _ordered_sum(gradient * weights, 0).squeeze(0)
        # Native tapes skip a completely dead node, including v-m overflow.
        da = torch.where((gradient == 0).all(), torch.zeros_like(da), da)
        return dm, da, dv


def memory_write(memory: torch.Tensor, address: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return _MemoryWrite.apply(memory, address, value)


class _Blend(torch.autograd.Function):
    @staticmethod
    def forward(probability, yes, no):
        return probability * yes if no is None else probability * yes + (1 - probability) * no

    @staticmethod
    def setup_context(ctx, inputs, output):
        probability, yes, no = inputs
        ctx.has_no = no is not None
        ctx.save_for_backward(probability, yes, torch.zeros_like(yes) if no is None else no)

    @staticmethod
    def backward(ctx, gradient):
        probability, yes, no = ctx.saved_tensors
        dp = gradient * (yes - no)
        # Native blend skips each zero upstream component before subtraction.
        dp = torch.where(gradient == 0, torch.zeros_like(dp), dp)
        dy = gradient * probability
        dn = gradient * (1 - probability) if ctx.has_no else None
        return _sum_to_shape(dp, probability.shape), dy, dn


def blend(probability: torch.Tensor, yes: torch.Tensor, no: torch.Tensor | None = None) -> torch.Tensor:
    return _Blend.apply(probability, yes, no)


class _Log(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(value):
        return torch.log(value)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, gradient):
        (value,) = ctx.saved_tensors
        result = gradient * value.reciprocal()
        return torch.where(gradient == 0, torch.zeros_like(result), result)


def log(value: torch.Tensor) -> torch.Tensor:
    return _Log.apply(value)


class _FiniteGradient(torch.autograd.Function):
    """A standard autograd boundary for the native finite-gradient contract."""

    generate_vmap_rule = True

    @staticmethod
    def forward(value, diagnostic_anchor):
        return value

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, gradient):
        torch._assert_async(torch.isfinite(gradient).all(), "nonfinite WeightLang gradient")
        return gradient, None


def check_gradient(value: torch.Tensor, diagnostic_anchor: torch.Tensor | None = None) -> torch.Tensor:
    """Attach backward validation to an already checked tensor, when recording."""
    if not torch.is_grad_enabled():
        # Identity returns must not expose an input/parameter's existing graph.
        return value.detach() if value.requires_grad else value
    # The local anchor retains native diagnostics for frozen storage and
    # constants without changing their requires_grad flag or accumulating .grad.
    return _FiniteGradient.apply(value, diagnostic_anchor)


def check_finite(value: torch.Tensor, diagnostic_anchor: torch.Tensor | None = None) -> torch.Tensor:
    """Check native finite activations and their backwards gradients."""
    torch._assert_async(torch.isfinite(value).all(), "nonfinite WeightLang tensor")
    return check_gradient(value, diagnostic_anchor)


def reset_gradient_anchor(module: torch.nn.Module) -> None:
    """Keep a moved diagnostic buffer a leaf, so deepcopy remains supported."""
    name = "_wgt_gradient_anchor"
    anchor = module._buffers[name]
    if not anchor.is_leaf or not anchor.requires_grad:
        module._buffers[name] = anchor.detach().requires_grad_(True)


class Linear(torch.nn.Linear):
    """nn.Linear with WGT's parameter-gradient validation at its boundaries."""

    def forward(self, value: torch.Tensor, diagnostic_anchor: torch.Tensor | None = None) -> torch.Tensor:
        weight = check_finite(self.weight, diagnostic_anchor)
        bias = check_finite(self.bias, diagnostic_anchor) if self.bias is not None else None
        return F.linear(value, weight, bias)


# A surrounding Python model can symbolically trace the sequential Linear
# subclass while retaining the validation as one ordinary call_function node.
torch.fx.wrap("check_finite")


def check_input(value: torch.Tensor, shape: tuple[int, ...], diagnostic_anchor: torch.Tensor | None = None) -> torch.Tensor:
    """Keep WGT's static trailing axes while accepting leading batch axes."""
    if not isinstance(value, torch.Tensor):
        raise TypeError("WeightLang models accept a torch.Tensor")
    if value.ndim < len(shape) or tuple(value.shape[-len(shape) :]) != shape:
        raise ValueError(f"expected input shape (..., {', '.join(map(str, shape))}), got {tuple(value.shape)}")
    if not value.is_floating_point():
        raise TypeError("WeightLang model inputs must have a floating point dtype")
    if value.numel() == 0:
        raise ValueError("WeightLang execution requires a nonempty batch")
    return check_finite(value, diagnostic_anchor)


def clip(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """WGT defines zero derivative at both clipping endpoints."""
    return torch.where(value > low, torch.where(value < high, value, high), low)


def probabilities(value: torch.Tensor) -> torch.Tensor:
    """Validate each category vector without changing its stored mass."""
    valid = (
        torch.isfinite(value).all()
        & ((value >= 0) & (value <= 1)).all()
        & ((_accumulate(value).sum(dim=-1) - 1).abs() <= 1e-5).all().to(value.device)
    )
    torch._assert_async(valid, "probabilities must be finite, in [0,1], and sum to 1 within 1e-5")
    return value
