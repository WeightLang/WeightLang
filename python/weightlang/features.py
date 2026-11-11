"""Sparse feature dictionaries for activation datasets and residual-preserving steering."""
import math
import hashlib
import struct
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class SparseAutoencoder(nn.Module):
    """A trainable ReLU feature dictionary with a linear decoder.

    Use ordinary PyTorch optimizers on loss(). Checkpoints use state_dict().
    Dictionary weights and semantic labels belong to the host experiment.
    """

    def __init__(self, width: int, features: int, *, device=None, dtype=None):
        super().__init__()
        if type(width) is not int or type(features) is not int or width <= 0 or features <= 0:
            raise ValueError("dictionary dimensions must be positive integers")
        self.width = width
        self.features = features
        self.encoder = nn.Linear(width, features, device=device, dtype=dtype)
        self.decoder = nn.Linear(features, width, device=device, dtype=dtype)

    @classmethod
    def from_wgt(cls, model):
        """Copy the trained WGT sparse-autoencoder example's four named tensors."""
        from contextlib import nullcontext
        with getattr(model, "_execution_lock", nullcontext()):
            names = getattr(model, "parameter_names", {})
            required = ("encoder", "encoder_bias", "decoder", "decoder_bias")
            if any(name not in names for name in required):
                raise ValueError("WGT dictionary requires encoder, encoder_bias, decoder and decoder_bias")
            tensors = [model.get_parameter(names[name]).detach().clone() for name in required]
        encoder, encoder_bias, decoder, decoder_bias = tensors
        if (encoder.ndim != 2 or decoder.shape != encoder.T.shape
                or encoder_bias.shape != (encoder.shape[0],)
                or decoder_bias.shape != (encoder.shape[1],)
                or any(t.device != encoder.device or t.dtype != encoder.dtype or not torch.isfinite(t).all()
                       for t in tensors)):
            raise ValueError("invalid WGT dictionary parameter schema")
        # Constructor initialization is immediately replaced by the trained WGT values.
        # Preserve the host CPU RNG while allocating the adapter.
        with torch.random.fork_rng(devices=[]):
            result = cls(encoder.shape[1], encoder.shape[0], device="cpu", dtype=encoder.dtype)
        result = result.to(device=encoder.device)
        with torch.no_grad():
            for target, value in zip((result.encoder.weight, result.encoder.bias,
                                      result.decoder.weight, result.decoder.bias), tensors):
                target.copy_(value)
        return result

    def encode(self, value):
        return F.relu(self.encoder(value))

    def decode(self, features):
        return self.decoder(features)

    def forward(self, value):
        return self.decode(self.encode(value))

    def loss(self, activations, *, sparsity=1e-3):
        """Mean squared reconstruction plus sparsity * mean absolute feature activation."""
        if type(sparsity) not in (float, int) or not math.isfinite(sparsity) or sparsity < 0:
            raise ValueError("sparsity must be finite and nonnegative")
        features = self.encode(activations)
        return F.mse_loss(self.decode(features), activations) + sparsity * features.abs().mean()

    def export(self):
        """Return the row-major native FeatureDictionary representation."""
        return {"width": self.width, "features": self.features,
                "encoder": self.encoder.weight.detach().cpu().flatten().tolist(),
                "encoder_bias": self.encoder.bias.detach().cpu().tolist(),
                "decoder": self.decoder.weight.detach().cpu().flatten().tolist()}

    @torch.no_grad()
    def intervene(self, activation, feature: int, value):
        """Replace one feature while retaining the original reconstruction residual."""
        if type(feature) is not int or not 0 <= feature < self.features:
            raise ValueError("feature index out of range")
        if activation.shape[-1:] != (self.width,) or not torch.isfinite(activation).all():
            raise ValueError("invalid dictionary activation")
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("feature value must be finite")
        before = self.encode(activation)
        delta = value - before[..., feature]
        patched = activation + delta.unsqueeze(-1) * self.decoder.weight[:, feature]
        if not torch.isfinite(patched).all():
            raise ValueError("nonfinite feature intervention")
        return patched


@torch.no_grad()
def feature_effects(model, value, dictionary: SparseAutoencoder, instruction: int, seed, *,
                    occurrence=0, max_features=128, max_work=None,
                    max_bytes=64 * 1024 * 1024, memory_bytes=None):
    """Measure output effects of ablating each feature at a selected dynamic boundary.

    All experiments use the same model snapshot. The total forward work is bounded.
    """
    from .program import ProgramModule, _limits
    from .research import Research, Intervention, _execute, _validate_value, _context, _tensor_payload
    if not isinstance(model, ProgramModule):
        raise TypeError("feature effects require a WGT program")
    if not isinstance(dictionary, SparseAutoencoder):
        raise TypeError("dictionary must be a SparseAutoencoder")
    if type(max_features) is not int or not 0 < dictionary.features <= max_features <= 128:
        raise ValueError("feature experiment budget exceeded")
    dictionary = deepcopy(dictionary)
    if type(instruction) is not int or type(occurrence) is not int or occurrence < 0:
        raise ValueError("invalid feature intervention target")
    _validate_value(seed, model.output_shape, value)
    maximum, memory = _limits(model, max_work, memory_bytes)
    with model._execution_lock:
        baseline = Research(max_bytes=max_bytes)
        output = _execute(model, value, baseline, maximum, memory)
        matches = [e for e in baseline.events if e.instruction == instruction and e.occurrence == occurrence]
        if not matches:
            raise ValueError("feature observation was not executed")
        activation = matches[0].before
        if tuple(activation.shape) != (dictionary.width,):
            raise ValueError("feature effects require a vector activation")
        work = baseline.work
        effects = []
        for feature in range(dictionary.features):
            patched = dictionary.intervene(activation, feature, 0.)
            session = Research(max_bytes=max_bytes,
                               interventions=[Intervention(instruction, occurrence, patched)])
            if work >= maximum:
                raise ValueError("feature experiment work budget exceeded")
            result = _execute(model, value, session, maximum - work, memory)
            work += session.work
            effect = ((output - result) * seed).sum().item()
            if not math.isfinite(effect):
                raise ValueError("nonfinite feature effect")
            effects.append(effect)
        context = _context(model, value, seed)
    digest = hashlib.sha256()
    digest.update(struct.pack("<QQ", dictionary.width, dictionary.features))
    digest.update(str(dictionary.encoder.weight.dtype).encode())
    for tensor in (dictionary.encoder.weight, dictionary.encoder.bias, dictionary.decoder.weight):
        digest.update(_tensor_payload(tensor))
    return {"output": output, "features": dictionary.encode(activation).detach(),
            "effects": effects, "work": work, "context": context, "instruction": instruction,
            "occurrence": occurrence, "dictionary_fingerprint": digest.hexdigest()}
