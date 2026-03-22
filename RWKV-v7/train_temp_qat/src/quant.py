########################################################################################################
# Per-channel weight quantization with STE for QAT
########################################################################################################

import torch
import torch.nn as nn
from torch.nn import functional as F


class _FakeQuantSTE(torch.autograd.Function):
    """Fake-quantize with Straight-Through Estimator for the backward pass.

    Forward: clamp(round(w / scale), qmin, qmax) * scale  (symmetric per-channel)
    Backward: gradient passes straight through (STE), zeroed where clamped.
    """
    @staticmethod
    def forward(ctx, w, scale, qmin, qmax):
        # scale: [out_features, 1], w: [out_features, in_features]
        w_scaled = w / scale
        w_clamped = torch.clamp(w_scaled, qmin, qmax)
        # save mask of values not clamped, for STE
        ctx.save_for_backward(w_clamped.eq(w_scaled))
        return torch.round(w_clamped) * scale

    @staticmethod
    def backward(ctx, grad_output):
        (in_range,) = ctx.saved_tensors
        # STE: pass gradient only where the value was not clamped
        return grad_output * in_range, None, None, None


class QuantizedLinear(nn.Linear):
    """Drop-in replacement for nn.Linear with optional per-channel weight quantization.

    Quantization is symmetric (zero-point == 0).  Scale is one value per output
    channel, stored as a non-trainable buffer.

    Args:
        in_features, out_features, bias: same as nn.Linear.
        n_bits: quantization bit-width (default 8).
        enable_quant: if False, behaves exactly like nn.Linear (default True).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 n_bits: int = 8, enable_quant: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        self.n_bits = n_bits
        self.enable_quant = enable_quant
        self.qmax = float(2 ** (n_bits - 1) - 1)
        self.qmin = -self.qmax
        # Per-channel scale: one value per output channel.
        # Stored as a buffer (not a trainable parameter).
        self.register_buffer('q_scale', torch.ones(out_features))

    @torch.no_grad()
    def init_quant_params(self):
        """Initialize q_scale from weight min/max (per output channel)."""
        w = self.weight.float()                    # [out, in]
        abs_max = w.abs().amax(dim=1).clamp(min=1e-8)   # [out]
        self.q_scale.copy_((abs_max / self.qmax).to(self.q_scale.dtype))

    def forward(self, x):
        if not self.enable_quant:
            return F.linear(x, self.weight, self.bias)

        scale = self.q_scale.to(dtype=self.weight.dtype).view(-1, 1)  # [out, 1]
        w_fq = _FakeQuantSTE.apply(self.weight, scale, self.qmin, self.qmax)
        return F.linear(x, w_fq, self.bias)

    def extra_repr(self):
        base = super().extra_repr()
        return f'{base}, n_bits={self.n_bits}, enable_quant={self.enable_quant}'
