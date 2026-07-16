#!/usr/bin/env python3
"""
RWKV-v7a PTQ quantization.

Rules (matching llama.cpp-style PTQ with custom block size):
- All 2-D matrices (including DeepEmb s_emb.weight / s_emb_x.weight) -> INT8.
- All vectors / scalars / 3-D bias-like tensors (e.g. s0, x_k, ln_x.weight) -> FP16.
- Per-block symmetric quantization, block size = 48 along the width (last dim K).
- If K < 48, block size = K (single block per row).
- If K > 48 and not divisible by 48, the last block is a tail of size K % 48.
- Each block scale stored as UE8M0 (unsigned 8-bit exponent-only float, bias 127).
- No per-channel quantization, no secondary scales.
"""

import argparse
import os
import time
import math
import torch
import numpy as np


def float_to_ue8m0(scales: torch.Tensor, round_mode: str = 'nearest') -> torch.Tensor:
    """
    Convert positive FP32 scales to UE8M0 uint8.
    UE8M0: 0 sign bits, 8 exponent bits, 0 mantissa bits, bias 127.
    Encoded value e represents 2^(e - 127).

    round_mode:
      - 'nearest': round log2(scale) to nearest integer (default).
      - 'ceil':    round up to the next UE8M0 representable power-of-two.
                   This guarantees the stored scale is >= the ideal scale,
                   so no INT8 clipping overflow occurs.
    """
    scales = scales.float().clamp(min=2.0 ** (-127))
    log_scales = torch.log2(scales)
    if round_mode == 'ceil':
        exponents = torch.ceil(log_scales)
    elif round_mode == 'nearest':
        exponents = torch.round(log_scales)
    else:
        raise ValueError(f"Unsupported round_mode: {round_mode}")
    # UE8M0 exponent range is [-127, 128]
    exponents = exponents.clamp(-127, 128) + 127
    return exponents.clamp(0, 255).to(torch.uint8)


def ue8m0_to_float(scales_u8: torch.Tensor) -> torch.Tensor:
    """Convert UE8M0 uint8 scales back to FP32."""
    exponents = scales_u8.to(torch.float32)
    return 2.0 ** (exponents - 127.0)


def quantize_matrix(w: torch.Tensor, block_size: int = 48, ue8m0_round: str = 'nearest',
                    scale_format: str = 'ue8m0') -> tuple:
    """
    Symmetric quantize a 2-D matrix (M, K) to INT8 with per-block scale (min-max).
    Returns (q, s) where q is int8 weights of shape (M, K) and s is scales of shape (M, num_blocks).
    scale_format: 'ue8m0' (uint8) or 'fp16' (torch.float16).
    """
    assert w.dim() == 2, f"quantize_matrix expects 2-D input, got {w.dim()}-D"
    assert scale_format in ('ue8m0', 'fp16')
    M, K = w.shape
    if K < block_size:
        block_size = K
    num_blocks = (K + block_size - 1) // block_size

    w_f = w.float()
    q = torch.empty((M, K), dtype=torch.int8, device=w.device)
    s = torch.empty((M, num_blocks), dtype=torch.uint8 if scale_format == 'ue8m0' else torch.float16, device=w.device)

    for b in range(num_blocks):
        start = b * block_size
        end = min((b + 1) * block_size, K)
        block = w_f[:, start:end]
        max_abs = block.abs().max(dim=1, keepdim=True).values
        # Avoid division by zero; zero blocks get a tiny scale which still dequantize to ~0.
        ideal_scale = (max_abs / 127.0).clamp_min(2.0 ** (-127)).squeeze(1)
        if scale_format == 'ue8m0':
            s_u8 = float_to_ue8m0(ideal_scale, round_mode=ue8m0_round)
            scale = ue8m0_to_float(s_u8).unsqueeze(1)
            s[:, b] = s_u8
        else:
            scale = ideal_scale.unsqueeze(1)
            s[:, b] = ideal_scale.to(torch.float16)
        q_block = torch.round(block / scale).clamp(-127, 127).to(torch.int8)
        q[:, start:end] = q_block

    return q, s


def _search_best_scale(w_blocks: torch.Tensor, percentiles: list, ue8m0_round: str = 'nearest',
                        scale_format: str = 'ue8m0'):
    """
    KL-inspired / MSE-optimal clipping: search per-block clipping threshold by scanning percentiles.
    w_blocks: (M, num_blocks, N)
    Returns (best_scale, best_idx) where best_scale has shape (M, num_blocks) and best_idx has shape (M, num_blocks)
    with the index into the percentiles list.
    """
    M, B, N = w_blocks.shape
    abs_w = w_blocks.abs()

    # Sort absolute values once; then any percentile is just an index.
    sorted_abs, _ = torch.sort(abs_w, dim=-1)  # (M, B, N)

    best_mse = None
    best_scale = None
    best_idx = None

    for idx, p in enumerate(percentiles):
        # k-th smallest (1-indexed) corresponding to percentile p
        k = min(N - 1, int(math.ceil(p / 100.0 * N)) - 1)
        T = sorted_abs[..., k]  # (M, B)
        ideal_scale = (T / 127.0).clamp_min(2.0 ** (-127))
        if scale_format == 'ue8m0':
            scale = ue8m0_to_float(float_to_ue8m0(ideal_scale, round_mode=ue8m0_round)).unsqueeze(-1)  # (M, B, 1)
        else:
            scale = ideal_scale.unsqueeze(-1)  # (M, B, 1)

        q = torch.round(w_blocks / scale).clamp(-127, 127)
        w_hat = q * scale
        mse = ((w_blocks - w_hat) ** 2).mean(dim=-1)  # (M, B)

        if best_mse is None:
            best_mse = mse
            best_scale = ideal_scale  # store the pre-rounding scale; encoding happens later
            best_idx = torch.full((M, B), idx, dtype=torch.int32, device=w_blocks.device)
        else:
            better = mse < best_mse
            best_mse = torch.where(better, mse, best_mse)
            best_scale = torch.where(better, ideal_scale, best_scale)
            best_idx = torch.where(better, idx, best_idx)

    return best_scale, best_idx


def quantize_matrix_kl(w: torch.Tensor, block_size: int = 48,
                       percentiles: list = None, ue8m0_round: str = 'nearest',
                       scale_format: str = 'ue8m0') -> tuple:
    """
    Symmetric quantize a 2-D matrix (M, K) to INT8 with per-block scale.
    Uses KL-inspired percentile search: for each block, try several clipping thresholds
    (percentiles of |w|) and pick the one that minimises reconstruction MSE.

    Returns (q, s, chosen_idx) where q/s are as usual and chosen_idx is a tensor of percentile indices.
    """
    if percentiles is None:
        percentiles = [100.0, 99.9, 99.5, 99.0, 98.0, 96.0, 94.0, 92.0, 90.0, 85.0, 80.0]

    assert w.dim() == 2, f"quantize_matrix_kl expects 2-D input, got {w.dim()}-D"
    assert scale_format in ('ue8m0', 'fp16')
    M, K = w.shape
    if K < block_size:
        block_size = K
    num_blocks = (K + block_size - 1) // block_size
    full_blocks = K // block_size
    tail = K % block_size

    w_f = w.float()
    q = torch.empty((M, K), dtype=torch.int8, device=w.device)
    s = torch.empty((M, num_blocks), dtype=torch.uint8 if scale_format == 'ue8m0' else torch.float16, device=w.device)
    chosen_idx = torch.empty((M, num_blocks), dtype=torch.int32, device=w.device)

    # Full blocks
    if full_blocks > 0:
        w_full = w_f[:, :full_blocks * block_size].reshape(M, full_blocks, block_size)
        best_scale, best_idx = _search_best_scale(w_full, percentiles, ue8m0_round=ue8m0_round, scale_format=scale_format)
        if scale_format == 'ue8m0':
            s_u8 = float_to_ue8m0(best_scale, round_mode=ue8m0_round)
            best_scale = ue8m0_to_float(s_u8)
            s[:, :full_blocks] = s_u8
        else:
            s[:, :full_blocks] = best_scale.to(torch.float16)
        q_full = torch.round(w_full / best_scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
        q[:, :full_blocks * block_size] = q_full.reshape(M, full_blocks * block_size)
        chosen_idx[:, :full_blocks] = best_idx

    # Tail block
    if tail > 0:
        w_tail = w_f[:, full_blocks * block_size:]
        best_scale, best_idx = _search_best_scale(w_tail.unsqueeze(1), percentiles, ue8m0_round=ue8m0_round, scale_format=scale_format)
        best_scale = best_scale.squeeze(1)
        best_idx = best_idx.squeeze(1)
        if scale_format == 'ue8m0':
            s_u8 = float_to_ue8m0(best_scale, round_mode=ue8m0_round)
            best_scale = ue8m0_to_float(s_u8)
            s[:, full_blocks:] = s_u8.unsqueeze(1)
        else:
            s[:, full_blocks:] = best_scale.to(torch.float16).unsqueeze(1)
        q_tail = torch.round(w_tail / best_scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
        q[:, full_blocks * block_size:] = q_tail
        chosen_idx[:, full_blocks:] = best_idx.unsqueeze(1)

    return q, s, chosen_idx


def dequantize_matrix(q: torch.Tensor, s: torch.Tensor, block_size: int = 48) -> torch.Tensor:
    """Dequantize INT8 matrix back to FP32 using UE8M0 or FP16 scales."""
    assert q.dim() == 2 and s.dim() == 2
    M, K = q.shape
    num_blocks = s.shape[1]
    if num_blocks == 1:
        block_size = K

    w = torch.empty((M, K), dtype=torch.float32, device=q.device)
    if s.dtype == torch.uint8:
        scales_f = ue8m0_to_float(s)
    else:
        scales_f = s.to(torch.float32)

    for b in range(num_blocks):
        start = b * block_size
        end = min((b + 1) * block_size, K)
        scale = scales_f[:, b:b + 1]
        w[:, start:end] = q[:, start:end].float() * scale

    return w


def is_matrix(t: torch.Tensor) -> bool:
    """Return True for genuine 2-D matrices (both dims > 1)."""
    return t.dim() == 2 and t.shape[0] > 1 and t.shape[1] > 1


def quantize_checkpoint(src_path: str, dst_path: str, block_size: int = 48,
                        per_channel: bool = False,
                        use_kl: bool = False, kl_percentiles: list = None,
                        ue8m0_round: str = 'nearest', scale_format: str = 'ue8m0',
                        verify: bool = True):
    print(f"Loading source checkpoint: {src_path}")
    z = torch.load(src_path, map_location='cpu')

    if per_channel:
        print("Using per-channel quantization (one scale per row)")
    quantize_fn = quantize_matrix_kl if use_kl else quantize_matrix
    quantize_kwargs = {'block_size': block_size, 'ue8m0_round': ue8m0_round, 'scale_format': scale_format}
    if use_kl:
        quantize_kwargs['percentiles'] = kl_percentiles
        print(f"Using KL-inspired percentile search, candidates: {kl_percentiles}")
    else:
        print("Using min-max quantization")
    print(f"Scale format: {scale_format}, UE8M0 round mode: {ue8m0_round}")

    qz = {}
    stats = []
    total_orig_bytes = 0
    total_q_bytes = 0
    all_chosen_idx = []

    scale_dtype = torch.uint8 if scale_format == 'ue8m0' else torch.float16
    scale_bytes = 1 if scale_format == 'ue8m0' else 2

    for k in sorted(z.keys()):
        t = z[k]
        if is_matrix(t):
            row_block_size = block_size if not per_channel else t.shape[1]
            fn_kwargs = dict(quantize_kwargs, block_size=row_block_size)
            if use_kl:
                q, s, chosen_idx = quantize_fn(t, **fn_kwargs)
                all_chosen_idx.append(chosen_idx.reshape(-1))
            else:
                q, s = quantize_fn(t, **fn_kwargs)
            qz[k] = {'q': q, 's': s}

            orig_bytes = t.numel() * 2  # assume original is 16-bit
            q_bytes = q.numel() + s.numel() * scale_bytes
            total_orig_bytes += orig_bytes
            total_q_bytes += q_bytes

            if verify:
                verify_block_size = row_block_size
                w_hat = dequantize_matrix(q, s, block_size=verify_block_size)
                w_f = t.float()
                mse = ((w_f - w_hat) ** 2).mean().item()
                max_err = (w_f - w_hat).abs().max().item()
                mean_sq = (w_f ** 2).mean().item()
                if mean_sq > 0 and mse > 0:
                    snr = 10 * math.log10(mean_sq / mse)
                elif mse == 0:
                    snr = float('inf')
                else:
                    snr = -float('inf')
                stats.append((k, tuple(t.shape), mse, max_err, snr, q_bytes / orig_bytes))
        else:
            # Vectors, scalars, 3-D bias-like tensors -> FP16
            qz[k] = t.squeeze().to(dtype=torch.float16)
            total_orig_bytes += t.numel() * 2
            total_q_bytes += t.numel() * 2

    # Store metadata
    qz['_meta'] = {
        'quantization': 'int8_perchannel' if per_channel else 'int8_block48',
        'block_size': block_size if not per_channel else None,
        'per_channel': per_channel,
        'source': src_path,
        'symmetric': True,
        'scale_format': scale_format,
        'ue8m0_round': ue8m0_round if scale_format == 'ue8m0' else None,
        'use_kl': use_kl,
        'kl_percentiles': kl_percentiles if use_kl else None,
    }

    print(f"Saving quantized checkpoint: {dst_path}")
    torch.save(qz, dst_path)

    print(f"\nSize summary:")
    print(f"  Original (assumed 16-bit): {total_orig_bytes / 1024**3:.3f} GB")
    print(f"  Quantized:                 {total_q_bytes / 1024**3:.3f} GB")
    print(f"  Ratio:                     {total_q_bytes / total_orig_bytes * 100:.1f}%")

    if verify and stats:
        print(f"\n{'Layer':<50} {'Shape':<18} {'MSE':>10} {'MaxErr':>10} {'SNR(dB)':>10} {'Ratio':>8}")
        for k, shape, mse, max_err, snr, ratio in stats:
            snr_str = 'inf' if snr == float('inf') else f"{snr:>10.2f}"
            print(f"{k:<50} {str(shape):<18} {mse:>10.2e} {max_err:>10.2e} {snr_str:>10} {ratio:>8.3f}")

        all_mse = [s[2] for s in stats]
        finite_snr = [s[4] for s in stats if s[4] != float('inf') and s[4] != -float('inf')]
        print(f"\nOverall MSE mean: {np.mean(all_mse):.2e}")
        if finite_snr:
            print(f"Overall SNR mean (finite): {np.mean(finite_snr):.2f} dB, min: {np.min(finite_snr):.2f} dB, max: {np.max(finite_snr):.2f} dB")

    if use_kl and all_chosen_idx:
        chosen = torch.cat(all_chosen_idx).cpu().numpy()
        print(f"\nChosen percentile histogram (total blocks: {len(chosen)}):")
        counts = np.bincount(chosen, minlength=len(kl_percentiles))
        for i, p in enumerate(kl_percentiles):
            pct = counts[i] / len(chosen) * 100
            print(f"  p{p:>5.1f}: {counts[i]:>10,} blocks ({pct:>5.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='RWKV-v7a INT8 block-48 UE8M0 PTQ')
    parser.add_argument('--src', type=str, default='/models/rwkv7a-g1d-0.1b-20260212-ctx8192.pth',
                        help='Source .pth checkpoint')
    parser.add_argument('--dst', type=str, default=None,
                        help='Output quantized checkpoint path')
    parser.add_argument('--block-size', type=int, default=48, help='Quantization block size')
    parser.add_argument('--per-channel', action='store_true',
                        help='Use per-channel quantization (one scale per row, ignores --block-size)')
    parser.add_argument('--kl', action='store_true',
                        help='Use KL-inspired percentile search instead of min-max')
    parser.add_argument('--kl-percentiles', type=float, nargs='+',
                        default=[100.0, 99.9, 99.5, 99.0, 98.0, 96.0, 94.0, 92.0, 90.0, 85.0, 80.0],
                        help='Percentile candidates for KL search (100 = min-max)')
    parser.add_argument('--ue8m0-round', type=str, default='nearest',
                        choices=['nearest', 'ceil'],
                        help="How to round the FP32 scale to UE8M0 (nearest/ceil)")
    parser.add_argument('--scale-format', type=str, default='ue8m0',
                        choices=['ue8m0', 'fp16'],
                        help="Storage format for per-block scales: ue8m0 (uint8) or fp16")
    parser.add_argument('--no-verify', action='store_true', help='Skip error verification')
    args = parser.parse_args()

    if args.dst is None:
        base, ext = os.path.splitext(args.src)
        suffix = '_q8'
        if args.per_channel:
            suffix += '_perchannel'
        if args.kl:
            suffix += '_kl'
        if args.scale_format != 'ue8m0':
            suffix += f'_{args.scale_format}scale'
        elif args.ue8m0_round != 'nearest':
            suffix += f'_{args.ue8m0_round}'
        args.dst = f"{base}{suffix}{ext}"

    t0 = time.time()
    quantize_checkpoint(args.src, args.dst, block_size=args.block_size,
                        per_channel=args.per_channel,
                        use_kl=args.kl, kl_percentiles=args.kl_percentiles,
                        ue8m0_round=args.ue8m0_round, scale_format=args.scale_format,
                        verify=not args.no_verify)
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
