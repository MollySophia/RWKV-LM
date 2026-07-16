#!/usr/bin/env python3
"""
End-to-end PPL evaluation for RWKV-v7a, supporting both original FP16 and quantized INT8 checkpoints.

Usage examples:
    python eval_ppl.py --model /models/rwkv7a-g1d-0.1b-20260212-ctx8192
    python eval_ppl.py --model /models/rwkv7a-g1d-0.1b-20260212-ctx8192 --quantized /models/rwkv7a-g1d-0.1b-20260212-ctx8192_q8
"""

import argparse
import json
import math
import os
import time
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch._C._jit_set_autocast_mode(False)

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyStatic = torch.jit.script

DTYPE = torch.half
HEAD_SIZE = 64


from torch.utils.cpp_extension import load
load(name="rwkv7_state_fwd_fp16", sources=["cuda/rwkv7_state_fwd_fp16.cpp", f"cuda/rwkv7_state_fwd_fp16.cu"], is_python_module=False,
     verbose=False, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])


class WKV_7(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b, elapsed_t):
        with torch.no_grad():
            T, C = r.size()
            H = C // HEAD_SIZE
            N = HEAD_SIZE
            assert HEAD_SIZE == C // H
            assert all(x.dtype == DTYPE for x in [state, r, w, k, v, a, b])
            assert all(x.is_contiguous() for x in [r, w, k, v, a, b])
            y = torch.empty((T, C), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
            if T == 1:
                torch.ops.rwkv7_state_fwd_fp16.forward_one(1, C, H, state, r, w, k, v, a, b, y, elapsed_t)
            else:
                torch.ops.rwkv7_state_fwd_fp16.forward_seq(1, T, C, H, state, r, w, k, v, a, b, y, elapsed_t)
            return y


def RWKV7_OP(state, r, w, k, v, a, b, elapsed_t):
    return WKV_7.apply(state, r, w, k, v, a, b, elapsed_t)


@MyStatic
def RWKV_x070_TMix_one(layer_id: int, H: int, N: int, x, x_prev, v_first, state,
                       x_r, x_w, x_k, x_v, x_a, x_g,
                       w0, w1, w2, a0, a1, a2, v0, v1, v2,
                       g1, g2, k_k, k_a, r_k,
                       R_, K_, V_, O_, ln_w, ln_b):
    xx = x_prev - x
    xr, xw, xk, xv, xa, xg = x + xx * x_r, x + xx * x_w, x + xx * x_k, x + xx * x_v, x + xx * x_a, x + xx * x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(H, N), dim=-1, p=2.0).view(H * N)
    k = k * (1 + (a - 1) * k_a)
    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
    w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float()))

    vk = v.view(H, N, 1) @ k.view(H, 1, N)
    ab = (-kk).view(H, N, 1) @ (kk * a).view(H, 1, N)
    state = state * w.view(H, 1, N) + state @ ab.float() + vk.float()
    xx = (state.to(dtype=x.dtype) @ r.view(H, N, 1))

    xx = torch.nn.functional.group_norm(xx.view(1, H * N), num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5).view(H * N)
    xx = xx + ((r * k * r_k).view(H, N).sum(dim=-1, keepdim=True) * v.view(H, N)).view(H * N)
    return (xx * g) @ O_, x, state, v_first


@MyStatic
def RWKV_x070_TMix_seq(layer_id: int, H: int, N: int, x, x_prev, v_first, state,
                       x_r, x_w, x_k, x_v, x_a, x_g,
                       w0, w1, w2, a0, a1, a2, v0, v1, v2,
                       g1, g2, k_k, k_a, r_k,
                       R_, K_, V_, O_, ln_w, ln_b):
    T = x.shape[0]
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1, :])) - x
    xr, xw, xk, xv, xa, xg = x + xx * x_r, x + xx * x_w, x + xx * x_k, x + xx * x_v, x + xx * x_a, x + xx * x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(T, H, N), dim=-1, p=2.0).view(T, H * N)
    k = k * (1 + (a - 1) * k_a)
    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    w = w0 + w
    xx = RWKV7_OP(state, r, w, k, v, -kk, kk * a, torch.zeros(1, dtype=torch.int32, device=state.device))

    xx = torch.nn.functional.group_norm(xx.view(T, H * N), num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5).view(T, H * N)
    xx = xx + ((r * k * r_k).view(T, H, N).sum(dim=-1, keepdim=True) * v.view(T, H, N)).view(T, H * N)
    return (xx * g) @ O_, x[-1, :], state, v_first


@MyStatic
def RWKV_x070_CMix_one(x, x_prev, x_k, K_, V_, semb_, s1_, s2_, s0_):
    xx = x_prev - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    ss = (x @ s1_) @ semb_.view(32, 32)
    k = k * ((ss @ s2_) + s0_)
    return k @ V_, x


@MyStatic
def RWKV_x070_CMix_seq(x, x_prev, x_k, K_, V_, semb_, s1_, s2_, s0_):
    T, C = x.shape
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1, :])) - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    ss = (x @ s1_).view(T, 1, 32) @ semb_.view(T, 32, 32)
    k = k * ((ss.view(T, 32) @ s2_) + s0_)
    return k @ V_, x[-1, :]


class RWKV_x070(MyModule):
    def __init__(self, args, z):
        super().__init__()
        self.args = args
        self.n_embd = args.n_embd
        self.n_layer = args.n_layer
        self.eval()

        self.n_head, self.head_size = z['blocks.0.att.r_k'].shape

        keys = list(z.keys())
        for k in keys:
            if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
                z[k] = z[k].t()
            z[k] = z[k].squeeze().to(dtype=DTYPE)
            if k.endswith('att.r_k'):
                z[k] = z[k].flatten()
        assert self.head_size == args.head_size

        z['emb.weight'] = F.layer_norm(z['emb.weight'], (args.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])

        for i in range(self.n_layer):
            z[f'blocks.{i}.ffn.s_emb.weight'] = z[f'blocks.{i}.ffn.s_emb.weight'] + z['emb.weight'] @ z[f'blocks.{i}.ffn.s_emb_x.weight'].t()

        z['blocks.0.att.v0'] = z['blocks.0.att.a0']
        z['blocks.0.att.v1'] = z['blocks.0.att.a1']
        z['blocks.0.att.v2'] = z['blocks.0.att.a2']

        self.z = z

    def forward(self, idx, state, full_output=False):
        if state is None:
            state = [None for _ in range(self.args.n_layer * 3)]
            for i in range(self.args.n_layer):
                state[i * 3 + 0] = torch.zeros(self.args.n_embd, dtype=DTYPE, requires_grad=False, device="cuda")
                state[i * 3 + 1] = torch.zeros((self.args.n_embd // self.args.head_size, self.args.head_size, self.args.head_size), dtype=DTYPE, requires_grad=False, device="cuda")
                state[i * 3 + 2] = torch.zeros(self.args.n_embd, dtype=DTYPE, requires_grad=False, device="cuda")

        if type(idx) is list:
            if len(idx) > 1:
                return self.forward_seq(idx, state, full_output)
            else:
                return self.forward_one(idx[0], state)
        else:
            return self.forward_one(idx, state)

    @MyFunction
    def forward_one(self, idx: int, state: List[torch.Tensor]):
        with torch.no_grad():
            z = self.z
            x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb + 'ln1.weight'], bias=z[bbb + 'ln1.bias'])

                xx, state[i * 3 + 0], state[i * 3 + 1], v_first = RWKV_x070_TMix_one(
                    i, self.n_head, self.head_size, xx, state[i * 3 + 0], v_first, state[i * 3 + 1],
                    z[att + 'x_r'], z[att + 'x_w'], z[att + 'x_k'], z[att + 'x_v'], z[att + 'x_a'], z[att + 'x_g'],
                    z[att + 'w0'], z[att + 'w1'], z[att + 'w2'], z[att + 'a0'], z[att + 'a1'], z[att + 'a2'],
                    z[att + 'v0'], z[att + 'v1'], z[att + 'v2'],
                    z[att + 'g1'], z[att + 'g2'], z[att + 'k_k'], z[att + 'k_a'], z[att + 'r_k'],
                    z[att + 'receptance.weight'], z[att + 'key.weight'], z[att + 'value.weight'], z[att + 'output.weight'],
                    z[att + 'ln_x.weight'], z[att + 'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb + 'ln2.weight'], bias=z[bbb + 'ln2.bias'])

                xx, state[i * 3 + 2] = RWKV_x070_CMix_one(
                    xx, state[i * 3 + 2], z[ffn + 'x_k'], z[ffn + 'key.weight'], z[ffn + 'value.weight'],
                    z[ffn + 's_emb.weight'][idx], z[ffn + 's1'], z[ffn + 's2'], z[ffn + 's0'])
                x = x + xx

            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x, state

    @MyFunction
    def forward_seq(self, idx: List[int], state: List[torch.Tensor], full_output: bool = False):
        with torch.no_grad():
            z = self.z
            x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb + 'ln1.weight'], bias=z[bbb + 'ln1.bias'])

                xx, state[i * 3 + 0], state[i * 3 + 1], v_first = RWKV_x070_TMix_seq(
                    i, self.n_head, self.head_size, xx, state[i * 3 + 0], v_first, state[i * 3 + 1],
                    z[att + 'x_r'], z[att + 'x_w'], z[att + 'x_k'], z[att + 'x_v'], z[att + 'x_a'], z[att + 'x_g'],
                    z[att + 'w0'], z[att + 'w1'], z[att + 'w2'], z[att + 'a0'], z[att + 'a1'], z[att + 'a2'],
                    z[att + 'v0'], z[att + 'v1'], z[att + 'v2'],
                    z[att + 'g1'], z[att + 'g2'], z[att + 'k_k'], z[att + 'k_a'], z[att + 'r_k'],
                    z[att + 'receptance.weight'], z[att + 'key.weight'], z[att + 'value.weight'], z[att + 'output.weight'],
                    z[att + 'ln_x.weight'], z[att + 'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb + 'ln2.weight'], bias=z[bbb + 'ln2.bias'])

                xx, state[i * 3 + 2] = RWKV_x070_CMix_seq(
                    xx, state[i * 3 + 2], z[ffn + 'x_k'], z[ffn + 'key.weight'], z[ffn + 'value.weight'],
                    z[ffn + 's_emb.weight'][idx], z[ffn + 's1'], z[ffn + 's2'], z[ffn + 's0'])
                x = x + xx

            if not full_output:
                x = x[-1, :]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x, state


class RWKV_TOKENIZER:
    def __init__(self, file_name):
        self.idx2token = {}
        sorted_tokens = []
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted_tokens += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        self.table = [[[] for _ in range(256)] for _ in range(256)]
        self.good = [set() for _ in range(256)]
        self.wlen = [0 for _ in range(256)]

        for i in reversed(range(len(sorted_tokens))):
            s = sorted_tokens[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> list:
        src_len = len(src)
        tokens = []
        i = 0
        while i < src_len:
            s = src[i: i + 1]
            if i < src_len - 1:
                s1 = int(src[i + 1])
                s0 = int(src[i])
                if s1 in self.good[s0]:
                    sss = src[i: i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)
        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')


def ue8m0_to_float(scales_u8: torch.Tensor) -> torch.Tensor:
    exponents = scales_u8.to(torch.float32)
    return 2.0 ** (exponents - 127.0)


def dequantize_matrix(q: torch.Tensor, s: torch.Tensor, block_size: int = 48) -> torch.Tensor:
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


def load_weights(model_path: str, quantized_path: str = None):
    """Load original or quantized checkpoint and prepare weights for the model."""
    if quantized_path:
        print(f"Loading quantized checkpoint: {quantized_path}")
        qz = torch.load(quantized_path + '.pth', map_location='cpu')
        meta = qz.pop('_meta', {})
        block_size = meta.get('block_size', 48)
        z = {}
        for k, v in qz.items():
            if isinstance(v, dict) and 'q' in v and 's' in v:
                z[k] = dequantize_matrix(v['q'], v['s'], block_size=block_size)
            else:
                z[k] = v.to(torch.float32)
        # Move everything to CUDA for the model
        z = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in z.items()}
        print(f"  Quantization metadata: {meta}")
    else:
        print(f"Loading original checkpoint: {model_path}")
        z = torch.load(model_path + '.pth', map_location='cuda')
    return z


def evaluate_lambada(model, tokenizer, dataset_path: str, max_samples: int = None):
    print(f'\nEvaluating LAMBADA from {dataset_path} ...')
    with open(dataset_path, "r", encoding="utf-8") as f:
        todo = [json.loads(line) for line in f]
    todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]

    if max_samples:
        todo = todo[:max_samples]

    xsum = 0.0
    xcnt = 0
    xacc = 0
    t0 = time.time()

    for d in todo:
        src = [0] + tokenizer.encode(d[0])
        dst = tokenizer.encode(d[1])

        out, _ = model.forward(src + dst, None, full_output=True)

        logits = 0.0
        correct = True
        for i in range(len(dst)):
            ooo = out[len(src) - 1 + i].float()
            probs = F.softmax(ooo, dim=-1)
            logits += math.log(probs[dst[i]].item())
            if torch.argmax(probs).item() != dst[i]:
                correct = False

        xcnt += 1
        xsum += logits
        xacc += 1 if correct else 0
        if xcnt % 100 == 0 or xcnt == len(todo):
            ppl = math.exp(-xsum / xcnt)
            acc = xacc / xcnt * 100
            print(f'  {xcnt}/{len(todo)}  ppl {ppl:.2f}  acc {acc:.2f}%  ({time.time()-t0:.1f}s)')

    final_ppl = math.exp(-xsum / xcnt)
    final_acc = xacc / xcnt * 100
    return final_ppl, final_acc


def evaluate_src_txt(model, tokenizer, src_path: str, max_tokens: int = None, report_every: int = 512):
    """Evaluate cumulative per-token perplexity on a flat token file.

    For a sequence of tokens [t_1, t_2, ..., t_T], at each position i the model
    outputs logits for t_i given t_1...t_{i-1}.  Cumulative PPL at position i is:

        PPL(i) = exp( -(1/i) * sum_{j=1..i} log P(t_j | t_<j) )

    We report PPL at every `report_every` token (and at the end), which lets you
    compare how a quantization/backend diverges from the baseline as the sequence
    length grows.
    """
    print(f'\nEvaluating src txt from {src_path} ...')
    with open(src_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    if max_tokens:
        tokens = tokens[:max_tokens]
    print(f'  total tokens: {len(tokens)}')

    # prepend a dummy BOS token (index 0) so the first real token is predicted
    idx = [0] + tokens
    out, _ = model.forward(idx, None, full_output=True)

    nll_sum = 0.0
    count = 0
    results = []
    t0 = time.time()
    for i in range(len(tokens)):
        ooo = out[i].float()
        probs = F.softmax(ooo, dim=-1)
        nll_sum += -math.log(probs[tokens[i]].item())
        count += 1
        if count % report_every == 0 or count == len(tokens):
            ppl = math.exp(nll_sum / count)
            results.append((count, ppl))
            print(f'  tokens {count:>6}: cumulative ppl {ppl:.4f}  ({time.time()-t0:.1f}s)')
    return results


def main():
    parser = argparse.ArgumentParser(description='RWKV-v7a LAMBADA PPL evaluation')
    parser.add_argument('--model', type=str, default='/models/rwkv7a-g1d-0.1b-20260212-ctx8192',
                        help='Original model path (without .pth)')
    parser.add_argument('--quantized', type=str, default=None,
                        help='Quantized model path (without .pth); if omitted, evaluate original')
    parser.add_argument('--dataset', type=str, default='misc/lambada_test.jsonl',
                        help='Path to LAMBADA JSONL dataset')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit number of LAMBADA samples for quick test')
    parser.add_argument('--src-txt', type=str, default=None,
                        help='Path to a text file for cumulative per-token PPL evaluation')
    parser.add_argument('--src-max-tokens', type=int, default=None,
                        help='Limit number of tokens for src txt evaluation')
    parser.add_argument('--src-report-every', type=int, default=512,
                        help='Report cumulative PPL every N tokens for src txt')
    parser.add_argument('--skip-lambada', action='store_true',
                        help='Skip LAMBADA evaluation and only run src txt')
    args = parser.parse_args()

    args_model = types.SimpleNamespace()
    args_model.MODEL_NAME = args.model
    args_model.n_layer = 12
    args_model.n_embd = 768
    args_model.vocab_size = 65536
    args_model.head_size = 64

    z = load_weights(args.model, args.quantized)
    model = RWKV_x070(args_model, z)

    tokenizer = RWKV_TOKENIZER("rwkv_vocab_v20230424.txt")

    if not args.skip_lambada:
        ppl, acc = evaluate_lambada(model, tokenizer, args.dataset, max_samples=args.max_samples)
        print(f'\nFinal: PPL = {ppl:.2f}, Accuracy = {acc:.2f}%')

    if args.src_txt:
        evaluate_src_txt(model, tokenizer, args.src_txt, max_tokens=args.src_max_tokens,
                         report_every=args.src_report_every)


if __name__ == '__main__':
    main()
