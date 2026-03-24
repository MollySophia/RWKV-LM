########################################################################################################
# Evaluate RWKV with different quantization settings on LAMBADA
########################################################################################################

import torch, types, os, gc, math, json
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
np.set_printoptions(precision=4, suppress=True, linewidth=200)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch._C._jit_set_autocast_mode(False)

import sys
sys.path.insert(0, '..')

args = types.SimpleNamespace()

DTYPE = torch.half
USE_CUDA_KERNEL = False  # Use pure Python implementation for compatibility
HEAD_SIZE = 64  # updated at runtime by load_model

def infer_args_from_checkpoint(path):
    """Infer model config from checkpoint shapes."""
    w = torch.load(path, map_location='cpu')
    a = types.SimpleNamespace()
    a.vocab_size  = w['emb.weight'].shape[0]
    a.n_embd      = w['emb.weight'].shape[1]
    a.n_layer     = max(int(k.split('.')[1]) for k in w if k.startswith('blocks.')) + 1
    a.dim_att     = w['blocks.0.att.receptance.weight'].shape[0]
    a.dim_ffn     = w['blocks.0.ffn.key.weight'].shape[0]
    a.head_size_a = w['blocks.0.att.r_k'].shape[1]
    a.D_DECAY_LORA = w['blocks.0.att.w1'].shape[1]
    a.D_AAA_LORA   = w['blocks.0.att.a1'].shape[1]
    a.D_MV_LORA    = w['blocks.0.att.v1'].shape[1]
    a.D_GATE_LORA  = w['blocks.0.att.g1'].shape[1]
    return a, w

MyModule = nn.Module
def MyFunction(x): return x
def MyStatic(x): return x

########################################################################################################
# Tokenizer
########################################################################################################

class RWKV_TOKENIZER():
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

        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(range(len(sorted_tokens))):
            s = sorted_tokens[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len = len(src)
        tokens = []
        i = 0
        while i < src_len:
            s = src[i:i+1]
            if i < src_len - 1:
                s1 = int(src[i+1])
                s0 = int(src[i])
                if s1 in self.good[s0]:
                    sss = src[i:i+self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)
        return tokens

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens)).decode('utf-8', errors='replace')

tokenizer = RWKV_TOKENIZER("../rwkv_vocab_v20230424.txt")

########################################################################################################
# CUDA Kernel
########################################################################################################

if USE_CUDA_KERNEL:
    from torch.utils.cpp_extension import load
    load(name="wkv7_eval", sources=["../cuda/wkv7_op.cpp", "../cuda/wkv7.cu"], is_python_module=False,
         verbose=False, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
    class WKV_7(torch.autograd.Function):
        @staticmethod
        def forward(ctx, r, w, k, v, a, b):
            B, T, C = r.size()
            H = C // HEAD_SIZE
            N = HEAD_SIZE
            assert HEAD_SIZE == C // H
            assert r.dtype == DTYPE
            assert all(x.is_contiguous() for x in [r,w,k,v,a,b])
            y = torch.empty((B, T, C), device=k.device, dtype=DTYPE, memory_format=torch.contiguous_format)
            torch.ops.wkv7_eval.forward(B, T, C, H, r, w, k, v, a, b, y)
            return y
    def RWKV7_OP(r, w, k, v, a, b):
        return WKV_7.apply(r, w, k, v, a, b)
else:
    def RWKV7_OP(r, w, k, v, a, b):
        B, T, C = r.size()
        H = C // HEAD_SIZE
        N = HEAD_SIZE
        r = r.view(B, T, H, N).float()
        k = k.view(B, T, H, N).float()
        v = v.view(B, T, H, N).float()
        a = a.view(B, T, H, N).float()
        b = b.view(B, T, H, N).float()
        w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
        out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
        state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)
        for t in range(T):
            kk = k[:, t, :].view(B, H, 1, N)
            rr = r[:, t, :].view(B, H, N, 1)
            vv = v[:, t, :].view(B, H, N, 1)
            aa = a[:, t, :].view(B, H, N, 1)
            bb = b[:, t, :].view(B, H, 1, N)
            state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
            out[:, t, :] = (state @ rr).view(B, H, N)
        return out.view(B, T, C).to(torch.half)

########################################################################################################
# Quantization Functions
########################################################################################################

def quantize_tensor(w, n_bits=8):
    """Per-channel symmetric quantization"""
    w = w.float()
    qmax = 2 ** (n_bits - 1) - 1
    abs_max = w.abs().amax(dim=1).clamp(min=1e-8)
    scale = abs_max / qmax
    w_scaled = w / scale.view(-1, 1)
    w_quant = torch.round(torch.clamp(w_scaled, -qmax, qmax))
    w_dequant = w_quant * scale.view(-1, 1)
    return w_dequant.to(torch.half), scale

def apply_quantization_to_model(model, n_bits=8, lmhead_bits=None, q_scales=None):
    """Apply quantization to all Linear layers in the model"""
    if lmhead_bits is None:
        lmhead_bits = n_bits
    if q_scales is None:
        q_scales = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight is not None:
            bits = lmhead_bits if name == 'head' else n_bits
            qmax = 2 ** (bits - 1) - 1

            if name in q_scales:
                # Use saved q_scale from checkpoint
                scale = q_scales[name].float().cuda()
                w = module.weight.data.float()
                w_scaled = w / scale.view(-1, 1)
                w_quant = torch.round(torch.clamp(w_scaled, -qmax, qmax))
                w_dequant = w_quant * scale.view(-1, 1)
                module.weight.data = w_dequant.to(torch.half)
            else:
                # Fall back to computing scale from min/max
                w_quant, _ = quantize_tensor(module.weight.data, bits)
                module.weight.data = w_quant
    return model

########################################################################################################
# Model Definition (from demo)
########################################################################################################

class RWKV_Tmix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        H = self.n_head
        N = self.head_size
        C = args.n_embd
        D_DECAY_LORA = args.D_DECAY_LORA
        D_AAA_LORA   = args.D_AAA_LORA
        D_MV_LORA    = args.D_MV_LORA
        D_GATE_LORA  = args.D_GATE_LORA

        self.x_r = nn.Parameter(torch.empty(1,1,C))
        self.x_w = nn.Parameter(torch.empty(1,1,C))
        self.x_k = nn.Parameter(torch.empty(1,1,C))
        self.x_v = nn.Parameter(torch.empty(1,1,C))
        self.x_a = nn.Parameter(torch.empty(1,1,C))
        self.x_g = nn.Parameter(torch.empty(1,1,C))
        self.w0 = nn.Parameter(torch.empty(1,1,C))
        self.w1 = nn.Parameter(torch.empty(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(torch.empty(D_DECAY_LORA, C))
        self.a0 = nn.Parameter(torch.empty(1,1,C))
        self.a1 = nn.Parameter(torch.empty(C, D_AAA_LORA))
        self.a2 = nn.Parameter(torch.empty(D_AAA_LORA, C))
        self.v0 = nn.Parameter(torch.empty(1,1,C))
        self.v1 = nn.Parameter(torch.empty(C, D_MV_LORA))
        self.v2 = nn.Parameter(torch.empty(D_MV_LORA, C))
        self.g1 = nn.Parameter(torch.empty(C, D_GATE_LORA))
        self.g2 = nn.Parameter(torch.empty(D_GATE_LORA, C))
        self.k_k = nn.Parameter(torch.empty(1,1,C))
        self.k_a = nn.Parameter(torch.empty(1,1,C))
        self.r_k = nn.Parameter(torch.empty(H,N))
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

    @MyFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g
        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)
        x = RWKV7_OP(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first

class RWKV_CMix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.empty(1,1,args.n_embd))
        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)
        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)
        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn
        x = x + self.ffn(self.ln2(x))
        return x, v_first

class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)
        x = self.ln_out(x)
        return self.head(x)

########################################################################################################
# Load and Evaluate
########################################################################################################

def load_model(model_path, quant_bits=None, lmhead_bits=None):
    """Load model, optionally apply quantization"""
    global HEAD_SIZE
    model_args, w = infer_args_from_checkpoint(model_path)
    HEAD_SIZE = model_args.head_size_a

    model = RWKV(model_args).cuda()

    # Extract q_scale values before removing them
    q_scales = {}
    keys = list(w.keys())
    for k in keys:
        if '.q_scale' in k:
            # Extract module name: e.g., "blocks.0.att.receptance.q_scale" -> "blocks.0.att.receptance"
            module_name = k.replace('.q_scale', '')
            q_scales[module_name] = w[k]
            del w[k]
            continue
        if 'qmin' in k or 'qmax' in k:
            del w[k]
            continue
        w[k] = w[k].float()

    model.load_state_dict(w, strict=False)
    model = model.half().cuda()

    # Apply quantization if specified
    if quant_bits is not None:
        head_bits = lmhead_bits if lmhead_bits is not None else quant_bits
        print(f"Applying {quant_bits}-bit quantization (lm_head: {head_bits}-bit)...")
        if len(q_scales) > 0:
            print(f"Using saved q_scale values from checkpoint ({len(q_scales)} scales)")
        model = apply_quantization_to_model(model, quant_bits, lmhead_bits, q_scales)

    model.eval()
    return model

def evaluate_lambada(model, max_samples=None):
    """Evaluate on LAMBADA dataset"""
    with open("../misc/lambada_test.jsonl", "r", encoding="utf-8") as f:
        todo = [json.loads(line) for line in f]
        todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]

    if max_samples:
        todo = todo[:max_samples]

    print(f'\nEvaluating LAMBADA ({len(todo)} samples)...')
    xsum = 0
    xcnt = 0
    xacc = 0

    with torch.no_grad():
        for d in todo:
            src = [0] + tokenizer.encode(d[0])
            dst = tokenizer.encode(d[1])

            logits = 0
            correct = True
            out = model(torch.tensor(src+dst).reshape(1,-1).cuda())

            for i in range(len(dst)):
                ooo = out[0,len(src)-1+i].float()
                probs = F.softmax(ooo, dim=-1)
                logits += math.log(max(probs[dst[i]].item(), 1e-10))
                if torch.argmax(probs).item() != dst[i]:
                    correct = False

            xcnt += 1
            xsum += logits
            xacc += 1 if correct else 0

            if xcnt % 100 == 0 or xcnt == len(todo):
                ppl = math.exp(-xsum / xcnt)
                acc = xacc / xcnt * 100
                print(f'{xcnt}/{len(todo)} | PPL: {ppl:.2f} | ACC: {acc:.2f}%')

    final_ppl = math.exp(-xsum / xcnt)
    final_acc = xacc / xcnt * 100
    return final_ppl, final_acc

########################################################################################################
# Main
########################################################################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model .pth file")
    parser.add_argument("--quant_bits", type=int, default=None, help="Quantization bits (None, 8, or 4)")
    parser.add_argument("--quant_bits_lmhead", type=int, default=None, help="Quantization bits for lm_head (defaults to --quant_bits)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for testing")
    args_eval = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Model: {args_eval.model}")
    print(f"Quantization: {args_eval.quant_bits if args_eval.quant_bits else 'None (FP16)'}-bit")
    print(f"{'='*60}")

    model = load_model(args_eval.model, args_eval.quant_bits, args_eval.quant_bits_lmhead)
    ppl, acc = evaluate_lambada(model, args_eval.max_samples)

    print(f"\n{'='*60}")
    print(f"Final Results:")
    print(f"  PPL:  {ppl:.2f}")
    print(f"  ACC:  {acc:.2f}%")
    print(f"{'='*60}")
