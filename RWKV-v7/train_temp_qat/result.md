# Static Quantization Results (Non-QAT)

**Model:** rwkv7-g1d-0.1b-20260129-ctx8192
**Dataset:** LAMBADA (5153 samples, full set)

| Configuration               | PPL       | ACC    |
|-----------------------------|-----------|--------|
| FP16 (baseline)             | 15.26     | 45.84% |
| INT8 (all layers)           | 15.72     | 45.10% |
| INT4 (lm_head INT8)         | 29499.51  | 1.63%  |

---

# QAT Results (1 Epoch, Lower LR + Pretrain Data)

**Training:** subsample_world_v35, 1 epoch (~18M tokens), LR 1e-6→1e-7, freeze_emb=1, warmup 50 steps
**QAT Config:** INT4 weights, INT8 lm_head, q_scale update every 10 steps

## Min/Max Analysis

- **q_scale count:** 73 tensors
- **Scale range:** [0.0003, 0.1348], mean=0.0036
- **Weight range:** [-2.344, 23.750]
- **INT4 clipping:** 30.10% of values exceed [-7, 7] range

## Evaluation Results

| Configuration                          | PPL      | ACC    | vs Baseline |
|----------------------------------------|----------|--------|-------------|
| QAT checkpoint, FP16 (no quant)        | 20.39    | 39.61% | +33% PPL    |
| QAT checkpoint, INT4 (with q_scale)    | 4076.07  | 2.41%  | ~270x PPL   |
| Original model, INT4 (static)          | 29499.51 | 1.63%  | ~2000x PPL  |

## Analysis

**Major improvement with lower LR + pretrain data:**
- FP16 PPL: 116.84 → 20.39 (5.7x better!)
- FP16 ACC: 23.11% → 39.61% (much closer to baseline 45.84%)
- QAT INT4: 93.9k → 4k PPL (23x better!)

**INT4 QAT vs Static:**
- QAT INT4 (4k PPL) is **7x better** than static INT4 (29.5k PPL)
- QAT is now actually helping, but still not practical

**Remaining issues:**
- 30% of weights still being clipped (STE gradient masking)
- Need even lower LR or more epochs to fully recover FP16 quality
- Consider LSQ-style learned scales instead of min/max

---

# QAT Run 3 — Even Lower LR + 2 Epochs

**Training:** subsample_world_v35, 2 epochs (~37M tokens), LR 5e-7→1e-7, warmup 100 steps

| Checkpoint | FP16 PPL | FP16 ACC | INT4 PPL | INT4 ACC |
|-----------|----------|----------|----------|----------|
| Epoch 0 (rwkv-0.pth) | 19.39 | 40.13% | 3304.37 | 3.24% |
| Epoch 1 (rwkv-1.pth) | 19.39 | 40.13% | 3304.37 | 3.24% |
| Final (rwkv-final.pth) | 19.39 | 40.13% | 3304.37 | 3.24% |

## Analysis

- Lower LR helps: FP16 PPL 20.39 → 19.39
- 2 epochs same as 1 epoch — converged quickly
- INT4: 4076 → 3304 (~20% improvement)
- Still 30% weights clipped

---

# Summary

| Method | FP16 PPL | INT4 PPL | INT4 vs Static |
|--------|----------|----------|----------------|
| Baseline (no QAT) | 15.26 | 29499 | — |
| QAT Run 1 (5e-6, FineTome) | 116.84 | 93964 | 3.2x worse |
| QAT Run 2 (1e-6, 1ep) | 20.39 | 4076 | 7x better |
| QAT Run 3 (5e-7, 2ep) | 19.39 | 3304 | 9x better |

