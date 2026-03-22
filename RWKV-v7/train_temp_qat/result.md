# Quantization Evaluation Results

**Model:** rwkv7-g1d-0.4b-20260210-ctx8192
**Dataset:** LAMBADA (5153 samples, full set)
**Evaluation:** Static (post-training) quantization — no QAT

## Static Quantization Results

| Configuration               | PPL    | ACC    |
|-----------------------------|--------|--------|
| FP16 (baseline)             |  7.96  | 57.02% |
| INT8 (all layers)           |  7.92  | 57.48% |
| INT4 (lm_head INT8)         | 30.57  | 29.94% |

## QAT Results

| Configuration                              | PPL    | ACC    |
|--------------------------------------------|--------|--------|
| INT4 QAT (1 epoch, lr 1e-6→1e-7, FP16 eval) | 346.35 | 11.60% |