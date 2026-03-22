### Int8 QAT
```bash
python train.py \
  --load_model /path/to/pretrained.pth \
  --qat 1 --qat_bits 8 \
  --wandb "your_project_name" \
  --proj_dir out/qat-8bit \
  --data_file "data/your_data" --data_type binidx \
  --ctx_len 512 --vocab_size 65536 \
  --n_layer 12 --n_embd 768 --head_size 64 \
  --lr_init 1e-6 --lr_final 1e-7 \
  --my_exit_tokens 184719070 \
  --micro_bsz 4 --epoch_save 5 \
  --accelerator gpu --devices 1 --precision bf16 \
  --strategy deepspeed_stage_2 --grad_cp 1
```

### Int4 QAT
```bash
python train.py \
  --load_model /path/to/pretrained.pth \
  --qat 1 --qat_bits 4 \
  --wandb "your_project_name" \
  --proj_dir out/qat-4bit \
  --data_file "data/your_data" --data_type binidx \
  --ctx_len 512 --vocab_size 65536 \
  --n_layer 12 --n_embd 768 --head_size 64 \
  --lr_init 1e-6 --lr_final 1e-7 \
  --my_exit_tokens 184719070 \
  --micro_bsz 4 --epoch_save 5 \
  --accelerator gpu --devices 1 --precision bf16 \
  --strategy deepspeed_stage_2 --grad_cp 1
```

## Key Parameters

| Parameter | Description |
|------|------|
| `--qat 1` | Enable QAT |
| `--qat_bits 8/4` | Quantization bits |
| `--lr_init/--lr_final` | Learning rate range (use small learning rate for finetuning/QAT) |
| `--my_exit_tokens` | Total training tokens, used for cosine decay |

## Model Evaluation

```bash
# Original model
python eval_quant.py --model /path/to/model.pth --max_samples 100

# Static quantization (no QAT)
python eval_quant.py --model /path/to/model.pth --quant_bits 8 --max_samples 100
python eval_quant.py --model /path/to/model.pth --quant_bits 4 --max_samples 100

# QAT trained model
python eval_quant.py --model out/qat-8bit/rwkv-final.pth --max_samples 100
```

## Output Files

- Training log: `out/<proj_dir>/train_log.txt`
- Checkpoint: `out/<proj_dir>/rwkv-{epoch}.pth`
- Final model: `out/<proj_dir>/rwkv-final.pth`
