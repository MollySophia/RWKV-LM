#!/bin/bash

MODEL_NAME="rwkv7-g1d-0.1b"
MODEL_PATH="/models/rwkv7-g1d-0.1b-20260129-ctx8192.pth"

N_LAYER=12
N_EMBD=768
DIM_FFN=3072
DIM_MV_LORA=32
DIM_GATE_LORA=128
VOCAB_SIZE=65536
HEAD_SIZE=64

# Training params
CTX_LEN=512
PROJ_DIR="out/qat-${MODEL_NAME}-test"

# QAT params
QAT=1
QAT_BITS=4
QAT_BITS_LMHEAD=8

# Optimization params
M_BSZ=4
LR_INIT="5e-6"
LR_FINAL="1e-7"
GRAD_CP=1
EPOCH_SAVE=1
WEIGHT_DECAY=0.001

# Data params - 1 epoch
DATA_FILE="data/finetome_"
# Dataset: 54,144,294 tokens, ctx_len=512 -> 105,750 slots
MAGIC_PRIME=105701  # largest 3n+2 prime <= 105750
MY_EXIT_TOKENS=20643840  # 1 epoch = 10080 steps * 4 bsz * 512 ctx_len

#######################################################################################################################

mkdir -p $PROJ_DIR

echo "Starting QAT training (1 epoch)..."
echo "Model: $MODEL_PATH"
echo "Output: $PROJ_DIR"
echo "QAT: $QAT, Bits: $QAT_BITS"

python train.py \
  --load_model "$MODEL_PATH" \
  --qat $QAT --qat_bits $QAT_BITS --qat_bits_lmhead $QAT_BITS_LMHEAD \
  --proj_dir $PROJ_DIR \
  --my_testing "x070" \
  --ctx_len $CTX_LEN \
  --train_stage 0 --epoch_count 100 --epoch_begin 0 \
  --data_file "$DATA_FILE" --data_type "binidx" --vocab_size $VOCAB_SIZE \
  --magic_prime $MAGIC_PRIME --my_exit_tokens $MY_EXIT_TOKENS \
  --num_nodes 1 --micro_bsz $M_BSZ \
  --n_layer $N_LAYER --n_embd $N_EMBD --dim_ffn $DIM_FFN --head_size $HEAD_SIZE \
  --dim_mv_lora $DIM_MV_LORA --dim_gate_lora $DIM_GATE_LORA \
  --freeze_emb 1 \
  --lr_init $LR_INIT --lr_final $LR_FINAL \
  --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 \
  --weight_decay $WEIGHT_DECAY --epoch_save $EPOCH_SAVE \
  --accelerator gpu --devices 1 --precision bf16 \
  --strategy deepspeed_stage_2 --grad_cp $GRAD_CP \
  --enable_progress_bar True --ds_bucket_mb 2 2>&1 | tee $PROJ_DIR/train.log
