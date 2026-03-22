# bezdarnost

Starting from the Naive Baseline (`train_gpt.py` snapshot from root).

## Configuration

- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 MLP_MULT=2 CONV_KERNEL_SIZE=4`
- Attention: Gated Delta Net (replaces softmax attention + RoPE)
- Tied embeddings: `TIE_EMBEDDINGS=1`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`
- Wallclock cap: `MAX_WALLCLOCK_SECONDS=600`

## Launch

```bash
source .venv/bin/activate

# Download dataset (if not already done)
python3 data/cached_challenge_fineweb.py --variant sp1024

# Single GPU
RUN_ID=bezdarnost \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
python3 records/track_10min_16mb/2026-03-22_bezdarnost/train_gpt.py

# Multi-GPU (8xH100)
NCCL_IB_DISABLE=1 \
RUN_ID=bezdarnost \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-03-22_bezdarnost/train_gpt.py
```

1xH100 SXM (10 min)

| ID | Change | val_bpb | Delta | Train iters | Params | Model weight | Code weigth | Comment |
| -- | ------ | ------- | ----- | ----------- | ------ | ------------ | ----------- | ------- |
| 0 | Baseline | 1.3076 | 0 | 1771 | 17059912 | 14580119 | 47688 |  |
| 1 | Square LeakyReLU | 1.3072 | -0.0004 | 1769 | 17059912 | 14589049 | 47708 | Use |
| 2 | 16d StandardRoPE + 16d DocRoPE | 1.3107 | +0.0035 | 1727 | 17059912 | 14529626 | 48788 | Revert |
| 3 | Full GatedDeltaNet | 1.4191 | +0.1119 | 701 | 21908176 | 14360260 | 46838 |  |
| 4 | Optimized (3) | 1.4206 | +0.1134 | 712 | 21908176 | 14464942 | 46255 |  |
| 5 | 1 Attn + 3 GatedDeltaNet | 1.3586 | +0.0496 | 874 | 20292088 | 14156098 | 51454 |  |
| 6 | Seq_len=1024 -> 2048 |  |  |  | 20292088 |  |  |  |
| 6 | Seq_len=2048 -> 4096 |  |  |  | 20292088 |  |  |  |
| 6 | Seq_len=4096 -> 8192 |  |  |  | 20292088 |  |  |  |
