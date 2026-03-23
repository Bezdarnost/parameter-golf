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
| 4 | Optimized (3) | 1.4206 | +0.0015 | 712 | 21908176 | 14464942 | 46255 |  |
| 5 | 1 Attn + 3 GatedDeltaNet | 1.3586 | -0.0620 | 874 | 20292088 | 14156098 | 51454 |  |
| 6 | Seq_len = 1024 -> 2048 | 1.3409 | -0.0177 | 849 | 20292088 | 13996852 | 51454 |  |
| 7 | Seq_len = 2048 -> 4096 | 1.3371 | -0.0038 | 806 | 20292088 | 13782822 | 51454 |  |
| 8 | Seq_len = 4096 -> 8192 | 1.3492 |  +0.0121 | 730 | 20292088 | 13403156 | 51454 |  |
| 9 | 1:1(Attn:GatedDeltaNet) | 1.3455 | -0.0037 | 744 | 19214696 | 12749627 | 51452 |  |
| 10 | Seq_len = 8192 -> 4096 | 1.3263 | -0.0192 | 885 | 19214696 | 13381858 | 51452 |  |
| 11 | Revert all to (1) | 1.3071 | -0.0192 | 1769 | 17059912 | 14613540 | 51448 |  |
| 12 | [Gated Attention](https://arxiv.org/abs/2505.06708) | 1.2978 | -0.0093 | 1688 | 19419208 | 16406930 | 51965 |  |
| 13 | [XSA](https://arxiv.org/abs/2603.09078) before Gate | 1.3012 | +0.0034 | 1599 | 19419208 | 16128353 | 52187 |  |
| 14 | [XSA](https://arxiv.org/abs/2603.09078) after Gate | 1.3016 | +0.0004 | 1564 | 19419208 | 15996241 | 52232 |  |
| 15 | Turn off the [Gate](https://arxiv.org/abs/2505.06708) in Attention | 1.3071 | +0.0055 | 1627 | 17059912 | 14238418 | 52233 | Remove the [XSA](https://arxiv.org/abs/2603.09078) |
| 16 | [Gated Attention](https://arxiv.org/abs/2505.06708) in each odd layer | 1.3037 | -0.0034 | 1692 | 18370632 | 15521988 | 52044 |  |
| 17 | [Gated Attention](https://arxiv.org/abs/2505.06708) in all layers, mlp_mult=2.0->1.0 |  |  |  | 14700616 |  |  |  |
| 17 | mlp_mult=1.0->1.5 |  |  |  | 14700616 |  |  |  |
