# MX Selective GEMM Test Results

## Test Status: ✅ WORKING (with fallback to WikiText)

### Fixed Issues:
1. ✅ **Linear Layer Replacement**: Fixed recursive replacement logic - now properly replaces 8 Linear layers in tinystories-33M
2. ✅ **Multi-GPU Usage**: Confirmed using all 8 GPUs ([0, 1, 2, 3, 4, 5, 6, 7])
3. ✅ **MxLinear Implementation**: Fixed `quantize_mx_op` call to include proper `axes` and `round` parameters
4. ✅ **MX Configuration**: Added `round='even'` to DEFAULT_MX_SPEC (OCP MX v1.0 standard)
5. ✅ **Dataset Split**: Changed from 'validation' to 'train' split for Pile dataset

### Current Behavior:
- **Baseline (FP32/BF16)**: PPL 2.39, Xent 0.8713, Entr 2.4317 ✅
- **MX Selective GEMM**: Successfully replaces 8 MLP Linear layers
- **GPU Utilization**: All 8 GPUs being used in parallel

### Known Limitation:
- **Pile Dataset**: Currently falls back to WikiText due to zstd compression support
  - Solution: Install `zstandard` package: `uv pip install zstandard`
  - Alternative: Use WikiText for testing (working fine)

## Verified Replacements (tinystories-33M):
```
Replaced 8 Linear layers with MxLinear:
- MLP layers: c_fc, c_proj in each transformer block
- Pattern matching: 'mlp.c_fc', 'mlp.c_proj'
```

## Multi-GPU Execution Verified:
```
2025-10-02 23:54:42,805 - INFO - Baseline  Using GPUs: [0, 1, 2, 3, 4, 5, 6, 7] with 1 worker(s)/device
```

## Next Steps:
1. Install zstandard for full Pile dataset support
2. Test with larger models (qwen2.5-7b, llama-3.1-8b)
3. Run full benchmark suite with all models
4. Compare MX vs baseline results

## Tier A Benchmark Snapshot — 2025-10-03
- **Log file**: `tierA_20251003.log` (root of `mx-scaling-tests` repo)
- **Setup**: Tier A (MLP-only) replacements, 200 Pile samples, batch size 4 (Qwen override 1), 8 GPUs.

| Model            | Variant      | PPL | Xent  | Entr  | Notes |
| ---------------- | ------------ | ---:| -----:| -----:| ----- |
| phi-1_5          | Baseline     | 2.38 | 0.8663 | 2.4644 | — |
| phi-1_5          | MX-Selective | 2.42 | 0.8851 | 2.5469 | 48 linear layers replaced |
| llama-3.1-8b     | Baseline     | 1.76 | 0.5670 | 1.5463 | — |
| llama-3.1-8b     | MX-Selective | 1.78 | 0.5755 | 1.6134 | 96 linear layers replaced |
| qwen2.5-14b      | Baseline     | 2.35 | 0.8533 | 2.1449 | Batch size override = 1 |
| qwen2.5-14b      | MX-Selective | 2.33 | 0.8438 | 2.1283 | 144 linear layers replaced |

**Backend availability**
- INT8 (bitsandbytes) — skipped: installed version too old (“pip install -U bitsandbytes”).
- NV-FP8 (Transformer Engine) — skipped: dependency/CUDA support missing during this run.

## Usage Commands:

### Quick Test (Small Model):
```bash
uv run python mx_lm_selective.py --models tinystories-33M --mlp_only --max_samples 50 --batch_size 2
```

### Full MLP-Only Test (Tier-A):
```bash
uv run python mx_lm_selective.py --models tinystories-33M qwen1.5-0.5B phi-1_5 --mlp_only --max_samples 1000
```

### Large Models (7B-30B):
```bash
uv run python mx_lm_selective.py --models qwen2.5-7b llama-3.1-8b --mlp_only --min_free_ratio 0.3
```

### Full Attention + MLP:
```bash
uv run python mx_lm_selective.py --models qwen2.5-7b --max_samples 1000
```

### All Linear Layers:
```bash
uv run python mx_lm_selective.py --models phi-1_5 --all_linears
```

## Model List:
- **Small**: tinystories-33M (✅ tested), qwen1.5-0.5B, phi-1_5
- **Medium**: qwen2.5-7b, llama-3.1-8b
- **Large**: qwen2.5-14b, phi-3-medium-14b, mixtral-8x7b, mpt-30b, yi-34b
