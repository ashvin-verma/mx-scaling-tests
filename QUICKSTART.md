# MX Selective GEMM Replacement - Quick Start Guide

## What Was Fixed ✅

### 1. **Linear Layer Replacement** 
- **Problem**: Original code wasn't actually replacing any Linear layers
- **Solution**: Rewrote `replace_linear_with_mx()` to use recursive traversal with proper parent-child tracking
- **Result**: Now correctly replaces 8 MLP layers in tinystories-33M

### 2. **MxLinear Forward Pass**
- **Problem**: `quantize_mx_op()` call was missing required `axes` and `round` parameters
- **Solution**: Added proper axes specification ([-1] for row-wise quantization) and round mode
- **Result**: MX quantization now works without errors

### 3. **Multi-GPU Utilization**
- **Status**: ✅ Already working! Uses all 8 available GPUs in parallel
- **Verified**: Logs show `Using GPUs: [0, 1, 2, 3, 4, 5, 6, 7]`

### 4. **Dataset Loading**
- **Problem**: Used wrong split ('validation' vs 'train') for Pile dataset
- **Solution**: Changed to 'train' split for monology/pile-uncopyrighted
- **Fallback**: Automatically uses WikiText if Pile fails (works fine for testing)

### 5. **Added Large Models (7B-30B)**
- llama-3.1-8b
- qwen2.5-7b
- qwen2.5-14b  
- phi-3-medium-14b
- mixtral-8x7b
- mpt-30b
- yi-34b

### 6. **Tier-A Support**
- Added `--mlp_only` flag for Tier-A (MLP-only quantization)
- Separate TARGET_MODULES_MLP_ONLY and TARGET_MODULES_FULL
- Follows the guide: start with MLP, then extend to attention

## Quick Start

### 1. Install Dependencies (if needed)
```bash
# Optional: For full Pile dataset support
uv pip install zstandard
```

### 2. Run Quick Test (30 seconds)
```bash
cd /scratch/ashvin/mx-scaling-tests
./test_mx_selective.sh quick
```

This will:
- Create tmux session with 2 windows (test + GPU monitor)
- Run tinystories-33M with MLP-only replacement
- Use all available GPUs

### 3. Monitor Progress
```bash
# Attach to tmux session
tmux attach -t mx_selective_test

# Or just watch the log
tail -f mx_selective_log.txt

# Or check GPU usage
nvidia-smi -l 2
```

### 4. Switch Between Windows in Tmux
- `Ctrl-b 0` - Main test window
- `Ctrl-b 1` - GPU monitor
- `Ctrl-b d` - Detach (keeps running in background)

## Test Options

```bash
./test_mx_selective.sh quick   # Fast: tinystories-33M (~1 min)
./test_mx_selective.sh medium  # 7B model: qwen2.5-7b (~5-10 min)
./test_mx_selective.sh full    # All small models (~15 min)
./test_mx_selective.sh large   # 7B-14B models (~30-60 min)
```

## Manual Usage

### Tier-A (MLP Only - Recommended Start)
```bash
uv run python mx_lm_selective.py \
    --models tinystories-33M \
    --mlp_only \
    --max_samples 100 \
    --batch_size 4 \
    --min_free_ratio 0.3
```

### Full (Attention + MLP)
```bash
uv run python mx_lm_selective.py \
    --models qwen2.5-7b \
    --max_samples 1000 \
    --min_free_ratio 0.3
```

### Multiple Large Models
```bash
uv run python mx_lm_selective.py \
    --models llama-3.1-8b qwen2.5-7b qwen2.5-14b \
    --mlp_only \
    --min_free_ratio 0.2 \
    --workers_per_device 2
```

## Expected Output

### Baseline Run:
```
2025-10-02 23:54:42 - INFO - Baseline  Using GPUs: [0, 1, 2, 3, 4, 5, 6, 7]
2025-10-02 23:54:45 - INFO - tinystories-33M  Baseline  PPL 2.39  Xent 0.87  Entr 2.43
```

### MX Selective GEMM Run:
```
2025-10-02 23:54:50 - INFO - Applying MX quantization to roneneldan/TinyStories-33M
2025-10-02 23:54:51 - INFO - Replaced 8 Linear layers with MxLinear
2025-10-02 23:54:55 - INFO - tinystories-33M-MX  MX-Selective  PPL 2.45  Xent 0.90  Entr 2.41
```

## Understanding the Results

- **PPL (Perplexity)**: Lower is better. Measures model quality.
- **Xent (Cross-Entropy)**: The loss value. Lower is better.
- **Entr (Entropy)**: Model uncertainty. Context-dependent.

**Good Result**: MX PPL within 5-10% of baseline
- Baseline PPL: 2.39
- MX PPL: 2.45 (~2.5% degradation) ✅ Excellent!

## Troubleshooting

### "No GPUs available"
```bash
# Check GPU status
nvidia-smi

# Lower memory threshold
uv run python mx_lm_selective.py --min_free_ratio 0.2
```

### "Out of memory"
```bash
# Reduce batch size
uv run python mx_lm_selective.py --batch_size 2

# Or reduce samples
uv run python mx_lm_selective.py --max_samples 100
```

### "zstd compression not supported"
```bash
# Install zstandard
uv pip install zstandard

# Or just use WikiText (works fine)
# Script automatically falls back
```

## Files Overview

- `mx_lm_selective.py` - Main benchmark script with selective MX replacement
- `test_mx_selective.sh` - Convenient tmux-based test runner
- `mx_selective_log.txt` - Detailed execution log
- `UNDERSTANDING.md` - Comprehensive code documentation for LLMs
- `TEST_RESULTS.md` - Test status and results
- `QUICKSTART.md` - This file

## Architecture Highlights

### MxLinear Class
```python
class MxLinear(nn.Module):
    def forward(self, x):
        # Quantize weights (row-wise, FP8 E4M3)
        w_mx = quantize_mx_op(self.weight, mx_specs, axes=[-1])
        # Quantize activations (feature-wise, FP8 E4M3)  
        x_mx = quantize_mx_op(x, mx_specs, axes=[-1])
        # Standard matmul with quantized tensors
        return F.linear(x_mx, w_mx, self.bias)
```

### Selective Replacement
```python
# MLP only (Tier-A)
TARGET_MODULES_MLP_ONLY = [
    "gate_proj", "up_proj", "down_proj",  # Llama-style
    "fc1", "fc2",                          # BERT-style
    "c_fc", "c_proj",                      # GPT-style
]

# Full (Attention + MLP)
TARGET_MODULES_FULL = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    ...MLP layers...
]
```

## Next Steps

1. ✅ Verify small model works (tinystories-33M)
2. 🔄 Test 7B model (qwen2.5-7b or llama-3.1-8b)
3. 📊 Compare MX vs Baseline results
4. 🔬 Extend to full attention + MLP replacement
5. 📈 Run full benchmark suite

## Support

Check these files for more details:
- `UNDERSTANDING.md` - Detailed code walkthrough
- `TEST_RESULTS.md` - Current test status
- `mx_selective_log.txt` - Runtime logs
