# MX Format Testing Guide

## Available MX Formats

The refactored `mx_lm_selective.py` now supports multiple MX precision formats for benchmarking:

### FP8 Formats
- **fp8_e4m3** (default): 8-bit float with 4-bit exponent, 3-bit mantissa
  - Best balance for most workloads
  - OCP MX v1.0 standard format
  
- **fp8_e5m2**: 8-bit float with 5-bit exponent, 2-bit mantissa
  - Wider dynamic range, less precision
  - Better for models with large activation magnitudes

### FP6 Formats
- **fp6_e3m2**: 6-bit float with 3-bit exponent, 2-bit mantissa
  - More aggressive quantization
  - Lower memory footprint

- **fp6_e2m3**: 6-bit float with 2-bit exponent, 3-bit mantissa
  - Narrower range, higher precision
  - May overflow on extreme values

### FP4 Format
- **fp4_e2m1**: 4-bit float with 2-bit exponent, 1-bit mantissa
  - Extreme quantization for memory-constrained scenarios
  - Significant accuracy degradation expected

## Usage

### Test a specific MX format:
```bash
uv run python mx_lm_selective.py \
  --models tinystories-33M \
  --mx_format fp8_e5m2 \
  --tier B \
  --max_samples 200 \
  --batch_size 4
```

### Compare multiple formats (run separately):
```bash
# FP8 E4M3 (default)
uv run python mx_lm_selective.py --models phi-1_5 --mx_format fp8_e4m3 --max_samples 100

# FP8 E5M2
uv run python mx_lm_selective.py --models phi-1_5 --mx_format fp8_e5m2 --max_samples 100

# FP6 E3M2
uv run python mx_lm_selective.py --models phi-1_5 --mx_format fp6_e3m2 --max_samples 100
```

### Skip MX to test only INT8/NVFP8:
```bash
uv run python mx_lm_selective.py \
  --models llama-3.1-8b \
  --skip_mx \
  --tier B \
  --max_samples 200
```

## Backend Status

### ✅ Working Backends
1. **Baseline (FP32/BF16)**: Full precision reference
2. **MX Formats**: All 5 format presets validated
3. **INT8 (bitsandbytes)**: Weight-only 8-bit quantization

### ❌ NV-FP8 (Transformer Engine)
**Status**: Binary incompatibility with current PyTorch 2.4.0

**Error**: 
```
ImportError: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```

**Cause**: The installed `transformer-engine` wheel was compiled against a different PyTorch C++ ABI version.

**Solutions**:
1. Build Transformer Engine from source with matching PyTorch version:
   ```bash
   pip uninstall transformer-engine
   git clone https://github.com/NVIDIA/TransformerEngine.git
   cd TransformerEngine
   export NVTE_FRAMEWORK=pytorch
   pip install .
   ```

2. Use a container with pre-compiled TE:
   ```bash
   docker run --gpus all -it nvcr.io/nvidia/pytorch:24.01-py3
   ```

3. Accept graceful skip — INT8 and MX formats still provide comprehensive benchmarks

## Refactoring Summary

### Key Improvements
1. **TE_AVAILABLE** flag for robust Transformer Engine detection
2. **MX_FORMAT_PRESETS** dict with 5 tested format configurations
3. **`--mx_format`** CLI arg to select precision at runtime
4. Enhanced error handling in `NvFp8Linear` and `replace_linear_with_nvfp8`
5. Better logging for skip reasons (shows which dependencies missing)
6. Updated `make_nvfp8_factory` to use `bfloat16` and proper recipe syntax

### Changed Files
- `mx_lm_selective.py`: Main refactoring with format presets and TE fixes

### Backward Compatibility
- Default behavior unchanged (`fp8_e4m3`, Tier B)
- All existing CLI flags preserved
- Logs now show selected MX format in header
