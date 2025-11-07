#!/bin/bash
# Quick monitoring script for large model tests

echo "=== MX Selective GEMM - Large Model Test Monitor ==="
echo ""
echo "Test started at: $(date)"
echo "Models: llama-3.1-8b, qwen2.5-7b, qwen2.5-14b"
echo "Configuration: MLP-only (Tier-A), 1000 samples from The Pile"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""
echo "==============================================="
echo ""

# Follow the log and show only important lines
tail -f mx_selective_log.txt | grep --line-buffered -E "(Evaluating:|Baseline.*PPL|MX-Selective.*PPL|Replaced.*Linear|BENCHMARK COMPLETE|Using GPUs)"
