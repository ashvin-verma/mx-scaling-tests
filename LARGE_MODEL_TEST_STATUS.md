# 🚀 Large Model Test — ✅ COMPLETE

**Completed**: October 3, 2025 10:37 UTC  
**Latest Command**: `uv run python mx_lm_selective.py --models llama-3.1-8b qwen2.5-14b --mlp_only --max_samples 1000 --batch_size 8 --workers_per_device 1`

Thanks to the multi-GPU updates (device_map + per-model batch sizing) both large models finished successfully. The Hugging Face caches now live on `/scratch/ashvin/hf_cache`, so future downloads have breathing room.

---

## ✅ Final Metrics (1000 The Pile samples)

| Model | Variant | PPL | Cross-Entropy | Entropy | Δ vs Baseline |
| --- | --- | --- | --- | --- | --- |
| llama-3.1-8b | Baseline | 1.73 | 0.5484 | 1.5210 | — |
|  | MX (MLP-only) | 1.75 | 0.5609 | 1.5969 | +1.4% PPL |
| qwen2.5-14b | Baseline | 2.38 | 0.8682 | 2.2031 | — |
|  | MX (MLP-only) | 2.36 | 0.8566 | 2.1757 | −0.8% PPL |

**Layer coverage**
- llama-3.1-8b: 96 linear layers swapped to `MxLinear`
- qwen2.5-14b: 144 linear layers swapped

---

## 🛠️ Implementation Notes

- Added multi-GPU-aware model factory: `device_map="balanced"` + per-GPU `max_memory`
- Logged device placement for transparency: Qwen 14B spanned all 8 GPUs (`cuda:0-7`)
- Introduced per-model batch overrides (Qwen 14B uses batch size 1 to avoid activation spikes)
- Cleaned up caches between runs and relocated Hugging Face cache off `/home`

---

## 📁 Artifacts

- `large_model_run.log` — fresh, successful run (kept in repo root)
- `large_model_run_20251003_qwen14b_oom*.log` — archived failed attempts for reference
- `mx_selective_log.txt` — rolling evaluation ledger (tail for quick summaries)

---

## ✅ Next Steps

1. Fold results into `FINAL_RESULTS.md` / `SUMMARY.md` with the new 8B + 14B numbers
2. Decide which model to tackle next (Mixtral, Phi-3-medium, Yi-34B)
3. Consider attention+MLP tier once selective MLP baselines are all green

No active jobs remain — GPUs are idle (`nvidia-smi` clean). Feel free to kick off the next batch when ready.
