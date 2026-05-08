# CURRENT — Active Task Queue

---

## In-Flight

| Status | Task | Notes |
|--------|------|-------|
| [~] | TRAIN+TEST v5.7 chain | bash PID 25799; CVG done, RVG train done + test running, MVM queued; logs: train/test_{cvg,rvg,mvm}_v57.log |
| [ ] | SHAP analysis v5.7 | After ALL DONE: `python -m backend.shap_analysis --mode all`; check SVO importance with new data |

## Pending

| Status | Task | Notes |
|--------|------|-------|
| [ ] | FEATURE: External Factual Grounding (EFG) | `backend/Features/Factual/getFactualGrounding.py`; DeBERTa-v3-base-mnli-fever-anli; per-sentence factuality score → n×m delta matrix; detects which text is more world-knowledge-supported |
| [ ] | BENCHMARK run | `python -m backend.benchmark_eval`; target ROC-AUC ≥ 0.85 |
| [ ] | STABILITY: Stratified val split by dataset source | Reduces inter-batch gradient variance |
| [ ] | STABILITY: Per-source loss weighting | Normalize calibration across STS-B/MNLI/HaluEval |
| [ ] | STABILITY: Augment MVM val set to ≥600 samples | Only 334 now → high variance |

## Standing Instructions

- End of each day: spawn Haiku agent to update README.MD (version, feature counts, AUC table)
- Training sequential: one process at a time (kill old chains before starting new ones)

## Notes

### v5.7 changes (current)
- **Removed:** `relation_triplet_recall` (Relex) — SHAP ~0 all modes, 98.6% trivial-1.0
- **Added:** FEVER (1000), SNLI (1000), SciTail (1000) for SVO-dense training data
- **Fixed:** `feature_map_to_tensor fill_missing=True` — entity/NLI/SVO timeouts no longer crash
- **Feature counts:** CVG=19, RVG=20, MVM=21 | spatial_size=32

### v5.7 results so far
- CVG: ROC-AUC **0.8053** / Acc 72.0% (baseline v5.6: 0.8203 / 74.1%) — slight regression
- RVG: train done (best val acc ~81.65%), test running
- MVM: queued

### Training params
- patience=15, num_epochs=75, warmup=5ep LinearLR + CosineAnnealingLR
- MSELoss, Adam; batch_size=32
- `_safe_extract()` timeout=90s for NLI/entity/SVO (fill_missing=True on timeout)
