# CURRENT — Active Task Queue

---

## In-Flight

| Status | Task | Notes |
|--------|------|-------|
| [ ] | BENCHMARK run | `python -m backend.benchmark_eval`; target ROC-AUC ≥ 0.85 on SummEval/FactCC/FRANK/AggreFact |

## Pending

| Status | Task | Notes |
|--------|------|-------|
| [ ] | STABILITY: Stratified val split by dataset source | Reduces inter-batch gradient variance |
| [ ] | STABILITY: Per-source loss weighting | Normalize calibration across STS-B/MNLI/HaluEval |
| [ ] | STABILITY: Augment MVM val set to ≥600 samples | Only ~386 now → high variance on test |

## Standing Instructions

- End of each day: spawn Haiku agent to update README.MD (version, feature counts, AUC table)
- Training sequential: one process at a time (kill old chains before starting new ones)

## Notes

### v5.8 — COMPLETE (2026-05-08, commit 90afd92)

**Changes vs v5.7:**
- Added EFG (External Factual Grounding) via `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` → 3 new feature maps: `efg_supports`, `efg_refutes`, `efg_factual_delta`
- Dropped low-SHAP features: `entity_product_value_prec` (CVG), `entity_percentage` (RVG+MVM)
- Feature counts: CVG=21, RVG=22, MVM=23 | spatial_size=32

**Final results:**

| Mode | ROC-AUC | Accuracy | F1 | MCC | vs v5.7 |
|------|---------|----------|----|-----|---------|
| CVG | 0.8130 | 72% | 0.723 | 0.448 | +0.008 AUC |
| RVG | 0.9283 | 85% | 0.854 | 0.700 | +0.028 AUC |
| MVM | 0.8969 | 81% | 0.816 | 0.612 | +0.005 AUC |

RVG benefited most (+2.8pp) — EFG factual grounding helps faithfulness evaluation.

### Training params
- patience=15, num_epochs=75, warmup=5ep LinearLR + CosineAnnealingLR
- MSELoss, Adam; batch_size=32
- `_safe_extract()` timeout=90s for NLI/entity/SVO (fill_missing=True on timeout)
- EFG cache: 71,975 pairs after v5.8 training run → future retrains skip prefill entirely
