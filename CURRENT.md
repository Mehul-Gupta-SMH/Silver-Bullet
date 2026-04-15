# CURRENT — Active Task Queue

---

## In-Flight

| Status | Task | Notes |
|--------|------|-------|
| [~] | RETRAIN RVG v5.5 | PID 21880; relex-base model; train_rvg.log |
| [~] | Chain: eval RVG → train MVM → eval MVM → eval CVG | PID 19900 (chain_train); auto-runs after RVG completes |
| [ ] | COMMIT v5.5 | After all evals: checkpoints + logs + frontend + fixes |

## Pending

| Status | Task | Notes |
|--------|------|-------|
| [ ] | BENCHMARK run | `python -m backend.benchmark_eval`; target ROC-AUC ≥ 0.85 |
| [ ] | STABILITY: Stratified val split by dataset source | Reduces inter-batch gradient variance |
| [ ] | STABILITY: Per-source loss weighting | Normalize calibration across STS-B/MNLI/HaluEval |
| [ ] | STABILITY: Augment MVM val set to ≥600 samples | Only 334 now → high variance |

## Standing Instructions

- End of each day: spawn Haiku agent to update README.MD (version, feature counts, AUC table)
- Training sequential: one process at a time

## Notes

- v5.5 feature counts: CVG=19, RVG=20, MVM=21 (each adds `relation_triplet_recall`)
- Relex model: switched large→base (`knowledgator/gliner-relex-base-v1.0`) — fits in 4.2GB free RAM
- Training params: patience=15, num_epochs=75, warmup=5ep LinearLR + CosineAnnealingLR
- 3 code fixes applied (2026-04-14): UnboundLocalError → `_free_extractor_models()` method; chainer false-positive → 500-byte tail check; launch → Python subprocess.Popen not bash &
