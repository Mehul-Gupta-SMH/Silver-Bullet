"""benchmark_eval.py — Evaluate SilverBullet against held-out human-annotated benchmarks.

Runs SilverBullet predictions on each benchmark in data/benchmarks/ and computes:
  - Pearson r  (linear correlation with human scores where available)
  - Spearman ρ (rank correlation — more robust to non-linearity)
  - ROC-AUC    (binary classification performance)
  - PR-AUC     (precision-recall AUC — better for imbalanced benchmarks)
  - Accuracy   (at threshold 0.5)

Benchmarks (validation only — never in training data):
  - summeval   → reference-vs-generated  (Likert 1-5 consistency → binary ≥3.5)
  - factcc     → reference-vs-generated  (CORRECT/INCORRECT claim faithfulness)
  - frank      → reference-vs-generated  (any error type → unfaithful)
  - aggrefact  → reference-vs-generated  (aggregated faithfulness labels)

Usage:
    python -m backend.benchmark_eval
    python -m backend.benchmark_eval --mode reference-vs-generated
    python -m backend.benchmark_eval --benchmarks summeval factcc
    python -m backend.benchmark_eval --output benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from tqdm import tqdm

from backend.predict import SimilarityPredictor

_ROOT       = Path(__file__).parent.parent
BENCHMARK_DIR = _ROOT / "data" / "benchmarks"
REPORT_DIR  = _ROOT / "benchmark_reports"

# Which mode to use per benchmark
_BENCHMARK_MODE: dict[str, str] = {
    "summeval":  "reference-vs-generated",
    "factcc":    "reference-vs-generated",
    "frank":     "reference-vs-generated",
    "aggrefact": "reference-vs-generated",
}


def _load_benchmark(name: str) -> list[dict]:
    path = BENCHMARK_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark '{name}' not found at {path}. "
            "Run `python -m backend.fetch_external_data` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))["data"]


def _run_benchmark(name: str, mode: str | None, predictor_cache: dict) -> dict:
    """Run SilverBullet on one benchmark and return metrics."""
    pairs = _load_benchmark(name)
    resolved_mode = mode or _BENCHMARK_MODE.get(name, "reference-vs-generated")

    if resolved_mode not in predictor_cache:
        print(f"  Loading predictor for mode: {resolved_mode}")
        predictor_cache[resolved_mode] = SimilarityPredictor(mode=resolved_mode)
    predictor = predictor_cache[resolved_mode]

    print(f"\n[{name}] {len(pairs)} pairs  mode={resolved_mode}")

    scores, labels, human_scores = [], [], []
    for pair in tqdm(pairs, desc=f"  Scoring {name}"):
        try:
            prob = predictor.predict_pair(pair["text1"], pair["text2"])
        except Exception:
            prob = 0.5
        scores.append(float(prob))
        labels.append(int(pair["label"]))
        if "human_score" in pair:
            human_scores.append(float(pair["human_score"]))

    scores  = np.array(scores)
    labels  = np.array(labels)

    metrics: dict = {"benchmark": name, "mode": resolved_mode, "n": len(pairs)}

    # ROC-AUC + PR-AUC (binary)
    if len(set(labels)) == 2:
        metrics["roc_auc"]  = float(roc_auc_score(labels, scores))
        metrics["pr_auc"]   = float(average_precision_score(labels, scores))
        metrics["accuracy"] = float(accuracy_score(labels, (scores >= 0.5).astype(int)))

    # Pearson + Spearman vs. binary labels
    r_p, _  = pearsonr(labels,  scores)
    r_s, _  = spearmanr(labels, scores)
    metrics["pearson_r_binary"]  = float(r_p)
    metrics["spearman_r_binary"] = float(r_s)

    # Pearson + Spearman vs. raw human scores (where available, e.g. SummEval Likert)
    if human_scores and len(human_scores) == len(scores):
        r_p_h, _ = pearsonr(human_scores,  scores)
        r_s_h, _ = spearmanr(human_scores, scores)
        metrics["pearson_r_human"]  = float(r_p_h)
        metrics["spearman_r_human"] = float(r_s_h)

    return metrics


def _print_metrics(m: dict) -> None:
    print(f"\n  {'Benchmark':<14} {m['benchmark']}  (n={m['n']}, mode={m['mode']})")
    if "roc_auc"  in m: print(f"  {'ROC-AUC':<14} {m['roc_auc']:.4f}")
    if "pr_auc"   in m: print(f"  {'PR-AUC':<14} {m['pr_auc']:.4f}")
    if "accuracy" in m: print(f"  {'Accuracy':<14} {m['accuracy']:.4f}")
    if "pearson_r_human"  in m: print(f"  {'Pearson r (h)':<14} {m['pearson_r_human']:.4f}")
    if "spearman_r_human" in m: print(f"  {'Spearman ρ (h)':<14} {m['spearman_r_human']:.4f}")
    if "pearson_r_binary"  in m: print(f"  {'Pearson r (b)':<14} {m['pearson_r_binary']:.4f}")
    if "spearman_r_binary" in m: print(f"  {'Spearman ρ (b)':<14} {m['spearman_r_binary']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SilverBullet against held-out human-annotated benchmarks"
    )
    parser.add_argument(
        "--benchmarks", nargs="+",
        default=list(_BENCHMARK_MODE.keys()),
        help="Which benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--mode", default=None,
        help="Override mode for all benchmarks (default: per-benchmark default)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to write JSON results (default: benchmark_reports/YYYY-MM-DD.json)",
    )
    args = parser.parse_args()

    available = [b for b in args.benchmarks if (BENCHMARK_DIR / f"{b}.json").exists()]
    missing   = [b for b in args.benchmarks if b not in available]
    if missing:
        print(f"[WARNING] Benchmarks not found (run fetch_external_data first): {missing}")
    if not available:
        raise SystemExit("No benchmark files found. Run `python -m backend.fetch_external_data` first.")

    predictor_cache: dict = {}
    all_metrics = []

    for name in available:
        m = _run_benchmark(name, args.mode, predictor_cache)
        _print_metrics(m)
        all_metrics.append(m)

    # Summary table
    print("\n" + "=" * 60)
    print(f"  {'Benchmark':<14} {'ROC-AUC':>8} {'PR-AUC':>8} {'Pearson r':>10} {'Spearman ρ':>11}")
    print("  " + "-" * 56)
    for m in all_metrics:
        roc  = f"{m.get('roc_auc',  float('nan')):.4f}"
        pra  = f"{m.get('pr_auc',   float('nan')):.4f}"
        pr_h = m.get("pearson_r_human",  m.get("pearson_r_binary",  float("nan")))
        sp_h = m.get("spearman_r_human", m.get("spearman_r_binary", float("nan")))
        print(f"  {m['benchmark']:<14} {roc:>8} {pra:>8} {pr_h:>10.4f} {sp_h:>11.4f}")
    print("=" * 60)

    # Save
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import date
    out_path = Path(args.output) if args.output else REPORT_DIR / f"{date.today()}.json"
    out_path.write_text(
        json.dumps({"results": all_metrics}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
