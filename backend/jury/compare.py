"""Jury vs CNN comparison report for SilverBullet.

Runs the CNN on the full test set (fast, local), then samples pairs
strategically and runs the LLM jury on that sample.  The result is a
structured JSON report surfacing four quadrants:

  AGREE_FAITHFUL      — both score ≥ threshold  (true positives)
  AGREE_HALLUCINATED  — both score < threshold  (true negatives)
  JURY_MISS           — jury HIGH, CNN LOW  (CNN misses hallucination)
  CNN_MISS            — jury LOW, CNN HIGH  (CNN false positive)

"JURY_MISS" and "CNN_MISS" are the interesting cases — they tell you
what each evaluator uniquely sees that the other doesn't.

Usage:
    python -m backend.jury.compare --mode context-vs-generated
    python -m backend.jury.compare --mode context-vs-generated --n 60 --threshold 0.5
    python -m backend.jury.compare --mode reference-vs-generated --n 40 --out jury_reports/
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample_pairs(
    pairs: list[list[str]],
    labels: list[float],
    cnn_scores: list[float],
    n: int,
    threshold: float,
    seed: int = 42,
) -> list[int]:
    """Return indices of *n* pairs sampled to maximise coverage of score regions.

    Strategy: split CNN scores into three bands (low / boundary / high) and
    sample roughly equally from each.  Boundary pairs (CNN score within ±0.15
    of the threshold) are oversampled because they are most likely to expose
    disagreements.  All indices are shuffled within each band before selection.
    """
    import random
    rng = random.Random(seed)

    low_idx    = [i for i, s in enumerate(cnn_scores) if s < threshold - 0.15]
    bound_idx  = [i for i, s in enumerate(cnn_scores)
                  if threshold - 0.15 <= s <= threshold + 0.15]
    high_idx   = [i for i, s in enumerate(cnn_scores) if s > threshold + 0.15]

    rng.shuffle(low_idx)
    rng.shuffle(bound_idx)
    rng.shuffle(high_idx)

    # Allocate: 40% boundary, 30% low, 30% high (clipped to band sizes)
    n_bound = min(int(n * 0.40), len(bound_idx))
    n_low   = min(int(n * 0.30), len(low_idx))
    n_high  = min(n - n_bound - n_low, len(high_idx))

    # Fill any gap left by small bands
    remaining = n - n_bound - n_low - n_high
    extras = (low_idx[n_low:] + high_idx[n_high:] + bound_idx[n_bound:])[:remaining]

    selected = bound_idx[:n_bound] + low_idx[:n_low] + high_idx[:n_high] + extras
    return sorted(set(selected))[:n]


# ---------------------------------------------------------------------------
# Quadrant classification
# ---------------------------------------------------------------------------

def _quadrant(cnn_score: float, jury_score: float, threshold: float) -> str:
    cnn_pos  = cnn_score  >= threshold
    jury_pos = jury_score >= threshold
    if cnn_pos and jury_pos:
        return "AGREE_FAITHFUL"
    if not cnn_pos and not jury_pos:
        return "AGREE_HALLUCINATED"
    if jury_pos and not cnn_pos:
        return "JURY_MISS"       # jury says faithful, CNN says not — CNN missed something
    return "CNN_MISS"            # CNN says faithful, jury says not — CNN is wrong


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    mode: str,
    n: int,
    threshold: float,
    out_dir: str,
    checkpoint: str | None = None,
) -> dict:
    from backend.train import load_json_data
    from backend.predict import SimilarityPredictor
    from backend.jury.jury_evaluator import JuryEvaluator

    # --- Load test data ---
    data_dir = f"data/{mode}"
    test_pairs, test_labels = load_json_data(f"{data_dir}/test.json")
    print(f"Loaded {len(test_pairs)} test pairs for mode '{mode}'")

    # --- CNN scores (full test set) ---
    ckpt = checkpoint or str(Path("models") / mode / "best.pth")
    print(f"Loading CNN from {ckpt} ...")
    predictor = SimilarityPredictor(ckpt)

    print("Running CNN on full test set ...")
    cnn_scores: list[float] = []
    for pair in tqdm(test_pairs, desc="CNN"):
        result = predictor.predict_pair(pair[0], pair[1])
        cnn_scores.append(result["probability"])

    # --- Sample pairs for jury ---
    n_actual = min(n, len(test_pairs))
    indices = _sample_pairs(test_pairs, test_labels, cnn_scores, n_actual, threshold)
    print(f"Sampled {len(indices)} pairs for jury evaluation "
          f"(low={sum(1 for i in indices if cnn_scores[i] < threshold - 0.15)}, "
          f"boundary={sum(1 for i in indices if abs(cnn_scores[i] - threshold) <= 0.15)}, "
          f"high={sum(1 for i in indices if cnn_scores[i] > threshold + 0.15)})")

    # --- Jury scores ---
    jury = JuryEvaluator()
    jury_results = []
    for idx in tqdm(indices, desc="Jury"):
        pair = test_pairs[idx]
        try:
            result = jury.evaluate(pair[0], pair[1], mode)  # type: ignore[arg-type]
            jury_results.append((idx, result))
        except RuntimeError as exc:
            print(f"  [WARN] jury failed for pair {idx}: {exc}")
            jury_results.append((idx, None))

    # --- Build comparison records ---
    records = []
    for idx, jury_result in jury_results:
        cnn_score  = cnn_scores[idx]
        true_label = test_labels[idx]

        if jury_result is None:
            continue

        jury_score = jury_result.score
        quad = _quadrant(cnn_score, jury_score, threshold)
        gap  = round(jury_score - cnn_score, 4)

        record = {
            "index":       idx,
            "text1":       test_pairs[idx][0],
            "text2":       test_pairs[idx][1],
            "true_label":  true_label,
            "cnn_score":   round(cnn_score, 4),
            "jury_score":  jury_score,
            "gap":         gap,          # jury - cnn (positive = jury more faithful)
            "quadrant":    quad,
            "jury_verdict": jury_result.verdict,
            "jury_model":   jury_result.model_used,
            "jury_questions": [
                {
                    "question":   q.question,
                    "answer":     q.answer,
                    "confidence": q.confidence,
                    "reasoning":  q.reasoning,
                }
                for q in jury_result.questions
            ],
        }
        records.append(record)

    # --- Aggregate stats ---
    quadrant_counts = {q: 0 for q in ["AGREE_FAITHFUL", "AGREE_HALLUCINATED",
                                       "JURY_MISS", "CNN_MISS"]}
    for r in records:
        quadrant_counts[r["quadrant"]] += 1

    n_total   = len(records)
    agreement = (quadrant_counts["AGREE_FAITHFUL"] +
                 quadrant_counts["AGREE_HALLUCINATED"]) / n_total if n_total else 0.0

    gaps = [r["gap"] for r in records]
    import statistics
    gap_stats = {
        "mean":   round(statistics.mean(gaps), 4) if gaps else 0.0,
        "stdev":  round(statistics.stdev(gaps), 4) if len(gaps) > 1 else 0.0,
        "max_jury_over_cnn":  round(max(gaps), 4) if gaps else 0.0,
        "max_cnn_over_jury":  round(-min(gaps), 4) if gaps else 0.0,
    }

    # Top disagreements by |gap|
    jury_misses = sorted(
        [r for r in records if r["quadrant"] == "JURY_MISS"],
        key=lambda r: abs(r["gap"]), reverse=True
    )
    cnn_misses = sorted(
        [r for r in records if r["quadrant"] == "CNN_MISS"],
        key=lambda r: abs(r["gap"]), reverse=True
    )

    # --- Build report ---
    report = {
        "meta": {
            "timestamp":   datetime.now().isoformat(),
            "mode":        mode,
            "checkpoint":  ckpt,
            "jury_model":  records[0]["jury_model"] if records else "unknown",
            "threshold":   threshold,
            "n_sampled":   len(records),
            "n_cnn_total": len(test_pairs),
        },
        "summary": {
            "agreement_rate":  round(agreement, 4),
            "quadrant_counts": quadrant_counts,
            "gap_stats":       gap_stats,
        },
        "jury_misses":  jury_misses[:10],   # top 10: jury faithful, CNN missed
        "cnn_misses":   cnn_misses[:10],    # top 10: CNN faithful, jury flagged
        "all_records":  records,
    }

    # --- Save ---
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = Path(out_dir) / f"jury_vs_cnn_{mode}_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved -> {out_path}")

    # --- Print summary ---
    print(f"\n{'─'*55}")
    print(f"  Mode        : {mode}")
    print(f"  Pairs       : {len(records)} evaluated  ({len(test_pairs)} total CNN)")
    print(f"  Agreement   : {agreement:.1%}  (threshold={threshold})")
    print(f"  Quadrants   :")
    for q, c in quadrant_counts.items():
        bar = "#" * c
        print(f"    {q:<22} {c:3d}  {bar}")
    print(f"\n  Gap (jury − CNN):  mean={gap_stats['mean']:+.3f}  "
          f"std={gap_stats['stdev']:.3f}")
    print(f"  Max jury > CNN  : +{gap_stats['max_jury_over_cnn']:.3f}  "
          f"(CNN misses faithfulness)")
    print(f"  Max CNN > jury  : +{gap_stats['max_cnn_over_jury']:.3f}  "
          f"(CNN false positive)")

    if jury_misses:
        print(f"\n  Top JURY_MISS (jury faithful, CNN missed) — gap:")
        for r in jury_misses[:3]:
            snippet = r["text2"][:80].replace("\n", " ")
            print(f"    [{r['gap']:+.3f}] CNN={r['cnn_score']:.2f} "
                  f"Jury={r['jury_score']:.2f}  \"{snippet}…\"")

    if cnn_misses:
        print(f"\n  Top CNN_MISS (CNN faithful, jury flagged) — gap:")
        for r in cnn_misses[:3]:
            snippet = r["text2"][:80].replace("\n", " ")
            print(f"    [{r['gap']:+.3f}] CNN={r['cnn_score']:.2f} "
                  f"Jury={r['jury_score']:.2f}  \"{snippet}…\"")
    print(f"{'─'*55}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare LLM jury scores vs CNN scores on a test split."
    )
    parser.add_argument(
        "--mode",
        choices=["context-vs-generated", "reference-vs-generated", "model-vs-model"],
        default="context-vs-generated",
        help="Evaluation mode (default: context-vs-generated)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of pairs to evaluate with the jury (default: 50)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for faithful/hallucinated classification (default: 0.5)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit CNN checkpoint path (overrides default models/{mode}/best.pth)",
    )
    parser.add_argument(
        "--out",
        default="jury_reports",
        help="Output directory for the comparison report (default: jury_reports/)",
    )
    args = parser.parse_args()

    run(
        mode=args.mode,
        n=args.n,
        threshold=args.threshold,
        out_dir=args.out,
        checkpoint=args.checkpoint,
    )
