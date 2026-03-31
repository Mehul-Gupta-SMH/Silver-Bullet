"""
SilverBullet vs Re-ranker Benchmark
====================================
Compares SilverBullet (TextSimilarityCNN) against one or more cross-encoder
re-ranker models on the held-out test split for each evaluation mode.

Metrics compared: accuracy, ROC-AUC, AUPRC, MCC, F1, Brier score, latency.
Failure-case analysis: pairs where SilverBullet and re-ranker disagree.

Usage:
    python -m backend.benchmark --mode context-vs-generated
    python -m backend.benchmark --all-modes
    python -m backend.benchmark --mode model-vs-model --n-latency 30
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from sentence_transformers import CrossEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, f1_score, brier_score_loss,
    roc_curve, precision_score, recall_score,
)
from torch.utils.data import DataLoader

from backend.predict import _load_model_from_checkpoint
from backend.train import TextSimilarityDataset, load_json_data

# ---------------------------------------------------------------------------
# Re-ranker registry
# ---------------------------------------------------------------------------
# Each entry: model_id -> {score_fn: callable(raw_output) -> float in [0,1]}
RERANKERS = {
    "cross-encoder/nli-deberta-v3-base": {
        "label": "NLI-DeBERTa-v3-base",
        "description": "NLI cross-encoder; entailment probability used as similarity score",
        "num_labels": 3,
        # label2id: contradiction=0, entailment=1, neutral=2
        "score_fn": lambda logits: softmax(logits, axis=-1)[:, 1],
    },
    "cross-encoder/stsb-roberta-base": {
        "label": "STS-RoBERTa-base",
        "description": "STS-B cross-encoder; raw score normalised to [0,1] via sigmoid",
        "num_labels": 1,
        "score_fn": lambda logits: 1 / (1 + np.exp(-np.array(logits).flatten())),
    },
}

VALID_MODES = ["context-vs-generated", "reference-vs-generated", "model-vs-model"]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_metrics(binary_true: np.ndarray, probs: np.ndarray) -> dict:
    preds_05 = (probs >= 0.5).astype(int)
    fpr, tpr, thresholds = roc_curve(binary_true, probs)
    youden_idx = int(np.argmax(tpr - fpr))
    opt_thresh = float(thresholds[youden_idx])
    preds_opt = (probs >= opt_thresh).astype(int)

    return {
        "accuracy":         float(accuracy_score(binary_true, preds_05)),
        "roc_auc":          float(roc_auc_score(binary_true, probs)),
        "auprc":            float(average_precision_score(binary_true, probs)),
        "mcc":              float(matthews_corrcoef(binary_true, preds_05)),
        "f1":               float(f1_score(binary_true, preds_05)),
        "precision":        float(precision_score(binary_true, preds_05)),
        "recall":           float(recall_score(binary_true, preds_05)),
        "brier_score":      float(brier_score_loss(binary_true, probs)),
        "optimal_threshold": opt_thresh,
        "f1_at_optimal":    float(f1_score(binary_true, preds_opt)),
    }


def _delta(sb_val: float, rr_val: float, lower_is_better: bool = False) -> str:
    diff = sb_val - rr_val
    if lower_is_better:
        diff = -diff
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.4f}"


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

def _measure_sb_latency(model, features_tensor: torch.Tensor, device, n_reps: int = 5) -> float:
    """Mean inference latency in ms/pair over n_reps passes (cache warm, model cold)."""
    model.eval()
    features_tensor = features_tensor.to(device)
    # Warmup
    with torch.no_grad():
        _ = model(features_tensor)
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(features_tensor)
        times.append((time.perf_counter() - t0) * 1000 / len(features_tensor))
    return float(np.mean(times))


def _measure_rr_latency(ce_model: CrossEncoder, pairs: list, n_reps: int = 5) -> float:
    """Mean inference latency in ms/pair over n_reps passes."""
    # Warmup
    ce_model.predict(pairs[:1])
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        ce_model.predict(pairs)
        times.append((time.perf_counter() - t0) * 1000 / len(pairs))
    return float(np.mean(times))


# ---------------------------------------------------------------------------
# Failure case analysis
# ---------------------------------------------------------------------------

def _failure_cases(
    pairs: list,
    true_labels: np.ndarray,
    sb_probs: np.ndarray,
    rr_probs: np.ndarray,
    rr_label: str,
    n: int = 10,
) -> dict:
    sb_preds = (sb_probs >= 0.5).astype(int)
    rr_preds = (rr_probs >= 0.5).astype(int)

    # SilverBullet wrong, re-ranker right
    sb_wrong_rr_right = [
        i for i in range(len(pairs))
        if sb_preds[i] != true_labels[i] and rr_preds[i] == true_labels[i]
    ]
    # SilverBullet right, re-ranker wrong
    sb_right_rr_wrong = [
        i for i in range(len(pairs))
        if sb_preds[i] == true_labels[i] and rr_preds[i] != true_labels[i]
    ]
    # Both wrong
    both_wrong = [
        i for i in range(len(pairs))
        if sb_preds[i] != true_labels[i] and rr_preds[i] != true_labels[i]
    ]

    def _fmt(indices, limit):
        return [
            {
                "text1":       pairs[i][0],
                "text2":       pairs[i][1],
                "true_label":  int(true_labels[i]),
                "sb_prob":     float(sb_probs[i]),
                "rr_prob":     float(rr_probs[i]),
                "sb_pred":     int(sb_preds[i]),
                "rr_pred":     int(rr_preds[i]),
            }
            for i in indices[:limit]
        ]

    return {
        f"sb_wrong_{rr_label}_right": {
            "count": len(sb_wrong_rr_right),
            "examples": _fmt(sb_wrong_rr_right, n),
        },
        f"sb_right_{rr_label}_wrong": {
            "count": len(sb_right_rr_wrong),
            "examples": _fmt(sb_right_rr_wrong, n),
        },
        "both_wrong": {
            "count": len(both_wrong),
            "examples": _fmt(both_wrong, n),
        },
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _markdown_report(result: dict) -> str:
    mode      = result["mode"]
    ts        = result["timestamp"]
    n_test    = result["n_test_pairs"]
    sb        = result["silverbullet"]
    rerankers = result["rerankers"]

    md = f"# Benchmark Report — {mode}\n"
    md += f"Generated: {ts} | Test pairs: {n_test}\n\n"

    # ── Metric comparison table ──────────────────────────────────────────────
    md += "## Metric Comparison\n\n"
    metrics_order = [
        ("accuracy",    "Accuracy",     False),
        ("roc_auc",     "ROC-AUC",      False),
        ("auprc",       "AUPRC",        False),
        ("mcc",         "MCC",          False),
        ("f1",          "F1 (@0.5)",    False),
        ("brier_score", "Brier Score",  True),
    ]
    header = "| Metric | SilverBullet |"
    sep    = "|--------|-------------|"
    for rr_id, rr_data in rerankers.items():
        lbl = rr_data["label"]
        header += f" {lbl} | Delta |"
        sep    += "--------|-------|"
    md += header + "\n" + sep + "\n"

    for key, name, lower_better in metrics_order:
        row = f"| {name} | {sb['metrics'][key]:.4f} |"
        for rr_id, rr_data in rerankers.items():
            rr_val = rr_data["metrics"][key]
            delta  = _delta(sb["metrics"][key], rr_val, lower_better)
            row   += f" {rr_val:.4f} | {delta} |"
        md += row + "\n"

    # ── Latency table ────────────────────────────────────────────────────────
    md += "\n## Inference Latency (ms / pair, warm cache)\n\n"
    md += "| Model | ms/pair |\n|-------|--------|\n"
    md += f"| SilverBullet (precomputed features) | {sb['latency_ms_per_pair']:.2f} |\n"
    for rr_id, rr_data in rerankers.items():
        md += f"| {rr_data['label']} | {rr_data['latency_ms_per_pair']:.2f} |\n"

    # ── Failure cases ────────────────────────────────────────────────────────
    md += "\n## Failure Case Analysis\n"
    for rr_id, rr_data in rerankers.items():
        fc = rr_data.get("failure_cases", {})
        md += f"\n### vs {rr_data['label']}\n"
        for key, bucket in fc.items():
            md += f"\n**{key.replace('_', ' ')}** ({bucket['count']} pairs)\n\n"
            for ex in bucket["examples"][:3]:
                md += (
                    f"- true={ex['true_label']}  "
                    f"SB={ex['sb_pred']} ({ex['sb_prob']:.3f})  "
                    f"RR={ex['rr_pred']} ({ex['rr_prob']:.3f})\n"
                    f"  - *text1:* {ex['text1'][:120]}\n"
                    f"  - *text2:* {ex['text2'][:120]}\n\n"
                )

    return md


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------

def run_benchmark(mode: str, n_latency: int = 20, reranker_ids: list = None) -> dict:
    if reranker_ids is None:
        reranker_ids = list(RERANKERS.keys())

    print(f"\n{'='*60}")
    print(f"Benchmarking mode: {mode}")
    print(f"{'='*60}")

    # ── Load SilverBullet ────────────────────────────────────────────────────
    ckpt_path = str(Path("models") / mode / "best.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = f"models/{mode}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    sb_model, arch = _load_model_from_checkpoint(checkpoint, device)
    sb_model.eval()
    print(f"SilverBullet checkpoint: {ckpt_path}  [{arch}]")

    # ── Load test data ───────────────────────────────────────────────────────
    data_dir = f"data/{mode}"
    test_pairs, test_labels_raw = load_json_data(f"{data_dir}/test.json")
    binary_true = (np.array(test_labels_raw).flatten() >= 0.5).astype(int)
    n_test = len(test_pairs)
    print(f"Test pairs: {n_test}")

    dataset = TextSimilarityDataset(test_pairs, test_labels_raw, use_cache=True)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False)

    # ── SilverBullet inference ───────────────────────────────────────────────
    print("Running SilverBullet inference...")
    sb_probs = []
    sb_model.eval()
    with torch.no_grad():
        for features, _ in loader:
            out = sb_model(features.to(device)).cpu().numpy().flatten()
            sb_probs.extend(out.tolist())
    sb_probs = np.array(sb_probs)
    sb_metrics = _compute_metrics(binary_true, sb_probs)

    # Latency: inference only on a sample (precomputed features)
    latency_features = dataset.features[:n_latency].to(device)
    sb_latency = _measure_sb_latency(sb_model, latency_features, device)

    result = {
        "timestamp":    datetime.now().isoformat(),
        "mode":         mode,
        "n_test_pairs": n_test,
        "n_latency":    n_latency,
        "device":       str(device),
        "silverbullet": {
            "checkpoint":         ckpt_path,
            "architecture":       arch,
            "metrics":            sb_metrics,
            "latency_ms_per_pair": sb_latency,
        },
        "rerankers": {},
    }

    # ── Re-ranker benchmarks ─────────────────────────────────────────────────
    raw_pairs = [(p[0], p[1]) for p in test_pairs]
    latency_pairs = raw_pairs[:n_latency]

    for rr_id in reranker_ids:
        cfg = RERANKERS[rr_id]
        print(f"\nLoading re-ranker: {rr_id}")
        ce = CrossEncoder(rr_id, max_length=512)

        print(f"  Running inference on {n_test} pairs...")
        t0 = time.perf_counter()
        raw_scores = ce.predict(raw_pairs, batch_size=32, show_progress_bar=True)
        inference_time = time.perf_counter() - t0
        print(f"  Done in {inference_time:.1f}s")

        rr_probs = cfg["score_fn"](raw_scores)
        rr_probs = np.clip(rr_probs, 0.0, 1.0)
        rr_metrics = _compute_metrics(binary_true, rr_probs)

        print(f"  Measuring latency ({n_latency} pairs, {5} reps)...")
        rr_latency = _measure_rr_latency(ce, latency_pairs)

        safe_label = cfg["label"].replace("/", "-").replace(" ", "_")
        failure = _failure_cases(
            test_pairs, binary_true, sb_probs, rr_probs,
            rr_label=safe_label,
        )

        result["rerankers"][rr_id] = {
            "label":              cfg["label"],
            "description":        cfg["description"],
            "metrics":            rr_metrics,
            "latency_ms_per_pair": rr_latency,
            "failure_cases":      failure,
        }

        # Print side-by-side
        print(f"\n  {'Metric':<20} {'SilverBullet':>14} {cfg['label']:>20} {'Delta':>8}")
        print(f"  {'-'*65}")
        for key, name, lower_better in [
            ("accuracy",    "Accuracy",    False),
            ("roc_auc",     "ROC-AUC",     False),
            ("auprc",       "AUPRC",       False),
            ("mcc",         "MCC",         False),
            ("f1",          "F1 (@0.5)",   False),
            ("brier_score", "Brier Score", True),
        ]:
            delta = _delta(sb_metrics[key], rr_metrics[key], lower_better)
            print(f"  {name:<20} {sb_metrics[key]:>14.4f} {rr_metrics[key]:>20.4f} {delta:>8}")
        print(f"\n  Latency (ms/pair): SilverBullet={sb_latency:.2f}  {cfg['label']}={rr_latency:.2f}")

    return result


# ---------------------------------------------------------------------------
# Report saving
# ---------------------------------------------------------------------------

def save_report(result: dict, output_dir: str = "benchmark_reports") -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = result["mode"]

    json_path = os.path.join(output_dir, f"benchmark_{mode}_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    md_path = os.path.join(output_dir, f"benchmark_{mode}_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_markdown_report(result))

    print(f"\nReport saved: {json_path}")
    print(f"Report saved: {md_path}")
    return json_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark SilverBullet vs cross-encoder re-rankers.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mode", choices=VALID_MODES, help="Single mode to benchmark.")
    group.add_argument("--all-modes", action="store_true", help="Benchmark all three modes.")
    parser.add_argument(
        "--rerankers", nargs="+", choices=list(RERANKERS.keys()), default=list(RERANKERS.keys()),
        help="Re-ranker model IDs to benchmark against (default: all).",
    )
    parser.add_argument(
        "--n-latency", type=int, default=20,
        help="Number of pairs used for latency measurement (default: 20).",
    )
    args = parser.parse_args()

    modes = VALID_MODES if args.all_modes else [args.mode]
    all_results = []

    for mode in modes:
        result = run_benchmark(mode, n_latency=args.n_latency, reranker_ids=args.rerankers)
        save_report(result)
        all_results.append(result)

    # ── Cross-mode summary ───────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("CROSS-MODE SUMMARY")
        print(f"{'='*60}")
        for res in all_results:
            sb_acc  = res["silverbullet"]["metrics"]["accuracy"]
            sb_auc  = res["silverbullet"]["metrics"]["roc_auc"]
            print(f"\n  {res['mode']}")
            print(f"    SilverBullet: acc={sb_acc:.4f}  auc={sb_auc:.4f}")
            for rr_id, rr in res["rerankers"].items():
                rr_acc = rr["metrics"]["accuracy"]
                rr_auc = rr["metrics"]["roc_auc"]
                delta_acc = _delta(sb_acc, rr_acc)
                delta_auc = _delta(sb_auc, rr_auc)
                print(f"    {rr['label']}: acc={rr_acc:.4f} ({delta_acc})  auc={rr_auc:.4f} ({delta_auc})")
