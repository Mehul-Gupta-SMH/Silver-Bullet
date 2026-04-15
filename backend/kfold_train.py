"""kfold_train.py — K-fold cross-validation training for SilverBullet.

Pools train + validate splits, runs stratified k-fold CV, reports mean ± std
metrics across folds, then evaluates the best fold's checkpoint on the held-out
test set.

Usage:
    python -m backend.kfold_train --mode context-vs-generated
    python -m backend.kfold_train --mode reference-vs-generated --k 5
    python -m backend.kfold_train --mode model-vs-model --k 10

Each fold saves a checkpoint to:
    models/{mode}/kfold_fold{i}_best.pth

A summary JSON is written to:
    training_reports/kfold_{mode}_{date}.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from datetime import date
from pathlib import Path

from torch.utils.data import DataLoader, Subset

from backend.feature_registry import get_feature_keys
from backend.model import TextSimilarityCNN
from backend.train import (
    TextSimilarityDataset,
    load_json_data,
    train_model,
    SPATIAL_SIZE,
)

_ROOT = Path(__file__).parent.parent


def _stratified_folds(labels: list[float], k: int, seed: int = 42) -> list[list[int]]:
    """Return k lists of indices, stratified by binary label."""
    import random
    rng = random.Random(seed)

    pos = [i for i, l in enumerate(labels) if l >= 0.5]
    neg = [i for i, l in enumerate(labels) if l < 0.5]
    rng.shuffle(pos)
    rng.shuffle(neg)

    folds: list[list[int]] = [[] for _ in range(k)]
    for bucket in (pos, neg):
        for j, idx in enumerate(bucket):
            folds[j % k].append(idx)

    return folds


def run_kfold(mode: str, k: int = 5) -> dict:
    data_dir = _ROOT / "data" / mode
    model_dir = _ROOT / "models" / mode
    model_dir.mkdir(parents=True, exist_ok=True)

    active_keys = get_feature_keys(mode)
    n_features = len(active_keys)
    print(f"\nMode: {mode}  |  k={k}  |  Features: {n_features}")

    # Pool train + validate (test stays held-out)
    train_pairs, train_labels = load_json_data(str(data_dir / "train.json"))
    val_pairs,   val_labels   = load_json_data(str(data_dir / "validate.json"))
    test_pairs,  test_labels  = load_json_data(str(data_dir / "test.json"))

    all_pairs  = train_pairs  + val_pairs
    all_labels = train_labels + val_labels
    n_pool = len(all_pairs)
    print(f"Pool: {n_pool} pairs  |  Test (held-out): {len(test_pairs)} pairs")

    # Build dataset once (all feature extraction + caching happens here)
    print("\nBuilding full pool dataset (this prefills all caches)...")
    pool_dataset = TextSimilarityDataset(all_pairs, all_labels, use_cache=True,
                                          feature_keys=active_keys)

    folds = _stratified_folds(all_labels, k=k)

    fold_metrics: list[dict] = []

    for fold_idx in range(k):
        val_indices   = folds[fold_idx]
        train_indices = [i for j, f in enumerate(folds) if j != fold_idx for i in f]

        print(f"\n{'='*60}")
        print(f"  Fold {fold_idx + 1}/{k}  |  train={len(train_indices)}  val={len(val_indices)}")
        print(f"{'='*60}")

        fold_train = Subset(pool_dataset, train_indices)
        fold_val   = Subset(pool_dataset, val_indices)

        train_loader = DataLoader(fold_train, batch_size=16, shuffle=True, drop_last=True)
        val_loader   = DataLoader(fold_val,   batch_size=16)

        best_ckpt = str(model_dir / f"kfold_fold{fold_idx + 1}_best.pth")

        fold_model = TextSimilarityCNN(
            num_features=pool_dataset.num_features,
            spatial_size=SPATIAL_SIZE,
            use_length_cond=False,
        )

        _, metrics = train_model(
            fold_model, train_loader, val_loader,
            best_ckpt=best_ckpt,
            mode=mode,
            label_smooth=0.0,
            feature_keys=active_keys,
        )

        fold_metrics.append({
            "fold": fold_idx + 1,
            "best_val_loss": metrics.get("best_val_loss"),
            "best_epoch":    metrics.get("best_epoch"),
            "checkpoint":    best_ckpt,
        })

    # Summary stats
    val_losses = [m["best_val_loss"] for m in fold_metrics if m["best_val_loss"] is not None]
    mean_loss  = statistics.mean(val_losses)
    std_loss   = statistics.stdev(val_losses) if len(val_losses) > 1 else 0.0

    print(f"\n{'='*60}")
    print(f"  K-fold summary ({k} folds)")
    print(f"  Val loss: {mean_loss:.4f} ± {std_loss:.4f}")
    for m in fold_metrics:
        print(f"    Fold {m['fold']}: val_loss={m['best_val_loss']:.4f}  best_epoch={m['best_epoch']}")
    print(f"{'='*60}")

    # Find best fold checkpoint and evaluate on held-out test
    best_fold = min(fold_metrics, key=lambda m: m["best_val_loss"] or float("inf"))
    print(f"\nBest fold: {best_fold['fold']}  (val_loss={best_fold['best_val_loss']:.4f})")
    print(f"Evaluating on held-out test set...")

    import torch
    from backend.predict import _load_model_from_checkpoint
    from backend.test import test_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(best_fold["checkpoint"], map_location=device, weights_only=False)
    best_model, _ = _load_model_from_checkpoint(checkpoint, device)
    feature_keys  = checkpoint.get("manifest", {}).get("features") or active_keys

    test_dataset = TextSimilarityDataset(test_pairs, test_labels, use_cache=True,
                                          feature_keys=feature_keys)
    test_loader  = DataLoader(test_dataset, batch_size=16)
    test_results = test_model(best_model, test_loader, test_pairs, device=device)

    summary = {
        "mode":       mode,
        "k":          k,
        "date":       str(date.today()),
        "pool_size":  n_pool,
        "test_size":  len(test_pairs),
        "fold_metrics": fold_metrics,
        "mean_val_loss": mean_loss,
        "std_val_loss":  std_loss,
        "best_fold":     best_fold["fold"],
        "test_results":  test_results,
    }

    report_dir = _ROOT / "training_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"kfold_{mode}_{date.today()}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved to {out_path}")

    return summary


if __name__ == "__main__":
    VALID_MODES = ["model-vs-model", "reference-vs-generated", "context-vs-generated"]

    parser = argparse.ArgumentParser(description="K-fold CV training for SilverBullet")
    parser.add_argument("--mode", choices=VALID_MODES, required=True,
                        help="Evaluation mode to train")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of folds (default: 5)")
    args = parser.parse_args()

    os.makedirs("cache", exist_ok=True)
    os.makedirs("training_reports", exist_ok=True)

    run_kfold(mode=args.mode, k=args.k)
