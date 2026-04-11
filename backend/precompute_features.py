"""
precompute_features.py
----------------------
Pre-populates the feature cache for every unique text pair across all data
splits and all modes, so subsequent training runs hit 100% cache and skip
feature extraction entirely.

Run once before training:
    python -m backend.precompute_features

Options:
    --data-dir PATH    Root data directory (default: data/)
    --cache-dir PATH   Feature cache directory (default: cache/)
    --workers N        Parallel worker processes for extraction (default: 1)
    --skip-cached      Skip pairs already in cache (default: True)
    --force            Recompute and overwrite all cached features

After this completes, run training with no extraction overhead:
    python -m backend.train --mode context-vs-generated
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from tqdm import tqdm

from backend.feature_cache import FeatureCache
from backend.Splitter.sentence_splitter import split_txt
from backend.Features.Semantic.getSemanticWeights import SemanticWeights
from backend.Features.Lexical.getLexicalWeights import LexicalWeights
from backend.Features.NLI.getNLIweights import NLIWeights
from backend.Features.EntityGroups.getOverlap import EntityMatch
from backend.Features.LCS.getLCSweights import LCSWeights
from backend.Features.Relations.getRelationWeights import RelationGrounding
from backend.train import feature_map_to_tensor

_MODES = ["model-vs-model", "reference-vs-generated", "context-vs-generated"]
_SPLITS = ["train", "validate", "test"]


def _collect_pairs(data_dir: Path) -> list[tuple[str, str]]:
    """Collect every unique (text1, text2) pair across all splits and modes."""
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []

    def _add(path: Path) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("data", []):
            key = (item["text1"], item["text2"])
            if key not in seen:
                seen.add(key)
                pairs.append(key)

    # General splits
    for split in _SPLITS:
        _add(data_dir / f"{split}.json")

    # Per-mode splits
    for mode in _MODES:
        for split in _SPLITS:
            _add(data_dir / mode / f"{split}.json")

    return pairs


def _build_extractors():
    """Load all feature extractor models once — reused across all pairs."""
    print("Loading feature extractors...")
    extractors = {
        "lexical":    LexicalWeights(),
        "semantic":   SemanticWeights(),
        "nli":        NLIWeights(),
        "entity":     EntityMatch(),
        "lcs":        LCSWeights(),
        "relations":  RelationGrounding(),
    }
    print("All extractors ready.\n")
    return extractors


def _extract(text1: str, text2: str, extractors: dict) -> list:
    """Run full feature pipeline for one pair. Returns stacked tensor as list."""
    sent1 = split_txt(text1)
    sent2 = split_txt(text2)

    feature_map = {}
    feature_map.update(extractors["lexical"].getFeatureMap(sent1, sent2))
    feature_map.update(extractors["semantic"].getFeatureMap(sent1, sent2))
    feature_map.update(extractors["nli"].getFeatureMap(sent1, sent2))
    feature_map.update(extractors["entity"].getFeatureMap(sent1, sent2))
    feature_map.update(extractors["lcs"].getFeatureMap(sent1, sent2))
    feature_map.update(extractors["relations"].getFeatureMap(sent1, sent2))

    stacked = feature_map_to_tensor(feature_map)   # [F, 64, 64]
    return stacked.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute feature cache for all data splits"
    )
    parser.add_argument("--data-dir",  default="data",  help="Root data directory")
    parser.add_argument("--cache-dir", default="cache", help="Feature cache directory")
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute and overwrite all cached features (default: skip cached)",
    )
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect all unique pairs ─────────────────────────────────────────────
    print(f"Scanning {data_dir} for pairs...")
    all_pairs = _collect_pairs(data_dir)
    print(f"Found {len(all_pairs)} unique pairs across all splits and modes.\n")

    # ── Filter already-cached pairs ──────────────────────────────────────────
    cache = FeatureCache(cache_dir=str(cache_dir))
    if not args.force:
        todo = [(t1, t2) for t1, t2 in all_pairs if cache.get_features(t1, t2) is None]
        cached_count = len(all_pairs) - len(todo)
        if cached_count:
            print(f"Skipping {cached_count} already-cached pairs. "
                  f"{len(todo)} remaining.\n")
    else:
        todo = all_pairs
        print(f"--force: recomputing all {len(todo)} pairs.\n")

    if not todo:
        print("All pairs already cached. Nothing to do.")
        print("Run training with:  python -m backend.train --mode <mode>")
        return

    # ── Load extractors once ─────────────────────────────────────────────────
    extractors = _build_extractors()

    # ── Extract & cache ──────────────────────────────────────────────────────
    errors = 0
    t0 = time.perf_counter()

    for text1, text2 in tqdm(todo, desc="Precomputing features", unit="pair"):
        try:
            features = _extract(text1, text2, extractors)
            cache.save_features(text1, text2, features)
        except Exception as exc:
            errors += 1
            tqdm.write(f"  [WARN] extraction failed: {exc!r}  (pair skipped)")

    elapsed = time.perf_counter() - t0
    done    = len(todo) - errors

    print(f"\nDone. {done} pairs cached in {elapsed:.0f}s "
          f"({elapsed/max(done,1):.1f}s/pair).  Errors: {errors}")
    if errors:
        print(f"  {errors} pairs failed — re-run to retry, or inspect warnings above.")
    print("\nTraining will now use 100% cache hits:")
    for mode in _MODES:
        print(f"  python -m backend.train --mode {mode}")


if __name__ == "__main__":
    main()
