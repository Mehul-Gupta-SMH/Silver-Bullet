"""One-shot script: patch svo_triplet_recall into cached RVG + MVM entries.

Reads every RVG and MVM data pair, finds cache entries that are dicts missing
'svo_triplet_recall', runs SVOGrounding on that pair, and writes the key back.
All other features are untouched — no Relex, no NLI, no GPU required.

Usage:
    python -m backend.prefill_svo [--modes rvg mvm] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from tqdm import tqdm

from backend.feature_cache import FeatureCache
from backend.Features.Relations.getSVOWeights import SVOGrounding
from backend.Splitter.sentence_splitter import split_txt

_MODE_DIR = {
    "rvg": "data/reference-vs-generated",
    "mvm": "data/model-vs-model",
    "cvg": "data/context-vs-generated",
}
_SPLITS = ["train", "validate", "test"]


def _collect_pairs(modes: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for mode in modes:
        d = pathlib.Path(_MODE_DIR[mode])
        for split in _SPLITS:
            p = d / f"{split}.json"
            if not p.exists():
                continue
            for row in json.loads(p.read_text())["data"]:
                pairs.append((row["text1"], row["text2"]))
    return list(dict.fromkeys(pairs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["rvg", "mvm"],
                        choices=list(_MODE_DIR), help="Modes to prefill")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count pairs that need patching without writing")
    args = parser.parse_args()

    cache = FeatureCache()
    svo   = SVOGrounding()

    all_pairs = _collect_pairs(args.modes)
    to_patch  = [
        (t1, t2) for t1, t2 in all_pairs
        if isinstance(cache.get_features(t1, t2), dict)
        and "svo_triplet_recall" not in cache.get_features(t1, t2)
    ]
    already   = len(all_pairs) - len(to_patch)

    print(f"Modes : {args.modes}")
    print(f"Pairs : {len(all_pairs)} total  |  {already} already have svo  |  {len(to_patch)} to patch")

    if args.dry_run or not to_patch:
        print("Dry-run / nothing to do — exiting.")
        return

    errors = 0
    t0 = time.perf_counter()

    for text1, text2 in tqdm(to_patch, desc="SVO prefill", unit="pair"):
        try:
            entry = cache.get_features(text1, text2)
            if not isinstance(entry, dict):
                continue
            sent1 = split_txt(text1)
            sent2 = split_txt(text2)
            fm    = svo.getFeatureMap(sent1, sent2)
            entry["svo_triplet_recall"] = fm["svo_triplet_recall"].tolist()
            cache.save_features(text1, text2, entry)
        except Exception as exc:
            errors += 1
            tqdm.write(f"  [WARN] {exc!r}")

    elapsed = time.perf_counter() - t0
    done    = len(to_patch) - errors
    print(f"\nDone. {done} patched in {elapsed:.0f}s  ({errors} errors)")


if __name__ == "__main__":
    main()
