"""
fetch_external_data.py
----------------------
Downloads public NLP datasets from HuggingFace and merges them with the
hand-crafted SilverBullet pairs to produce larger train/validate/test splits.

Datasets
  General (added to all mode splits):
    STS-B   – sentence-pair similarity, score ≥ 3.5 → label=1
    MNLI    – entailment(0)→1, neutral(1)/contradiction(2)→0

  Mode-specific:
    model-vs-model        – QQP  (duplicate question pairs, 1=duplicate)
    reference-vs-generated – QNLI (question-answer entailment, 0=entailment→1)
    context-vs-generated  – HaluEval QA ((knowledge, right)→1, (knowledge, hallucinated)→0)

Usage
    python -m backend.fetch_external_data
    python -m backend.fetch_external_data --max-per-source 300 --seed 99

The script:
  1. Downloads each dataset via the HuggingFace `datasets` library.
  2. Filters degenerate pairs (too short / too long / identical texts).
  3. Samples up to --max-per-source pairs, balanced label=0/1.
  4. Caches raw downloads to data/external/.
  5. Merges with hand-crafted pairs from generate_data.py.
  6. Re-saves train/validate/test splits (70/15/15) to data/ and data/{mode}/.

Delete data/external/ and re-run to force a fresh download.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

try:
    import datasets as hf_datasets
except ImportError:
    raise SystemExit(
        "The 'datasets' package is required for this script.\n"
        "Install with:  pip install 'datasets>=2.0'"
    )

from backend.generate_data import (
    PAIRS,
    MODE_PAIRS_MODEL_VS_MODEL,
    MODE_PAIRS_REFERENCE_VS_GENERATED,
    MODE_PAIRS_CONTEXT_VS_GENERATED,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent
DATA_DIR = _ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"

# ---------------------------------------------------------------------------
# Filtering & sampling helpers
# ---------------------------------------------------------------------------

_MIN_LEN = 20
_MAX_LEN = 4_000


def _filter(pairs: list[dict]) -> list[dict]:
    """Remove degenerate pairs: too short, too long, or identical texts."""
    out = []
    for p in pairs:
        t1, t2 = str(p["text1"]).strip(), str(p["text2"]).strip()
        if len(t1) < _MIN_LEN or len(t2) < _MIN_LEN:
            continue
        if len(t1) > _MAX_LEN or len(t2) > _MAX_LEN:
            continue
        if t1 == t2:
            continue
        out.append({"text1": t1, "text2": t2, "label": int(p["label"])})
    return out


def _balanced_sample(pairs: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample up to n pairs with balanced label=0 / label=1."""
    pos = [p for p in pairs if p["label"] == 1]
    neg = [p for p in pairs if p["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    half = n // 2
    return pos[:half] + neg[:half]


# ---------------------------------------------------------------------------
# Per-dataset loaders
# ---------------------------------------------------------------------------

def _load_stsb(max_n: int, rng: random.Random) -> list[dict]:
    """GLUE STS-B: score ≥ 3.5 → label=1.  Covers graded / partial similarity."""
    print("  [STS-B] downloading…")
    ds = hf_datasets.load_dataset("glue", "stsb", split="train")
    pairs = [
        {"text1": r["sentence1"], "text2": r["sentence2"], "label": 1 if r["label"] >= 3.5 else 0}
        for r in ds
    ]
    pairs = _filter(pairs)
    sampled = _balanced_sample(pairs, max_n, rng)
    print(f"  [STS-B] {len(sampled)} pairs  "
          f"(pos={sum(p['label'] for p in sampled)}, neg={sum(1-p['label'] for p in sampled)})")
    return sampled


def _load_mnli(max_n: int, rng: random.Random) -> list[dict]:
    """GLUE MNLI: entailment(0)→1, neutral(1)/contradiction(2)→0."""
    print("  [MNLI]  downloading…")
    ds = hf_datasets.load_dataset("glue", "mnli", split="train")
    pairs = []
    for r in ds:
        if r["label"] == -1:          # skip unlabelled rows
            continue
        pairs.append({
            "text1": r["premise"],
            "text2": r["hypothesis"],
            "label": 1 if r["label"] == 0 else 0,
        })
    pairs = _filter(pairs)
    sampled = _balanced_sample(pairs, max_n, rng)
    print(f"  [MNLI]  {len(sampled)} pairs  "
          f"(pos={sum(p['label'] for p in sampled)}, neg={sum(1-p['label'] for p in sampled)})")
    return sampled


def _load_qqp(max_n: int, rng: random.Random) -> list[dict]:
    """GLUE QQP: duplicate(1)→label=1.  Proxy for model-vs-model agreement."""
    print("  [QQP]   downloading…")
    ds = hf_datasets.load_dataset("glue", "qqp", split="train")
    pairs = [
        {"text1": r["question1"], "text2": r["question2"], "label": int(r["label"])}
        for r in ds
    ]
    pairs = _filter(pairs)
    sampled = _balanced_sample(pairs, max_n, rng)
    print(f"  [QQP]   {len(sampled)} pairs  "
          f"(pos={sum(p['label'] for p in sampled)}, neg={sum(1-p['label'] for p in sampled)})")
    return sampled


def _load_qnli(max_n: int, rng: random.Random) -> list[dict]:
    """GLUE QNLI: entailment(0)→label=1.  Proxy for reference-vs-generated faithfulness."""
    print("  [QNLI]  downloading…")
    ds = hf_datasets.load_dataset("glue", "qnli", split="train")
    pairs = [
        {"text1": r["question"], "text2": r["sentence"], "label": 1 if r["label"] == 0 else 0}
        for r in ds
    ]
    pairs = _filter(pairs)
    sampled = _balanced_sample(pairs, max_n, rng)
    print(f"  [QNLI]  {len(sampled)} pairs  "
          f"(pos={sum(p['label'] for p in sampled)}, neg={sum(1-p['label'] for p in sampled)})")
    return sampled


def _load_halueval(max_n: int, rng: random.Random) -> list[dict]:
    """HaluEval QA: (knowledge, right_answer)→1, (knowledge, hallucinated_answer)→0.
    Directly mirrors context-vs-generated hallucination detection."""
    print("  [HaluEval] downloading…")
    ds = None
    for repo in ("pminervini/HaluEval", "HaluEval/HaluEval"):
        try:
            ds = hf_datasets.load_dataset(repo, "qa_samples", split="data")
            break
        except Exception:
            continue
    if ds is None:
        print("  [HaluEval] WARNING: dataset not found on HuggingFace — skipping.")
        return []

    pairs = []
    for r in ds:
        knowledge = str(r.get("knowledge", "")).strip()
        right = str(r.get("right_answer", "")).strip()
        hallucinated = str(r.get("hallucinated_answer", "")).strip()
        if knowledge and right:
            pairs.append({"text1": knowledge, "text2": right, "label": 1})
        if knowledge and hallucinated:
            pairs.append({"text1": knowledge, "text2": hallucinated, "label": 0})

    pairs = _filter(pairs)
    sampled = _balanced_sample(pairs, max_n, rng)
    print(f"  [HaluEval] {len(sampled)} pairs  "
          f"(pos={sum(p['label'] for p in sampled)}, neg={sum(1-p['label'] for p in sampled)})")
    return sampled


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache(path: Path) -> list[dict] | None:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["data"]
    return None


def _save_cache(path: Path, pairs: list[dict]) -> None:
    path.write_text(json.dumps({"data": pairs}, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Split & save
# ---------------------------------------------------------------------------

def _split_and_save(pairs: list[dict], out_dir: Path, rng: random.Random) -> None:
    """Shuffle and save 70/15/15 train/validate/test JSON files."""
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    splits = {
        "train":    shuffled[:n_train],
        "validate": shuffled[n_train:n_train + n_val],
        "test":     shuffled[n_train + n_val:],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps({"data": data}, indent=2, ensure_ascii=False), encoding="utf-8")
        pos = sum(p["label"] for p in data)
        print(f"    {name:10s}: {len(data):5d} pairs  (pos={pos}, neg={len(data)-pos})")


def _to_dicts(tuples: list[tuple]) -> list[dict]:
    """Convert generate_data.py (text1, text2, label, kind) tuples → dicts."""
    return [{"text1": t1, "text2": t2, "label": lbl} for t1, t2, lbl, _ in tuples]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch external NLP datasets and merge into SilverBullet training splits"
    )
    parser.add_argument(
        "--max-per-source", type=int, default=400,
        help="Max pairs sampled per source, balanced label=0/1 (default: 400)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if cached copies exist in data/external/",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    n = args.max_per_source
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Download / load cache ────────────────────────────────────────────────

    def _get(name: str, loader) -> list[dict]:
        cache = EXTERNAL_DIR / f"{name}.json"
        if not args.force:
            cached = _load_cache(cache)
            if cached is not None:
                print(f"  [{name.upper()}] loaded from cache ({len(cached)} pairs)")
                return cached
        pairs = loader(n, rng)
        _save_cache(cache, pairs)
        return pairs

    print("\n── General datasets ────────────────────────────────────────────────")
    stsb = _get("stsb", _load_stsb)
    mnli = _get("mnli", _load_mnli)

    print("\n── Mode-specific datasets ──────────────────────────────────────────")
    qqp      = _get("qqp",      _load_qqp)
    qnli     = _get("qnli",     _load_qnli)
    halueval = _get("halueval", _load_halueval)

    # ── Hand-crafted pairs ───────────────────────────────────────────────────

    handcrafted = _to_dicts(PAIRS)
    mode_handcrafted = {
        "model-vs-model":         _to_dicts(MODE_PAIRS_MODEL_VS_MODEL),
        "reference-vs-generated": _to_dicts(MODE_PAIRS_REFERENCE_VS_GENERATED),
        "context-vs-generated":   _to_dicts(MODE_PAIRS_CONTEXT_VS_GENERATED),
    }
    mode_external = {
        "model-vs-model":         qqp,
        "reference-vs-generated": qnli,
        "context-vs-generated":   halueval,
    }
    general_external = stsb + mnli

    # ── Merge & save ─────────────────────────────────────────────────────────

    print("\n── General splits ──────────────────────────────────────────────────")
    general_all = handcrafted + general_external
    print(f"  Total: {len(general_all)} pairs  "
          f"(hand-crafted={len(handcrafted)}, external={len(general_external)})")
    _split_and_save(general_all, DATA_DIR, random.Random(args.seed))

    for mode in ("model-vs-model", "reference-vs-generated", "context-vs-generated"):
        mode_all = handcrafted + general_external + mode_handcrafted[mode] + mode_external[mode]
        print(f"\n── {mode} splits {'─' * max(1, 52 - len(mode))}")
        print(f"  Total: {len(mode_all)} pairs  "
              f"(hand-crafted={len(handcrafted) + len(mode_handcrafted[mode])}, "
              f"external={len(general_external) + len(mode_external[mode])})")
        _split_and_save(mode_all, DATA_DIR / mode, random.Random(args.seed))

    print("\n✓ Done. Retrain with:  python -m backend.train --mode <mode>")


if __name__ == "__main__":
    main()
