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
        print("  [HaluEval] WARNING: dataset not found on HuggingFace -- skipping.")
        return []

    pairs = []
    for r in ds:
        # pminervini/HaluEval: knowledge + answer + hallucination ('yes'/'no')
        knowledge = str(r.get("knowledge", "")).strip()
        answer = str(r.get("answer", "")).strip()
        hallucination = str(r.get("hallucination", "")).strip().lower()
        if knowledge and answer and hallucination in ("yes", "no"):
            pairs.append({"text1": knowledge, "text2": answer,
                          "label": 0 if hallucination == "yes" else 1})

    pairs = _filter(pairs)
    sampled = _balanced_sample(pairs, max_n, rng)
    print(f"  [HaluEval] {len(sampled)} pairs  "
          f"(pos={sum(p['label'] for p in sampled)}, neg={sum(1-p['label'] for p in sampled)})")
    return sampled


def _load_halueval_summarization(max_n: int, rng: random.Random) -> list[dict]:
    """HaluEval summarization: (document, summary, hallucination=yes/no).
    hallucination=no → label=1 (faithful), hallucination=yes → label=0 (hallucinated).
    Extends CvG coverage with abstractive summarisation hallucinations."""
    print("  [HaluEval-Sum] downloading…")
    ds = None
    for repo in ("pminervini/HaluEval", "HaluEval/HaluEval"):
        try:
            ds = hf_datasets.load_dataset(repo, "summarization_samples", split="data")
            break
        except Exception:
            continue
    if ds is None:
        print("  [HaluEval-Sum] WARNING: dataset not found on HuggingFace -- skipping.")
        return []

    pairs = []
    for r in ds:
        document = str(r.get("document", "")).strip()
        summary = str(r.get("summary", "")).strip()
        hallucination = str(r.get("hallucination", "")).strip().lower()
        if document and summary and hallucination in ("yes", "no"):
            pairs.append({"text1": document, "text2": summary,
                          "label": 0 if hallucination == "yes" else 1})

    pairs = _filter(pairs)
    sampled = _balanced_sample(pairs, max_n, rng)
    print(f"  [HaluEval-Sum] {len(sampled)} pairs  "
          f"(pos={sum(p['label'] for p in sampled)}, neg={sum(1-p['label'] for p in sampled)})")
    return sampled


def _load_halueval_dialogue(max_n: int, rng: random.Random) -> list[dict]:
    """HaluEval dialogue: (knowledge, response, hallucination=yes/no).
    hallucination=no → label=1 (grounded), hallucination=yes → label=0 (hallucinated).
    Provides conversational grounding pairs for context-vs-generated."""
    print("  [HaluEval-Dial] downloading…")
    ds = None
    for repo in ("pminervini/HaluEval", "HaluEval/HaluEval"):
        try:
            ds = hf_datasets.load_dataset(repo, "dialogue_samples", split="data")
            break
        except Exception:
            continue
    if ds is None:
        print("  [HaluEval-Dial] WARNING: dataset not found on HuggingFace -- skipping.")
        return []

    pairs = []
    for r in ds:
        knowledge = str(r.get("knowledge", "")).strip()
        response = str(r.get("response", "")).strip()
        hallucination = str(r.get("hallucination", "")).strip().lower()
        if knowledge and response and hallucination in ("yes", "no"):
            pairs.append({"text1": knowledge, "text2": response,
                          "label": 0 if hallucination == "yes" else 1})

    pairs = _filter(pairs)
    sampled = _balanced_sample(pairs, max_n, rng)
    print(f"  [HaluEval-Dial] {len(sampled)} pairs  "
          f"(pos={sum(p['label'] for p in sampled)}, neg={sum(1-p['label'] for p in sampled)})")
    return sampled


def _load_paws(max_n: int, rng: random.Random) -> list[dict]:
    """PAWS: Paraphrase Adversaries from Word Scrambling.
    label=1 (paraphrase) and adversarial label=0 (non-paraphrase with high lexical overlap).
    Used for both model-vs-model (surface-similar but semantically different) and
    reference-vs-generated (generated matches reference or diverges)."""
    print("  [PAWS]  downloading…")
    try:
        ds = hf_datasets.load_dataset(
            "google-research-datasets/paws", "labeled_final", split="train"
        )
    except Exception:
        try:
            ds = hf_datasets.load_dataset("paws", "labeled_final", split="train")
        except Exception:
            print("  [PAWS]  WARNING: dataset not found on HuggingFace -- skipping.")
            return []

    pairs = [
        {"text1": str(r["sentence1"]).strip(), "text2": str(r["sentence2"]).strip(),
         "label": int(r["label"])}
        for r in ds
    ]
    pairs = _filter(pairs)
    sampled = _balanced_sample(pairs, max_n, rng)
    print(f"  [PAWS]  {len(sampled)} pairs  "
          f"(pos={sum(p['label'] for p in sampled)}, neg={sum(1-p['label'] for p in sampled)})")
    return sampled


def _load_mrpc(max_n: int, rng: random.Random) -> list[dict]:
    """GLUE MRPC: Microsoft Research Paraphrase Corpus.
    label=1 (semantically equivalent), label=0 (not equivalent).
    Good proxy for model-vs-model agreement on short factual sentences."""
    print("  [MRPC]  downloading…")
    try:
        ds = hf_datasets.load_dataset("glue", "mrpc", split="train")
    except Exception:
        print("  [MRPC]  WARNING: dataset not found on HuggingFace -- skipping.")
        return []

    pairs = [
        {"text1": str(r["sentence1"]).strip(), "text2": str(r["sentence2"]).strip(),
         "label": int(r["label"])}
        for r in ds
    ]
    pairs = _filter(pairs)
    sampled = _balanced_sample(pairs, max_n, rng)
    print(f"  [MRPC]  {len(sampled)} pairs  "
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
        help="Max pairs sampled per non-HaluEval source, balanced label=0/1 (default: 400)",
    )
    parser.add_argument(
        "--halueval-max", type=int, default=700,
        help="Max pairs sampled per HaluEval source for cvg (default: 700). "
             "HaluEval is the only on-task cvg source so a higher limit is used.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if cached copies exist in data/external/",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    n        = args.max_per_source
    n_halue  = args.halueval_max
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Download / load cache ────────────────────────────────────────────────

    def _get(name: str, loader, limit: int) -> list[dict]:
        cache = EXTERNAL_DIR / f"{name}.json"
        if not args.force:
            cached = _load_cache(cache)
            if cached is not None and len(cached) >= limit:
                print(f"  [{name.upper()}] loaded from cache ({len(cached)} pairs)")
                return cached
        pairs = loader(limit, rng)
        _save_cache(cache, pairs)
        return pairs

    print("\n-- General datasets ----------------------------------------------------")
    stsb = _get("stsb", _load_stsb, n)
    mnli = _get("mnli", _load_mnli, n)

    print("\n-- Mode-specific datasets ----------------------------------------------")
    qqp            = _get("qqp",            _load_qqp,                       n)
    qnli           = _get("qnli",           _load_qnli,                      n)
    halueval       = _get("halueval",       _load_halueval,                  n_halue)
    halueval_sum   = _get("halueval_sum",   _load_halueval_summarization,    n_halue)
    halueval_dial  = _get("halueval_dial",  _load_halueval_dialogue,         n_halue)
    paws           = _get("paws",           _load_paws,                      n)
    mrpc           = _get("mrpc",           _load_mrpc,                      n)

    # ── Hand-crafted pairs ───────────────────────────────────────────────────

    handcrafted = _to_dicts(PAIRS)
    mode_handcrafted = {
        "model-vs-model":         _to_dicts(MODE_PAIRS_MODEL_VS_MODEL),
        "reference-vs-generated": _to_dicts(MODE_PAIRS_REFERENCE_VS_GENERATED),
        "context-vs-generated":   _to_dicts(MODE_PAIRS_CONTEXT_VS_GENERATED),
    }
    mode_external = {
        # QQP: duplicate questions (model agreement proxy)
        # PAWS: adversarial paraphrases (surface-similar but semantically distinct pairs)
        # MRPC: sentence-level paraphrase equivalence
        "model-vs-model":         qqp + paws + mrpc,
        # QNLI: question-answer entailment (reference faithfulness proxy)
        # PAWS: paraphrase adversarial (generated matches reference or diverges)
        "reference-vs-generated": qnli + paws,
        # HaluEval QA/Sum/Dial: the only on-task sources for hallucination detection.
        # STS-B and MNLI are excluded from cvg: they measure similarity / logical
        # entailment, not context-grounding, producing label noise for this mode.
        "context-vs-generated":   halueval + halueval_sum + halueval_dial,
    }
    general_external = stsb + mnli
    # STS-B + MNLI are valid general proxies for rvg/mvm but noise for cvg.
    mode_general = {
        "model-vs-model":         general_external,
        "reference-vs-generated": general_external,
        "context-vs-generated":   [],
    }

    # ── Merge & save ─────────────────────────────────────────────────────────

    print("\n-- General splits ------------------------------------------------------")
    general_all = handcrafted + general_external
    print(f"  Total: {len(general_all)} pairs  "
          f"(hand-crafted={len(handcrafted)}, external={len(general_external)})")
    _split_and_save(general_all, DATA_DIR, random.Random(args.seed))

    for mode in ("model-vs-model", "reference-vs-generated", "context-vs-generated"):
        gen   = mode_general[mode]
        mode_all = handcrafted + gen + mode_handcrafted[mode] + mode_external[mode]
        n_hand = len(handcrafted) + len(mode_handcrafted[mode])
        n_ext  = len(gen) + len(mode_external[mode])
        print(f"\n-- {mode} splits {'-' * max(1, 52 - len(mode))}")
        print(f"  Total: {len(mode_all)} pairs  (hand-crafted={n_hand}, external={n_ext})")
        _split_and_save(mode_all, DATA_DIR / mode, random.Random(args.seed))

    print("\nDone. Retrain with:  python -m backend.train --mode <mode>")


if __name__ == "__main__":
    main()
