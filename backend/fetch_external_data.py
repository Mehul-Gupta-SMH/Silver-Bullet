"""
fetch_external_data.py
----------------------
Downloads public NLP datasets from HuggingFace and merges them with the
hand-crafted SilverBullet pairs to produce larger train/validate/test splits.

Datasets
  General (added to all mode splits):
    STS-B   – sentence-pair similarity, score ≥ 3.5 → label=1
    MNLI    – entailment(0)→1, neutral(1)/contradiction(2)→0

  context-vs-generated (hallucination detection):
    HaluEval QA/Sum/Dial – (knowledge, right)→1, (knowledge, hallucinated)→0
    RAGTruth             – real LLM outputs (GPT-4/Llama2/Mistral); passage+response pairs
    FaithDial            – dialogue grounded to Wikipedia knowledge; faithful/unfaithful per turn
    SummaC               – document+summary pairs; multi-source faithfulness consistency

  reference-vs-generated (faithfulness eval):
    QNLI   – question-answer entailment, 0=entailment→1
    PAWS   – adversarial paraphrases (surface-similar but semantically distinct)
    ANLI   – adversarially constructed NLI, R3 (hardest round); entailment→1, contradiction→0
    WiCE   – Wikipedia claim+evidence; supported→1, partial→0 (conservative), refuted→0
    SummaC – shared with cvg above

  model-vs-model (agreement scoring):
    QQP    – duplicate question pairs, 1=duplicate
    PAWS   – adversarial paraphrases
    MRPC   – sentence-level paraphrase equivalence

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
    """Sample up to n pairs with balanced label=0 / label=1.

    Uses a deterministic sort key derived from the pair content so that
    increasing *n* extends the sample rather than producing a wholly
    different set (stable superset property).  *rng* is used only to
    break ties and is not advanced, preserving the global RNG state for
    subsequent callers.
    """
    import hashlib

    def _stable_key(p: dict) -> str:
        return hashlib.md5((str(p["text1"]) + str(p["text2"])).encode()).hexdigest()

    pos = sorted([p for p in pairs if p["label"] == 1], key=_stable_key)
    neg = sorted([p for p in pairs if p["label"] == 0], key=_stable_key)
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


def _load_ragtruth(max_n: int, rng: random.Random) -> list[dict]:
    """RAGTruth: real GPT-4/Llama2/Mistral outputs across QA/summarization/data2text tasks.
    text1=source passage/document, text2=model response.
    Hallucinated → label=0, faithful → label=1."""
    print("  [RAGTruth] downloading…")
    ds = None
    for repo in ("wentingzhao/RAGTruth", "plaguss/ragtruth"):
        try:
            ds = hf_datasets.load_dataset(repo, split="train")
            break
        except Exception:
            continue
    if ds is None:
        print("  [RAGTruth] WARNING: not found on HuggingFace -- skipping.")
        return []

    pairs = []
    for r in ds:
        context  = str(r.get("source") or r.get("passage") or r.get("document") or "").strip()
        response = str(r.get("response") or r.get("summary") or "").strip()
        # Probe label field: True/1/"yes" → hallucinated → label=0 for us
        raw = r.get("is_hallucination") or r.get("hallucination") or r.get("label")
        if not context or not response or raw is None:
            continue
        if isinstance(raw, bool):
            label = 0 if raw else 1
        elif isinstance(raw, str):
            label = 0 if raw.lower() in ("yes", "true", "hallucinated", "1") else 1
        elif isinstance(raw, (int, float)):
            # RAGTruth convention: 1=hallucinated; invert to our 1=faithful convention
            label = 0 if int(raw) == 1 else 1
        else:
            continue
        pairs.append({"text1": context, "text2": response, "label": label})

    pairs = _filter(pairs)
    if not pairs:
        print("  [RAGTruth] WARNING: 0 valid pairs extracted — verify field names.")
        return []
    sampled = _balanced_sample(pairs, max_n, rng)
    print(f"  [RAGTruth] {len(sampled)} pairs  "
          f"(pos={sum(p['label'] for p in sampled)}, neg={sum(1-p['label'] for p in sampled)})")
    return sampled


def _load_faithdial(max_n: int, rng: random.Random) -> list[dict]:
    """FaithDial: dialogue grounded to Wikipedia knowledge snippets.
    text1=knowledge, text2=wizard response.  Faithful → label=1, hallucinated → label=0.
    Hallucination is signalled by BEGIN_HALLUCINATION / [H] markers in the response."""
    print("  [FaithDial] downloading…")
    try:
        ds = hf_datasets.load_dataset("McGill-NLP/FaithDial", split="train")
    except Exception:
        print("  [FaithDial] WARNING: not found on HuggingFace -- skipping.")
        return []

    pairs = []
    for r in ds:
        knowledge = str(r.get("knowledge") or "").strip()
        # Response may be in 'response', 'utterance', or nested 'history[-1]'
        response = str(
            r.get("response") or r.get("utterance") or
            (r.get("history") or [""])[-1] or ""
        ).strip()
        if not knowledge or not response:
            continue
        # Hallucination markers used in FaithDial annotation
        hallucinated = any(
            marker in response
            for marker in ("BEGIN HALLUCINATION", "END HALLUCINATION", "[H]", "<H>")
        )
        # Also check a dedicated label field if present
        raw_label = r.get("label") or r.get("hallucination") or r.get("faithful")
        if raw_label is not None and not isinstance(raw_label, str):
            # Prefer explicit label over text marker heuristic
            if isinstance(raw_label, bool):
                hallucinated = raw_label  # True = hallucinated
            elif isinstance(raw_label, (int, float)):
                hallucinated = bool(int(raw_label))
        pairs.append({
            "text1": knowledge,
            "text2": response.replace("BEGIN HALLUCINATION", "").replace("END HALLUCINATION", "").strip(),
            "label": 0 if hallucinated else 1,
        })

    pairs = _filter(pairs)
    if not pairs:
        print("  [FaithDial] WARNING: 0 valid pairs extracted — verify field names.")
        return []
    sampled = _balanced_sample(pairs, max_n, rng)
    print(f"  [FaithDial] {len(sampled)} pairs  "
          f"(pos={sum(p['label'] for p in sampled)}, neg={sum(1-p['label'] for p in sampled)})")
    return sampled


def _load_anli(max_n: int, rng: random.Random) -> list[dict]:
    """ANLI Round 3: adversarially constructed NLI — harder than MNLI/QNLI.
    text1=premise, text2=hypothesis.  entailment(0)→label=1, contradiction(2)→label=0.
    Neutral (1) is dropped.  Mapped to reference-vs-generated faithfulness."""
    print("  [ANLI-R3]  downloading…")
    try:
        ds = hf_datasets.load_dataset("facebook/anli", split="train_r3")
    except Exception:
        print("  [ANLI-R3]  WARNING: not found on HuggingFace -- skipping.")
        return []

    pairs = []
    for r in ds:
        if r["label"] == 1:   # neutral — skip
            continue
        pairs.append({
            "text1": str(r["premise"]).strip(),
            "text2": str(r["hypothesis"]).strip(),
            "label": 1 if r["label"] == 0 else 0,   # 0=entailment→1, 2=contradiction→0
        })

    pairs = _filter(pairs)
    sampled = _balanced_sample(pairs, max_n, rng)
    print(f"  [ANLI-R3]  {len(sampled)} pairs  "
          f"(pos={sum(p['label'] for p in sampled)}, neg={sum(1-p['label'] for p in sampled)})")
    return sampled


def _load_wice(max_n: int, rng: random.Random) -> list[dict]:
    """WiCE: Wikipedia claim+evidence fine-grained entailment.
    text1=evidence, text2=claim.  supported→1, partial→0 (conservative), refuted→0.
    Adversarial hard negatives built-in.  Mapped to reference-vs-generated."""
    print("  [WiCE]     downloading…")
    try:
        ds = hf_datasets.load_dataset("google/wice", split="train")
    except Exception:
        print("  [WiCE]     WARNING: not found on HuggingFace -- skipping.")
        return []

    pairs = []
    for r in ds:
        # Field names: evidence/claim or premise/hypothesis — probe both
        evidence = str(r.get("evidence") or r.get("premise") or r.get("text") or "").strip()
        claim    = str(r.get("claim") or r.get("hypothesis") or "").strip()
        raw      = str(r.get("label") or r.get("entailment") or "").strip().lower()
        if not evidence or not claim or not raw:
            continue
        if raw in ("supported", "entailment", "1", "true"):
            label = 1
        else:
            label = 0   # refuted or partial → 0 (conservative)
        pairs.append({"text1": evidence, "text2": claim, "label": label})

    pairs = _filter(pairs)
    if not pairs:
        print("  [WiCE]     WARNING: 0 valid pairs extracted — verify field names.")
        return []
    sampled = _balanced_sample(pairs, max_n, rng)
    print(f"  [WiCE]     {len(sampled)} pairs  "
          f"(pos={sum(p['label'] for p in sampled)}, neg={sum(1-p['label'] for p in sampled)})")
    return sampled


def _load_summac(max_n: int, rng: random.Random) -> list[dict]:
    """SummaC: document+summary pairs with binary factual consistency labels.
    text1=document, text2=summary.  consistent→label=1, inconsistent→label=0.
    Multi-source (XSum, CNN/DM, etc.); high-precision annotator-consensus labels."""
    print("  [SummaC]   downloading…")
    try:
        ds = hf_datasets.load_dataset("lhoestq/summac", split="train")
    except Exception:
        print("  [SummaC]   WARNING: not found on HuggingFace -- skipping.")
        return []

    pairs = []
    for r in ds:
        doc     = str(r.get("document") or r.get("doc") or r.get("text") or "").strip()
        summary = str(r.get("summary") or r.get("claim") or "").strip()
        raw     = r.get("label") or r.get("consistent") or r.get("score")
        if not doc or not summary or raw is None:
            continue
        if isinstance(raw, bool):
            label = 1 if raw else 0
        elif isinstance(raw, str):
            label = 1 if raw.lower() in ("1", "true", "consistent") else 0
        elif isinstance(raw, (int, float)):
            label = 1 if float(raw) >= 0.5 else 0
        else:
            continue
        pairs.append({"text1": doc, "text2": summary, "label": label})

    pairs = _filter(pairs)
    if not pairs:
        print("  [SummaC]   WARNING: 0 valid pairs extracted — verify field names.")
        return []
    sampled = _balanced_sample(pairs, max_n, rng)
    print(f"  [SummaC]   {len(sampled)} pairs  "
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
    ragtruth       = _get("ragtruth",       _load_ragtruth,                  n)
    faithdial      = _get("faithdial",      _load_faithdial,                 n)
    anli           = _get("anli",           _load_anli,                      n)
    wice           = _get("wice",           _load_wice,                      n)
    summac         = _get("summac",         _load_summac,                    n)

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
        # ANLI-R3: adversarially constructed NLI (harder than MNLI/QNLI)
        # WiCE: Wikipedia claim+evidence fine-grained entailment
        # SummaC: document+summary factual consistency
        "reference-vs-generated": qnli + paws + anli + wice + summac,
        # HaluEval QA/Sum/Dial: core on-task hallucination detection sources.
        # RAGTruth: real LLM outputs (GPT-4/Llama2/Mistral) across task types.
        # FaithDial: dialogue grounded to Wikipedia knowledge snippets.
        # SummaC: shared with rvg — abstractive summary faithfulness.
        # STS-B and MNLI are excluded from cvg: they measure similarity / logical
        # entailment, not context-grounding, producing label noise for this mode.
        "context-vs-generated":   halueval + halueval_sum + halueval_dial + ragtruth + faithdial + summac,
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
