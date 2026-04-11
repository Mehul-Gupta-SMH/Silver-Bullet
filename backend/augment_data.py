"""
augment_data.py
---------------
Generates adversarial hard-negative pairs and appends them to the mode-specific
train splits.  Two augmentation strategies:

1. Numeric-swap
   Take a label=1 pair whose text2 contains at least one numeric token.
   Replace that token with a plausible but different value.
   Result: same topic/entities, different numbers → label=0.
   Targets the `numeric_jaccard` gap identified in feature analysis.

2. Entity-swap
   Take a label=1 pair whose text2 contains at least one proper noun.
   Replace that proper noun with a different one drawn from a cross-pair pool.
   Result: same structure/numbers, swapped entity → label=0.
   Targets the `entity_grounding_recall` / `entity_value_prec` gap.

Usage:
    python -m backend.augment_data [--modes all] [--per-mode 200] [--dry-run]

Appends new pairs to data/{mode}/train.json only.
Does NOT modify validate or test splits.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
MODES = ["context-vs-generated", "reference-vs-generated", "model-vs-model"]
SEED = 99
random.seed(SEED)

# ---------------------------------------------------------------------------
# Numeric extraction / replacement
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(
    r"""
    (?:[\$£€¥])?              # optional currency
    \b
    \d[\d,\.]*                # integer or decimal
    \s*
    (?:%|[KkMmBbTt](?:illion|n)?\b)?  # optional suffix
    """,
    re.VERBOSE,
)

# Replacement pools keyed by rough magnitude bucket
_NUMERIC_POOLS: dict[str, list[str]] = {
    "pct":   ["12%", "23%", "37%", "48%", "61%", "74%", "89%", "95%", "3%", "17%"],
    "small": ["3", "7", "11", "14", "19", "23", "31", "47"],
    "mid":   ["120", "250", "480", "750", "1,200", "3,400", "7,800"],
    "large": ["$1.2 million", "$4.5 million", "$8 billion", "$15 billion", "$200 million"],
    "year":  ["1987", "1993", "2001", "2008", "2015", "2021", "2024"],
    "speed": ["120 m/s", "340 m/s", "1,480 m/s", "299 km/s"],
}


def _bucket(token: str) -> str:
    t = token.strip().lower()
    if "%" in t:
        return "pct"
    if re.search(r"\d{4}", t) and int(re.search(r"\d{4}", t).group()) > 1800:
        return "year"
    if re.search(r"m/s|km/s", t):
        return "speed"
    digits = re.sub(r"[^\d]", "", t)
    if not digits:
        return "small"
    v = int(digits)
    if v > 100_000:
        return "large"
    if v > 999:
        return "mid"
    return "small"


def numeric_swap(text: str, original: str) -> str | None:
    """Replace *original* numeric token in *text* with a different plausible value."""
    bucket = _bucket(original)
    pool = [x for x in _NUMERIC_POOLS[bucket] if x.lower() != original.lower().strip()]
    if not pool:
        return None
    replacement = random.choice(pool)
    # Replace only the first occurrence to keep the swap localised
    return text.replace(original, replacement, 1)


# ---------------------------------------------------------------------------
# Proper-noun extraction (no model — capitalisation heuristic)
# ---------------------------------------------------------------------------
_PROPER_RE = re.compile(
    r"""
    (?<![.!?]\s)           # not sentence-start (preceded by sentence-ending punct)
    (?<!\A)                # not very start of string
    \b([A-Z][a-z]+         # first capitalised word
    (?:\s+[A-Z][a-z]+)*)   # optional further capitalised words
    \b
    """,
    re.VERBOSE,
)

# Exclusion list — common sentence-initial / title words that aren't entities
_STOPWORDS = {
    "The", "A", "An", "This", "That", "These", "Those", "It", "He", "She",
    "They", "We", "You", "I", "In", "On", "At", "By", "For", "Of", "To",
    "And", "Or", "But", "With", "From", "As", "Is", "Are", "Was", "Were",
    "Be", "Been", "Being", "Have", "Has", "Had", "Do", "Does", "Did",
    "Will", "Would", "Could", "Should", "May", "Might", "Must", "Shall",
    "Not", "No", "So", "If", "When", "Where", "Which", "Who", "What",
    "How", "Each", "All", "Both", "Some", "Many", "Most", "More",
    "During", "After", "Before", "Over", "Under", "Between", "Among",
    "Through", "About", "Against", "Within", "Without", "According",
    "Based", "Using", "Given", "Since", "Until", "While", "Because",
}


def _extract_proper_nouns(text: str) -> list[str]:
    candidates = _PROPER_RE.findall(text)
    return [c for c in candidates if c not in _STOPWORDS and len(c) > 2]


def _build_entity_pool(all_pairs: list[dict]) -> list[str]:
    """Mine all training text2 strings for proper nouns → pool for swaps."""
    pool: set[str] = set()
    for p in all_pairs:
        for field in ("text1", "text2"):
            pool.update(_extract_proper_nouns(p.get(field, "")))
    # Keep only tokens seen in multiple texts (more likely true entities)
    counts: dict[str, int] = {}
    for p in all_pairs:
        for field in ("text1", "text2"):
            seen_in_pair: set[str] = set()
            for noun in _extract_proper_nouns(p.get(field, "")):
                if noun not in seen_in_pair:
                    counts[noun] = counts.get(noun, 0) + 1
                    seen_in_pair.add(noun)
    return [n for n, c in counts.items() if c >= 2]


def entity_swap(text: str, entity: str, pool: list[str]) -> str | None:
    """Replace *entity* in *text* with a different entity from *pool*."""
    candidates = [e for e in pool if e != entity and e not in text]
    if not candidates:
        return None
    replacement = random.choice(candidates)
    return text.replace(entity, replacement, 1)


# ---------------------------------------------------------------------------
# Augmentation logic
# ---------------------------------------------------------------------------

def _augment_split(
    pairs: list[dict],
    entity_pool: list[str],
    target_new: int,
) -> list[dict]:
    """Generate up to *target_new* adversarial pairs from positive *pairs*."""
    positives = [p for p in pairs if p.get("label") == 1]
    random.shuffle(positives)

    new_pairs: list[dict] = []
    seen_texts: set[str] = {p["text2"] for p in pairs}  # avoid duplicates

    # --- Numeric-swap ---
    for pair in positives:
        if len(new_pairs) >= target_new // 2:
            break
        text2 = pair.get("text2", "")
        tokens = _NUM_RE.findall(text2)
        tokens = [t.strip() for t in tokens if t.strip()]
        if not tokens:
            continue
        token = random.choice(tokens)
        swapped = numeric_swap(text2, token)
        if swapped is None or swapped == text2 or swapped in seen_texts:
            continue
        seen_texts.add(swapped)
        new_pairs.append({
            "text1": pair["text1"],
            "text2": swapped,
            "label": 0,
            "_aug": "numeric_swap",
        })

    # --- Entity-swap ---
    for pair in positives:
        if len(new_pairs) >= target_new:
            break
        text2 = pair.get("text2", "")
        entities = _extract_proper_nouns(text2)
        if not entities:
            continue
        entity = random.choice(entities)
        swapped = entity_swap(text2, entity, entity_pool)
        if swapped is None or swapped == text2 or swapped in seen_texts:
            continue
        seen_texts.add(swapped)
        new_pairs.append({
            "text1": pair["text1"],
            "text2": swapped,
            "label": 0,
            "_aug": "entity_swap",
        })

    return new_pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(modes: list[str], per_mode: int, dry_run: bool) -> None:
    for mode in modes:
        train_path = DATA_DIR / mode / "train.json"
        if not train_path.exists():
            print(f"[SKIP] {train_path} not found")
            continue

        data = json.loads(train_path.read_text(encoding="utf-8"))
        original_pairs = data["data"]
        n_before = len(original_pairs)

        entity_pool = _build_entity_pool(original_pairs)
        new_pairs = _augment_split(original_pairs, entity_pool, per_mode)

        numeric_count = sum(1 for p in new_pairs if p.get("_aug") == "numeric_swap")
        entity_count  = sum(1 for p in new_pairs if p.get("_aug") == "entity_swap")

        print(
            f"{mode}: +{len(new_pairs)} pairs "
            f"(numeric_swap={numeric_count}, entity_swap={entity_count}) "
            f"-> total {n_before + len(new_pairs)}"
        )

        if dry_run:
            # Print a few examples (encode safely for Windows terminals)
            for p in new_pairs[:3]:
                sample = p['text2'][:120].encode('ascii', errors='replace').decode('ascii')
                print(f"  [{p['_aug']}] T2: {sample}")
            continue

        # Strip internal _aug metadata before saving
        clean_new = [{k: v for k, v in p.items() if k != "_aug"} for p in new_pairs]
        data["data"] = original_pairs + clean_new
        train_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Saved -> {train_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial data augmentation")
    parser.add_argument(
        "--modes", nargs="+", default=MODES,
        choices=MODES + ["all"],
        help="Which mode splits to augment (default: all three)",
    )
    parser.add_argument(
        "--per-mode", type=int, default=200,
        help="Target number of new pairs per mode (default: 200)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be added without writing files",
    )
    args = parser.parse_args()
    modes = MODES if "all" in args.modes else args.modes
    run(modes, args.per_mode, args.dry_run)
