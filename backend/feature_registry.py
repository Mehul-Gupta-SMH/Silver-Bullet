"""Canonical feature key registry for SilverBullet.

Defines the ordered list of feature map keys that the current pipeline produces.
This order must match the dict-insertion order used in:
  - train.py  (TextSimilarityDataset)
  - predict.py (predict_pair_breakdown)

Embedding the manifest in every checkpoint lets the loader detect feature-set
drift before running inference rather than silently producing wrong scores.

When adding or removing a feature extractor:
  1. Update FEATURE_KEYS below (in the same insertion order as in train.py).
  2. Delete ./cache/ to force feature recomputation.
  3. Retrain from scratch — num_features changes make old checkpoints incompatible.
"""

from __future__ import annotations

from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Entity types extracted by GLiNER (getOverlap.py)
# Imported by getOverlap.py so the registry is the single source of truth.
# Adding a type here automatically adds a feature map — no other file needs
# changing except deleting ./cache/ and retraining.
# ---------------------------------------------------------------------------

ENTITY_TYPES: list[str] = [
    # Named entities
    "person", "organization", "location", "product", "event",
    # Legal / linguistic
    "law", "language",
    # Temporal
    "date", "time", "duration",
    # Quantitative — critical for hallucination / numeric grounding
    "number", "quantity", "percentage", "money",
]

# Feature key per entity type: "entity_person", "entity_organization", …
ENTITY_FEATURE_KEYS: list[str] = [f"entity_{t}" for t in ENTITY_TYPES]

# ---------------------------------------------------------------------------
# Canonical feature key list
# Insertion order matches the .update() sequence in train.py / predict.py:
#   LexicalWeights -> SemanticWeights -> NLIWeights -> EntityMatch -> LCSWeights
# ---------------------------------------------------------------------------

FEATURE_KEYS: list[str] = [
    # Lexical (5) — getLexicalWeights.py
    "jaccard",
    "dice",
    "cosine",
    "rouge",
    "rouge3",
    # Semantic cosine similarity (2) — getSemanticWeights.py
    "mixedbread-ai/mxbai-embed-large-v1",
    "Qwen/Qwen3-Embedding-0.6B",
    # Semantic soft alignment (4) — getSemanticWeights.py (__calc_soft_alignment__)
    "SOFT_ROW_mixedbread-ai/mxbai-embed-large-v1",
    "SOFT_COL_mixedbread-ai/mxbai-embed-large-v1",
    "SOFT_ROW_Qwen/Qwen3-Embedding-0.6B",
    "SOFT_COL_Qwen/Qwen3-Embedding-0.6B",
    # Semantic coverage — BERTScore-style max-pool precision / recall (4)
    # PREC: for each sentence in text2, its best-match similarity to text1
    # REC:  for each sentence in text1, its best-match similarity to text2
    "PREC_mixedbread-ai/mxbai-embed-large-v1",
    "REC_mixedbread-ai/mxbai-embed-large-v1",
    "PREC_Qwen/Qwen3-Embedding-0.6B",
    "REC_Qwen/Qwen3-Embedding-0.6B",
    # NLI (3) — getNLIweights.py
    "entailment",
    "neutral",
    "contradiction",
    # Entity per type (14) — getOverlap.py
    # Replaces the single aggregate EntityMismatch with one agreement map per type.
    # Numeric / temporal types (date, time, number, money, …) serve as the
    # factual-grounding feature previously handled by regex; GLiNER extracts them.
    *ENTITY_FEATURE_KEYS,
    # LCS (2) — getLCSweights.py
    "lcs_token",
    "lcs_char",
]

VERSION      = "3.0"
SPATIAL_SIZE = 32   # side length of resized feature maps (resize_matrix target_size)


def build_manifest() -> dict:
    """Return a manifest dict suitable for embedding in a checkpoint."""
    return {
        "version":      VERSION,
        "features":     list(FEATURE_KEYS),
        "num_features": len(FEATURE_KEYS),
        "spatial_size": SPATIAL_SIZE,
        "created_at":   datetime.now(timezone.utc).isoformat(),
    }


def validate_manifest(manifest: dict | None) -> None:
    """Raise RuntimeError if *manifest* is incompatible with the current pipeline.

    A missing manifest (None or empty) is treated as a warning-only case so that
    checkpoints saved before this feature was added still load.

    Args:
        manifest: The 'manifest' sub-dict from a loaded checkpoint, or None.

    Raises:
        RuntimeError: If the feature list in *manifest* differs from FEATURE_KEYS.
    """
    if not manifest:
        return  # pre-manifest checkpoint — nothing to validate

    ckpt_features = manifest.get("features")
    if ckpt_features is None:
        return  # manifest present but no feature list — old format, skip

    if ckpt_features == FEATURE_KEYS:
        return  # all good

    added    = [k for k in FEATURE_KEYS  if k not in ckpt_features]
    removed  = [k for k in ckpt_features if k not in FEATURE_KEYS]
    reordered = (not added and not removed) and ckpt_features != FEATURE_KEYS

    lines = [
        "Checkpoint feature manifest does not match the current pipeline.",
        f"  Checkpoint ({len(ckpt_features)} features): {ckpt_features}",
        f"  Current    ({len(FEATURE_KEYS)} features): {FEATURE_KEYS}",
    ]
    if added:
        lines.append(f"  Added since checkpoint was saved   : {added}")
    if removed:
        lines.append(f"  Removed since checkpoint was saved : {removed}")
    if reordered:
        lines.append("  Feature order has changed.")
    lines.append("Delete ./cache/ and retrain to generate a compatible checkpoint.")

    raise RuntimeError("\n".join(lines))
