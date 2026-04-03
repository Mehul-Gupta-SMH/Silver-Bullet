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
    # Lexical (2 of 5) — getLexicalWeights.py
    # Dropped in v4.1 (ablation n=6293, correlated redundancy cross-r ≥ 0.93 with dice):
    #   jaccard (cross-r=0.983), cosine (cross-r=0.938), rouge (cross-r=0.934)
    # rouge3 retained: max_cross_r=0.836 with jaccard (below 0.85 threshold — independent signal)
    "dice",
    "rouge3",
    # Semantic coverage — BERTScore-style PREC (2 of 6) — getSemanticWeights.py
    # Dropped in v4.1 (ablation n=6293, within-cluster cross-r ≥ 0.966):
    #   mxbai cosine (cross-r=0.967 with REC_mxbai), REC_mxbai (cross-r=0.967)
    #   Qwen cosine  (cross-r=0.966 with REC_Qwen),  REC_Qwen  (cross-r=0.966)
    # PREC retained in each group: highest per-group Pearson r in v4.0b ablation.
    # SOFT_ROW/SOFT_COL dropped in v4.0a (ablation p=0.47-0.80 — confirmed noise).
    "PREC_mixedbread-ai/mxbai-embed-large-v1",
    "PREC_Qwen/Qwen3-Embedding-0.6B",
    # NLI (3) — getNLIweights.py
    "entailment",
    "neutral",
    "contradiction",
    # Entity per type (6 of 14) — getOverlap.py
    # Dropped in v4.0b (ablation n=6293, p ≥ 0.09, no Bonferroni significance):
    #   entity_person (p=0.494), entity_organization (p=0.763),
    #   entity_event (p=0.521), entity_language (p=0.093),
    #   entity_date (p=0.278), entity_number (p=0.170),
    #   entity_quantity (p=0.151), entity_money (p=0.360)
    "entity_location",
    "entity_product",
    "entity_law",
    "entity_time",
    "entity_duration",
    "entity_percentage",
    # LCS (2) — getLCSweights.py
    "lcs_token",
    "lcs_char",
]

VERSION      = "4.1"
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
