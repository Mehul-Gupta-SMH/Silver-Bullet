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
# Entity types selected for per-type value-overlap features (v5.2)
# Criteria: GLiNER zero-shot reliability + distinct matchable string values +
#           not redundant with numeric_jaccard (excludes number/quantity/money).
# Fuzzy thresholds (used in getOverlap._fuzzy_in): percentage needs strictest
#   matching (0.95) because "25%" ≠ "26%"; dates/times strict (0.90) because
#   year/hour errors are meaningful; product/location/duration more lenient.
# ---------------------------------------------------------------------------
ENTITY_VALUE_TYPES: list[str] = [
    "location",    # threshold=0.85 — "Paris" ≠ "Berlin", typos OK
    "product",     # threshold=0.80 — brand variants ("Tesla Model 3" vs "Tesla")
    "date",        # threshold=0.90 — "2023" ≠ "2024" must fail
    "time",        # threshold=0.90 — "3 PM" ≠ "9 AM" must fail
    "duration",    # threshold=0.88 — "3 hours" ≠ "3 days" must fail
    "percentage",  # threshold=0.95 — "25%" ≠ "26%" must fail (strictest)
]

# Per-type value keys: "entity_location_value_prec", "entity_location_value_rec", …
ENTITY_VALUE_KEYS: list[str] = [
    key
    for t in ENTITY_VALUE_TYPES
    for key in (f"entity_{t}_value_prec", f"entity_{t}_value_rec")
]

# ---------------------------------------------------------------------------
# Canonical feature key list
# Insertion order matches the .update() sequence in train.py / predict.py:
#   LexicalWeights -> SemanticWeights -> NLIWeights -> EntityMatch -> LCSWeights
# ---------------------------------------------------------------------------

FEATURE_KEYS: list[str] = [
    # Lexical (4 of 5) — getLexicalWeights.py
    # Dropped in v4.1 (ablation n=6293, correlated redundancy cross-r ≥ 0.93 with dice):
    #   cosine (cross-r=0.938)
    # rouge (unigram) restored in v4.2: CVG label-r=-0.183 (3rd-ranked), much stronger
    #   than dice (-0.114) and rouge3 (-0.082) for hallucination. Cross-r does not imply
    #   equal discriminative power per mode.
    # jaccard restored in v4.3: CVG label-r=-0.124, RVG=+0.213 — meaningful across modes;
    #   cross-r=0.983 with dice was misleading (same direction bias, different magnitude).
    # rouge3 retained: max_cross_r=0.836 with jaccard (below 0.85 — independent signal)
    "dice",
    "rouge3",
    "rouge",
    "jaccard",
    # Semantic (4 of 6) — getSemanticWeights.py
    # mxbai cosine restored in v4.4: CVG label-r=+0.098, RVG=+0.352, MVM=+0.439.
    #   Full n×m pairwise structure; CNN learns spatial patterns PREC/REC can't capture.
    # Qwen cosine NOT restored: lower label-r than mxbai across all modes.
    # REC_Qwen NOT restored: cross-r with PREC_Qwen = +0.999 on RVG — fully redundant.
    # SOFT_ROW/SOFT_COL dropped in v4.0a (ablation p=0.47-0.80 — confirmed noise).
    "mixedbread-ai/mxbai-embed-large-v1",
    "PREC_mixedbread-ai/mxbai-embed-large-v1",
    "REC_mixedbread-ai/mxbai-embed-large-v1",
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
    # entity_time: ablation p=0.829 (NOISE tier) but removing it cost CVG -3pt accuracy.
    #   PC04 shows entity_time loads +0.44 alongside entity_duration — model uses them as
    #   a pair. Marginal label-r is zero but interaction effect is real. RESTORED in v4.5.
    "entity_location",
    "entity_product",
    "entity_law",
    "entity_time",
    "entity_duration",
    "entity_percentage",
    # LCS (2) — getLCSweights.py
    "lcs_token",
    "lcs_char",
    # Entity value overlap (2) — getOverlap.py (v5.1)
    # entity_value_prec: fraction of text2 entity strings found in text1 (grounding)
    # entity_value_rec:  fraction of text1 entity strings covered by text2 (coverage)
    "entity_value_prec",
    "entity_value_rec",
    # Numeric grounding (1) — getNumericGrounding.py (v5.1)
    # Jaccard over normalised number sets: catches wrong/hallucinated numbers
    "numeric_jaccard",
    # Per-type entity value overlap (12) — getOverlap.py (v5.2)
    # Extends entity_value_prec/rec to operate per entity type with calibrated
    # fuzzy thresholds; captures type-specific hallucinations (location swap,
    # date distortion, wrong percentage) that the flat combined features dilute.
    # Skipped types: number/quantity/money (numeric_jaccard is superior),
    #   person (brittle fuzzy on name variants), law (ultra-sparse),
    #   event/language (inconsistent GLiNER extraction).
    "entity_location_value_prec",
    "entity_location_value_rec",
    "entity_product_value_prec",
    "entity_product_value_rec",
    "entity_date_value_prec",
    "entity_date_value_rec",
    "entity_time_value_prec",
    "entity_time_value_rec",
    "entity_duration_value_prec",
    "entity_duration_value_rec",
    "entity_percentage_value_prec",
    "entity_percentage_value_rec",
    # Entity grounding recall (1) — getRelationWeights.py (v5.4)
    # Recall of text1's named entities in text2 sentences.  Catches factual
    # hallucinations where entities and their relationships differ: if text1
    # asserts "Steve Jobs founded Apple" and text2 says "Tim Cook founded Apple",
    # the entity recall for the Jobs sentence drops toward 0.  Complements
    # entity_value_rec (which measures the inverse direction: text2→text1).
    "entity_grounding_recall",
    # SVO triplet recall (1) — getSVOWeights.py (v5.6)
    # spaCy dep-tree SVO extraction; ~100x faster than Relex; ablation target.
    "svo_triplet_recall",
    # EFG: External Factual Grounding (3) — getFactualGrounding.py (v5.8)
    # DeBERTa-v3-base-mnli-fever-anli cross-sentence factual support scores.
    # efg_supports: P(s2 supported by s1), forward direction
    # efg_refutes: P(s2 refuted by s1), forward direction
    # efg_factual_delta: efg_supports(fwd) − efg_supports(bwd); captures which
    #   text is more factually authoritative than the other
    "efg_supports",
    "efg_refutes",
    "efg_factual_delta",
]

VERSION      = "5.8"
SPATIAL_SIZE = 32   # side length of resized feature maps (resize_matrix target_size)

# ---------------------------------------------------------------------------
# Mode-specific feature baskets (v5.0)
# Each mode trains on only the features that carry statistically significant
# signal for that task (per-mode Pearson r analysis, v4.4 ablation n≈2k/mode).
#
# CVG (hallucination):  entity features all p≥0.066 — dropped entirely.
# RVG (faithfulness):   entity_product (p=0.019) + entity_percentage (p=0.033) kept.
# MVM (agreement):      entity_percentage (p<0.001) kept; entity_product borderline.
# ---------------------------------------------------------------------------

FEATURE_KEYS_BY_MODE: dict[str, list[str]] = {
    "context-vs-generated": [
        # Lexical (4)
        "dice", "rouge3", "rouge", "jaccard",
        # Semantic (4)
        "mixedbread-ai/mxbai-embed-large-v1",
        "PREC_mixedbread-ai/mxbai-embed-large-v1",
        "REC_mixedbread-ai/mxbai-embed-large-v1",
        "PREC_Qwen/Qwen3-Embedding-0.6B",
        # NLI (3) — dominant signal for CVG
        "entailment", "neutral", "contradiction",
        # Type-count entity dropped: all p≥0.066 (see v5.0 notes)
        # Value-level entity (2) — v5.1: string identity not type count
        "entity_value_prec",   # text2 entity strings grounded in text1
        "entity_value_rec",    # text1 entity strings covered by text2
        # LCS (2)
        "lcs_token", "lcs_char",
        # Numeric grounding (1) — v5.1
        "numeric_jaccard",
        # Per-type entity value (0) — v5.3: entity_product_value_prec dropped in v5.8
        # SHAP rank 19/19 (mean|SHAP|=2e-5, 9× weaker than median); Pearson r was
        # significant in ablation but the trained model does not use it — dropped.
        # Entity grounding recall (1) — v5.4
        # Recall-direction entity overlap: fraction of text1 entities grounded in text2.
        # Relevant for all modes — catches entity substitution/omission hallucinations.
        "entity_grounding_recall",
        # SVO triplet recall (1) — v5.6
        "svo_triplet_recall",
        # EFG (3) — v5.8: FEVER-tuned factual support scores + directional delta
        "efg_supports",
        "efg_refutes",
        "efg_factual_delta",
    ],
    "reference-vs-generated": [
        # Lexical (4)
        "dice", "rouge3", "rouge", "jaccard",
        # Semantic (4) — strongest group for RVG
        "mixedbread-ai/mxbai-embed-large-v1",
        "PREC_mixedbread-ai/mxbai-embed-large-v1",
        "REC_mixedbread-ai/mxbai-embed-large-v1",
        "PREC_Qwen/Qwen3-Embedding-0.6B",
        # NLI (3)
        "entailment", "neutral", "contradiction",
        # Entity type-count (1) — entity_product borderline kept; entity_percentage dropped
        # entity_percentage: v5.8 SHAP rank 20/20 (mean|SHAP|=7e-7, ~288× weaker than #1);
        #   mean_feat_val=0.99 (nearly constant at 1.0 — both-empty→1.0 on sparse text).
        "entity_product",     # p=0.019, SHAP rank 19/20 — borderline, kept
        # Entity value (2) — v5.1
        "entity_value_prec",
        "entity_value_rec",
        # LCS (2)
        "lcs_token", "lcs_char",
        # Numeric grounding (1) — v5.1
        "numeric_jaccard",
        # Per-type entity value (0) — v5.3: no per-type features survive Bonferroni for RVG
        # Ablation n=1831: entity_time (p=0.750/0.973 NOISE), location (p=0.136/0.283),
        # date/duration (p>0.14), product_prec (p=0.384), percentage_prec (p=0.129).
        # entity_product_value_rec (p=0.008) and entity_percentage_value_rec (p=0.013)
        # are nominally * but do not survive Bonferroni α=1.67e-03.
        # Both-empty→1.0 on sparse reference texts adds collective noise.
        # Entity grounding recall (1) — v5.4
        "entity_grounding_recall",
        # SVO triplet recall (1) — v5.6
        "svo_triplet_recall",
        # EFG (3) — v5.8
        "efg_supports",
        "efg_refutes",
        "efg_factual_delta",
    ],
    "model-vs-model": [
        # Lexical (4)
        "dice", "rouge3", "rouge", "jaccard",
        # Semantic (4)
        "mixedbread-ai/mxbai-embed-large-v1",
        "PREC_mixedbread-ai/mxbai-embed-large-v1",
        "REC_mixedbread-ai/mxbai-embed-large-v1",
        "PREC_Qwen/Qwen3-Embedding-0.6B",
        # NLI (3)
        "entailment", "neutral", "contradiction",
        # Entity type-count (0) — entity_percentage dropped in v5.8
        # SHAP rank 21/21 (mean|SHAP|=6.7e-7, ~251× weaker than #1);
        #   mean_feat_val=0.989 (essentially constant — both-empty→1.0 fills most pairs).
        # Entity value (2) — v5.1
        "entity_value_prec",
        "entity_value_rec",
        # LCS (2)
        "lcs_token", "lcs_char",
        # Numeric grounding (1) — v5.1
        "numeric_jaccard",
        # Per-type entity value (2) — v5.3: only Bonferroni-significant per-type features
        # entity_percentage_value_prec: p=4.25e-07 (***), r=+0.107 in MVM ablation n=2231
        # entity_percentage_value_rec:  p=1.23e-08 (***), r=+0.120 in MVM ablation n=2231
        # entity_time: DROP (p=0.927/0.957 NOISE)
        # entity_location: MARGINAL (p=0.013/0.035) — NS after Bonferroni α=2.17e-03
        "entity_percentage_value_prec",
        "entity_percentage_value_rec",
        # Entity grounding recall (1) — v5.4
        "entity_grounding_recall",
        # SVO triplet recall (1) — v5.6
        "svo_triplet_recall",
        # EFG (3) — v5.8
        "efg_supports",
        "efg_refutes",
        "efg_factual_delta",
    ],
}


def get_feature_keys(mode: str | None = None) -> list[str]:
    """Return the feature key list for *mode*, or the global FEATURE_KEYS fallback.

    Args:
        mode: One of the three evaluation mode strings, or None for general/legacy.

    Returns:
        The mode-specific feature key list, or FEATURE_KEYS if mode is unknown/None.
    """
    if mode is None:
        return FEATURE_KEYS
    return FEATURE_KEYS_BY_MODE.get(mode, FEATURE_KEYS)


def build_manifest(feature_keys: list[str] | None = None) -> dict:
    """Return a manifest dict suitable for embedding in a checkpoint.

    Args:
        feature_keys: The feature key list used for this training run.
                      Defaults to the global FEATURE_KEYS.
    """
    keys = feature_keys if feature_keys is not None else FEATURE_KEYS
    return {
        "version":      VERSION,
        "features":     list(keys),
        "num_features": len(keys),
        "spatial_size": SPATIAL_SIZE,
        "created_at":   datetime.now(timezone.utc).isoformat(),
    }


def validate_manifest(manifest: dict | None, expected_keys: list[str] | None = None) -> None:
    """Raise RuntimeError if *manifest* is incompatible with the current pipeline.

    A missing manifest (None or empty) is treated as a warning-only case so that
    checkpoints saved before this feature was added still load.

    Args:
        manifest: The 'manifest' sub-dict from a loaded checkpoint, or None.
        expected_keys: The feature list to validate against. Defaults to the global
                       FEATURE_KEYS. Pass the mode-specific list for per-mode checkpoints.

    Raises:
        RuntimeError: If the feature list in *manifest* differs from *expected_keys*.
    """
    if not manifest:
        return  # pre-manifest checkpoint — nothing to validate

    ckpt_features = manifest.get("features")
    if ckpt_features is None:
        return  # manifest present but no feature list — old format, skip

    reference = expected_keys if expected_keys is not None else FEATURE_KEYS

    if ckpt_features == reference:
        return  # all good

    added    = [k for k in reference     if k not in ckpt_features]
    removed  = [k for k in ckpt_features if k not in reference]
    reordered = (not added and not removed) and ckpt_features != reference

    lines = [
        "Checkpoint feature manifest does not match the current pipeline.",
        f"  Checkpoint ({len(ckpt_features)} features): {ckpt_features}",
        f"  Expected   ({len(reference)} features): {reference}",
    ]
    if added:
        lines.append(f"  Added since checkpoint was saved   : {added}")
    if removed:
        lines.append(f"  Removed since checkpoint was saved : {removed}")
    if reordered:
        lines.append("  Feature order has changed.")
    lines.append("Delete ./cache/ and retrain to generate a compatible checkpoint.")

    raise RuntimeError("\n".join(lines))
