import difflib
from collections import Counter

from gliner import GLiNER
from tqdm import tqdm

from backend.feature_registry import ENTITY_TYPES, ENTITY_FEATURE_KEYS, ENTITY_VALUE_TYPES
from backend.Postprocess.__addpad import resize_matrix

# ---------------------------------------------------------------------------
# Per-type fuzzy match thresholds (v5.2)
# Higher threshold = stricter matching = fewer false positives.
# Percentages and dates must fail on 1-unit differences ("25%" ≠ "26%").
# Product/brand names allow variant spacing/suffix drift.
# ---------------------------------------------------------------------------
_FUZZY_THRESHOLDS: dict[str, float] = {
    "location":   0.85,
    "product":    0.80,
    "date":       0.90,
    "time":       0.90,
    "duration":   0.88,
    "percentage": 0.95,
}


def _type_agreement(c1: int, c2: int) -> float:
    """Normalised count agreement for a single entity type, in [0, 1].

    Both zero  → 1.0  (both texts agree: no such entity present)
    Equal      → 1.0  (exact count match)
    One zero   → 0.0  (one text has entities of this type, the other does not)
    Partial    → min/max  (e.g. counts 2 vs 3 → 0.67)
    """
    if c1 == 0 and c2 == 0:
        return 1.0
    return min(c1, c2) / max(c1, c2)


def _fuzzy_in(needle: str, haystack: list[str], threshold: float = 0.82) -> bool:
    """Return True if *needle* fuzzy-matches any string in *haystack*.

    Uses SequenceMatcher ratio so "George Osborne" matches "Osborne",
    "US" matches "United States", etc.  Case-insensitive.
    """
    n = needle.lower()
    for s in haystack:
        if difflib.SequenceMatcher(None, n, s.lower()).ratio() >= threshold:
            return True
    return False


def _per_type_value_overlap(
    strs1: dict[str, list[str]],
    strs2: dict[str, list[str]],
) -> dict[str, tuple[float, float]]:
    """Compute per-type (precision, recall) for each type in ENTITY_VALUE_TYPES.

    Uses type-specific fuzzy thresholds from _FUZZY_THRESHOLDS so that
    percentages and dates require near-exact matches while product names
    allow surface form variation.

    Returns:
        Dict mapping entity_type → (prec, rec).
        Both empty → (1.0, 1.0); one side empty → (1.0, 0.0) or (0.0, 0.0).
    """
    result: dict[str, tuple[float, float]] = {}
    for etype in ENTITY_VALUE_TYPES:
        threshold = _FUZZY_THRESHOLDS.get(etype, 0.82)
        s1 = strs1.get(etype, [])
        s2 = strs2.get(etype, [])
        if not s1 and not s2:
            result[etype] = (1.0, 1.0)
        else:
            prec = (
                sum(1 for s in s2 if _fuzzy_in(s, s1, threshold)) / len(s2)
                if s2 else 1.0
            )
            rec = (
                sum(1 for s in s1 if _fuzzy_in(s, s2, threshold)) / len(s1)
                if s1 else 0.0
            )
            result[etype] = (prec, rec)
    return result


def _value_overlap(strings1: list[str], strings2: list[str]) -> tuple[float, float]:
    """Compute (precision, recall) of entity value overlap.

    Precision = fraction of text2 entity strings grounded in text1.
    Recall    = fraction of text1 entity strings covered by text2.
    Both zero entities → (1.0, 1.0) — agree on absence.
    text2 adds entities text1 lacks → precision = 0.0 (potential hallucination).
    """
    if not strings1 and not strings2:
        return 1.0, 1.0
    prec = (sum(1 for s in strings2 if _fuzzy_in(s, strings1)) / len(strings2)
            if strings2 else 1.0)
    rec  = (sum(1 for s in strings1 if _fuzzy_in(s, strings2)) / len(strings1)
            if strings1 else 0.0)
    return prec, rec


class EntityMatch:
    # Shared across all instances — GLiNER loaded once per process
    _model_cache: dict = {}
    # Sentence-level NER cache: sentence → {entity_type: count}
    # Keys follow ENTITY_TYPES so the cache stays valid as long as ENTITY_TYPES is unchanged.
    _entity_cache: dict = {}
    # Sentence-level string cache: sentence → {entity_type: [string1, string2, ...]}
    # Populated alongside _entity_cache. Used for entity_value_prec/rec features.
    _entity_strings_cache: dict = {}

    def __init__(
        self,
        MODEL: str = "knowledgator/modern-gliner-bi-base-v1.0",
        THRESHOLD: float = 0.3,
    ):
        self.MODEL = MODEL
        self.THRESHOLD = THRESHOLD
        self._reset_state()

    def _reset_state(self):
        """Reset per-call buffers. Preserves the loaded GLiNER model cache."""
        self.phrase1_list = None
        self.phrase1_entity_cnts = None
        self.phrase1_entity_strs = None
        self.phrase2_list = None
        self.phrase2_entity_cnts = None
        self.phrase2_entity_strs = None
        self.comparison_weights = {k: [] for k in ENTITY_FEATURE_KEYS}
        self.comparison_weights["entity_value_prec"] = []
        self.comparison_weights["entity_value_rec"]  = []
        for _t in ENTITY_VALUE_TYPES:
            self.comparison_weights[f"entity_{_t}_value_prec"] = []
            self.comparison_weights[f"entity_{_t}_value_rec"]  = []

    def __load_model__(self):
        EntityMatch._model_cache[self.MODEL] = GLiNER.from_pretrained(
            self.MODEL,
            cache_dir=f"Features/EntityGroups/model/{self.MODEL}/",
        )

    # Maximum character length fed to GLiNER per sentence.
    # Long documents (e.g. HaluEval-Sum source texts) are truncated before NER to
    # avoid OOM when batching many texts. Key entities appear early in a document;
    # 512 chars covers ~100 tokens which is ample for entity extraction.
    _MAX_NER_CHARS: int = 512

    def _batch_get_entities(self, texts: list[str]) -> list[dict]:
        """Return entity-count dicts for each text, using the sentence-level cache.

        Only sentences absent from _entity_cache are sent to GLiNER; results are
        stored back so repeated sentences across training pairs are never re-processed.
        Entity types are driven by ENTITY_TYPES from feature_registry.
        Long texts are truncated to _MAX_NER_CHARS before inference.
        """
        new_texts = [t for t in texts if t not in EntityMatch._entity_cache]

        if new_texts:
            if self.MODEL not in EntityMatch._model_cache:
                self.__load_model__()

            model = EntityMatch._model_cache[self.MODEL]
            truncated = [t[: self._MAX_NER_CHARS] for t in new_texts]
            if hasattr(model, "run"):
                raw = model.run(truncated, ENTITY_TYPES, threshold=self.THRESHOLD)
            elif hasattr(model, "batch_predict_entities"):
                raw = model.batch_predict_entities(
                    truncated, ENTITY_TYPES, threshold=self.THRESHOLD
                )
            else:
                raw = [
                    model.predict_entities(t, ENTITY_TYPES, threshold=self.THRESHOLD)
                    for t in truncated
                ]

            for text, entities_val in zip(new_texts, raw):
                counts = Counter([ent["label"] for ent in entities_val])
                EntityMatch._entity_cache[text] = {
                    t: counts.get(t, 0) for t in ENTITY_TYPES
                }
                # Store entity strings (text values) grouped by type for value-overlap.
                strings: dict[str, list[str]] = {t: [] for t in ENTITY_TYPES}
                for ent in entities_val:
                    if ent["label"] in strings:
                        strings[ent["label"]].append(ent["text"])
                EntityMatch._entity_strings_cache[text] = strings

        return [EntityMatch._entity_cache[t] for t in texts]

    def __compute_phrase_entity__(self):
        all_texts = self.phrase1_list + self.phrase2_list
        all_counts = self._batch_get_entities(all_texts)
        n1 = len(self.phrase1_list)
        self.phrase1_entity_cnts = all_counts[:n1]
        self.phrase2_entity_cnts = all_counts[n1:]
        # _batch_get_entities also populates _entity_strings_cache
        self.phrase1_entity_strs = [EntityMatch._entity_strings_cache.get(t, {}) for t in self.phrase1_list]
        self.phrase2_entity_strs = [EntityMatch._entity_strings_cache.get(t, {}) for t in self.phrase2_list]

        # One n×m agreement matrix per entity type + value-overlap matrices
        per_type: dict[str, list[list[float]]] = {
            k: [] for k in ENTITY_FEATURE_KEYS
        }
        prec_matrix: list[list[float]] = []
        rec_matrix:  list[list[float]] = []
        # Per-type value matrices (v5.2): one prec + one rec matrix per ENTITY_VALUE_TYPES entry
        per_type_val_prec: dict[str, list[list[float]]] = {t: [] for t in ENTITY_VALUE_TYPES}
        per_type_val_rec:  dict[str, list[list[float]]] = {t: [] for t in ENTITY_VALUE_TYPES}

        for dict1, strs1 in tqdm(
            zip(self.phrase1_entity_cnts, self.phrase1_entity_strs),
            total=len(self.phrase1_entity_cnts),
            desc="Entity maps",
        ):
            row_type:     dict[str, list[float]] = {k: [] for k in ENTITY_FEATURE_KEYS}
            row_prec:     list[float] = []
            row_rec:      list[float] = []
            row_val_prec: dict[str, list[float]] = {t: [] for t in ENTITY_VALUE_TYPES}
            row_val_rec:  dict[str, list[float]] = {t: [] for t in ENTITY_VALUE_TYPES}

            for dict2, strs2 in zip(self.phrase2_entity_cnts, self.phrase2_entity_strs):
                # Type-count agreement (existing features)
                for entity_type, key in zip(ENTITY_TYPES, ENTITY_FEATURE_KEYS):
                    row_type[key].append(
                        _type_agreement(dict1.get(entity_type, 0), dict2.get(entity_type, 0))
                    )
                # Value-overlap: flatten all entity strings across types
                all_s1 = [s for vals in strs1.values() for s in vals]
                all_s2 = [s for vals in strs2.values() for s in vals]
                prec, rec = _value_overlap(all_s1, all_s2)
                row_prec.append(prec)
                row_rec.append(rec)
                # Per-type value overlap (v5.2)
                type_overlaps = _per_type_value_overlap(strs1, strs2)
                for t in ENTITY_VALUE_TYPES:
                    p, r = type_overlaps[t]
                    row_val_prec[t].append(p)
                    row_val_rec[t].append(r)

            for key in ENTITY_FEATURE_KEYS:
                per_type[key].append(row_type[key])
            prec_matrix.append(row_prec)
            rec_matrix.append(row_rec)
            for t in ENTITY_VALUE_TYPES:
                per_type_val_prec[t].append(row_val_prec[t])
                per_type_val_rec[t].append(row_val_rec[t])

        per_type["entity_value_prec"] = prec_matrix
        per_type["entity_value_rec"]  = rec_matrix
        for t in ENTITY_VALUE_TYPES:
            per_type[f"entity_{t}_value_prec"] = per_type_val_prec[t]
            per_type[f"entity_{t}_value_rec"]  = per_type_val_rec[t]
        self.comparison_weights = per_type

    def __post_process_weights__(self):
        for key in self.comparison_weights:
            self.comparison_weights[key] = resize_matrix(self.comparison_weights[key])

    def getFeatureMap(self, phrase1_list: list, phrase2_list: list) -> dict:
        self._reset_state()
        self.phrase1_list, self.phrase2_list = phrase1_list, phrase2_list
        self.__compute_phrase_entity__()
        self.__post_process_weights__()
        return self.comparison_weights


if __name__ == "__main__":
    sample = [
        "Apple released the iPhone 15 in September 2023 for $999.",
        "Tim Cook announced record profits of 25% growth last quarter.",
    ]
    obj = EntityMatch()
    maps = obj.getFeatureMap(sample, sample)
    for k, v in maps.items():
        print(f"{k}: min={v.min():.2f}  max={v.max():.2f}")
