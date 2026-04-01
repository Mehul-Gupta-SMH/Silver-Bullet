from collections import Counter

from gliner import GLiNER
from tqdm import tqdm

from backend.feature_registry import ENTITY_TYPES, ENTITY_FEATURE_KEYS
from backend.Postprocess.__addpad import resize_matrix


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


class EntityMatch:
    # Shared across all instances — GLiNER loaded once per process
    _model_cache: dict = {}
    # Sentence-level NER cache: sentence → {entity_type: count}
    # Keys follow ENTITY_TYPES so the cache stays valid as long as ENTITY_TYPES is unchanged.
    _entity_cache: dict = {}

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
        self.phrase2_list = None
        self.phrase2_entity_cnts = None
        self.comparison_weights = {k: [] for k in ENTITY_FEATURE_KEYS}

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

        return [EntityMatch._entity_cache[t] for t in texts]

    def __compute_phrase_entity__(self):
        all_texts = self.phrase1_list + self.phrase2_list
        all_counts = self._batch_get_entities(all_texts)
        n1 = len(self.phrase1_list)
        self.phrase1_entity_cnts = all_counts[:n1]
        self.phrase2_entity_cnts = all_counts[n1:]

        # One n×m agreement matrix per entity type
        per_type: dict[str, list[list[float]]] = {k: [] for k in ENTITY_FEATURE_KEYS}
        for dict1 in tqdm(self.phrase1_entity_cnts, desc="Entity maps"):
            row_accum: dict[str, list[float]] = {k: [] for k in ENTITY_FEATURE_KEYS}
            for dict2 in self.phrase2_entity_cnts:
                for entity_type, key in zip(ENTITY_TYPES, ENTITY_FEATURE_KEYS):
                    row_accum[key].append(
                        _type_agreement(dict1.get(entity_type, 0), dict2.get(entity_type, 0))
                    )
            for key in ENTITY_FEATURE_KEYS:
                per_type[key].append(row_accum[key])

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
