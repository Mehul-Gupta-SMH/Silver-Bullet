"""Entity grounding recall feature extractor for SilverBullet.

Uses the existing GLiNER entity model (knowledgator/modern-gliner-bi-base-v1.0)
to extract (entity_text, entity_type) pairs from each sentence, then computes
recall of text1's entities in text2 — i.e. what fraction of entities claimed by
text1's sentence i are grounded in text2's sentence j.

This is a "relation-grounding" proxy: when both sentences discuss the same fact,
their named entities will overlap.  When text2 hallucinates a different entity
or omits one present in text1, the score falls below 1.0.

Feature produced:
    "entity_grounding_recall"
        n×m matrix (resized to [32,32]) where cell (i,j) =
        |entities_in_sent1_i ∩ entities_in_sent2_j| / |entities_in_sent1_i|

        Both empty → 1.0  (both sentences contain no named entities — they agree
                            on the absence of specific entities)
        sent1 has entities but sent2 has none → 0.0  (sent2 fails to ground any
                            entity asserted by sent1 — strong hallucination signal)
        sent1 empty, sent2 non-empty → 1.0  (sent1 makes no assertions;
                            the recall convention does not penalise text2 additions)

Reuses EntityMatch._model_cache and EntityMatch._entity_strings_cache so the
GLiNER model is loaded at most once per process regardless of import order.

Interface:
    RelationGrounding().getFeatureMap(phrase_list1, phrase_list2)
        → {"entity_grounding_recall": Tensor[32, 32]}
"""

from __future__ import annotations

import difflib
from typing import List

from backend.Postprocess.__addpad import resize_matrix
from backend.feature_registry import ENTITY_TYPES


# ---------------------------------------------------------------------------
# Fuzzy match helper (same as getOverlap._fuzzy_in but local to avoid circular
# imports — both live in the same Features layer).
# ---------------------------------------------------------------------------

def _fuzzy_match(needle: str, haystack: list[str], threshold: float = 0.82) -> bool:
    """Return True if *needle* fuzzy-matches any string in *haystack*.

    Case-insensitive SequenceMatcher ratio.  Threshold 0.82 is the project-wide
    default from getOverlap.py — allows name variant drift ("George Osborne" ≈
    "Osborne") while rejecting semantically distinct strings.
    """
    n = needle.lower()
    for s in haystack:
        if difflib.SequenceMatcher(None, n, s.lower()).ratio() >= threshold:
            return True
    return False


def _entity_grounding_recall(
    entities1: list[str],
    entities2: list[str],
) -> float:
    """Recall of sent1 entity strings in sent2.

    Returns fraction of entities1 that fuzzy-match at least one entity in
    entities2.  Edge cases follow the docstring convention above.
    """
    if not entities1:
        return 1.0   # sent1 makes no entity claims — trivially grounded
    if not entities2:
        return 0.0   # sent1 has entities but sent2 has none — not grounded

    matched = sum(1 for e in entities1 if _fuzzy_match(e, entities2))
    return matched / len(entities1)


class RelationGrounding:
    """Compute entity-grounding-recall feature maps for sentence-pair inputs.

    Shares the GLiNER model cache with EntityMatch — the model is loaded at most
    once per process even if both extractors are active in the same training run.

    Follows the standard SilverBullet feature extractor interface:
        getFeatureMap(phrase_list1, phrase_list2) → dict[str, Tensor]
    """

    # -----------------------------------------------------------------------
    # Class-level caches shared with EntityMatch (same keys, different class
    # attribute — Python looks up on the class, so we store locally and fall
    # back to EntityMatch's cache if already populated).
    # -----------------------------------------------------------------------
    _model_cache: dict = {}
    _entity_strings_cache: dict = {}

    # GLiNER model name — must match EntityMatch.MODEL default so the two
    # extractors share the same loaded weights object.
    MODEL: str = "knowledgator/modern-gliner-bi-base-v1.0"
    THRESHOLD: float = 0.3
    _MAX_NER_CHARS: int = 512   # same cap as EntityMatch

    def __init__(self):
        self._reset_state()

    def _reset_state(self):
        self.phrase_list1: list[str] | None = None
        self.phrase_list2: list[str] | None = None
        self._weight_matrix: list[list[float]] = []

    # ------------------------------------------------------------------
    # Model loading — lazy, class-level singleton
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load the GLiNER model into _model_cache if not already present.

        Prefers the EntityMatch cache so the model object is shared.
        """
        if self.MODEL in RelationGrounding._model_cache:
            return

        # Also check EntityMatch cache to avoid double-loading
        try:
            from backend.Features.EntityGroups.getOverlap import EntityMatch
            if self.MODEL in EntityMatch._model_cache:
                RelationGrounding._model_cache[self.MODEL] = (
                    EntityMatch._model_cache[self.MODEL]
                )
                RelationGrounding._entity_strings_cache = (
                    EntityMatch._entity_strings_cache
                )
                return
        except ImportError:
            pass

        from gliner import GLiNER
        RelationGrounding._model_cache[self.MODEL] = GLiNER.from_pretrained(
            self.MODEL,
            cache_dir=f"Features/EntityGroups/model/{self.MODEL}/",
        )

    # ------------------------------------------------------------------
    # Entity extraction with sentence-level caching
    # ------------------------------------------------------------------

    def _get_entity_strings(self, texts: list[str]) -> list[list[str]]:
        """Return flat list of entity text strings for each sentence in *texts*.

        Results are stored in _entity_strings_cache (keyed by sentence text) so
        sentences that were already processed by EntityMatch are free lookups.

        Args:
            texts: List of sentence strings.

        Returns:
            List of entity-string lists, one per input sentence.
        """
        # Sync with EntityMatch cache if available
        try:
            from backend.Features.EntityGroups.getOverlap import EntityMatch
            RelationGrounding._entity_strings_cache = EntityMatch._entity_strings_cache
        except ImportError:
            pass

        new_texts = [
            t for t in texts
            if t not in RelationGrounding._entity_strings_cache
        ]

        if new_texts:
            self._ensure_model()
            model = RelationGrounding._model_cache[self.MODEL]
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

            for text, entities in zip(new_texts, raw):
                strings: dict[str, list[str]] = {t: [] for t in ENTITY_TYPES}
                for ent in entities:
                    if ent["label"] in strings:
                        strings[ent["label"]].append(ent["text"])
                RelationGrounding._entity_strings_cache[text] = strings

            # Write-back to EntityMatch cache to keep them in sync
            try:
                from backend.Features.EntityGroups.getOverlap import EntityMatch
                EntityMatch._entity_strings_cache.update(
                    RelationGrounding._entity_strings_cache
                )
            except ImportError:
                pass

        return [
            [s for vals in RelationGrounding._entity_strings_cache[t].values() for s in vals]
            for t in texts
        ]

    # ------------------------------------------------------------------
    # Matrix computation
    # ------------------------------------------------------------------

    def _compute_matrix(self) -> None:
        """Fill _weight_matrix with n×m recall scores."""
        all_texts = self.phrase_list1 + self.phrase_list2
        all_entity_strings = self._get_entity_strings(all_texts)

        n1 = len(self.phrase_list1)
        entities1 = all_entity_strings[:n1]
        entities2 = all_entity_strings[n1:]

        self._weight_matrix = [
            [_entity_grounding_recall(e1, e2) for e2 in entities2]
            for e1 in entities1
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def getFeatureMap(
        self,
        phrase_list1: List[str],
        phrase_list2: List[str],
    ) -> dict:
        """Compute entity-grounding-recall feature map.

        Args:
            phrase_list1: Sentences from text1 (n sentences).
            phrase_list2: Sentences from text2 (m sentences).

        Returns:
            {"entity_grounding_recall": Tensor[32, 32]}
        """
        self._reset_state()
        self.phrase_list1 = phrase_list1
        self.phrase_list2 = phrase_list2
        self._compute_matrix()
        return {
            "entity_grounding_recall": resize_matrix(self._weight_matrix)
        }


if __name__ == "__main__":
    t1 = [
        "Apple was founded by Steve Jobs in Cupertino, California.",
        "The company released the iPhone in 2007.",
    ]
    t2 = [
        "Apple was co-founded by Steve Jobs and Steve Wozniak.",
        "The iPhone was announced by Tim Cook in 2023.",
    ]

    obj = RelationGrounding()
    maps = obj.getFeatureMap(t1, t2)
    for k, v in maps.items():
        print(f"{k}: shape={v.shape}  min={v.min():.3f}  max={v.max():.3f}  mean={v.mean():.3f}")
