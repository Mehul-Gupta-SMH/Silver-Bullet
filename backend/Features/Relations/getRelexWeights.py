"""Relation triplet recall feature extractor for SilverBullet (v5.5).

Uses `knowledgator/gliner-relex-large-v1.0` — a zero-shot joint NER + relation
extraction model — to extract (head, relation_type, tail) triplets from each
sentence, then computes how many of text1's triplets are grounded in text2.

Why this is stronger than entity_grounding_recall
--------------------------------------------------
entity_grounding_recall checks whether entity *strings* from text1 appear in
text2.  It fires on "Apple" regardless of whether text2 says
    "Apple was founded by Steve Jobs"  (correct)
    "Apple was acquired by Steve Jobs" (wrong relation — hallucination)

Relation triplet recall checks (head, relation_type, tail) jointly, so the
above two cases produce different scores.

Feature produced
----------------
"relation_triplet_recall"
    n×m matrix (resized to [32, 32]) where cell (i, j) =
    |triplets_in_sent1_i that have a match in triplets_in_sent2_j|
    / |triplets_in_sent1_i|

    Triplet matching:
        • relation_type must be an exact string match
        • tail must fuzzy-match (SequenceMatcher ratio ≥ 0.82)
        • head is ignored — we're comparing within the same cross-sentence pair;
          the head is usually the subject of the same factual claim.

    Edge cases:
        • sent1 has no triplets → 1.0  (no factual claim to violate)
        • sent1 has triplets, sent2 has none → 0.0  (claim ungrounded)

Relation types used
-------------------
A broad zero-shot set covering common hallucination failure modes:
    located_in, founded_by, created_by, authored_by, part_of, affiliated_with,
    employed_by, born_in, died_in, happened_in, dated, measured_as,
    owned_by, produced_by, caused_by, described_as, related_to

Model note
----------
gliner-relex-large-v1.0 uses the same `gliner` package as the entity extractor
but is a separate checkpoint.  It is loaded lazily into a class-level cache so
it is instantiated at most once per process.  It does NOT share weights with
EntityMatch or RelationGrounding (different model name).
"""

from __future__ import annotations

import difflib
from typing import List

from backend.Postprocess.__addpad import resize_matrix
from backend.cache_db import CacheDB

# ---------------------------------------------------------------------------
# Relation types (zero-shot labels passed to the model at inference time)
# ---------------------------------------------------------------------------

RELATION_TYPES: list[str] = [
    "located_in",
    "founded_by",
    "created_by",
    "authored_by",
    "part_of",
    "affiliated_with",
    "employed_by",
    "born_in",
    "died_in",
    "happened_in",
    "dated",
    "measured_as",
    "owned_by",
    "produced_by",
    "caused_by",
    "described_as",
    "related_to",
]

# Entity types used for the joint NER step of the relex model
ENTITY_TYPES: list[str] = [
    "person", "organization", "location", "product",
    "event", "date", "number", "quantity",
]

# Fuzzy match threshold for tail comparison
_TAIL_THRESHOLD: float = 0.82

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fuzzy_match(a: str, b: str, threshold: float = _TAIL_THRESHOLD) -> bool:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


def _triplet_recall(
    triplets1: list[tuple[str, str, str]],
    triplets2: list[tuple[str, str, str]],
) -> float:
    """Recall of text1 triplets in text2.

    A triplet (head1, rel, tail1) is matched if text2 contains a triplet
    (head2, rel, tail2) where:
        rel is identical AND tail1 fuzzy-matches tail2.

    Args:
        triplets1: List of (head, relation_type, tail) from sentence 1.
        triplets2: List of (head, relation_type, tail) from sentence 2.

    Returns:
        float in [0, 1].
    """
    if not triplets1:
        return 1.0
    if not triplets2:
        return 0.0

    tails_by_rel: dict[str, list[str]] = {}
    for _, rel, tail in triplets2:
        tails_by_rel.setdefault(rel, []).append(tail)

    matched = 0
    for _, rel, tail in triplets1:
        candidates = tails_by_rel.get(rel, [])
        if any(_fuzzy_match(tail, c) for c in candidates):
            matched += 1

    return matched / len(triplets1)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class RelexGrounding:
    """Compute relation-triplet-recall feature maps for sentence-pair inputs.

    Uses `knowledgator/gliner-relex-large-v1.0` (joint NER + RE) to extract
    (head, relation_type, tail) triplets, then computes pairwise recall.

    Interface:
        RelexGrounding().getFeatureMap(phrase_list1, phrase_list2)
            → {"relation_triplet_recall": Tensor[32, 32]}
    """

    MODEL: str = "knowledgator/gliner-relex-large-v1.0"
    ENTITY_THRESHOLD: float = 0.3
    RELATION_THRESHOLD: float = 0.5
    _MAX_CHARS: int = 512

    # Class-level model cache — loaded at most once per process
    _model_cache: dict = {}
    # Sentence-level triplet cache — keyed by sentence text
    _triplet_cache: dict[str, list[tuple[str, str, str]]] = {}
    _disk_cache_loaded: bool = False

    @classmethod
    def load_triplet_cache(cls) -> None:
        """Bulk-load all relation triplets from SQLite (once per process)."""
        if cls._disk_cache_loaded:
            return
        cls._disk_cache_loaded = True
        try:
            rows = CacheDB.get().load_all_triplets()
            for sent, triplets in rows.items():
                cls._triplet_cache[sent] = [tuple(t) for t in triplets]
            print(f"[RelexCache] loaded {len(rows)} persisted sentence triplets from SQLite")
        except Exception as e:
            print(f"[RelexCache] warning: could not load persistent cache ({e}) — starting fresh")

    @classmethod
    def save_triplet_cache(cls, new_sentences: list[str]) -> None:
        """Persist newly computed triplets to SQLite.

        Args:
            new_sentences: Sentences that were just computed and should be saved.
        """
        if not new_sentences:
            return
        try:
            triplets_list = [list(cls._triplet_cache[s]) for s in new_sentences]
            CacheDB.get().save_triplets_batch(new_sentences, triplets_list)
        except Exception as e:
            print(f"[RelexCache] warning: could not save to SQLite ({e})")

    def __init__(self):
        self._reset_state()

    def _reset_state(self):
        self.phrase_list1: list[str] | None = None
        self.phrase_list2: list[str] | None = None
        self._weight_matrix: list[list[float]] = []

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self.MODEL in RelexGrounding._model_cache:
            return
        from gliner import GLiNER
        RelexGrounding._model_cache[self.MODEL] = GLiNER.from_pretrained(
            self.MODEL,
            cache_dir=f"Features/Relations/model/{self.MODEL}/",
        )

    # ------------------------------------------------------------------
    # Triplet extraction
    # ------------------------------------------------------------------

    def _extract_triplets(self, texts: list[str]) -> list[list[tuple[str, str, str]]]:
        """Return (head, relation_type, tail) triplets for each sentence.

        Results are cached by sentence text to avoid redundant inference.

        Args:
            texts: List of sentence strings.

        Returns:
            List of triplet lists, one per input sentence.
        """
        RelexGrounding.load_triplet_cache()

        new_texts = [t for t in texts if t not in RelexGrounding._triplet_cache]

        if new_texts:
            self._ensure_model()
            model = RelexGrounding._model_cache[self.MODEL]
            truncated = [t[: self._MAX_CHARS] for t in new_texts]

            # Step 1: batch entity extraction across all new sentences at once.
            try:
                if hasattr(model, "batch_predict_entities"):
                    batch_entities = model.batch_predict_entities(
                        truncated, ENTITY_TYPES, threshold=self.ENTITY_THRESHOLD
                    )
                else:
                    batch_entities = [
                        model.predict_entities(t, ENTITY_TYPES, threshold=self.ENTITY_THRESHOLD)
                        for t in truncated
                    ]
            except Exception:
                batch_entities = [[] for _ in truncated]

            # Step 2: relation extraction per sentence (requires per-text entity list).
            # predict_relations is inherently per-text so this loop is unavoidable,
            # but entity inference (the expensive transformer pass) is batched above.
            for trunc, entities in zip(truncated, batch_entities):
                triplets: list[tuple[str, str, str]] = []
                try:
                    if entities and hasattr(model, "predict_relations"):
                        relations = model.predict_relations(
                            trunc,
                            entities,
                            RELATION_TYPES,
                            threshold=self.RELATION_THRESHOLD,
                        )
                        for rel in relations:
                            head = rel.get("head_text", rel.get("head", {}).get("text", ""))
                            tail = rel.get("tail_text", rel.get("tail", {}).get("text", ""))
                            rtype = rel.get("relation_text", rel.get("relation", ""))
                            if head and tail and rtype:
                                triplets.append((head, rtype, tail))
                except Exception:
                    pass  # model inference failure → empty triplet list for this sentence
                RelexGrounding._triplet_cache[trunc] = triplets

            RelexGrounding.save_triplet_cache(truncated)

        return [RelexGrounding._triplet_cache.get(t[: self._MAX_CHARS], []) for t in texts]

    # ------------------------------------------------------------------
    # Matrix computation
    # ------------------------------------------------------------------

    def _compute_matrix(self) -> None:
        all_texts = self.phrase_list1 + self.phrase_list2
        all_triplets = self._extract_triplets(all_texts)
        n1 = len(self.phrase_list1)
        triplets1 = all_triplets[:n1]
        triplets2 = all_triplets[n1:]
        self._weight_matrix = [
            [_triplet_recall(t1, t2) for t2 in triplets2]
            for t1 in triplets1
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def getFeatureMap(
        self,
        phrase_list1: List[str],
        phrase_list2: List[str],
    ) -> dict:
        """Compute relation-triplet-recall feature map.

        Args:
            phrase_list1: Sentences from text1 (n sentences).
            phrase_list2: Sentences from text2 (m sentences).

        Returns:
            {"relation_triplet_recall": Tensor[32, 32]}
        """
        self._reset_state()
        self.phrase_list1 = phrase_list1
        self.phrase_list2 = phrase_list2
        self._compute_matrix()
        return {"relation_triplet_recall": resize_matrix(self._weight_matrix)}


if __name__ == "__main__":
    t1 = [
        "Apple was founded by Steve Jobs in Cupertino, California in 1976.",
        "The company released the iPhone in 2007.",
    ]
    t2 = [
        "Apple was founded by Bill Gates in Seattle in 1975.",   # entity + date hallucination
        "The iPhone was announced by Apple in 2007.",            # relation change
    ]

    obj = RelexGrounding()
    maps = obj.getFeatureMap(t1, t2)
    for k, v in maps.items():
        print(f"{k}: shape={v.shape}")
        print(f"  Raw 2x2 corner:")
        print(f"    {v[0,0].item():.3f}  {v[0,1].item():.3f}")
        print(f"    {v[1,0].item():.3f}  {v[1,1].item():.3f}")
