"""SVO (Subject-Verb-Object) relation recall feature extractor for SilverBullet.

Extracts (subject, verb_lemma, object) triples from each sentence using spaCy
dependency parsing, then computes pairwise recall — same semantics as
RelexGrounding but ~100x faster (milliseconds vs seconds per sentence).

Why SVO alongside Relex
-----------------------
GLiNER Relex is accurate on complex zero-shot relation types but has O(n²) span
enumeration overhead that makes bulk prefill impractical on commodity hardware.
spaCy SVO covers the most common hallucination failure modes (predicate inversion,
entity swap, numeric error) via simple dep-tree traversal with no model inference
cost beyond the parser itself.

Running both in parallel lets an ablation study determine whether Relex adds signal
beyond what the fast SVO path already captures.

Feature produced
----------------
"svo_triplet_recall"
    n×m matrix (resized to [32, 32]) where cell (i, j) =
    |SVO triplets in sent1_i that have a match in sent2_j|
    / |SVO triplets in sent1_i|

    Triplet matching:
        • verb lemma must be an exact string match
        • object must fuzzy-match (SequenceMatcher ratio ≥ 0.82)
        • subject is ignored (same rationale as Relex head-ignore)

    Edge cases:
        • sent1 has no SVO triplets → 1.0  (no factual claim to violate)
        • sent1 has triplets, sent2 has none → 0.0  (claim ungrounded)

SVO extraction
--------------
Uses spaCy `en_core_web_sm` (dep parser only — NER/tok2vec not needed).
Subject tokens: dep_ in {nsubj, nsubjpass, csubj, csubjpass}
Verb: the syntactic head of the subject (root of the predicate subtree)
Object tokens: dep_ in {dobj, pobj, attr, dative, oprd} that are children of the verb

For compound objects ("New York City"), the full noun chunk text is used.
Negation: if the verb has a "neg" child, the verb lemma is prefixed with "not_"
so "Apple did not found X" produces ("apple", "not_found", "x") — distinct from
("apple", "found", "x").
"""

from __future__ import annotations

import difflib
from typing import List

from backend.Postprocess.__addpad import resize_matrix

_TAIL_THRESHOLD: float = 0.82

_SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
_OBJECT_DEPS  = {"dobj", "pobj", "attr", "dative", "oprd"}


def _fuzzy_match(a: str, b: str, threshold: float = _TAIL_THRESHOLD) -> bool:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


def _triplet_recall(
    triplets1: list[tuple[str, str, str]],
    triplets2: list[tuple[str, str, str]],
) -> float:
    if not triplets1:
        return 1.0
    if not triplets2:
        return 0.0

    tails_by_verb: dict[str, list[str]] = {}
    for _, verb, obj in triplets2:
        tails_by_verb.setdefault(verb, []).append(obj)

    matched = 0
    for _, verb, obj in triplets1:
        candidates = tails_by_verb.get(verb, [])
        if any(_fuzzy_match(obj, c) for c in candidates):
            matched += 1

    return matched / len(triplets1)


def _extract_svo(doc) -> list[tuple[str, str, str]]:
    """Extract (subject_text, verb_lemma, object_text) triples from a spaCy Doc.

    Iterates over verbs (not subjects) so compound proper-noun subjects like
    'Steve Jobs' are resolved via noun chunks rather than individual dep tokens.
    """
    triplets: list[tuple[str, str, str]] = []
    chunks_by_root = {chunk.root: chunk.text for chunk in doc.noun_chunks}

    for token in doc:
        if token.pos_ not in ("VERB",) or token.dep_ in ("aux", "auxpass", "relcl"):
            continue

        subj_token = next((c for c in token.children if c.dep_ in _SUBJECT_DEPS), None)
        if subj_token is None:
            continue

        subj_text = chunks_by_root.get(subj_token, subj_token.text)
        neg = any(c.dep_ == "neg" for c in token.children)
        verb_lemma = ("not_" if neg else "") + token.lemma_.lower()

        for child in token.children:
            if child.dep_ not in _OBJECT_DEPS:
                continue
            obj_text = chunks_by_root.get(child, child.text)
            triplets.append((subj_text.lower(), verb_lemma, obj_text.lower()))

    return triplets


class SVOGrounding:
    """Compute SVO-based relation-recall feature maps using spaCy dep parsing.

    Interface mirrors RelexGrounding — drop-in for ablation comparison.

    Usage:
        SVOGrounding().getFeatureMap(phrase_list1, phrase_list2)
            → {"svo_triplet_recall": Tensor[32, 32]}
    """

    MODEL: str = "en_core_web_sm"

    # Class-level model cache — loaded at most once per process
    _nlp_cache: dict = {}

    def __init__(self):
        self.phrase_list1: list[str] | None = None
        self.phrase_list2: list[str] | None = None
        self._weight_matrix: list[list[float]] = []

    def _ensure_model(self):
        if self.MODEL not in SVOGrounding._nlp_cache:
            import spacy
            try:
                nlp = spacy.load(self.MODEL, disable=["ner"])
            except OSError:
                from spacy.cli import download
                download(self.MODEL)
                nlp = spacy.load(self.MODEL, disable=["ner"])
            SVOGrounding._nlp_cache[self.MODEL] = nlp

    def _extract_triplets(self, texts: list[str]) -> list[list[tuple[str, str, str]]]:
        self._ensure_model()
        nlp = SVOGrounding._nlp_cache[self.MODEL]
        return [_extract_svo(doc) for doc in nlp.pipe(texts, batch_size=64)]

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

    def getFeatureMap(
        self,
        phrase_list1: List[str],
        phrase_list2: List[str],
    ) -> dict:
        """Compute SVO triplet-recall feature map.

        Args:
            phrase_list1: Sentences from text1 (n sentences).
            phrase_list2: Sentences from text2 (m sentences).

        Returns:
            {"svo_triplet_recall": Tensor[32, 32]}
        """
        self.phrase_list1 = phrase_list1
        self.phrase_list2 = phrase_list2
        self._compute_matrix()
        return {"svo_triplet_recall": resize_matrix(self._weight_matrix)}


if __name__ == "__main__":
    # Active-voice pairs — SVO extraction works best on SVO sentences.
    # t1[0] says "Jobs founded Apple"; t2[0] halluccinates the founder → recall=0
    # t1[1] says "company released iPhone"; t2[1] agrees → recall=1
    t1 = [
        "Steve Jobs founded Apple in Cupertino.",
        "The company released the iPhone in 2007.",
    ]
    t2 = [
        "Bill Gates founded Apple in Seattle.",    # wrong founder — predicate match, tail mismatch
        "Apple released the iPhone in 2007.",      # faithful restatement
    ]

    obj = SVOGrounding()
    maps = obj.getFeatureMap(t1, t2)
    for k, v in maps.items():
        print(f"{k}: shape={v.shape}")
        # Sample at quadrant centres of the 32x32 resized matrix (not [0,0] corner)
        print(f"  Quadrant samples (expect: TL~1.0, TR~0.0, BL~0.0, BR~1.0):")
        print(f"    TL={v[8,8].item():.3f}  TR={v[8,24].item():.3f}")
        print(f"    BL={v[24,8].item():.3f}  BR={v[24,24].item():.3f}")
