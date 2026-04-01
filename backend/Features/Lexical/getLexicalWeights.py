"""
This module provides functionality to compute lexical weights for phrase pairs
based on their token overlap. It defines a class `LexicalWeights` that takes two
lists of phrases, tokenizes them, and calculates a weight matrix indicating the
degree of lexical similarity between each pair of phrases.
"""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from typing import List, Tuple, Dict
from transformers import AutoTokenizer
from backend.Postprocess.__addpad import resize_matrix


class LexicalWeights:
    """
    Computes lexical weights for phrase pairs based on token overlap.
    The weights are calculated using various similarity metrics such as Jaccard,
    Dice, Cosine, and ROUGE.
    Attributes:
        phrase_list1 (List[str]): List of phrases (strings) from the first group.
        phrase_list2 (List[str]): List of phrases (strings) from the second group.
        phrase_tokens_list1 (List[List[str]]): Tokenized phrases from the first group.
        phrase_tokens_list2 (List[List[str]]): Tokenized phrases from the second group.
        weight_matrix (Dict[str, List[List[float]]]): A dictionary containing weight matrices
            for different similarity metrics.
    Methods:
        getFeatureMap() -> List[List[Dict[int, Dict[str, float]]]]:
            Computes and returns the weight matrix for the phrase pairs.
    """

    _TOKENIZER_REPO = "mixedbread-ai/mxbai-embed-large-v1"

    # Class-level tokenizer cache shared across all instances
    _tokenizer_cache = None

    def __init__(self):
        self._reset_state()

    def _reset_state(self):
        """Reset per-call data without discarding the tokenizer cache."""
        self.phrase_list1 = None
        self.phrase_tokens_list1 = None
        self.phrase_list2 = None
        self.phrase_tokens_list2 = None
        self.weight_matrix = {
            'jaccard': [],
            'dice': [],
            'cosine': [],
            'rouge': [],
            'rouge3': [],
        }

    def __load_model__(self):
        if LexicalWeights._tokenizer_cache is None:
            LexicalWeights._tokenizer_cache = AutoTokenizer.from_pretrained(
                self._TOKENIZER_REPO,
                use_fast=True,
                add_eos_token=False,
                add_bos_token=False,
                token=None,
                cache_dir=f'Features/Lexical/tokeniser_cache/{self._TOKENIZER_REPO}'
            )

    def sp_tokenize(self, text: str) -> List[str]:
        """SentencePiece tokens as strings (e.g., '▁quick', 'ly')."""
        self.__load_model__()
        return LexicalWeights._tokenizer_cache.tokenize(text, add_special_tokens=False)

    def sp_tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """Batch tokenization — single Rust call for all texts."""
        self.__load_model__()
        tok = LexicalWeights._tokenizer_cache
        enc = tok(texts, add_special_tokens=False, return_attention_mask=False)
        return [tok.convert_ids_to_tokens(ids) for ids in enc["input_ids"]]

    @staticmethod
    def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    @staticmethod
    def jaccard(a, b):
        """Jaccard similarity using sets (not counts)."""
        A, B = set(a), set(b)
        return len(A & B) / len(A | B) if (A or B) else 0.0

    @staticmethod
    def dice(a, b):
        """Dice similarity using sets (not counts)."""
        A, B = set(a), set(b)
        return (2*len(A & B)) / (len(A) + len(B)) if (A or B) else 0.0

    @staticmethod
    def cosine_counts(a, b):
        """Cosine similarity using counts (not sets)."""
        ca, cb = Counter(a), Counter(b)
        keys = set(ca) | set(cb)
        dot = sum(ca[k]*cb[k] for k in keys)
        na = sqrt(sum(v*v for v in ca.values()))
        nb = sqrt(sum(v*v for v in cb.values()))
        return dot/(na*nb) if na and nb else 0.0

    @staticmethod
    def rouge_counts(ref, cand):
        """ROUGE-N recall using counts (not sets)."""
        r, c = Counter(ref), Counter(cand)
        overlap = sum(min(r[k], c[k]) for k in r)
        total = sum(r.values())
        return overlap/total if total else 0.0

    def __compute_token_lists__(self):
        """Batch-tokenizes both phrase lists in two Rust-backed calls."""
        all_tokens = self.sp_tokenize_batch(self.phrase_list1 + self.phrase_list2)
        n = len(self.phrase_list1)
        self.phrase_tokens_list1 = all_tokens[:n]
        self.phrase_tokens_list2 = all_tokens[n:]

    def _compute_row(self, tokens1: List[str]) -> Tuple:
        row_j, row_d, row_c, row_r, row_r3 = [], [], [], [], []
        tri1 = self.ngrams(tokens1, 3)
        for tokens2 in self.phrase_tokens_list2:
            row_j.append(self.jaccard(tokens1, tokens2))
            row_d.append(self.dice(tokens1, tokens2))
            row_c.append(self.cosine_counts(tokens1, tokens2))
            row_r.append(self.rouge_counts(tokens1, tokens2))
            row_r3.append(self.rouge_counts(tri1, self.ngrams(tokens2, 3)))
        return row_j, row_d, row_c, row_r, row_r3

    def __compute_weights__(self):
        """Computes the weight matrix in parallel across rows."""
        with ThreadPoolExecutor() as ex:
            for row_j, row_d, row_c, row_r, row_r3 in ex.map(self._compute_row, self.phrase_tokens_list1):
                self.weight_matrix['jaccard'].append(row_j)
                self.weight_matrix['dice'].append(row_d)
                self.weight_matrix['cosine'].append(row_c)
                self.weight_matrix['rouge'].append(row_r)
                self.weight_matrix['rouge3'].append(row_r3)

    def __post_process_weights__(self):
        for key in self.weight_matrix:
            self.weight_matrix[key] = resize_matrix(self.weight_matrix[key])

    def getFeatureMap(self, phrase_list1, phrase_list2):
        """Computes and returns the weight matrix for the phrase pairs."""
        self._reset_state()

        self.phrase_list1, self.phrase_list2 = phrase_list1, phrase_list2

        self.__compute_token_lists__()
        self.__compute_weights__()
        self.__post_process_weights__()
        return self.weight_matrix


if __name__ == '__main__':
    s1 = "The quick brown fox jumps over the lazy dog."
    s2 = "A quick brown fox leaped over a very lazy dog."
    LexicalWeights_Obj = LexicalWeights()
    print(LexicalWeights_Obj.getFeatureMap([s1, s2], [s1, s2, s1]))
