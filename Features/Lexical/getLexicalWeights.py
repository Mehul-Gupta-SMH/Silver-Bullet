"""
This module provides functionality to compute lexical weights for phrase pairs
based on their token overlap. It defines a class `LexicalWeights` that takes two
lists of phrases, tokenizes them, and calculates a weight matrix indicating the
degree of lexical similarity between each pair of phrases.
"""

from collections import Counter
from math import sqrt
from typing import List, Tuple, Dict
from transformers import AutoTokenizer
from tqdm import tqdm
from Postprocess.__addpad import pad_matrix


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
    def __init__(self):
        """
        Initializes the LexicalWeights object with two lists of phrases.
        :return: None
        """
        self.phrase_list1 = None
        self.phrase_tokens_list1 = None

        self.phrase_list2 = None
        self.phrase_tokens_list2 = None

        self.weight_matrix = {
            'jaccard': [],
            'dice': [],
            'cosine': [],
            'rouge': []
        }

    def __load_model__(self):
        repo_id = "mixedbread-ai/mxbai-embed-large-v1"
        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            use_fast=True,                 # get the fast Rust tokenizer
            add_eos_token=False,           # we don't want special tokens for similarity
            add_bos_token=False,
            token=None,                    # or your HF token string; or set HF_TOKEN env var
            cache_dir=f'/Features/Lexical/tokeniser_cache/{repo_id}'
        )

        self.tokenizer_cache = tokenizer

    def sp_tokenize(self, text: str) -> List[str]:
        """SentencePiece tokens as strings (e.g., '▁quick', 'ly')."""
        self.__load_model__()
        return self.tokenizer_cache.tokenize(text, add_special_tokens=False)

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
        """Tokenizes the phrases in both lists using SentencePiece."""
        self.phrase_tokens_list1 = [self.sp_tokenize(p) for p in self.phrase_list1]
        self.phrase_tokens_list2 = [self.sp_tokenize(p) for p in self.phrase_list2]

    def __compute_weights__(self):
        """Computes the weight matrix for all phrase pairs using various similarity metrics."""
        for tokens1 in self.phrase_tokens_list1:
            row_jaccard, row_dice, row_cosine, row_rouge = [], [], [], []

            for tokens2 in self.phrase_tokens_list2:
                row_jaccard.append(self.jaccard(tokens1, tokens2))
                row_dice.append(self.dice(tokens1, tokens2))
                row_cosine.append(self.cosine_counts(tokens1, tokens2))
                row_rouge.append(self.rouge_counts(tokens1, tokens2))

            self.weight_matrix['jaccard'].append(row_jaccard)
            self.weight_matrix['dice'].append(row_dice)
            self.weight_matrix['cosine'].append(row_cosine)
            self.weight_matrix['rouge'].append(row_rouge)

    def __post_process_weights__(self):
        for key in self.weight_matrix:
            self.weight_matrix[key] = pad_matrix(self.weight_matrix[key])

    def getFeatureMap(self, phrase_list1, phrase_list2):
        """Computes and returns the weight matrix for the phrase pairs."""
        self.__init__()

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


