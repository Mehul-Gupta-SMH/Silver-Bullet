"""
LCS feature extractor.

Computes the normalised Longest Common Subsequence (LCS) similarity between
every pair of sentences across two sentence groups, producing an n×m matrix
that is then padded to 64×64.

Two variants are stored:
    "lcs_token"  — LCS over whitespace-split tokens     (word-level)
    "lcs_char"   — LCS over individual characters        (character-level)

Score formula: len(LCS) / max(len(seq1), len(seq2))
  → 0.0  when the sequences share nothing
  → 1.0  when one is a subsequence of the other (or they are identical)
"""

from typing import List
from tqdm import tqdm
from Postprocess.__addpad import pad_matrix


def _lcs_length(seq1: list, seq2: list) -> int:
    """Dynamic-programming LCS length in O(n·m) time, O(min(n,m)) space."""
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1   # ensure seq2 is the shorter one

    m = len(seq2)
    prev = [0] * (m + 1)

    for item in seq1:
        curr = [0] * (m + 1)
        for j, b in enumerate(seq2):
            if item == b:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(curr[j], prev[j + 1])
        prev = curr

    return prev[m]


def _normalised_lcs(seq1: list, seq2: list) -> float:
    """Normalised LCS similarity ∈ [0, 1]."""
    denom = max(len(seq1), len(seq2))
    if denom == 0:
        return 0.0
    return _lcs_length(seq1, seq2) / denom


class LCSWeights:
    """Computes token-level and character-level LCS similarity matrices."""

    def __init__(self):
        self._reset_state()

    def _reset_state(self):
        self.phrase_list1 = None
        self.phrase_list2 = None
        self.comparison_weights = {
            'lcs_token': [],
            'lcs_char':  [],
        }

    def __compute_weights__(self):
        for phrase1 in tqdm(self.phrase_list1, desc="LCS Weights"):
            tok1  = phrase1.lower().split()
            char1 = list(phrase1.lower())

            row_tok, row_char = [], []
            for phrase2 in self.phrase_list2:
                tok2  = phrase2.lower().split()
                char2 = list(phrase2.lower())

                row_tok.append(_normalised_lcs(tok1, tok2))
                row_char.append(_normalised_lcs(char1, char2))

            self.comparison_weights['lcs_token'].append(row_tok)
            self.comparison_weights['lcs_char'].append(row_char)

    def __post_process_weights__(self):
        for key in self.comparison_weights:
            self.comparison_weights[key] = pad_matrix(self.comparison_weights[key])

    def getFeatureMap(self, phrase_list1: List[str], phrase_list2: List[str]) -> dict:
        """Compute LCS feature maps for two sentence groups.

        Args:
            phrase_list1: Sentences from text 1.
            phrase_list2: Sentences from text 2.
        Returns:
            dict with keys "lcs_token" and "lcs_char",
            each a torch.Tensor of shape [64, 64].
        """
        self._reset_state()
        self.phrase_list1, self.phrase_list2 = phrase_list1, phrase_list2
        self.__compute_weights__()
        self.__post_process_weights__()
        return self.comparison_weights


if __name__ == '__main__':
    s1 = ["The quick brown fox jumps over the lazy dog.",
          "Pack my box with five dozen liquor jugs."]
    s2 = ["A quick brown fox leaped over a very lazy dog.",
          "Pack my box with a dozen liquor jugs.",
          "The fox was not lazy at all."]

    obj = LCSWeights()
    maps = obj.getFeatureMap(s1, s2)
    for k, v in maps.items():
        print(f"{k}: {v.shape}")
        print(v[:3, :4])
