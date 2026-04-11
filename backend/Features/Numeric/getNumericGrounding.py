"""Numeric grounding feature extractor for SilverBullet.

Extracts numeric tokens from each sentence and computes Jaccard overlap
between the number sets of text1 and text2 sentences.

Catches hallucinations where the topic is right but specific numbers,
statistics, or monetary amounts are wrong — a class of errors not captured
by the lexical (token overlap) or semantic (embedding cosine) features.

Feature produced:
    "numeric_jaccard"  — n×m Jaccard overlap matrix over normalised
                         numeric token sets. Higher = more numeric agreement.

Normalisation rules:
    - Currency symbols ($, £, €, ¥) stripped
    - Commas removed from numbers (1,000 → 1000)
    - Multiplier suffixes collapsed (K→×1e3, M→×1e6, B/bn→×1e9, T→×1e12)
    - Percentages kept as "N%" after stripping spaces ("25 %" → "25%")
    - Decimals and integers kept verbatim after the above transforms
    - All tokens lower-cased and stripped of trailing punctuation
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

from backend.Postprocess.__addpad import resize_matrix

# ---------------------------------------------------------------------------
# Regex: match numeric tokens (optionally prefixed with currency symbol)
# Captures: currency? + digits/decimal + optional multiplier/percent
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(
    r"""
    (?:[\$£€¥])?          # optional leading currency symbol
    \d[\d,\.]*            # integer or decimal (possibly with thousand-sep commas)
    \s*                   # optional space before suffix
    (?:                   # optional suffix group
        %                 # percentage
      | [Kk](?!\w)        # K (thousands) — not followed by word char (avoids "km")
      | [Mm](?!\w)        # M (millions)
      | [Bb](?:n|illion)? # B / Bn / Billion
      | [Tt](?:n|rillion)?# T / Tn / Trillion
    )?
    """,
    re.VERBOSE,
)

# Multiplier table (lower-case suffix → multiplier)
_MULTIPLIERS = {
    "k":        1_000,
    "m":        1_000_000,
    "b":        1_000_000_000,
    "bn":       1_000_000_000,
    "billion":  1_000_000_000,
    "t":        1_000_000_000_000,
    "tn":       1_000_000_000_000,
    "trillion": 1_000_000_000_000,
}


def _normalise_number(raw: str) -> str:
    """Return a canonical string for a raw numeric token.

    Examples:
        "$8 billion" → "8000000000"
        "£1,200"     → "1200"
        "25%"        → "25%"
        "3.14"       → "3.14"
        "2023"       → "2023"
    """
    s = raw.strip().lstrip("$£€¥").replace(",", "").strip()

    # Handle percentage — keep as "N%"
    if s.endswith("%"):
        try:
            return f"{float(s[:-1].strip())}%"
        except ValueError:
            return s

    # Separate numeric part from suffix
    suffix_match = re.search(r'([a-zA-Z]+(?:illion|n)?)\s*$', s)
    suffix = ""
    num_part = s
    if suffix_match:
        suffix = suffix_match.group(1).lower()
        num_part = s[:suffix_match.start()].strip()

    try:
        value = float(num_part)
    except ValueError:
        return raw.strip().lower()

    multiplier = _MULTIPLIERS.get(suffix, 1)
    result = value * multiplier
    # Return as integer string if whole number, else float string
    if result == int(result):
        return str(int(result))
    return str(result)


def _extract_numbers(text: str) -> list[str]:
    """Return sorted list of normalised numeric tokens found in *text*."""
    tokens = _NUM_RE.findall(text)
    normalised = [_normalise_number(t) for t in tokens if t.strip()]
    # De-duplicate but preserve multiplicity: use sorted list (not set) so
    # "3 and 3" is represented as ["3", "3"] and matches "3 and 3" exactly.
    return sorted(normalised)


def _numeric_jaccard(nums1: list[str], nums2: list[str]) -> float:
    """Jaccard similarity over multisets of normalised number strings.

    Both empty → 1.0 (both sentences make no numeric claims — agree on absence).
    One empty, other non-empty → 0.0 (one side introduces numeric claims).
    """
    if not nums1 and not nums2:
        return 1.0
    if not nums1 or not nums2:
        return 0.0

    from collections import Counter
    c1, c2 = Counter(nums1), Counter(nums2)
    intersection = sum((c1 & c2).values())
    union        = sum((c1 | c2).values())
    return intersection / union if union else 0.0


class NumericGrounding:
    """Compute numeric grounding feature maps for sentence-pair inputs.

    Follows the same interface as all other SilverBullet feature extractors:
        getFeatureMap(phrase_list1, phrase_list2) → dict[str, Tensor]

    Output:
        "numeric_jaccard" — [n, m] Jaccard overlap over normalised number sets.
                            Values in [0, 1].  Higher = more numeric agreement.
    """

    def __init__(self):
        self._reset_state()

    def _reset_state(self):
        self.phrase_list1 = None
        self.phrase_list2 = None
        self._nums1: list[list[str]] = []
        self._nums2: list[list[str]] = []
        self.weight_matrix: dict[str, list[list[float]]] = {"numeric_jaccard": []}

    def _extract_all(self):
        """Batch-extract normalised number lists for both phrase lists."""
        with ThreadPoolExecutor() as ex:
            self._nums1 = list(ex.map(_extract_numbers, self.phrase_list1))
            self._nums2 = list(ex.map(_extract_numbers, self.phrase_list2))

    def _compute_matrix(self):
        matrix: list[list[float]] = []
        for nums1 in self._nums1:
            row = [_numeric_jaccard(nums1, nums2) for nums2 in self._nums2]
            matrix.append(row)
        self.weight_matrix["numeric_jaccard"] = matrix

    def getFeatureMap(self, phrase_list1: list[str], phrase_list2: list[str]) -> dict:
        """Compute and return the numeric grounding feature map.

        Args:
            phrase_list1: Sentences from text1 (n sentences).
            phrase_list2: Sentences from text2 (m sentences).

        Returns:
            {"numeric_jaccard": Tensor[32, 32]}
        """
        self._reset_state()
        self.phrase_list1 = phrase_list1
        self.phrase_list2 = phrase_list2
        self._extract_all()
        self._compute_matrix()
        for key in self.weight_matrix:
            self.weight_matrix[key] = resize_matrix(self.weight_matrix[key])
        return self.weight_matrix


if __name__ == "__main__":
    t1 = ["George Osborne was challenged 18 times to explain the extra £8 billion for the NHS."]
    t2 = ["George Osborne refused to explain the £8 billion NHS funding commitment."]
    t3 = ["George Osborne refused to explain the £80 billion NHS funding commitment."]

    obj = NumericGrounding()
    correct = obj.getFeatureMap(t1, t2)
    wrong   = obj.getFeatureMap(t1, t3)
    print("Same number:", correct["numeric_jaccard"].mean().item())   # expect ~1.0
    print("Wrong number:", wrong["numeric_jaccard"].mean().item())    # expect ~0.0
