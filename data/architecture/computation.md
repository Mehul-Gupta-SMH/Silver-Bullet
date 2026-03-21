# SilverBullet — Score Computation

## Overview

Given two texts, the pipeline produces a scalar similarity score in [0, 1].
Each feature extractor builds an **n×m matrix** (n sentences from text1, m sentences from text2),
which is then reduced to a single score for display and fed into the CNN as a 64×64 padded tensor.

---

## Full Pipeline

```mermaid
flowchart TD
    A["text1, text2"] --> B["split_txt()\nSentence splitter + coref resolution"]
    B --> C["sent_group1 : List[str]\n(n sentences)"]
    B --> D["sent_group2 : List[str]\n(m sentences)"]

    C & D --> E1["SemanticWeights\n mxbai + Qwen3\n→ 6 maps"]
    C & D --> E2["LexicalWeights\n jaccard / dice / cosine / rouge\n→ 4 maps"]
    C & D --> E3["NLIWeights\n entailment / neutral / contradiction\n→ 3 maps"]
    C & D --> E4["EntityMatch\n EntityMismatch\n→ 1 map"]
    C & D --> E5["LCSWeights\n lcs_token / lcs_char\n→ 2 maps"]

    E1 & E2 & E3 & E4 & E5 --> F["16 raw n×m matrices"]
    F --> G["pad_matrix()\neach n×m → 64×64 zero-padded tensor"]
    G --> H["stack → [1, 16, 64, 64]"]
    H --> I["TextSimilarityCNN\nConv2d × 3 → FC → sigmoid"]
    I --> J["score ∈ [0, 1]"]
```

---

## n×m Matrix Structure

Every feature extractor produces a matrix where cell `[i][j]` is the
similarity between sentence `i` from text1 and sentence `j` from text2.

```mermaid
block-beta
    columns 5
    A[""] B["s2[0]"] C["s2[1]"] D["s2[2]"] E["s2[3]"]
    F["s1[0]"] G["0.9"] H["0.3"] I["0.2"] J["0.1"]
    K["s1[1]"] L["0.2"] M["0.8"] N["0.7"] O["0.1"]
    P["s1[2]"] Q["0.1"] R["0.2"] S["0.3"] T["0.6"]
```

---

## Feature Score Reduction (mean-max-row)

For the breakdown panel, each n×m matrix is reduced to a single scalar
using `_mean_max_row` (`predict.py:141`):

```mermaid
flowchart LR
    A["n×m matrix\n(cropped to actual\nsentence counts)"]
    --> B["For each row i\n→ take max across columns\n best match score for s1[i]"]
    --> C["[max_row_0, max_row_1, ..., max_row_n]"]
    --> D["mean()\n→ scalar score ∈ [0, 1]"]
```

### Example (3 sentences × 4 sentences)

```mermaid
flowchart LR
    subgraph Matrix ["n×m matrix"]
        R0["s1[0]: [0.9, 0.3, 0.2, 0.1]"]
        R1["s1[1]: [0.2, 0.8, 0.7, 0.1]"]
        R2["s1[2]: [0.1, 0.2, 0.3, 0.6]"]
    end

    subgraph RowMax ["row max"]
        M0["0.9"]
        M1["0.8"]
        M2["0.6"]
    end

    subgraph Result ["score"]
        S["mean(0.9, 0.8, 0.6)\n= 0.767"]
    end

    R0 --> M0
    R1 --> M1
    R2 --> M2
    M0 & M1 & M2 --> S
```

**Interpretation:** *"On average, how well does each sentence in text1 find its best counterpart in text2?"*

---

## Asymmetry

The reduction is row-oriented (text1-centric). text2 sentences only influence the
score when they happen to be the best match for a text1 sentence.

```mermaid
flowchart TD
    A["sentence in text2 with no counterpart\n in text1"]
    --> B{"Does any s1[i] row\nmax land on this column?"}
    B -- Yes --> C["Indirectly influences score"]
    B -- No --> D["Invisible to feature_scores\nbut flagged in divergent_in_2\nif max_col < 0.5"]
```

This makes the score well-suited for **hallucination detection**
(`text1 = source context`, `text2 = LLM output`) where the primary concern is
whether every output sentence is grounded in the source — but it can
overestimate similarity when text2 adds fabricated sentences that still
partially match text1.

---

## Divergence Detection (separate from feature_scores)

```mermaid
flowchart LR
    A["alignment matrix\n(mxbai cosine sim only)"]

    A --> B["max_row[i] = max over columns\nfor each sentence in text1"]
    A --> C["max_col[j] = max over rows\nfor each sentence in text2"]

    B --> D{"< 0.5 ?"}
    C --> E{"< 0.5 ?"}

    D -- Yes --> F["divergent_in_1\n(text1 sentence has no\ncounterpart in text2)"]
    D -- No  --> G["covered"]
    E -- Yes --> H["divergent_in_2\n(text2 sentence has no\ncounterpart in text1)"]
    E -- No  --> I["covered"]
```

> Divergence uses only the **mxbai cosine similarity** map, not all 16 maps.
> Threshold is hardcoded at `THRESH = 0.5` (`predict.py:131`).

---

## Feature Scores Displayed in the UI

| Label | Matrix key | Signal |
|---|---|---|
| Semantic (mxbai) | `mixedbread-ai/mxbai-embed-large-v1` | Dense cosine similarity |
| Semantic (Qwen3) | `Qwen/Qwen3-Embedding-0.6B` | Dense cosine similarity (2nd model) |
| Lexical ROUGE | `rouge` | ROUGE-1 F1 over SentencePiece tokens |
| Lexical Jaccard | `jaccard` | Jaccard index over token sets |
| NLI Entailment | `entailment` | roberta-large-mnli entailment probability |
| LCS Token | `lcs_token` | Normalised longest common subsequence |

The remaining 10 maps (`SOFT_ROW`, `SOFT_COL`, `dice`, `cosine`, `neutral`,
`contradiction`, `EntityMismatch`, `lcs_char`) are used by the CNN but not
displayed in the breakdown panel.

---

## Padding to 64×64

```mermaid
flowchart LR
    A["raw n×m matrix\n(up to 64×64)"] --> B["pad_matrix()\nPostprocess/__addpad.py"]
    B --> C["64×64 tensor\n(zero-padded, rows/cols\ntruncated if > 64)"]
    C --> D["stacked with other 15 maps\n→ [16, 64, 64] input to CNN"]
```

> Inputs exceeding 64 sentences are silently truncated (not crashed).
> See `Postprocess/__addpad.py`.
