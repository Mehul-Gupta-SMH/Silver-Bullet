 # Silver Bullet
------
<img src="assets/icon.png" alt="Project Banner" width="100%" style="max-width:600px; height:auto;">

Silver Bullet is an evaluation framework inspired by Reward Models used in RLHF.
Given two text passages, it produces a similarity / faithfulness score in **[0, 1]** that can be interpreted differently depending on the use case:

| Pair type | Score interpretation |
|-----------|----------------------|
| Context + LLM answer | Faithfulness — does the answer stay true to the source? |
| LLM-A answer + LLM-B answer | Comparability — how interchangeable are the two models? |
| Generated answer + Ground truth | Alignment — how close is the output to the accepted answer? |

-----------------

## How It Works

Silver Bullet compares texts at the **sentence level**, not as whole-document embeddings.

1. Both texts are split into sentences and coreference-resolved (pronouns → named entities).
2. Every sentence in text 1 is compared against every sentence in text 2, producing an **n×m similarity matrix** per feature type.
3. All matrices are padded to **64×64** and stacked as `[F, 64, 64]` feature channels.
4. A trained **Conv2D CNN** classifies the stacked maps into a score.

```
Text pair
  │
  ▼
split_txt()  —  sentence split + coref resolution (GPT-4o-mini / local LLM)
  │
  ▼
Feature extractors  →  n×m matrices  →  pad to [64×64]
  ├─ Semantic  (mxbai-embed-large-v1, Qwen3-Embedding-0.6B)  →  6 maps
  ├─ Lexical   (Jaccard, Dice, Cosine, ROUGE)                 →  4 maps
  ├─ NLI       (RoBERTa-large-MNLI)                           →  3 maps
  ├─ Entity    (GLiNER NER — entity count mismatch)           →  1 map
  └─ LCS       (token-level + char-level LCS similarity)      →  2 maps
                                                    TOTAL: 16 channels
  │
  ▼
TextSimilarityCNN  [B, 16, 64, 64]
  Conv2d × 3  →  MaxPool × 3  →  FC  →  sigmoid
  │
  ▼
Score  ∈ [0, 1]
```

-----------------

## Features

- **Semantic Analysis** — cosine similarity + soft row/column alignment over two transformer embedding models
- **Lexical Analysis** — token-overlap metrics (Jaccard, Dice, Cosine, ROUGE) via SentencePiece tokenisation
- **NLI Analysis** — entailment / neutral / contradiction probabilities per sentence pair
- **Entity Overlap** — GLiNER-based named entity count mismatch
- **LCS Similarity** — normalised Longest Common Subsequence at token and character level
- **Coreference Resolution** — pronoun → named entity substitution before feature extraction
- **Spatial CNN** — Conv2D model preserves the 2D alignment structure of sentence-pair matrices
- **Feature Caching** — MD5-keyed disk cache avoids recomputing expensive features mid-training
- **Training Reports** — per-epoch JSON + Markdown reports with loss, accuracy, hardware info
- **Early Stopping** — configurable patience with best-checkpoint saving

-----------------

## Installation

```bash
git clone https://github.com/Mehul-Gupta-SMH/Silver-Bullet.git
cd Silver-Bullet
pip install -r requirements.txt
mkdir -p data cache training_reports test_reports
```

### Configuration

Copy the example config and fill in your tokens:

```bash
cp resources/config.yaml.example resources/config.yaml
# edit resources/config.yaml with your keys
```

Or set environment variables (these take precedence over the YAML file):

```bash
export SB_HF_TOKEN=hf_...
export SB_OPENAI_TOKEN=sk-...
```

> **Security:** `resources/config.yaml` is gitignored. Never commit real tokens.

-----------------

## Usage

### Feature Extraction Demo

```bash
python example.py
```

Runs the full feature pipeline on sample paragraph pairs and prints each feature map's shape and value range.

### Training

Prepare `data/train.json` and `data/validate.json` in this format:

```json
{
  "data": [
    { "text1": "...", "text2": "...", "label": 1 },
    { "text1": "...", "text2": "...", "label": 0 }
  ]
}
```

Then:

```bash
python train.py
```

- Feature cache is written to `./cache/` (delete it before retraining with a changed feature set)
- Best checkpoint saved to `best_model.pth`
- Training reports saved to `training_reports/`

### Testing

```bash
python test.py
```

Produces accuracy, ROC-AUC, average precision, confusion matrix, and per-sample predictions in `test_reports/`.

### Inference

```bash
python predict.py --model best_model.pth --input input.json --output predictions.json [--device cuda|cpu]
```

Input JSON accepts either format:

```json
[["text1", "text2"], ["text3", "text4"]]
```

```json
{ "data": [{ "text1": "...", "text2": "..." }] }
```

Output:

```json
[
  { "prediction": 1, "probability": 0.92, "text1": "...", "text2": "..." }
]
```

-----------------

## Project Structure

```
Silver-Bullet/
├── data/                         # train.json, validate.json, test.json
├── resources/
│   ├── config.yaml               # secrets (gitignored)
│   ├── config.yaml.example       # template — commit this, not config.yaml
│   └── getConfig.py
├── Preprocess/coref/
│   └── resolveEntity.py          # EntityResolver — coreference via GPT-4o-mini or local LLM
├── Splitter/
│   └── sentence_splitter.py      # split_txt(text, resolver=None)
├── Features/
│   ├── Semantic/                 # 6 maps — embedding cosine sim + soft alignment
│   ├── Lexical/                  # 4 maps — token overlap metrics
│   ├── NLI/                      # 3 maps — entailment probabilities
│   ├── EntityGroups/             # 1 map  — entity count mismatch
│   └── LCS/                      # 2 maps — token + char LCS similarity
├── Postprocess/
│   └── __addpad.py               # pad_matrix() — n×m → 64×64, truncates if larger
├── model.py                      # TextSimilarityCNN — Conv2D over [F, 64, 64]
├── train.py                      # training pipeline
├── test.py                       # evaluation pipeline
├── predict.py                    # inference
├── example.py                    # feature extraction demo
├── feature_cache.py              # MD5-keyed disk cache
├── training_report.py            # JSON + Markdown report generation
├── CLAUDE.md                     # architecture reference for Claude Code
├── AGENT.md                      # per-module API reference for AI agents
└── TASK.md                       # task log
```

-----------------

## Model Architecture

`TextSimilarityCNN` takes stacked feature maps as `[batch, F, 64, 64]` input:

```
[B, F, 64, 64]
  Conv2d(F→128)  + BN + ReLU + MaxPool   →  [B, 128, 32, 32]
  Conv2d(128→64) + BN + ReLU + MaxPool   →  [B,  64, 16, 16]
  Conv2d(64→32)  + BN + ReLU + MaxPool   →  [B,  32,  8,  8]
  Flatten                                 →  [B, 2048]
  FC(2048→128)   + BN + ReLU + Dropout
  FC(128→1)      + sigmoid               →  score ∈ [0, 1]
```

The 2D spatial structure of the sentence-pair similarity matrices is preserved through the convolutional layers, allowing the model to learn alignment patterns directly from the feature maps.

### Checkpoints

Every checkpoint includes `num_features` so the model can be reloaded without re-running the feature pipeline:

| File | Contents |
|------|----------|
| `best_model.pth` | Best validation-loss checkpoint (updated during training) |
| `model_weights_{ts}_best.pth` | Copy of best checkpoint at end of run |
| `model_weights_{ts}_final.pth` | Final epoch checkpoint |

-----------------

## Adding a New Feature Extractor

1. Create `Features/{Name}/get{Name}Weights.py`
2. Implement `getFeatureMap(list1, list2) -> dict[str, torch.Tensor[64,64]]` following the `_reset_state()` pattern (see `AGENT.md` for details)
3. Add to `train.py` and `example.py`
4. Delete `./cache/` and retrain (new features change `num_features`)

-----------------

## License

Specify your license here.

---
*For questions or contributions, please open an issue or pull request.*
