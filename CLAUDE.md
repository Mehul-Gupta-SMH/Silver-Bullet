# SilverBullet — Claude Code Instructions

## What This Project Is

SilverBullet is a **learned text similarity / faithfulness scoring system**.
Given two texts, it produces a score in [0, 1].

**Use cases:**
- LLM faithfulness: `text1 = LLM output`, `text2 = source context`
- LLM comparison: `text1 = LLM-A output`, `text2 = LLM-B output`
- RL alignment: `text1 = generated answer`, `text2 = ground-truth answer`

**Closest analogue:** cross-encoder / re-ranker, but with explicit multi-signal feature engineering before the learned head.

---

## Repository Layout

```
Silver-Bullet/
├── data/                         # train.json, validate.json, test.json
│   └── *.json                    # {"data": [{"text1":..., "text2":..., "label": 0|1}]}
│
├── resources/
│   ├── config.yaml               # API keys by environment (DEVELOPMENT / STAGING / PRODUCTION)
│   └── getConfig.py              # getVal(env='DEVELOPMENT') -> dict
│
├── Preprocess/
│   └── coref/
│       └── resolveEntity.py      # EntityResolver — pronoun → named entity via GPT-4o-mini or local LLM
│
├── Splitter/
│   └── sentence_splitter.py      # split_txt(text) -> List[str]  (regex split + coref per sentence)
│
├── Features/
│   ├── Semantic/
│   │   ├── __generate_semantic_features.py   # SemanticFeatures — SentenceTransformer encoding
│   │   └── getSemanticWeights.py             # SemanticWeights.getFeatureMap() -> 6 padded 64×64 tensors
│   ├── Lexical/
│   │   └── getLexicalWeights.py              # LexicalWeights.getFeatureMap() -> 4 padded 64×64 tensors
│   ├── NLI/
│   │   └── getNLIweights.py                  # NLIWeights.getFeatureMap() -> 3 padded 64×64 tensors
│   └── EntityGroups/
│       └── getOverlap.py                     # EntityMatch.getFeatureMap() -> 1 padded 64×64 tensor
│
├── Postprocess/
│   ├── __addpad.py               # pad_matrix(mat, target_size=64) -> torch.Tensor [64,64]
│   └── postprocess.py            # thin wrapper around pad_matrix
│
├── feature_cache.py              # FeatureCache — MD5-keyed JSON cache in ./cache/
├── model.py                      # TextSimilarityCNN — FC reduction → Conv1D → FC → sigmoid
├── train.py                      # TextSimilarityDataset + train_model() entry point
├── test.py                       # TestReport + test_model() entry point
├── predict.py                    # SimilarityPredictor — batch/single inference
├── training_report.py            # TrainingReport — per-epoch JSON + Markdown reports
├── example.py                    # Manual feature pipeline demo (no model)
│
├── best_model.pth                # Saved best checkpoint (gitignored ideally)
├── requirements.txt
└── .sbvenv/                      # Local virtualenv (gitignored)
```

---

## Full Data Flow

```
Raw text pair (text1, text2)
         │
         ▼
  split_txt()  ←  Splitter/sentence_splitter.py
    • Regex split on period/newline
    • Each sentence → EntityResolver.resolve() (coref via GPT-4o-mini or Gemma)
    → [sent_group1: List[str], sent_group2: List[str]]
         │
         ▼
  Feature Extraction  (all run on sent_group1 × sent_group2 → n×m matrices)
    ┌─ SemanticWeights.getFeatureMap()
    │    mxbai-embed-large-v1    → cosine sim, SOFT_ROW, SOFT_COL   (3 maps)
    │    Qwen3-Embedding-0.6B    → cosine sim, SOFT_ROW, SOFT_COL   (3 maps)
    ├─ LexicalWeights.getFeatureMap()
    │    SentencePiece tokens    → jaccard, dice, cosine, rouge      (4 maps)
    ├─ NLIWeights.getFeatureMap()
    │    roberta-large-mnli      → entailment, neutral, contradiction (3 maps)
    └─ EntityMatch.getFeatureMap()
         GLiNER NER              → EntityMismatch                    (1 map)
                                                           TOTAL: 11 maps
         │
         ▼
  pad_matrix() — each n×m → 64×64 zero-padded tensor
  flatten each 64×64 → 4096 floats
  concatenate all → feature_vector  (11 × 4096 = 45,056 floats, approximately)
         │
         ▼
  TextSimilarityCNN (model.py)
    FC:     input_dim → 4096 → 1024 → 256
    Conv1D: [1, 256] → [128, 256] → [64, 64]  (with max-pool ×2)
    FC:     4096 → 128 → 1
    sigmoid → score ∈ [0, 1]
```

> **Note:** The actual feature count depends on which soft-alignment maps are generated.
> The model `input_dim` is determined dynamically at train time from the first sample.

---

## Running the Project

```bash
# Activate virtualenv (Windows)
.\.sbvenv\Scripts\activate

# Train
python train.py

# Test
python test.py

# Predict (batch)
python predict.py --model best_model.pth --input data/test.json --output predictions.json

# Feature pipeline demo (no model)
python example.py
```

---

## Environment & Config

Config is loaded from `resources/config.yaml` via `resources/getConfig.py`:

```python
from resources.getConfig import getVal
config = getVal(env='DEVELOPMENT')   # keys: url, api_key, hf_token, openai_token
```

**IMPORTANT — Security Issue:**
`config.yaml` currently contains hardcoded API keys committed to the repo.
Until this is fixed, do NOT commit new secrets. The correct fix is:
1. Load tokens from environment variables (`os.environ`)
2. Add `resources/config.yaml` to `.gitignore`
3. Provide a `resources/config.yaml.example` template

---

## Key Models Used

| Component | Model | Loaded By |
|-----------|-------|-----------|
| Semantic embedding | `mixedbread-ai/mxbai-embed-large-v1` | `SemanticFeatures` via `sentence_transformers` |
| Semantic embedding | `Qwen/Qwen3-Embedding-0.6B` | `SemanticFeatures` via `sentence_transformers` |
| Lexical tokenizer | `mixedbread-ai/mxbai-embed-large-v1` (tokenizer only) | `LexicalWeights` via `AutoTokenizer` |
| NLI | `FacebookAI/roberta-large-mnli` | `NLIWeights` via `AutoModelForSequenceClassification` |
| NER | `knowledgator/modern-gliner-bi-base-v1.0` | `EntityMatch` via `GLiNER` |
| Coref resolution | `gpt-4o-mini` (default) or `google/gemma-3-270m` | `EntityResolver` |

All HuggingFace models cache to subdirectories under their respective `Features/` or `Preprocess/` folders.
Semantic models use a class-level `_model_cache` dict — they are only loaded once per process.

---

## Data Format

```json
{
  "data": [
    { "text1": "...", "text2": "...", "label": 1 },
    { "text1": "...", "text2": "...", "label": 0 }
  ]
}
```

Labels: `1` = similar/faithful, `0` = not similar/unfaithful.

---

## Known Architectural Issues (see TASK.md for full list)

1. **Conv1D not Conv2D**: `model.py` flattens 64×64 maps to 1D before the CNN. The spatial structure of the n×m similarity matrix is discarded. Should use `Conv2D` on stacked `[F, 64, 64]` channel maps.

2. **Coref resolver overhead**: `split_txt()` creates a fresh `EntityResolver` (model load + API calls) on every invocation. In training, this means a GPT-4o-mini call per sentence per sample. Resolver should be instantiated once and passed in, or disabled during training.

3. **Lexical tokenizer reload**: `LexicalWeights.sp_tokenize()` calls `__load_model__()` on every call. Add `if not hasattr(self, 'tokenizer_cache')` guard.

4. **`getFeatureMap` calls `self.__init__()`**: Resets object state. Works only because model caches are set as extra attributes not in `__init__`. Fragile — prefer a `_reset_state()` method that clears only data, not model cache.

5. **64-sentence hard cap**: `pad_matrix` raises `ValueError` if either text has more than 64 sentences. Document this clearly; add a truncation or chunking strategy.

6. **Binary training**: BCE on 0/1 labels. For continuous faithfulness scoring, consider MSELoss on float labels in [0,1].

7. **Missing LCS feature**: Planned in design doc but not implemented.

---

## Coding Conventions

- Python 3.10+, standard library + requirements.txt
- Each feature extractor exposes a single public method: `getFeatureMap(list1, list2) -> dict[str, torch.Tensor[64,64]]`
- All feature matrices are padded to `64×64` before leaving the `Features/` layer
- Model inputs are flat float32 tensors; the dataset flattens and concatenates all feature maps
- Reports (training, test) are saved as both `.json` and `.md` to `training_reports/` and `test_reports/`
- Feature cache stored in `./cache/` as `{md5_hash}.json` — safe to delete to force recompute
- Do not hardcode absolute paths; use relative paths from project root or `Path(__file__).parent`

---

## Dependencies

```
transformers~=4.56.0
tqdm~=4.67.1
torch~=2.8.0
gliner~=0.2.22
PyYAML~=6.0.2
huggingface-hub~=0.34.4
openai~=1.107.2
sentence-transformers   (implied by SemanticFeatures)
sklearn                 (used in test.py)
```