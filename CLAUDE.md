# SilverBullet — Claude Code Instructions

## What This Project Is

SilverBullet is a **real-time LLM evaluation benchmark**.
Given two texts and an evaluation mode, it produces a faithfulness / agreement score in [0, 1].

**Evaluation modes (each has its own trained model):**
| Mode ID | text1 | text2 | Use case |
|---------|-------|-------|---------|
| `context-vs-generated` | Source context / RAG chunk | LLM answer | Hallucination detection, RAG groundedness |
| `reference-vs-generated` | Ground-truth reference | Generated answer | Faithfulness evaluation |
| `model-vs-model` | Model A output | Model B output | Model agreement scoring |

**Closest analogue:** cross-encoder / re-ranker with explicit multi-signal feature engineering before the learned head — designed for near-real-time inference in LLM pipelines.

---

## Repository Layout

```
Silver-Bullet/
├── backend/                          # ALL Python backend code
│   ├── __init__.py
│   ├── api/
│   │   ├── main.py                   # FastAPI app — endpoints, rate limiting, CORS, lifespan
│   │   ├── schemas.py                # Pydantic models — EvaluationMode literal + mode field on all requests
│   │   ├── dependencies.py           # get_predictor(mode) — lru_cache(maxsize=3), per-mode checkpoint resolution
│   │   └── middleware.py             # RequestIDMiddleware + LoggingMiddleware (structured JSON logs)
│   │
│   ├── Features/
│   │   ├── Semantic/
│   │   │   ├── __generate_semantic_features.py   # SemanticFeatures — SentenceTransformer encoding
│   │   │   └── getSemanticWeights.py             # SemanticWeights.getFeatureMap() → 6 padded 64×64 tensors
│   │   ├── Lexical/
│   │   │   └── getLexicalWeights.py              # LexicalWeights.getFeatureMap() → 4 padded 64×64 tensors
│   │   ├── NLI/
│   │   │   └── getNLIweights.py                  # NLIWeights.getFeatureMap() → 3 padded 64×64 tensors
│   │   ├── EntityGroups/
│   │   │   └── getOverlap.py                     # EntityMatch.getFeatureMap() → 1 padded 64×64 tensor
│   │   └── LCS/
│   │       └── getLCSweights.py                  # LCSWeights.getFeatureMap() → 2 padded 64×64 tensors
│   │
│   ├── Splitter/
│   │   └── sentence_splitter.py      # split_txt(text) → List[str]  (regex split + coref per sentence)
│   │
│   ├── Preprocess/
│   │   └── coref/
│   │       └── resolveEntity.py      # EntityResolver — pronoun → named entity via GPT-4o-mini or local LLM
│   │
│   ├── Postprocess/
│   │   ├── __addpad.py               # pad_matrix(mat, target_size=64) → torch.Tensor [64,64]
│   │   └── postprocess.py            # thin wrapper around pad_matrix
│   │
│   ├── resources/
│   │   ├── config.yaml               # API keys (gitignored — use env vars)
│   │   ├── config.yaml.example       # Template
│   │   └── getConfig.py              # getVal(env='DEVELOPMENT') → dict
│   │
│   ├── model.py                      # TextSimilarityCNN (Conv2D) + TextSimilarityCNNLegacy (Conv1D compat)
│   ├── train.py                      # TextSimilarityDataset + train_model(); --mode arg
│   ├── test.py                       # TestReport + test_model(); --mode arg
│   ├── predict.py                    # SimilarityPredictor — single/batch/breakdown inference
│   ├── feature_cache.py              # FeatureCache — MD5-keyed JSON cache in ./cache/
│   ├── feature_registry.py           # FEATURE_KEYS (16 ordered keys) + build_manifest() + validate_manifest()
│   ├── training_report.py            # TrainingReport — per-epoch JSON + Markdown reports
│   ├── generate_data.py              # Generates data/ + data/{mode}/ splits
│   └── example.py                    # Manual feature pipeline demo (no model)
│
├── frontend/                         # React + TypeScript + Vite UI
│   ├── src/
│   │   ├── App.tsx                   # Shell: header, tabs (Single Eval / Batch Eval / Experiments)
│   │   ├── config/modes.ts           # 3 evaluation modes with interpret() + descriptions
│   │   ├── types/index.ts            # ComparisonMode, PredictionResult, BreakdownResult, HealthResponse
│   │   ├── services/api.ts           # predictPair(t1,t2,mode), predictBatch(pairs,mode), healthCheck()
│   │   └── components/               # PairScorer, BreakdownPanel, BatchScorer, ScoreGauge, …
│   └── public/favicon.svg
│
├── tests/
│   ├── conftest.py                   # pytest fixtures — mock predictor, TestClient
│   └── test_api.py                   # 25 API tests: health, pair, batch, breakdown, mode validation, CORS
│
├── data/                             # Train/validate/test splits
│   ├── train.json                    # General dataset (194 pairs)
│   ├── validate.json
│   ├── test.json
│   ├── model-vs-model/               # Mode-specific splits (209 pairs each)
│   ├── reference-vs-generated/
│   └── context-vs-generated/
│
├── models/                           # Per-mode checkpoints (gitignored)
│   ├── model-vs-model.pth
│   ├── reference-vs-generated.pth
│   └── context-vs-generated.pth
│
├── best_model.pth                    # General fallback checkpoint
├── requirements.txt
├── pyproject.toml
└── .sbvenv/                          # Local virtualenv (gitignored)
```

---

## Full Data Flow

```
Raw text pair (text1, text2)  +  mode
         │
         ▼
  API: get_predictor(mode)  →  loads models/{mode}.pth (or best_model.pth fallback)
         │
         ▼
  split_txt()  ←  backend/Splitter/sentence_splitter.py
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
    ├─ EntityMatch.getFeatureMap()
    │    GLiNER NER              → EntityMismatch                    (1 map)
    └─ LCSWeights.getFeatureMap()
         built-in DP             → lcs_token, lcs_char              (2 maps)
                                                           TOTAL: 16 maps
         │
         ▼
  pad_matrix() — each n×m → 64×64 zero-padded tensor
  stack all maps → [F=16, 64, 64] tensor (canonical order from FEATURE_KEYS)
         │
         ▼
  TextSimilarityCNN (backend/model.py)
    Conv2d: [F, 64, 64] → [128, 32, 32] → [64, 16, 16] → [32, 8, 8]
    flatten → [batch, 2048]
    FC:  2048 → 128 → 1
    sigmoid → score ∈ [0, 1]
```

> `num_features` (F) and the feature manifest are serialised into every checkpoint.
> `predict.py` validates the manifest on load. Do not hardcode `num_features`.

---

## API Endpoints

| Method | Path | Rate limit | Description |
|--------|------|-----------|-------------|
| GET | `/api/v1/health` | — | Service health + per-mode model load state |
| POST | `/api/v1/predict/pair` | 60/min | Single pair → `{prediction, probability}` |
| POST | `/api/v1/predict/pair/breakdown` | 20/min | Single pair → full divergence analysis |
| POST | `/api/v1/predict/batch` | 60/min | Up to 100 pairs → list of scores |
| POST | `/api/v1/predict/batch/breakdown` | 10/min | Up to 10 pairs → breakdown (expensive) |

Docs: `http://localhost:8000/api/v1/docs`

All prediction endpoints require a `mode` field (default: `context-vs-generated`):

```json
{ "text1": "...", "text2": "...", "mode": "context-vs-generated" }
```

### Health response shape
```json
{
  "status": "ok",
  "model_loaded": true,
  "models": {
    "model-vs-model": true,
    "reference-vs-generated": false,
    "context-vs-generated": true
  }
}
```

### Breakdown response shape
```json
{
  "prediction": 0,
  "probability": 0.32,
  "sentences1": ["sent A", "sent B"],
  "sentences2": ["sent X", "sent Y", "sent Z"],
  "alignment": [[0.9, 0.3, 0.1], [0.2, 0.8, 0.4]],
  "divergent_in_1": [],
  "divergent_in_2": [2],
  "feature_scores": {
    "Semantic (mxbai)": 0.85,
    "NLI Entailment": 0.55,
    "LCS Token": 0.68
  }
}
```

---

## Running the Project

```bash
# Activate virtualenv (Windows)
.\.sbvenv\Scripts\activate

# Generate training data (general + per-mode splits)
python -m backend.generate_data

# Train — general model
python -m backend.train

# Train — mode-specific models
python -m backend.train --mode model-vs-model
python -m backend.train --mode reference-vs-generated
python -m backend.train --mode context-vs-generated

# Evaluate
python -m backend.test
python -m backend.test --mode context-vs-generated

# Predict (batch CLI)
python -m backend.predict --model best_model.pth --input data/test.json --output predictions.json

# API server
uvicorn backend.api.main:app --reload
# Docs at http://localhost:8000/api/v1/docs

# Run tests
pytest tests/ -v

# Frontend (separate terminal)
cd frontend && npm run dev
# Proxies /api → http://localhost:8000
```

---

## Training Data (`backend/generate_data.py`)

General dataset: 194 pairs (70/15/15 split):

| Category | Count | Description |
|----------|-------|-------------|
| `positive` (label=1) | ~104 | Paraphrases, faithful summaries, equivalent code — 14 domains |
| `hard_neg` (label=0) | ~54 | Same topic/structure but wrong numbers, negated claims, hallucinated additions |
| `soft_neg` (label=0) | ~36 | Clearly unrelated domain pairs |

Per-mode datasets: 209 pairs each (general + 15 mode-specific hard negatives):

| Mode | Hard negative characteristics |
|------|-------------------------------|
| `model-vs-model` | Models diverge on key facts, contradictory conclusions |
| `reference-vs-generated` | Generated answer adds wrong facts or wrong numbers |
| `context-vs-generated` | Answer introduces entities/claims not in context |

To regenerate: `python -m backend.generate_data` → overwrites `data/` and `data/{mode}/`.
Delete `./cache/` before retraining if feature extractors changed.

---

## Environment & Config

```python
from backend.resources.getConfig import getVal
config = getVal(env='DEVELOPMENT')   # keys: url, api_key, hf_token, openai_token
```

Preferred: set `SB_HF_TOKEN` and `SB_OPENAI_TOKEN` as environment variables.
`backend/resources/config.yaml` is gitignored. Never commit real tokens.

Per-mode model checkpoint env vars:
- `MODEL_PATH_MODEL_VS_MODEL`
- `MODEL_PATH_REFERENCE_VS_GENERATED`
- `MODEL_PATH_CONTEXT_VS_GENERATED`
- `MODEL_PATH` (general fallback)

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

All HuggingFace models cache to subdirectories under `backend/Features/`.
Semantic models use a class-level `_model_cache` — loaded once per process.

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

Labels: `1` = similar/faithful/grounded, `0` = different/unfaithful/hallucinated.
MSELoss is used during training — continuous float labels (e.g. 0.7) are also supported.

---

## Coding Conventions

- Python 3.10+, standard library + requirements.txt
- All backend code lives under `backend/` — import as `from backend.X import Y`
- Each feature extractor: `getFeatureMap(list1, list2) -> dict[str, torch.Tensor[64,64]]`
- All feature matrices padded to `64×64` before leaving `Features/` layer
- Model inputs are `[batch, F, 64, 64]` float32 tensors stacked by `feature_map_to_tensor()` in `backend/train.py`
- Feature keys must match `FEATURE_KEYS` in `backend/feature_registry.py` exactly
- Reports saved as `.json` + `.md` to `training_reports/` and `test_reports/`
- Feature cache in `./cache/{md5}.json` — safe to delete to force recompute
- No hardcoded absolute paths; use relative paths or `Path(__file__).parent`

---

## Known Issues

1. **Coref resolver overhead**: `split_txt()` creates a fresh `EntityResolver` per call. Disable during training or pass a shared resolver instance.
2. **64-sentence cap**: `pad_matrix` silently truncates inputs exceeding 64 sentences.
3. **config.yaml secrets**: Use env vars (`SB_HF_TOKEN`, `SB_OPENAI_TOKEN`) — never commit the YAML file.

---

## Dependencies (pinned in requirements.txt and pyproject.toml)

```
torch==2.8.0
transformers==4.51.0        # pinned — gliner==0.2.22 requires <=4.51.0
sentence-transformers==5.1.0
gliner==0.2.22
fastapi==0.135.1
uvicorn==0.41.0
slowapi==0.1.9
pydantic==2.11.9
scikit-learn==1.7.1
numpy==2.3.2
openai==1.107.2
PyYAML==6.0.2
```
