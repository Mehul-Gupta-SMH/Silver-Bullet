# SilverBullet — Claude Code Instructions

## What This Project Is

SilverBullet is a **real-time LLM evaluation benchmark**.
Given two texts, it produces a faithfulness / agreement score in [0, 1].

**Primary use cases:**
- **Hallucination detection**: `text1 = source context`, `text2 = LLM answer` — is the answer grounded?
- **Model agreement scoring**: `text1 = GPT-4o output`, `text2 = Claude output` — do models agree?
- **Faithfulness evaluation**: `text1 = ground-truth reference`, `text2 = generated answer` — is it faithful?
- **RAG groundedness**: verify every claim in a generated answer is supported by retrieved documents

**Closest analogue:** cross-encoder / re-ranker with explicit multi-signal feature engineering before the learned head — designed for near-real-time inference in LLM pipelines.

---

## Repository Layout

```
Silver-Bullet/
├── data/                         # train.json, validate.json, test.json
│   └── *.json                    # {"data": [{"text1":..., "text2":..., "label": 0|1}]}
│
├── generate_data.py              # Generates data/ from hardcoded positive/hard-neg/soft-neg pairs
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
│   ├── EntityGroups/
│   │   └── getOverlap.py                     # EntityMatch.getFeatureMap() -> 1 padded 64×64 tensor
│   └── LCS/
│       └── getLCSweights.py                  # LCSWeights.getFeatureMap() -> 2 padded 64×64 tensors
│
├── Postprocess/
│   ├── __addpad.py               # pad_matrix(mat, target_size=64) -> torch.Tensor [64,64]
│   └── postprocess.py            # thin wrapper around pad_matrix
│
├── feature_cache.py              # FeatureCache — MD5-keyed JSON cache in ./cache/
├── model.py                      # TextSimilarityCNN (Conv2D) + TextSimilarityCNNLegacy (Conv1D compat)
├── train.py                      # TextSimilarityDataset + train_model() entry point
├── test.py                       # TestReport + test_model() entry point
├── predict.py                    # SimilarityPredictor — batch/single/breakdown inference
├── training_report.py            # TrainingReport — per-epoch JSON + Markdown reports
├── example.py                    # Manual feature pipeline demo (no model)
│
├── api/
│   ├── main.py                   # FastAPI app — /api/v1/health, /predict/pair, /predict/pair/breakdown, /predict/batch
│   ├── schemas.py                # Pydantic models: PairRequest/Response, BatchRequest/Response, BreakdownResponse
│   ├── dependencies.py           # get_predictor() singleton via @lru_cache
│   └── middleware.py             # RequestIDMiddleware + LoggingMiddleware (structured JSON logs)
│
├── frontend/
│   ├── index.html                # Title: "SilverBullet — LLM Evaluation Benchmark"
│   ├── public/favicon.svg        # Violet bullseye SVG favicon
│   └── src/
│       ├── App.tsx               # Shell: header, tabs (Single Eval / Batch Eval / Experiments), footer
│       ├── config/modes.ts       # 3 evaluation modes with interpret() + descriptions
│       ├── types/index.ts        # PredictionResult, BreakdownResult, BatchResponse, HealthResponse
│       ├── services/api.ts       # predictPair(), predictPairBreakdown(), predictBatch(), healthCheck()
│       ├── components/
│       │   ├── PairScorer.tsx    # Single eval UI — textareas, score, "Drill Down" button
│       │   ├── BreakdownPanel.tsx # Divergence analysis — sentence colour map + feature bars
│       │   ├── BatchScorer.tsx   # Batch eval UI + distribution chart
│       │   ├── ScoreGauge.tsx    # Score bar with prediction badge
│       │   ├── ComparisonModeSelector.tsx  # Evaluation mode picker
│       │   ├── FeaturePanel.tsx  # Collapsible feature family reference (5 families, 16 maps)
│       │   ├── ExperimentsPanel.tsx        # Saved experiments with re-run
│       │   ├── TestCasePanel.tsx           # Pre-built test case library
│       │   └── SaveExperimentForm.tsx      # Name + notes form
│       └── data/testCases.ts     # 17 pair + 2 batch test cases incl. pipeline examples (ex-1, ex-2)
│
├── tests/
│   ├── conftest.py               # pytest fixtures — mock predictor, TestClient
│   └── test_api.py               # API tests: health, pair, batch, breakdown, CORS, validation
│
├── best_model.pth                # Saved best checkpoint
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
    ├─ EntityMatch.getFeatureMap()
    │    GLiNER NER              → EntityMismatch                    (1 map)
    └─ LCSWeights.getFeatureMap()
         built-in DP             → lcs_token, lcs_char              (2 maps)
                                                           TOTAL: 16 maps
         │
         ▼
  pad_matrix() — each n×m → 64×64 zero-padded tensor
  stack all maps → [F=16, 64, 64] tensor
         │
         ▼
  TextSimilarityCNN (model.py)
    Conv2d: [F, 64, 64] → [128, 32, 32] → [64, 16, 16] → [32, 8, 8]
    flatten → [batch, 2048]
    FC:  2048 → 128 → 1
    sigmoid → score ∈ [0, 1]
```

> `num_features` (F) is serialised into every checkpoint so `predict.py` can reconstruct
> the model without re-running the pipeline. Do not hardcode it.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Service health + model load state |
| POST | `/api/v1/predict/pair` | Single pair → `{prediction, probability}` |
| POST | `/api/v1/predict/pair/breakdown` | Single pair → full divergence analysis (slow — reruns pipeline) |
| POST | `/api/v1/predict/batch` | Up to 100 pairs → list of scores |

Docs: `http://localhost:8000/api/v1/docs`

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
    "Semantic (Qwen3)": 0.81,
    "Lexical ROUGE": 0.72,
    "Lexical Jaccard": 0.61,
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

# Generate training data
python generate_data.py

# Train
python train.py

# Test
python test.py

# Predict (batch)
python predict.py --model best_model.pth --input data/test.json --output predictions.json

# API server
uvicorn api.main:app --reload
# Docs at http://localhost:8000/api/v1/docs

# Frontend (separate terminal)
cd frontend && npm run dev
# Proxies /api → http://localhost:8000
```

---

## Training Data (`generate_data.py`)

Three pair categories, 194 pairs total (70/15/15 split):

| Category | Count | Description |
|----------|-------|-------------|
| `positive` (label=1) | ~104 | Paraphrases, faithful summaries, equivalent code — 14 domains |
| `hard_neg` (label=0) | ~54 | Same topic/structure but wrong numbers, negated claims, hallucinated additions |
| `soft_neg` (label=0) | ~36 | Clearly unrelated domain pairs |

To regenerate: `python generate_data.py` → overwrites `data/*.json`.
Delete `./cache/` before retraining if feature extractors changed.

---

## Environment & Config

```python
from resources.getConfig import getVal
config = getVal(env='DEVELOPMENT')   # keys: url, api_key, hf_token, openai_token
```

**IMPORTANT — Security:** `config.yaml` contains hardcoded tokens. Do NOT commit changes to it.
Fix: load from `os.environ`; add to `.gitignore`; provide `config.yaml.example`.

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

All HuggingFace models cache to subdirectories under their respective `Features/` folders.
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

---

## Coding Conventions

- Python 3.10+, standard library + requirements.txt
- Each feature extractor: `getFeatureMap(list1, list2) -> dict[str, torch.Tensor[64,64]]`
- All feature matrices padded to `64×64` before leaving `Features/` layer
- Model inputs are `[batch, F, 64, 64]` float32 tensors stacked by `feature_map_to_tensor()` in `train.py`
- Reports saved as `.json` + `.md` to `training_reports/` and `test_reports/`
- Feature cache in `./cache/{md5}.json` — safe to delete to force recompute
- No hardcoded absolute paths; use relative paths or `Path(__file__).parent`

---

## Known Issues (see TASK.md for full list)

1. **Coref resolver overhead**: `split_txt()` creates a fresh `EntityResolver` per call (model load + API). Disable during training or pass a shared resolver instance.
2. **64-sentence cap**: `pad_matrix` silently truncates inputs exceeding 64 sentences (fixed from hard crash).
3. **Binary training only**: BCE on 0/1 labels — continuous float faithfulness labels not yet supported.
4. **config.yaml secrets**: Live API keys committed to repo. Move to env vars.

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
fastapi
uvicorn
slowapi
sentence-transformers
sklearn
```
