# SilverBullet — LLM Evaluation Benchmark

<img src="assets/icon.png" alt="SilverBullet" width="100%" style="max-width:600px; height:auto;">

Real-time faithfulness and agreement scoring for LLM outputs.
Given two texts it returns a score in **[0, 1]** using a Conv2D model trained on 16 multi-signal sentence-pair feature maps.

| Evaluation mode | text1 | text2 | Score meaning |
|-----------------|-------|-------|---------------|
| `context-vs-generated` | Source context / RAG chunk | LLM-generated answer | Groundedness — is the answer supported by the context? |
| `reference-vs-generated` | Ground-truth reference answer | Generated answer | Faithfulness — does the answer match the reference? |
| `model-vs-model` | Model A output | Model B output | Agreement — do the models produce equivalent responses? |

Each mode has its own trained model checkpoint for best accuracy.

---

## How It Works

SilverBullet compares texts at the **sentence level**, not as whole-document embeddings.

```
Text pair  (text1, text2)
     │
     ▼
split_txt()  ──  sentence split + coreference resolution (GPT-4o-mini / local LLM)
     │
     ▼
Feature extractors  →  n×m matrices  →  pad to 64×64
  ├─ Semantic   mxbai-embed-large-v1 + Qwen3-Embedding-0.6B    →  6 maps
  ├─ Lexical    Jaccard · Dice · Cosine · ROUGE                 →  4 maps
  ├─ NLI        RoBERTa-large-MNLI entailment probs             →  3 maps
  ├─ Entity     GLiNER NER — entity count mismatch              →  1 map
  └─ LCS        token-level + char-level LCS similarity         →  2 maps
                                                       TOTAL: 16 channels
     │
     ▼
TextSimilarityCNN  [B, 16, 64, 64]
  Conv2d × 3 + MaxPool × 3  →  flatten  →  FC  →  sigmoid
     │
     ▼
Score ∈ [0, 1]
```

---

## Repository Layout

```
Silver-Bullet/
├── backend/                        # All Python backend code
│   ├── api/                        # FastAPI application
│   │   ├── main.py                 # App factory, endpoints, rate limiting
│   │   ├── schemas.py              # Pydantic request/response models
│   │   ├── dependencies.py         # Per-mode predictor singleton (lru_cache)
│   │   └── middleware.py           # RequestID + structured JSON logging
│   ├── Features/
│   │   ├── Semantic/               # 6 maps — embedding cosine + soft alignment
│   │   ├── Lexical/                # 4 maps — token overlap (SentencePiece)
│   │   ├── NLI/                    # 3 maps — entailment probabilities
│   │   ├── EntityGroups/           # 1 map  — GLiNER entity mismatch
│   │   └── LCS/                    # 2 maps — token + char LCS
│   ├── Splitter/
│   │   └── sentence_splitter.py    # split_txt() — regex split + coref per sentence
│   ├── Preprocess/coref/
│   │   └── resolveEntity.py        # EntityResolver — GPT-4o-mini or local Gemma
│   ├── Postprocess/
│   │   └── __addpad.py             # pad_matrix() — n×m → 64×64 (truncates at 64)
│   ├── resources/
│   │   ├── config.yaml             # API keys (gitignored — use env vars instead)
│   │   ├── config.yaml.example     # Template — commit this, not config.yaml
│   │   └── getConfig.py            # getVal(env) → dict
│   ├── model.py                    # TextSimilarityCNN (Conv2D)
│   ├── predict.py                  # SimilarityPredictor — single / batch / breakdown
│   ├── train.py                    # Training pipeline with early stopping + MSELoss
│   ├── test.py                     # Evaluation pipeline — metrics + TestReport
│   ├── feature_cache.py            # MD5-keyed JSON cache in ./cache/
│   ├── feature_registry.py         # FEATURE_KEYS + checkpoint manifest
│   ├── training_report.py          # Per-epoch JSON + Markdown reports
│   ├── generate_data.py            # Generates data/ + data/{mode}/ splits
│   └── example.py                  # Feature pipeline demo (no model)
├── frontend/                       # React + TypeScript + Vite UI
├── tests/                          # pytest API tests (mocked predictor)
├── data/                           # train.json, validate.json, test.json
│   ├── model-vs-model/             # Mode-specific splits
│   ├── reference-vs-generated/
│   └── context-vs-generated/
├── models/                         # Per-mode checkpoints (gitignored)
├── cache/                          # Feature cache (gitignored)
├── training_reports/               # Training reports (gitignored)
└── test_reports/                   # Test reports (gitignored)
```

---

## Installation

```bash
git clone https://github.com/Mehul-Gupta-SMH/Silver-Bullet.git
cd Silver-Bullet

python -m venv .sbvenv
# Windows
.sbvenv\Scripts\activate
# macOS / Linux
source .sbvenv/bin/activate

pip install -r requirements.txt
mkdir -p data models cache training_reports test_reports
```

### Configuration

Tokens are read from environment variables (recommended) or a YAML file.

```bash
# Recommended — environment variables
export SB_HF_TOKEN=hf_...
export SB_OPENAI_TOKEN=sk-...

# Alternative — YAML file (gitignored)
cp backend/resources/config.yaml.example backend/resources/config.yaml
# edit backend/resources/config.yaml
```

> **Security:** `backend/resources/config.yaml` is gitignored. Never commit real tokens.

---

## Usage

### Generate training data

```bash
python -m backend.generate_data
```

Creates `data/{train,validate,test}.json` (general) and per-mode splits under `data/{mode}/`.

#### Expand with external datasets (recommended)

Downloads and merges public NLP datasets (STS-B, MNLI, QQP, QNLI, HaluEval QA) to reach ~1 400 pairs per mode:

```bash
python -m backend.fetch_external_data
```

Options: `--max-per-source 400` (default), `--force` to re-download.

### Pre-compute features (recommended before training)

Extracts all 16 feature maps for every pair across all splits and caches them to disk. Subsequent training runs load from cache at ~95 pairs/sec instead of recomputing (~1–4 s/pair):

```bash
python -m backend.precompute_features
# --force  to recompute and overwrite existing cache entries
```

### Train

Train the general model or a mode-specific model:

```bash
# General model (fallback for all modes)
python -m backend.train

# Mode-specific models (recommended — better accuracy per use case)
python -m backend.train --mode model-vs-model
python -m backend.train --mode reference-vs-generated
python -m backend.train --mode context-vs-generated
```

Checkpoints are saved to:
- General: `best_model.pth`
- Per-mode: `models/{mode}.pth`

Feature cache is written to `./cache/` — delete it before retraining with a changed feature set.

If MLflow is running (see [Monitoring](#monitoring)), each run is automatically tracked and the best-weights model is registered in the Model Registry as `silverbullet-{mode}`.

### Evaluate

```bash
# General model
python -m backend.test

# Mode-specific model
python -m backend.test --mode context-vs-generated
```

Reports saved to `test_reports/` — accuracy, ROC-AUC, average precision, confusion matrix, per-sample predictions. If MLflow is running, test metrics are also logged as a separate run in the same experiment.

### Inference (CLI)

```bash
python -m backend.predict --model best_model.pth --input input.json --output predictions.json
```

Input format (either):
```json
[["text1", "text2"], ["text3", "text4"]]
```
```json
{ "data": [{ "text1": "...", "text2": "..." }] }
```

### Feature extraction demo

```bash
python -m backend.example
```

---

## API Server

```bash
uvicorn backend.api.main:app --reload
# Docs: http://localhost:8000/api/v1/docs
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Service health + per-mode model load state |
| POST | `/api/v1/predict/pair` | Single pair → score |
| POST | `/api/v1/predict/pair/breakdown` | Single pair → sentence-level divergence analysis |
| POST | `/api/v1/predict/batch` | Up to 100 pairs → scores |
| POST | `/api/v1/predict/batch/breakdown` | Up to 10 pairs → breakdown (expensive) |

All prediction endpoints accept a `mode` field (default: `context-vs-generated`):

```bash
curl -X POST http://localhost:8000/api/v1/predict/pair \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "The sky is blue.",
    "text2": "The sky appears blue in colour.",
    "mode": "context-vs-generated"
  }'
```

Response:
```json
{ "prediction": 1, "probability": 0.91 }
```

Rate limits: 60 req/min global · 20/min on breakdown · 10/min on batch breakdown.

### Frontend

```bash
cd frontend && npm install && npm run dev
# http://localhost:5173  (proxies /api → http://localhost:8000)
```

---

## Monitoring

SilverBullet integrates with MLflow, Prometheus, and Grafana for training observability.

### MLflow

Start the tracking server before (or during) training — epoch metrics are buffered and replayed automatically on late connect:

```bash
mlflow server \
  --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow/mlflow.db \
  --default-artifact-root ./mlflow/artifacts
# UI: http://127.0.0.1:5000
```

**What gets tracked per training run:**
- Hyperparameters (epochs, lr, batch size, optimizer, num_features)
- Per-epoch: `train_loss`, `val_loss`, `accuracy`
- Final: `best_val_loss`, `best_epoch`
- Checkpoint artifacts (final + best `.pth` files)
- **Model Registry**: best-weights model registered as `silverbullet-{mode}` (version increments on each retrain)

**Test runs** (`python -m backend.test --mode ...`) log `test_accuracy`, `test_roc_auc`, `test_avg_precision` as a separate run in the same experiment.

### Prometheus + Grafana (optional, real-time)

Live per-epoch metrics pushed to Prometheus pushgateway during training:

```bash
cp .env.example .env
docker compose up prometheus pushgateway grafana
# Grafana: http://localhost:3000  (admin/admin)
```

The provisioned dashboard (`monitoring/grafana/dashboards/training.json`) shows train/val loss curves and accuracy, filterable by mode and run.

---

## Docker

```bash
docker build -t silver-bullet .
docker run -p 8000:8000 \
  -e SB_HF_TOKEN=hf_... \
  -e SB_OPENAI_TOKEN=sk-... \
  silver-bullet
```

Or with Docker Compose:

```bash
cp .env.example .env   # fill in tokens
docker compose up
```

A Docker image is automatically published to GitHub Container Registry on every push to `master`:

```bash
docker pull ghcr.io/mehul-gupta-smh/silver-bullet:latest
```

---

## Model Architecture

`TextSimilarityCNN` takes stacked feature maps as `[batch, 16, 64, 64]` input:

```
[B, 16, 64, 64]
  Conv2d(16→128) + BN + ReLU + MaxPool   →  [B, 128, 32, 32]
  Conv2d(128→64) + BN + ReLU + MaxPool   →  [B,  64, 16, 16]
  Conv2d(64→32)  + BN + ReLU + MaxPool   →  [B,  32,  8,  8]
  Flatten                                 →  [B, 2048]
  FC(2048→128)   + BN + ReLU + Dropout
  FC(128→1)      + sigmoid               →  score ∈ [0, 1]
```

The 2D spatial structure of the sentence-pair similarity matrices is preserved through the convolutional layers, allowing the model to learn alignment patterns directly from the feature maps.

Training uses **MSELoss** (supports continuous float faithfulness labels as well as binary 0/1).

### Checkpoints

Every checkpoint embeds `num_features` and a feature manifest so the model can be reloaded and validated without re-running the pipeline.

| File | Created by | Used by |
|------|-----------|---------|
| `best_model.pth` | `python -m backend.train` | fallback for all modes |
| `models/model-vs-model.pth` | `train --mode model-vs-model` | `mode=model-vs-model` requests |
| `models/reference-vs-generated.pth` | `train --mode reference-vs-generated` | `mode=reference-vs-generated` requests |
| `models/context-vs-generated.pth` | `train --mode context-vs-generated` | `mode=context-vs-generated` requests |

---

## Adding a New Feature Extractor

1. Create `backend/Features/{Name}/get{Name}Weights.py`
2. Implement `getFeatureMap(list1, list2) -> dict[str, torch.Tensor[64,64]]` using the `_reset_state()` pattern (see `AGENT.md`)
3. Register the new feature keys in `backend/feature_registry.py` → `FEATURE_KEYS`
4. Add the extractor to `backend/train.py` and `backend/example.py`
5. Delete `./cache/` and retrain (new features change `num_features` and the manifest)

---

## Tests

```bash
pytest tests/ -v
```

25 API tests covering health, prediction endpoints, validation, rate-limit handler, and CORS. All tests use a mocked predictor — no model file required.

---

## CI / CD

| Workflow | Trigger | Action |
|----------|---------|--------|
| CI (`.github/workflows/ci.yml`) | Every push / PR | lint → api-tests → frontend type-check + vitest |
| CD (`.github/workflows/cd.yml`) | Push to `master` | Build + push Docker image to GHCR |

---

## License

Specify your license here.

---

*For questions or contributions, open an issue or pull request.*
