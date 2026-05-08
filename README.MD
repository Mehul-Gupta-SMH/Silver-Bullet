# SilverBullet — LLM Evaluation Benchmark

<img src="assets/icon.png" alt="SilverBullet" width="100%" style="max-width:600px; height:auto;">

Real-time faithfulness and agreement scoring for LLM outputs.
Given two texts it returns a score in **[0, 1]** using a Conv2D model trained on 21–23 multi-signal sentence-pair feature maps (v5.8).

| Evaluation mode | text1 | text2 | Score meaning |
|-----------------|-------|-------|---------------|
| `context-vs-generated` | Source context / RAG chunk | LLM-generated answer | Groundedness — is the answer supported by the context? |
| `reference-vs-generated` | Ground-truth reference answer | Generated answer | Faithfulness — does the answer match the reference? |
| `model-vs-model` | Model A output | Model B output | Agreement — do the models produce equivalent responses? |

Each mode has its own trained model checkpoint for best accuracy.

---

## Demos

### Model-vs-Model — Pair Scoring + Drill-Down

Scores a high-agreement pair (Python overview) then a divergent pair (Eiffel Tower hallucinations), opens the drill-down panel, and walks through sentence alignment, per-signal feature score bars, and severity-coded misalignment diagnostics.

<video src="docs/videos/demo-mvm-pair-drilldown.webm" width="100%" controls muted></video>

### Model-vs-Model — Batch Evaluation

Uploads a 3-pair fixture file, runs batch scoring, reviews the score-distribution chart and results table, then saves the experiment.

<video src="docs/videos/demo-mvm-batch.webm" width="100%" controls muted></video>

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
Feature extractors  →  n×m matrices  →  bilinear resize to 32×32
  ├─ Semantic   mxbai-embed-large-v1 + Qwen3-Embedding-0.6B    →  4 maps  [batched, SQLite embedding cache]
  ├─ Lexical    Jaccard · Dice · ROUGE · ROUGE-3                →  4 maps  [batch tokenise + parallel rows]
  ├─ NLI        RoBERTa-large-MNLI entailment probs             →  3 maps  [batched, SQLite pair cache]
  ├─ Entity     GLiNER NER — type-count + per-type value overlap →  up to 9 maps  [batch NER + Bonferroni pruning, SQLite cache]
  ├─ LCS        token-level + char-level LCS similarity         →  2 maps  [parallel rows]
  ├─ Numeric    normalised numeric token Jaccard                →  1 map   [pure regex, no model]
  ├─ Relations  entity grounding recall (GLiNER, shared cache)  →  1 map   [shares EntityGroups model]
  ├─ SVO        spaCy dep-tree subject-verb-object triplet recall →  1 map  [~100× faster than Relex]
  └─ EFG        DeBERTa-v3-base-mnli-fever-anli factual grounding →  3 maps [supports, refutes, factual_delta; v5.8 new]
                                      TOTAL: 21–23 channels (mode-specific basket, v5.8)
     │
     ▼
TextSimilarityCNN  [B, F, 32, 32]  (F = 21–23, mode-specific, v5.8)
  Conv2d(F→128)  + BN + ReLU + MaxPool  →  [B, 128, 16, 16]
  Conv2d(128→64) + BN + ReLU + MaxPool  →  [B,  64,  8,  8]
  Conv2d(64→32)  + BN + ReLU + MaxPool  →  [B,  32,  4,  4]
  Flatten → FC(512→128) + BN + Dropout → FC(128→1) + sigmoid
     │
     ▼
Score ∈ [0, 1]
```

> **Bilinear resize vs zero-padding:** Each n×m matrix is resized to 32×32 using bilinear interpolation so every cell carries a real signal regardless of input length. Zero-padding (the previous approach) produced ~99% sparse maps for short texts, causing the CNN to deflate scores for short-but-similar inputs.

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
│   │   ├── Semantic/               # 4 maps — embedding cosine + soft alignment
│   │   ├── Lexical/                # 4 maps — token overlap (SentencePiece)
│   │   ├── NLI/                    # 3 maps — entailment probabilities
│   │   ├── EntityGroups/           # up to 9 maps — GLiNER NER mismatch + per-type value overlap
│   │   ├── LCS/                    # 2 maps — token + char LCS
│   │   ├── Numeric/                # 1 map  — normalised numeric token Jaccard
│   │   ├── Relations/              # 1 map  — entity grounding recall (GLiNER shared cache)
│   │   ├── SVO/                    # 1 map  — spaCy dep-tree SVO triplet recall
│   │   └── Factual/                # 3 maps — DeBERTa-v3 factual grounding (v5.8 new)
│   ├── Splitter/
│   │   └── sentence_splitter.py    # split_txt() — regex split + coref per sentence
│   ├── Preprocess/coref/
│   │   └── resolveEntity.py        # EntityResolver — GPT-4o-mini or local Gemma
│   ├── Postprocess/
│   │   └── __addpad.py             # resize_matrix() — bilinear resize n×m → 32×32
│   ├── resources/
│   │   ├── config.yaml             # API keys (gitignored — use env vars instead)
│   │   ├── config.yaml.example     # Template — commit this, not config.yaml
│   │   └── getConfig.py            # getVal(env) → dict
│   ├── model.py                    # TextSimilarityCNN (Conv2D)
│   ├── predict.py                  # SimilarityPredictor — single / batch / breakdown
│   ├── train.py                    # Training pipeline with early stopping + MSELoss
│   ├── test.py                     # Evaluation pipeline — metrics + TestReport
│   ├── cache_db.py                 # SQLite-backed unified cache (cache/silverbullet.db)
   ├── feature_cache.py            # MD5-keyed feature cache (wraps CacheDB)
│   ├── feature_registry.py         # FEATURE_KEYS + checkpoint manifest
│   ├── training_report.py          # Per-epoch JSON + Markdown reports
│   ├── generate_data.py            # Generates data/ + data/{mode}/ splits
│   ├── fetch_external_data.py      # Downloads STS-B/MNLI/RAGTruth/MedHallu/etc.
│   ├── augment_data.py             # Adversarial hard-negative augmentation (numeric/entity swap)
│   ├── precompute_features.py      # Batch feature cache pre-fill before training
│   └── example.py                  # Feature pipeline demo (no model)
├── analysis_reports/               # Feature pattern analysis (confident correct vs wrong)
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

Downloads and merges public NLP datasets to reach ~6 000–10 000 pairs per mode:

```bash
python -m backend.fetch_external_data
```

Options: `--max-per-source 400` (default), `--force` to re-download.

#### Add adversarial hard negatives (recommended)

Generates numeric-swap and entity-swap hard negatives from existing label=1 pairs and appends them to the training splits. Targets entity substitution and numeric hallucination failure modes identified in feature analysis:

```bash
python -m backend.augment_data          # +200 pairs per mode (default)
python -m backend.augment_data --per-mode 300 --dry-run   # preview without writing
```

### Pre-compute features (recommended before training)

Extracts all feature maps for every pair across all splits and caches them to disk. Subsequent training runs load from cache; new features added later are computed incrementally (only missing keys are recomputed and merged — no need to clear the cache):

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
- Per-mode: `models/{mode}/best.pth` (active) + `models/{mode}/{ts}_best.pth` / `{ts}_final.pth` (archived per run)

Feature cache is written to `./cache/`. The cache is **incremental** — adding a new feature extractor does not require clearing the cache. The training pipeline detects missing keys per entry and recomputes only those, then merges and saves the updated entry.

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

`TextSimilarityCNN` takes stacked feature maps as `[batch, F, 32, 32]` input (F = 21–23, mode-specific):

```
[B, F, 32, 32]  (F = 21–23, mode-specific basket, v5.8)
  Conv2d(F→128)  + BN + ReLU + MaxPool   →  [B, 128, 16, 16]
  Conv2d(128→64) + BN + ReLU + MaxPool   →  [B,  64,  8,  8]
  Conv2d(64→32)  + BN + ReLU + MaxPool   →  [B,  32,  4,  4]
  Flatten                                 →  [B, 512]
  FC(512→128)    + BN + ReLU + Dropout
  FC(128→1)      + sigmoid               →  score ∈ [0, 1]
```

The 2D spatial structure of the sentence-pair similarity matrices is preserved through the convolutional layers, allowing the model to learn alignment patterns directly from the feature maps.

Training uses **MSELoss** (supports continuous float faithfulness labels as well as binary 0/1).

---

## Model Performance

**v5.8** (feature pipeline with EFG, 21–23 channels per mode):

| Mode | Test ROC-AUC | Accuracy | F1 | MCC | Feature count | Train pairs |
|------|-------------|----------|----|-----|--------------|-------------|
| `context-vs-generated` | 0.8130 | 72% | 0.723 | 0.448 | 21 | ~7,941 |
| `reference-vs-generated` | 0.9283 | 85% | 0.854 | 0.700 | 22 | ~9,676 |
| `model-vs-model` | 0.8969 | 81% | 0.816 | 0.612 | 23 | ~7,000 |

**v5.7** (final, 19–20 channels per mode): CVG 0.8053/71.98%, RVG 0.9001/81.54%, MVM 0.8919/81.66%

v5.6 baseline (for reference): CVG 0.8203/74.1%, RVG 0.8907/81.4%, MVM 0.8943/82.9%

**v5.8 changes:**
- Added EFG (External Factual Grounding) via `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` → 3 new feature maps (supports, refutes, factual_delta)
- Dropped low-SHAP features (SHAP-driven pruning): entity_product_value_prec (CVG), entity_percentage (RVG+MVM) — near-zero importance
- Total global features: 39 (was 36)

Key findings from SHAP feature analysis:
- **NLI neutral score** is the single most discriminative feature across all modes
- **Entity value precision/recall** is the strongest signal for RVG failures (entity substitution)
- **Numeric Jaccard** catches hallucinations where topic matches but numbers differ
- **SVO triplet recall** added in v5.6 to capture verb-grounded factual claim mismatches

All three mode-specific checkpoints are tracked in MLflow and registered in the Model Registry as `silverbullet-{mode}`.

---

## Benchmark & Roadmap

### External benchmark results (v5.8, RVG mode)

Evaluated on four held-out human-annotated faithfulness benchmarks — none used during training:

| Benchmark | n | ROC-AUC | PR-AUC | Pearson r | Spearman ρ | Notes |
|-----------|---|---------|--------|-----------|------------|-------|
| [FRANK](https://huggingface.co/datasets/mtc/frank-test-set-with-faithfulness-annotation) | 1,000 | **0.8137** | 0.7615 | — | — | Structured factual errors in summarisation |
| [AggreFact](https://huggingface.co/datasets/NinaCalvi/llm_aggrefact_pre_aug9) | 1,000 | 0.7310 | 0.7025 | 0.389 | 0.400 | Aggregated LLM faithfulness labels |
| [FactCC](https://huggingface.co/datasets/pminervini/FactCC) | 268 | 0.7302 | **0.9416** | — | — | Claim-level factual consistency |
| [SummEval](https://huggingface.co/datasets/mteb/summeval) | 654 | 0.6905 | 0.8500 | 0.293 | 0.280 | Human quality ratings (continuous, harder target) |

All four benchmarks use `reference-vs-generated` mode. SummEval's lower ROC-AUC reflects its continuous human opinion scores rather than clean binary faithfulness labels.

### Planned: Re-ranker baseline comparison

The next evaluation milestone is benchmarking SilverBullet against a cross-encoder re-ranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` or `cross-encoder/nli-deberta-v3-base`) on the task of sentence-pair alignment scoring.

**Why a re-ranker?** Cross-encoders are the closest off-the-shelf analogue — they jointly encode both texts and produce a relevance/entailment score. SilverBullet differs in that it operates at sentence-pair level with explicit multi-signal feature engineering before the learned head, making it interpretable and mode-aware.

**Benchmark plan:**
1. Run both models on the same held-out test sets for all three modes
2. Compare ROC-AUC, average precision, accuracy @ 0.5 threshold
3. Measure inference latency (ms/pair) to assess real-time viability
4. Identify failure cases where SilverBullet outperforms and underperforms the re-ranker

### Implemented: External Factual Grounding (EFG) feature (v5.8)

The EFG extractor (`backend/Features/Factual/getFactualGrounding.py`) uses `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (FEVER-trained, ~180M params, local, no retrieval) to assess per-sentence factual grounding. 

Produces 3 feature maps per mode:
- `efg_supports` — model's confidence that text2 is supported by text1
- `efg_refutes` — model's confidence that text2 is refuted by text1
- `efg_factual_delta` — asymmetric factual delta: forward entailment (text1→text2) minus backward entailment (text2→text1)

Addresses the gap where inter-text agreement alone cannot determine which text is factually correct. Especially useful for RVG and MVM where there is no guaranteed ground-truth context.

### Planned: Training data improvements

Current training data: hand-crafted (~213 pairs) + external corpora (STS-B, MNLI, QQP, QNLI, HaluEval, FEVER, SNLI, SciTail). Current scale: CVG ~7,941 / RVG ~9,676 / MVM ~7,000 pairs.

### Checkpoints

Every checkpoint embeds `num_features` and a feature manifest so the model can be reloaded and validated without re-running the pipeline.

| File | Created by | Used by |
|------|-----------|---------|
| `best_model.pth` | `python -m backend.train` | fallback for all modes |
| `models/model-vs-model/best.pth` | `train --mode model-vs-model` | `mode=model-vs-model` requests |
| `models/reference-vs-generated/best.pth` | `train --mode reference-vs-generated` | `mode=reference-vs-generated` requests |
| `models/context-vs-generated/best.pth` | `train --mode context-vs-generated` | `mode=context-vs-generated` requests |

---

## Adding a New Feature Extractor

1. Create `backend/Features/{Name}/get{Name}Weights.py`
2. Implement `getFeatureMap(list1, list2) -> dict[str, torch.Tensor[32,32]]` using the `_reset_state()` pattern (see `AGENT.md`)
3. Register the new feature keys in `backend/feature_registry.py` → `FEATURE_KEYS`
4. Add the extractor to `backend/train.py` and `backend/example.py`
5. Retrain — the cache is incremental and does not need to be cleared. Missing keys are computed and merged per entry on the first training run.

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
