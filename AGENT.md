# SilverBullet — Agent Guide

This file is the authoritative reference for AI agents working on this codebase.
Read this before making any changes. Cross-reference `CLAUDE.md` for architecture
and `TASK.md` for current work items.

---

## Project Identity

SilverBullet is a **real-time LLM evaluation benchmark** — not a generic text similarity tool.
It scores faithfulness, model agreement, and RAG groundedness between two texts using a
Conv2D model trained on 19–21 multi-signal sentence-pair feature maps (v5.7).

**Three evaluation modes, each with its own trained model:**
- `context-vs-generated` — hallucination detection / RAG groundedness
- `reference-vs-generated` — faithfulness against a reference answer
- `model-vs-model` — agreement between two LLM outputs

**Positioning:** Designed for near-real-time use inside LLM pipelines. The API adds ~100–200 ms on cached pairs.

**All Python backend code is under `backend/`.** Import with `from backend.X import Y`.

---

## Module Reference

### `backend/Splitter/sentence_splitter.py`
**Entry point for text preprocessing.**

```python
from backend.Splitter.sentence_splitter import split_txt
sentences = split_txt("My name is Mehul. He is a good person.")
# → ["My name is Mehul.", "Mehul is a good person."]  (coref resolved)
```

- Splits on `(?<!\d)\.(?!\d)` and `\n`
- Each sentence passed through `EntityResolver.resolve()` — pronoun → named entity
- **Performance:** Creates a new `EntityResolver` per call. Expensive in batch contexts —
  instantiate resolver once and pass it in, or disable coref during training.

---

### `backend/Preprocess/coref/resolveEntity.py`
**Coreference resolution via LLM.**

```python
from backend.Preprocess.coref.resolveEntity import EntityResolver
resolver = EntityResolver(model="gpt-4o-mini")   # or local Gemma
resolved = resolver.resolve("He went to the store.")
```

- `gpt-4o`, `gpt-4o-mini` etc. → `MODEL_TYPE = 'api'` (OpenAI client)
- Any other string → `MODEL_TYPE = 'local'` (HuggingFace causal LM)
- HF `login()` is deferred to first local model load via `_ensure_hf_login()`

---

### `backend/Features/Semantic/getSemanticWeights.py`
**Produces 6 feature maps from sentence embeddings.**

```python
from backend.Features.Semantic.getSemanticWeights import SemanticWeights
weights = SemanticWeights().getFeatureMap(sent_group1, sent_group2)
# keys: "mixedbread-ai/mxbai-embed-large-v1",
#       "Qwen/Qwen3-Embedding-0.6B",
#       "SOFT_ROW_<model>", "SOFT_COL_<model>"
# values: torch.Tensor shape [64, 64]
```

- `SemanticFeatures` uses a **class-level `_model_cache`** — loaded once per process.
- `_reset_state()` clears sentence data but never touches `_model_cache`.

---

### `backend/Features/Lexical/getLexicalWeights.py`
**Produces 4 feature maps from SentencePiece token overlap.**

```python
from backend.Features.Lexical.getLexicalWeights import LexicalWeights
weights = LexicalWeights().getFeatureMap(sent_group1, sent_group2)
# keys: "jaccard", "dice", "cosine", "rouge"
```

- Class-level `_tokenizer_cache` — tokenizer loaded once per process.
- Metrics operate on token sets/counts, not embeddings — zero inference cost.

---

### `backend/Features/NLI/getNLIweights.py`
**Produces 3 feature maps from textual entailment probabilities.**

```python
from backend.Features.NLI.getNLIweights import NLIWeights
weights = NLIWeights().getFeatureMap(sent_group1, sent_group2)
# keys: "entailment", "neutral", "contradiction"
```

- Model: `FacebookAI/roberta-large-mnli`
- Batched inference (batch_size=64). Model stored in `__model_cache__` (instance attr).
- `_reset_state()` clears buffers but leaves `__model_cache__` intact.

---

### `backend/Features/EntityGroups/getOverlap.py`
**Produces up to 9 feature maps from GLiNER named entity extraction.**

```python
from backend.Features.EntityGroups.getOverlap import EntityMatch
weights = EntityMatch().getFeatureMap(sent_group1, sent_group2)
# keys (mode-specific subset):
#   type-count: "entity_location", "entity_product", "entity_law", ...
#   flat value: "entity_value_prec", "entity_value_rec"
#   per-type:   "entity_location_value_prec", "entity_percentage_value_rec", ...
```

- GLiNER model `knowledgator/modern-gliner-bi-base-v1.0`
- 14 entity types; Bonferroni-pruned per mode (see `feature_registry.py`)
- Value overlap uses fuzzy string matching with type-calibrated thresholds (0.80–0.95)
- Results cached to SQLite (`CacheDB.entities`) — warm cache skips NER entirely

---

### `backend/Features/LCS/getLCSweights.py`
**Produces 2 feature maps via dynamic-programming LCS (no model).**

```python
from backend.Features.LCS.getLCSweights import LCSWeights
weights = LCSWeights().getFeatureMap(sent_group1, sent_group2)
# keys: "lcs_token", "lcs_char"
# score = len(LCS) / max(len(seq1), len(seq2))  ∈ [0, 1]
```

- No model load, O(n·m) DP in pure Python. Cheapest extractor.

---

### `backend/Postprocess/__addpad.py`
**Pads any n×m matrix to 64×64.**

```python
from backend.Postprocess.__addpad import pad_matrix
t = pad_matrix([[1,2],[3,4]], target_size=64)
# → torch.Tensor shape [64, 64]
```

- Truncates inputs with > 64 rows or > 64 columns (silently drops overflow sentences).
- **Hard constraint:** texts are limited to 64 sentences each.

---

### `backend/cache_db.py` — `CacheDB` (SQLite unified cache)
**Single SQLite database backing all caches. Replaces four old file-based caches.**

```python
from backend.cache_db import CacheDB
db = CacheDB.get()   # process-wide singleton

# Embeddings
db.load_all_embeddings(model_name)          # → dict[sentence, np.ndarray]
db.save_embeddings_batch(model, sents, embs)

# NLI pairs
db.load_all_nli()                           # → dict[(s1,s2), (ent, neu, con)]
db.save_nli_batch(rows)

# Entity cache
db.load_all_entities()                      # → dict[sentence, (counts, strings)]
db.save_entities_batch(sents, counts, strings)

# Feature maps
cached = db.get_features(text1, text2)     # → dict | None
db.save_features(text1, text2, feature_dict)
```

- DB file: `cache/silverbullet.db`, WAL journal mode, thread-local connections.
- Auto-migrates old `{md5}.json`, `.npz`, and JSON files on first run.
- `FeatureCache` in `feature_cache.py` is a thin wrapper that delegates to `CacheDB`.

### `backend/feature_cache.py`
Thin compatibility wrapper over `CacheDB`. Existing code that uses `FeatureCache` still works.

---

### `backend/feature_registry.py`
**Canonical feature key ordering + checkpoint manifest. VERSION = "5.7"**

```python
from backend.feature_registry import FEATURE_KEYS, FEATURE_KEYS_BY_MODE, build_manifest, validate_manifest, get_feature_keys
# FEATURE_KEYS: 36 global keys (all extractors)
# FEATURE_KEYS_BY_MODE: mode-specific subsets — CVG=19, RVG=20, MVM=21
# get_feature_keys(mode) → mode-specific list or global fallback
# build_manifest(feature_keys) → dict with version, features, num_features, spatial_size, created_at
# validate_manifest(manifest, expected_keys) → raises RuntimeError if mismatch
```

- `feature_map_to_tensor(feature_map, feature_keys, fill_missing=False)` in `train.py` stacks maps in key order.
  - `fill_missing=True`: missing keys (e.g. from extractor timeouts) get zero tensors instead of crashing.
- Every checkpoint embeds `manifest` dict (`features` key = list of feature names).
- `predict.py` calls `validate_manifest` on load to catch stale checkpoints before inference.
- `SPATIAL_SIZE = 32` — feature maps are bilinear-resized to 32×32 (not zero-padded to 64×64).

---

### `backend/model.py` — `TextSimilarityCNN` + `TextSimilarityCNNLegacy`

**Current architecture (Conv2D, spatial_size=32):**
```python
from backend.model import TextSimilarityCNN
model = TextSimilarityCNN(num_features=19)  # or 20/21 per mode
# Input:  [batch, num_features, 32, 32]
# Output: [batch, 1]  sigmoid → [0, 1]
```

```
Conv2d(F→128, k=3) + BN + MaxPool(2) → [B, 128, 16, 16]
Conv2d(128→64, k=3) + BN + MaxPool(2) → [B, 64,   8,  8]
Conv2d(64→32,  k=3) + BN + MaxPool(2) → [B, 32,   4,  4]
flatten → [B, 512]
FC(512→128) + BN + Dropout
FC(128→1) → sigmoid
```

**Legacy architecture (Conv1D)** — `TextSimilarityCNNLegacy`:
Kept for backward-compatibility with old checkpoints.
Auto-detected in `predict.py` via presence of `fc_reduce1.weight` in state dict.

---

### `backend/train.py`

```bash
python -m backend.train                          # general model → best_model.pth
python -m backend.train --mode context-vs-generated  # → models/context-vs-generated.pth
```

Key classes/functions:
- `feature_map_to_tensor(feature_map, feature_keys, fill_missing=False)` → `torch.Tensor [F, 32, 32]` — stacks maps in key order; `fill_missing=True` fills missing keys with zeros instead of raising
- `TextSimilarityDataset(pairs, labels, use_cache=True, feature_keys=None)` — runs full feature extraction; uses `_safe_extract()` (90s timeout per extractor group) to avoid hanging on NLI/entity/SVO
- `_safe_extract(extractors_map, sg1, sg2, timeout=90)` — wraps extractor calls in ThreadPoolExecutor; returns empty dict on timeout (caller uses `fill_missing=True`)
- `train_model(model, train_loader, val_loader, best_ckpt='best_model.pth', mode='general')` — MSELoss + Adam + early stopping (patience=15)
- `_Tracker(mode, params)` — optional MLflow + Prometheus tracking; buffers epochs and replays on late-connect

Data loaded from `data/{mode}/` if `--mode` given, else `data/`.

**MLflow behaviour:**
- If `http://localhost:5000` is up at training start → metrics streamed live
- If not → epochs buffered in `_Tracker._history`; at `finish()` it retries, connects, and bulk-replays
- On successful finish: best-weights model logged via `mlflow.pytorch.log_model` and registered in Model Registry as `silverbullet-{mode}`

---

### `backend/test.py`

```bash
python -m backend.test                           # data/test.json + best_model.pth
python -m backend.test --mode context-vs-generated  # data/context-vs-generated/test.json + models/context-vs-generated.pth
python -m backend.test --checkpoint path/to/model.pth  # explicit checkpoint override
```

Outputs: accuracy, ROC-AUC, average precision, confusion matrix, per-sample predictions → `test_reports/`.

If MLflow is running, `_log_test_to_mlflow()` creates a separate run in the same experiment with `test_accuracy`, `test_roc_auc`, `test_avg_precision`, and the test report JSON as an artifact.

---

### `backend/predict.py` — `SimilarityPredictor`

```python
from backend.predict import SimilarityPredictor
predictor = SimilarityPredictor("best_model.pth")

# Fast path (uses feature cache)
result = predictor.predict_pair(text1, text2)
# → {"prediction": 0|1, "probability": float, "text1": ..., "text2": ...}

# Batch
results = predictor.predict_batch([[t1, t2], ...])

# Breakdown (slow — reruns pipeline without cache)
bd = predictor.predict_pair_breakdown(text1, text2)
# → {
#     "prediction": int, "probability": float,
#     "sentences1": [...], "sentences2": [...],
#     "alignment": [[float]], "divergent_in_1": [int], "divergent_in_2": [int],
#     "feature_scores": {"Semantic (mxbai)": float, ...}
#   }

# Batch breakdown (up to 10 pairs)
bds = predictor.predict_batch_breakdown([[t1, t2], ...])
```

---

### `backend/api/` — FastAPI application

```
GET  /api/v1/health                        → HealthResponse  (per-mode model load state)
POST /api/v1/predict/pair                  → PairResponse    (60/min)
POST /api/v1/predict/pair/breakdown        → BreakdownResponse  (20/min — slow)
POST /api/v1/predict/batch                 → BatchResponse   (60/min, max 100 pairs)
POST /api/v1/predict/batch/breakdown       → BatchBreakdownResponse  (10/min, max 10 pairs)
```

All POST endpoints accept `mode: EvaluationMode` (default `"context-vs-generated"`).

**`backend/api/dependencies.py`** — `get_predictor(mode: str)`:
- `@lru_cache(maxsize=3)` — one `SimilarityPredictor` per mode, cached for the process lifetime
- Checkpoint resolution order: `MODEL_PATH_<MODE>` env var → `models/{mode}/best.pth` → `models/{mode}.pth` (legacy) → `MODEL_PATH` / `best_model.pth`
- No `sys.path` manipulation needed — `backend` is a proper package on the project root path

**`backend/api/schemas.py`** — `EvaluationMode = Literal["model-vs-model", "reference-vs-generated", "context-vs-generated"]`

Entry point for uvicorn: `uvicorn backend.api.main:app --reload`

---

### `backend/generate_data.py`

```bash
python -m backend.generate_data
```

Generates ~213 hand-crafted pairs across 15 domains:
- `data/{train,validate,test}.json` — general pairs (70/15/15 split)
- `data/{mode}/{train,validate,test}.json` — general + mode-specific hard negatives

Pair taxonomy: `positive` (label=1), `hard_neg` (label=0), `soft_neg` (label=0).

**Important:** module-level code is wrapped in `main()` + `if __name__ == "__main__"`. Do not execute on import.

---

### `backend/fetch_external_data.py`

```bash
python -m backend.fetch_external_data        # --max-per-source 400 --force
```

Downloads and merges public NLP datasets. Caches raw downloads to `data/external/`.

| Source | Mode | Pairs | Notes |
|--------|------|-------|-------|
| STS-B, MNLI | general | ~1400 | score≥3.5 → 1; entailment → 1 |
| QQP | model-vs-model | ~1400 | duplicate → 1 |
| QNLI | reference-vs-generated | ~1400 | entailment → 1 |
| HaluEval QA | context-vs-generated | ~1400 | hallucination=yes → 0 |
| FEVER (`copenlu/fever_gold_evidence`) | CVG + RVG | +1000 each | supported → 1 |
| SNLI | reference-vs-generated | +1000 | entailment → 1, contradiction → 0 |
| SciTail (`allenai/scitail`) | reference-vs-generated | +1000 | entails → 1, neutral → 0 |

Result: CVG ~7,941 / RVG ~9,676 / MVM ~7,000 pairs.

**Note:** `datasets>=3.x` dropped loading-script support. FEVER must use `copenlu/fever_gold_evidence` (Parquet mirror); the original `fever/v1.0` path fails.

---

### `backend/precompute_features.py`

```bash
python -m backend.precompute_features        # --force to recompute all
```

Collects every unique `(text1, text2)` pair across all splits and modes, deduplicates, loads all feature extractors once, and saves feature dicts to `CacheDB` (SQLite). Run once before training to avoid per-pair extraction overhead. After warm cache, hits load at ~100 pairs/sec vs ~1–100 s/pair live (depending on extractor timeouts).

---

## Common Patterns & Pitfalls

### Pattern: Feature extractor lazy model loading
```python
def __calc_weights__(self):
    if not hasattr(self, "__model_cache__"):
        self.__load_model__()
    # use self.__model_cache__
```
All extractors use this. `_reset_state()` never sets `__model_cache__`, so the model survives between `getFeatureMap` calls.

### PITFALL: Do not set `self.__model_cache__ = None` in `__init__`
This breaks the `hasattr` guard — model reloads on every call.

### Pattern: Consistent `getFeatureMap` interface
```python
def getFeatureMap(self, list1: List[str], list2: List[str]) -> Dict[str, torch.Tensor]:
    self._reset_state()
    ...
    return self.comparison_weights   # dict of {name: Tensor[64,64]}
```

### PITFALL: Key collisions in `feature_map.update()`
`train.py` merges all extractor outputs with `.update()`. If two extractors share a key, one silently overwrites the other. Verify uniqueness in `FEATURE_KEYS` when adding a new extractor.

### PITFALL: Feature set changes invalidate cache AND checkpoints
Changing which extractors run (or their key order) invalidates:
1. `./cache/` — delete to force recompute
2. `best_model.pth` / `models/*.pth` — retrain from scratch (`num_features` changes, manifest mismatch)

---

## Adding a New Feature Extractor

1. Create `backend/Features/{Name}/get{Name}Weights.py`
2. Implement with `_reset_state()`, `__load_model__()`, `getFeatureMap()` following the existing pattern
3. Add new key(s) to `FEATURE_KEYS` (and relevant mode baskets in `FEATURE_KEYS_BY_MODE`) in `backend/feature_registry.py`
4. Import and call in `backend/train.py` (`TextSimilarityDataset.__init__`) and `backend/predict.py`
5. Add a `_prefill_*_cache()` in `train.py` if the extractor is slow — new extractors run sequentially per-pair without prefill, blocking training for hours
6. Prefill functions must call `CacheDB.save_*()` explicitly — they bypass `getFeatureMap()` so nothing persists automatically
7. Wrap the `getFeatureMap` call in `_safe_extract()` if the extractor can hang (NLI/GLiNER/external model pattern)
8. Add to `backend/example.py` for manual testing
9. Clear `./cache/` (or let the incremental merge handle it) and retrain all mode-specific models

### Planned: External Factual Grounding (EFG)
`backend/Features/Factual/getFactualGrounding.py` using `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`.
Per-sentence factuality scores → n×m delta matrix. Detects which text is more world-knowledge-supported (complements agreement signal). Implement after v5.7 SHAP analysis.

---

## Directory Outputs (auto-created at runtime)

| Path | Created By | Contents |
|------|-----------|----------|
| `./cache/silverbullet.db` | `CacheDB` | SQLite WAL — embeddings, NLI pairs, entities, triplets, feature maps |
| `./training_reports/` | `TrainingReport` | `training_report_{ts}_{current\|final}.{json\|md}` |
| `./test_reports/` | `TestReport` | `test_report_{ts}.{json\|md}` |
| `best_model.pth` | `backend/train.py` | General fallback checkpoint |
| `models/{mode}/best.pth` | `backend/train.py --mode` | Active per-mode checkpoint (API loads this) |
| `models/{mode}/{ts}_best.pth` | `backend/train.py --mode` | Archived best weights per run |
| `models/{mode}/{ts}_final.pth` | `backend/train.py --mode` | Archived final weights per run |

---

## Security Notes

- `backend/resources/config.yaml` contains live API tokens — **do not commit** (gitignored)
- Prefer env vars: `SB_HF_TOKEN`, `SB_OPENAI_TOKEN`, `MODEL_PATH_*`
- Model cache dirs use relative paths from project root — verify on target OS
