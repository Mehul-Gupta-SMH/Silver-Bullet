# SilverBullet — Agent Guide

This file is the authoritative reference for AI agents working on this codebase.
Read this before making any changes. Cross-reference `CLAUDE.md` for architecture
and `TASK.md` for current work items.

---

## Project Identity

SilverBullet is a **real-time LLM evaluation benchmark** — not a generic text similarity tool.
It scores faithfulness, model agreement, and RAG groundedness between two texts using a
Conv2D model trained on 16 multi-signal sentence-pair feature maps.

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
**Produces 1 feature map from named entity count differences.**

```python
from backend.Features.EntityGroups.getOverlap import EntityMatch
weights = EntityMatch().getFeatureMap(sent_group1, sent_group2)
# keys: "EntityMismatch"   (values are ≤ 0; 0 = perfect entity match)
```

- GLiNER model `knowledgator/modern-gliner-bi-base-v1.0`
- Entity types: person, org, location, date, event, art, work_of_art, law, language
- Cell value = `sum(-abs(countA[e] - countB[e]) for e in entity_types)`

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

### `backend/feature_cache.py`
**Disk-based feature cache keyed by MD5 of text pair.**

```python
from backend.feature_cache import FeatureCache
cache = FeatureCache(cache_dir='cache')
cached = cache.get_features(text1, text2)   # None if miss
cache.save_features(text1, text2, stacked_tensor.tolist())
```

- Cache stores the full stacked `[F, 64, 64]` tensor as a JSON list.
- Order-sensitive: `(A, B)` ≠ `(B, A)`.
- Delete `./cache/` to force full recompute (required after changing feature set).

---

### `backend/feature_registry.py`
**Canonical feature key ordering + checkpoint manifest.**

```python
from backend.feature_registry import FEATURE_KEYS, build_manifest, validate_manifest
# FEATURE_KEYS: list of 16 strings in stable order
# build_manifest() → dict with version, feature_keys, timestamp
# validate_manifest(checkpoint_manifest) → raises if incompatible
```

- `feature_map_to_tensor()` in `train.py` stacks maps in `FEATURE_KEYS` order — channel index is always stable.
- Every checkpoint embeds `manifest` dict. `predict.py` calls `validate_manifest` on load.

---

### `backend/model.py` — `TextSimilarityCNN` + `TextSimilarityCNNLegacy`

**Current architecture (Conv2D):**
```python
from backend.model import TextSimilarityCNN
model = TextSimilarityCNN(num_features=16)
# Input:  [batch, num_features, 64, 64]
# Output: [batch, 1]  sigmoid → [0, 1]
```

```
Conv2d(F→128, k=3) + BN + MaxPool(2) → [B, 128, 32, 32]
Conv2d(128→64, k=3) + BN + MaxPool(2) → [B, 64,  16, 16]
Conv2d(64→32,  k=3) + BN + MaxPool(2) → [B, 32,   8,  8]
flatten → [B, 2048]
FC(2048→128) + BN + Dropout
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
- `feature_map_to_tensor(feature_map)` → `torch.Tensor [F, 64, 64]` — stacks maps in `FEATURE_KEYS` order
- `TextSimilarityDataset(pairs, labels, use_cache=True)` — runs full feature extraction
- `train_model(model, train_loader, val_loader, best_ckpt='best_model.pth', mode='general')` — MSELoss + Adam + early stopping
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
- Checkpoint resolution order: `MODEL_PATH_<MODE>` env var → `models/{mode}.pth` → `MODEL_PATH` / `best_model.pth`
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

Downloads STS-B, MNLI (general), QQP (model-vs-model), QNLI (reference-vs-generated), HaluEval QA (context-vs-generated) via HuggingFace `datasets`. Caches raw downloads to `data/external/`. Merges with hand-crafted pairs and re-saves all splits. Result: ~1 400 pairs per mode (1001/214/216).

HaluEval schema: `{knowledge, question, answer, hallucination: "yes"/"no"}` → `label = 0 if hallucination == "yes" else 1`.

---

### `backend/precompute_features.py`

```bash
python -m backend.precompute_features        # --force to recompute all
```

Collects every unique `(text1, text2)` pair across all splits and modes, deduplicates, loads all 5 feature extractors once, and saves tensors to `./cache/`. Run once before training to avoid per-pair extraction overhead during `TextSimilarityDataset.__init__`. After precompute, cache hits load at ~95 pairs/sec vs ~1–4 s/pair live.

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
3. Add new key(s) to `FEATURE_KEYS` in `backend/feature_registry.py`
4. Import and call in `backend/train.py` (`TextSimilarityDataset.__init__`) and `backend/predict.py`
5. Add to `backend/example.py` for manual testing
6. Delete `./cache/`, retrain all mode-specific models

---

## Directory Outputs (auto-created at runtime)

| Path | Created By | Contents |
|------|-----------|----------|
| `./cache/` | `FeatureCache` | `{md5}.json` — stacked `[F,64,64]` tensors per text pair |
| `./training_reports/` | `TrainingReport` | `training_report_{ts}_{current\|final}.{json\|md}` |
| `./test_reports/` | `TestReport` | `test_report_{ts}.{json\|md}` |
| `best_model.pth` | `backend/train.py` | General fallback checkpoint |
| `models/{mode}.pth` | `backend/train.py --mode` | Per-mode checkpoint |
| `model_weights_{ts}_{final\|best}.pth` | `backend/train.py` | Archived weights per run |

---

## Security Notes

- `backend/resources/config.yaml` contains live API tokens — **do not commit** (gitignored)
- Prefer env vars: `SB_HF_TOKEN`, `SB_OPENAI_TOKEN`, `MODEL_PATH_*`
- Model cache dirs use relative paths from project root — verify on target OS
