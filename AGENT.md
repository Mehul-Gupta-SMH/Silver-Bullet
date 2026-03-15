# SilverBullet — Agent Guide

This file is the authoritative reference for AI agents working on this codebase.
Read this before making any changes. Cross-reference `CLAUDE.md` for architecture
and `TASK.md` for current work items.

---

## Project Identity

SilverBullet is a **real-time LLM evaluation benchmark** — not a generic text similarity tool.
It scores faithfulness, model agreement, and RAG groundedness between two texts using a
Conv2D model trained on 16 multi-signal sentence-pair feature maps.

**Positioning:** Designed for near-real-time use inside LLM pipelines (hallucination guards,
model comparison, RL reward signals). The API adds ~100–200 ms on cached pairs.

---

## Module Reference

### `Splitter/sentence_splitter.py`
**Entry point for text preprocessing.**

```python
from Splitter.sentence_splitter import split_txt
sentences = split_txt("My name is Mehul. He is a good person.")
# → ["My name is Mehul.", "Mehul is a good person."]  (coref resolved)
```

- Splits on `(?<!\d)\.(?!\d)` and `\n`
- Each sentence passed through `EntityResolver.resolve()` — pronoun → named entity
- **Performance:** Creates a new `EntityResolver` per call. Expensive in batch contexts —
  instantiate resolver once and pass it in, or disable coref during training.

---

### `Preprocess/coref/resolveEntity.py`
**Coreference resolution via LLM.**

```python
resolver = EntityResolver(model="gpt-4o-mini")   # or local Gemma
resolved = resolver.resolve("He went to the store.")
```

- `gpt-4o`, `gpt-4o-mini` etc. → `MODEL_TYPE = 'api'` (OpenAI client)
- Any other string → `MODEL_TYPE = 'local'` (HuggingFace causal LM)
- HF `login()` runs at import — ensure `hf_token` is valid before importing

---

### `Features/Semantic/getSemanticWeights.py`
**Produces 6 feature maps from sentence embeddings.**

```python
weights = SemanticWeights().getFeatureMap(sent_group1, sent_group2)
# keys: "mixedbread-ai/mxbai-embed-large-v1",
#       "Qwen/Qwen3-Embedding-0.6B",
#       "SOFT_ROW_<model>", "SOFT_COL_<model>"
# values: torch.Tensor shape [64, 64]
```

- `SemanticFeatures` uses a **class-level `_model_cache`** — loaded once per process.
- `_reset_state()` clears sentence data but never touches `_model_cache`.

---

### `Features/Lexical/getLexicalWeights.py`
**Produces 4 feature maps from SentencePiece token overlap.**

```python
weights = LexicalWeights().getFeatureMap(sent_group1, sent_group2)
# keys: "jaccard", "dice", "cosine", "rouge"
```

- Class-level `_tokenizer_cache` — tokenizer loaded once per process.
- Metrics operate on token sets/counts, not embeddings — zero inference cost.

---

### `Features/NLI/getNLIweights.py`
**Produces 3 feature maps from textual entailment probabilities.**

```python
weights = NLIWeights().getFeatureMap(sent_group1, sent_group2)
# keys: "entailment", "neutral", "contradiction"
```

- Model: `FacebookAI/roberta-large-mnli`
- Batched inference (batch_size=64). Model stored in `__model_cache__` (instance attr).
- `_reset_state()` clears buffers but leaves `__model_cache__` intact.

---

### `Features/EntityGroups/getOverlap.py`
**Produces 1 feature map from named entity count differences.**

```python
weights = EntityMatch().getFeatureMap(sent_group1, sent_group2)
# keys: "EntityMismatch"   (values are ≤ 0; 0 = perfect entity match)
```

- GLiNER model `knowledgator/modern-gliner-bi-base-v1.0`
- Entity types: person, org, location, date, event, art, work_of_art, law, language
- Cell value = `sum(-abs(countA[e] - countB[e]) for e in entity_types)`

---

### `Features/LCS/getLCSweights.py`
**Produces 2 feature maps via dynamic-programming LCS (no model).**

```python
weights = LCSWeights().getFeatureMap(sent_group1, sent_group2)
# keys: "lcs_token", "lcs_char"
# score = len(LCS) / max(len(seq1), len(seq2))  ∈ [0, 1]
```

- No model load, O(n·m) DP in pure Python. Cheapest extractor.

---

### `Postprocess/__addpad.py`
**Pads any n×m matrix to 64×64.**

```python
from Postprocess.__addpad import pad_matrix
t = pad_matrix([[1,2],[3,4]], target_size=64)
# → torch.Tensor shape [64, 64]
```

- Truncates inputs with > 64 rows or > 64 columns (silently drops overflow sentences).
- **Hard constraint:** texts are limited to 64 sentences each.

---

### `feature_cache.py`
**Disk-based feature cache keyed by MD5 of text pair.**

```python
cache = FeatureCache(cache_dir='cache')
cached = cache.get_features(text1, text2)   # None if miss
cache.save_features(text1, text2, stacked_tensor.tolist())
```

- Cache stores the full stacked `[F, 64, 64]` tensor as a JSON list.
- Order-sensitive: `(A, B)` ≠ `(B, A)`.
- Delete `./cache/` to force full recompute (required after changing feature set).

---

### `model.py` — `TextSimilarityCNN` (current) + `TextSimilarityCNNLegacy`

**Current architecture (Conv2D):**
```python
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
Kept for backward-compatibility so old `best_model.pth` checkpoints still load.
Auto-detected in `predict.py` via presence of `fc_reduce1.weight` in state dict.

**Checkpoint format:**
```python
{
    'epoch': int,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'loss': float,
    'num_features': int,   # required for Conv2D; absent in legacy checkpoints
}
```

---

### `train.py`

Key classes/functions:
- `feature_map_to_tensor(feature_map)` → `torch.Tensor [F, 64, 64]` — stacks all maps in insertion order
- `TextSimilarityDataset(pairs, labels, use_cache=True)` — runs full feature extraction, stores `[N, F, 64, 64]`
- `train_model(model, train_loader, val_loader, num_epochs=50, patience=5)` — Adam + BCELoss + early stopping

Checkpoints saved:
- `best_model.pth` — best val-loss checkpoint, overwritten each improvement
- `model_weights_{ts}_final.pth` / `model_weights_{ts}_best.pth` — archived at end

`num_features` is derived from `train_dataset.num_features` and serialised into every checkpoint.

---

### `predict.py` — `SimilarityPredictor`

```python
predictor = SimilarityPredictor("best_model.pth")

# Fast path (uses feature cache)
result = predictor.predict_pair(text1, text2)
# → {"prediction": 0|1, "probability": float, "text1": ..., "text2": ...}

# Batch
results = predictor.predict_batch([[t1, t2], ...])

# Breakdown (slow — reruns pipeline, extracts intermediate data)
bd = predictor.predict_pair_breakdown(text1, text2)
# → {
#     "prediction": int, "probability": float,
#     "sentences1": [...], "sentences2": [...],
#     "alignment": [[float]], "divergent_in_1": [int], "divergent_in_2": [int],
#     "feature_scores": {"Semantic (mxbai)": float, ...}
#   }
```

`_load_model_from_checkpoint(checkpoint, device)` auto-detects legacy vs Conv2D via state dict keys.

---

### `api/` — FastAPI application

```
GET  /api/v1/health                   → HealthResponse
POST /api/v1/predict/pair             → PairResponse          (fast, uses cache)
POST /api/v1/predict/pair/breakdown   → BreakdownResponse     (slow, full pipeline)
POST /api/v1/predict/batch            → BatchResponse         (up to 100 pairs)
```

- `get_predictor()` in `dependencies.py` is `@lru_cache` — singleton per process
- `MODEL_PATH` env var overrides default `best_model.pth`
- Structured JSON logging via `LoggingMiddleware`; request IDs via `RequestIDMiddleware`
- CORS: `localhost:5173` + `ALLOWED_ORIGINS` env var
- Docs: `/api/v1/docs` (Swagger), `/api/v1/redoc`

---

### `frontend/`

Stack: React + TypeScript + Vite + TailwindCSS

Key components:
- `PairScorer` — single eval form; after scoring shows "Drill Down — Impact & Divergence" button
- `BreakdownPanel` — renders divergence analysis: two-column sentence list (colour-coded by max alignment), orphaned-sentence summary, feature-score bars
- `BatchScorer` — up to 100 pairs, distribution chart
- `ScoreGauge` — visual score bar + Similar/Different badge
- `TestCasePanel` — pre-built test cases incl. `ex-1`/`ex-2` (from `example.py`)
- `ExperimentsPanel` — save, annotate, re-run scored pairs

Vite proxy: `/api` → `http://localhost:8000` (configured in `frontend/vite.config.ts`).

---

### `generate_data.py`

Generates `data/train.json`, `data/validate.json`, `data/test.json` from hardcoded pairs.

```
python generate_data.py
```

Pair taxonomy:
- `positive` (label=1): paraphrases, faithful summaries, equivalent code — 14 domains
- `hard_neg` (label=0): same topic, wrong number/date/entity, negated claim, partial hallucination
- `soft_neg` (label=0): completely different domains

Run this before training whenever you want to change the dataset. Delete `./cache/` too.

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
This breaks the hasattr guard — model reloads on every call.

### Pattern: Consistent `getFeatureMap` interface
```python
def getFeatureMap(self, list1: List[str], list2: List[str]) -> Dict[str, torch.Tensor]:
    self._reset_state()
    ...
    return self.comparison_weights   # dict of {name: Tensor[64,64]}
```

### PITFALL: Key collisions in `feature_map.update()`
`train.py` merges all extractor outputs with `.update()`. If two extractors share a key, one silently overwrites the other. Verify uniqueness when adding a new extractor.

### PITFALL: Feature set changes invalidate cached tensors AND checkpoints
Changing which extractors run (or their key order) invalidates:
1. `./cache/` — delete to force recompute
2. `best_model.pth` — retrain from scratch (`num_features` changes)

---

## Adding a New Feature Extractor

1. Create `Features/{Name}/get{Name}Weights.py`
2. Implement with `_reset_state()`, `__load_model__()`, `getFeatureMap()` following existing pattern
3. Import and call in `train.py` (`TextSimilarityDataset.__init__`) and `predict.py` (`predict_pair_breakdown`)
4. Add to `example.py` for manual testing
5. Verify no key collision with existing feature names
6. Delete `./cache/`, retrain model

---

## Directory Outputs (auto-created at runtime)

| Path | Created By | Contents |
|------|-----------|----------|
| `./cache/` | `FeatureCache` | `{md5}.json` — stacked `[F,64,64]` tensors per text pair |
| `./training_reports/` | `TrainingReport` | `training_report_{ts}_{current\|final}.{json\|md}` |
| `./test_reports/` | `TestReport` | `test_report_{ts}.{json\|md}` |
| `best_model.pth` | `train.py` | Best validation checkpoint |
| `model_weights_{ts}_{final\|best}.pth` | `train.py` | Archived weights per run |

---

## Security Notes

- `resources/config.yaml` contains live API tokens — **do not commit**
- `getConfig.py` has a hardcoded Windows absolute path fallback — do not replicate
- Model cache dirs use relative paths from project root — verify on target OS
