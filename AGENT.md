# SilverBullet — Agent Guide

This file is the authoritative reference for AI agents working on this codebase.
Read this before making any changes. Cross-reference `CLAUDE.md` for architecture
and `TASK.md` for current work items.

---

## Project Purpose

SilverBullet learns to score the similarity or faithfulness between two text passages.
It works by splitting texts into sentences, computing cross-sentence feature matrices
using multiple NLP signals, and feeding those matrices to a trained CNN classifier.

The key insight is the **n×m sentence-pair matrix**: rather than comparing two texts as
whole embeddings, every sentence in text1 is compared against every sentence in text2,
capturing fine-grained alignment information.

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
- Each sentence is passed through `EntityResolver.resolve()` to replace pronouns with named entities
- **Performance note:** Creates a new `EntityResolver` on every call — this triggers model load + API calls. In batch contexts, instantiate `EntityResolver` once and pass it in, or disable coref during training.

---

### `Preprocess/coref/resolveEntity.py`
**Coreference resolution via LLM.**

```python
resolver = EntityResolver(model="gpt-4o-mini")   # or "google/gemma-3-270m" for local
resolved = resolver.resolve("He went to the store.")
```

- `model` in `['gpt-4o', 'gpt-4o-mini', ...]` → `MODEL_TYPE = 'api'` (uses OpenAI client)
- Any other model string → `MODEL_TYPE = 'local'` (loads via `AutoModelForCausalLM`)
- API key sourced from `resources/config.yaml` via `getVal()['openai_token']`
- HF token sourced from `getVal()['hf_token']`
- Module-level `login(token=...)` runs on import — ensure HF token is valid before importing

---

### `Features/Semantic/getSemanticWeights.py`
**Produces 6 feature maps from sentence embeddings.**

```python
weights = SemanticWeights().getFeatureMap(sent_group1, sent_group2)
# keys: "mixedbread-ai/mxbai-embed-large-v1",
#       "Qwen/Qwen3-Embedding-0.6B",
#       "SOFT_ROW_<model>", "SOFT_COL_<model>"  (softmax alignments)
# values: torch.Tensor shape [64, 64]
```

Internal flow: `SemanticFeatures.run()` → encode both groups → `cos_sim` pairwise → softmax along rows and cols → `pad_matrix`.

`SemanticFeatures` uses a **class-level `_model_cache`** — models are loaded once per process and shared across all instances. Do not clear this cache unless necessary.

---

### `Features/Lexical/getLexicalWeights.py`
**Produces 4 feature maps from token overlap metrics.**

```python
weights = LexicalWeights().getFeatureMap(sent_group1, sent_group2)
# keys: "jaccard", "dice", "cosine", "rouge"
# values: torch.Tensor shape [64, 64]
```

- Uses `mxbai-embed-large-v1` tokenizer (SentencePiece) — tokenizer only, no model weights
- **Bug:** `sp_tokenize()` calls `__load_model__()` unconditionally — tokenizer reloads every call. Fix: add `if not hasattr(self, 'tokenizer_cache')` guard.
- Metrics operate on token sets/counts, not embeddings

---

### `Features/NLI/getNLIweights.py`
**Produces 3 feature maps from textual entailment probabilities.**

```python
weights = NLIWeights().getFeatureMap(sent_group1, sent_group2)
# keys: "entailment", "neutral", "contradiction"
# values: torch.Tensor shape [64, 64]
```

- Model: `FacebookAI/roberta-large-mnli`
- Processes in batches of 64 pairs (configurable via `self.__batch_size__`)
- Model cached in `__model_cache__` instance attribute (set only once due to `hasattr` guard)
- **Note:** `getFeatureMap` calls `self.__init__()` which resets `self.comparison_weights = {}` but does NOT delete `__model_cache__` — model survives. Do not change `__init__` without verifying this.

---

### `Features/EntityGroups/getOverlap.py`
**Produces 1 feature map from named entity count differences.**

```python
weights = EntityMatch().getFeatureMap(sent_group1, sent_group2)
# keys: "EntityMismatch"
# values: torch.Tensor shape [64, 64]   (values are negative integers, 0 = perfect match)
```

- Uses GLiNER model `knowledgator/modern-gliner-bi-base-v1.0`
- Entity types: person, organization, location, date, event, art, work_of_art, law, language
- Score per cell = `sum(-abs(count_A[entity] - count_B[entity]) for entity in types)` — closer to 0 is better
- Same `__init__` / `__model_cache__` pattern as NLI

---

### `Postprocess/__addpad.py`
**Pads any n×m matrix to 64×64.**

```python
from Postprocess.__addpad import pad_matrix
t = pad_matrix([[1,2],[3,4]], target_size=64, pad_value=0)
# → torch.Tensor shape [64, 64]
```

- **Hard constraint:** Input matrix must not exceed 64 rows or 64 columns. Raises `IndexError` (via F.pad negative pad values) if it does.
- This means texts are limited to **64 sentences each**.

---

### `feature_cache.py`
**Disk-based feature cache.**

```python
cache = FeatureCache(cache_dir='cache')
cached = cache.get_features(text1, text2)   # None if not found
cache.save_features(text1, text2, feature_vector_list)
```

- Key = `md5(text1 + "|||" + text2)`
- Stored as JSON in `cache/{key}.json`
- Cache is **order-sensitive**: `get_features(A, B) != get_features(B, A)`
- Safe to delete `./cache/` to force full recompute

---

### `model.py` — `TextSimilarityCNN`

```python
model = TextSimilarityCNN(input_dim=45056)   # dynamic, set from first training sample
output = model(feature_tensor)               # → scalar in [0, 1]
```

Architecture:
```
FC(input_dim → 4096) + BN + Dropout
FC(4096 → 1024)      + BN + Dropout
FC(1024 → 256)       + BN + Dropout
unsqueeze(1)  →  [batch, 1, 256]
Conv1d(1→128, k=3) + ReLU + MaxPool(2)
Conv1d(128→64, k=3) + ReLU + MaxPool(2)
flatten  →  [batch, 64*64]
FC(4096 → 128)       + BN + Dropout
FC(128 → 1)
sigmoid → [0, 1]
```

**Known issue:** The architecture calls itself a CNN but processes a 1D sequence after flattening 2D matrices. The spatial n×m structure is lost. Correct fix is `Conv2d` over a `[batch, F, 64, 64]` input where F = number of feature maps.

---

### `train.py`

**Key classes/functions:**
- `load_json_data(path)` → `(pairs, labels)`
- `TextSimilarityDataset(pairs, labels, use_cache=True)` — runs full feature extraction at init, stores flat tensors
- `train_model(model, train_loader, val_loader, num_epochs=50, patience=5)` — Adam + BCELoss + early stopping

Checkpoints saved:
- `best_model.pth` — best validation loss checkpoint (overwritten each improvement)
- `model_weights_{timestamp}_final.pth` — final epoch weights
- `model_weights_{timestamp}_best.pth` — best epoch weights (copy at training end)
- `training_reports/training_report_{timestamp}_current.json` — updated each epoch
- `training_reports/training_report_{timestamp}_final.json` — at end

**`input_dim` is dynamic** — it is derived from `len(train_dataset[0][0])` and passed to `TextSimilarityCNN`. Never hardcode it.

---

### `predict.py` — `SimilarityPredictor`

```python
predictor = SimilarityPredictor("best_model.pth")
result = predictor.predict_pair(text1, text2)
# → {"prediction": 0|1, "probability": float, "text1": ..., "text2": ...}

results = predictor.predict_batch([[t1, t2], ...])
```

**Known issue:** Constructor creates `TextSimilarityDataset([["temp", "temp"]], [0])` just to infer `input_dim`. This triggers the full feature pipeline on garbage input. Should serialize `input_dim` into the checkpoint instead.

---

### `resources/getConfig.py`

```python
from resources.getConfig import getVal
cfg = getVal(env='DEVELOPMENT')
# cfg['hf_token'], cfg['openai_token'], cfg['api_key']
```

- Falls back to a hardcoded absolute path on failure — this must be removed
- Until secrets are moved to env vars, keep `resources/config.yaml` out of commits

---

## Common Patterns & Pitfalls

### Pattern: Feature extractor with lazy model loading
```python
def __calc_weights__(self):
    if not hasattr(self, "__model_cache__"):
        self.__load_model__()
    # use self.__model_cache__
```
All feature extractors follow this pattern. The model survives `getFeatureMap` calls because `self.__init__()` does not set `__model_cache__`, so the `hasattr` guard prevents reload.

### Pitfall: Adding attributes in `__init__` that shadow model cache
If you add `self.__model_cache__ = None` to any `__init__`, the lazy-load guard will break and the model will reload on every `getFeatureMap` call.

### Pattern: Consistent `getFeatureMap` interface
Every feature extractor exposes:
```python
def getFeatureMap(self, list1: List[str], list2: List[str]) -> Dict[str, torch.Tensor]:
```
Returns dict of `{feature_name: torch.Tensor[64, 64]}`. The keys must be unique across all extractors since `train.py` merges them with `feature_map.update(...)`.

### Pitfall: Key collisions in feature_map.update()
If two extractors return the same key, one will silently overwrite the other in `train.py`. All current keys are unique but verify when adding new extractors.

### Pitfall: Feature vector length must be consistent between train and predict
The model's `input_dim` is fixed at training time. Adding or removing feature extractors invalidates saved checkpoints. When changing features, retrain from scratch.

---

## Adding a New Feature Extractor

1. Create `Features/{Name}/get{Name}Weights.py`
2. Implement class with:
   - `__init__`: initialize state vars (NO model loading here)
   - `__load_model__`: lazy model load, stores in `self.__model_cache__`
   - `__calc_weights__`: builds `self.comparison_weights` dict
   - `__post_process_weights__`: calls `pad_matrix` on each entry
   - `getFeatureMap(list1, list2) -> dict`: calls `self.__init__()` to reset, then runs pipeline
3. Add to `train.py`, `predict.py` (via `TextSimilarityDataset`), and `example.py`
4. Verify key names don't collide with existing feature keys
5. Delete `./cache/` to force recompute with new features
6. Retrain model (new features change `input_dim`)

---

## Directory Outputs (auto-created at runtime)

| Path | Created By | Contents |
|------|-----------|----------|
| `./cache/` | `FeatureCache` | `{md5}.json` — flat feature vectors per text pair |
| `./training_reports/` | `TrainingReport` | `training_report_{ts}_{current|final}.{json|md}` |
| `./test_reports/` | `TestReport` | `test_report_{ts}.{json|md}` |
| `best_model.pth` | `train.py` | Best validation checkpoint |
| `model_weights_{ts}_{final|best}.pth` | `train.py` | Final/best weights per run |

---

## Security Notes

- `resources/config.yaml` contains live API tokens — **do not commit changes to this file**
- `getConfig.py` contains a hardcoded absolute Windows path as a fallback — do not replicate this pattern
- Model cache directories (`/Features/.../weights_cache/`) use absolute-style paths starting with `/` — on Windows these resolve relative to the drive root. Verify paths work on target OS before deploying.