# SilverBullet — Task Log

## Project Origin (2025-03-15)

> Original design writeup from `ToDo.md`

**Goal:** Build a methodology to compare texts in different contexts and produce a score usable downstream.

| text1 | text2 | Score interpretation |
|-------|-------|----------------------|
| LLM-generated answer | Source context | Faithfulness / RAG groundedness |
| LLM 1 output | LLM 2 output | How comparable the models are |
| LLM-generated answer | Accepted RL answer | Alignment to accepted answer |

**Approach:**
- For `n`-sentence text 1 and `m`-sentence text 2, build an n×m matrix of pairwise scores using multiple signals: cosine similarity of sentence embeddings (domain-specialised models), LCS (longest common subsequence), entity mapping (same entity referred or not)
- This produces F feature maps of shape n×m, fed into a CNN classifier → single score ∈ [0, 1]
- Trained model can act as a teacher for a lighter student model that operates on whole-text embeddings

**Closest analogue:** cross-encoder / re-ranker (cross-entropy between two texts)

---

## Change Log

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-03-11 | [x] | Project design and architecture defined | `ToDo.md` |
| 2026-03-11 | [x] | Sentence splitter with coref resolution | `Splitter/sentence_splitter.py`, `Preprocess/coref/resolveEntity.py` |
| 2026-03-11 | [x] | Semantic feature extractor | `Features/Semantic/getSemanticWeights.py`, `Features/Semantic/__generate_semantic_features.py` |
| 2026-03-11 | [x] | Lexical feature extractor | `Features/Lexical/getLexicalWeights.py` |
| 2026-03-11 | [x] | NLI feature extractor | `Features/NLI/getNLIweights.py` |
| 2026-03-11 | [x] | Entity overlap feature extractor | `Features/EntityGroups/getOverlap.py` |
| 2026-03-11 | [x] | Pad/postprocess n×m matrices to 64×64 | `Postprocess/__addpad.py`, `Postprocess/postprocess.py` |
| 2026-03-11 | [x] | TextSimilarityCNN model architecture | `model.py` |
| 2026-03-11 | [x] | Training pipeline with early stopping | `train.py` |
| 2026-03-11 | [x] | Feature caching (MD5-keyed JSON on disk) | `feature_cache.py` |
| 2026-03-11 | [x] | Training report (JSON + Markdown) | `training_report.py` |
| 2026-03-11 | [x] | Test/evaluation pipeline with full metrics | `test.py` |
| 2026-03-11 | [x] | Inference / batch predict pipeline | `predict.py` |
| 2026-03-11 | [x] | Config loading from YAML | `resources/config.yaml`, `resources/getConfig.py` |

## Batch 1 — Quick Independent Fixes
| 2026-03-11 | [x] | CRITICAL: Env vars for secrets; `config.yaml.example` template; remove abs path | `resources/getConfig.py`, `resources/config.yaml.example` |
| 2026-03-11 | [x] | BUG: Lexical tokenizer reloads on every `sp_tokenize()` — class-level cache + guard | `Features/Lexical/getLexicalWeights.py` |
| 2026-03-11 | [x] | IMPROVEMENT: `pad_matrix` — truncation instead of crash for >64 sentences | `Postprocess/__addpad.py` |

## Batch 2 — Medium Complexity
| 2026-03-11 | [x] | BUG: Coref resolver re-instantiated on every `split_txt` — `resolver` param injected | `Splitter/sentence_splitter.py` |
| 2026-03-11 | [x] | REFACTOR: `getFeatureMap` called `self.__init__()` — replaced with `_reset_state()` on all extractors | `Features/Semantic/`, `Features/Lexical/`, `Features/NLI/`, `Features/EntityGroups/` |
| 2026-03-11 | [x] | IMPROVEMENT: `num_features` serialised into all checkpoints; `predict.py` loads it directly | `train.py`, `predict.py`, `test.py` |

## Batch 3 — Model Architecture
| 2026-03-11 | [x] | BUG: Conv1D → Conv2D — features stacked as [F, 64, 64] channels, spatial structure preserved | `model.py`, `train.py` |

## Batch 4 — New Feature
| 2026-03-11 | [x] | FEATURE: LCS extractor — `lcs_token` + `lcs_char` maps, no external deps | `Features/LCS/getLCSweights.py`, `train.py`, `example.py` |

## Batch 5 — Hardening & Polish (PRs #17–#21, all merged 2026-03-12)
| 2026-03-12 | [x] | Unit 1+2: API input validation + OpenAPI docs — max_length, batch cap, startup check, global exception handler, Field descriptions, tags, redoc | `api/main.py`, `api/schemas.py`, `api/dependencies.py` |
| 2026-03-12 | [x] | Unit 3: Docker HEALTHCHECK + .env.example + env_file in compose | `Dockerfile`, `docker-compose.yml`, `.env.example` |
| 2026-03-12 | [x] | Unit 4: pytest API tests — conftest fixtures (mock predictor, TestClient), 7 tests covering health/pair/batch/CORS/validation | `tests/__init__.py`, `tests/conftest.py`, `tests/test_api.py`, `requirements.txt` |
| 2026-03-12 | [x] | Unit 5: Frontend error UX — ErrorBoundary, loading spinner, dismissible error alert, .env.example | `frontend/src/components/ErrorBoundary.tsx`, `frontend/src/components/PairScorer.tsx`, `frontend/src/components/BatchScorer.tsx`, `frontend/src/App.tsx`, `frontend/.env.example` |
| 2026-03-12 | [x] | Unit 6: Frontend vitest tests — ScoreGauge (5 tests), PairScorer (5 tests), jsdom setup | `frontend/vite.config.ts`, `frontend/src/test/setup.ts`, `frontend/src/components/ScoreGauge.test.tsx`, `frontend/src/components/PairScorer.test.tsx`, `frontend/package.json` |
| 2026-03-12 | [x] | Post-merge fix: rewrote api/main.py, schemas.py, dependencies.py (duplicate code from sequential PR merges); fixed conftest.py import ordering (pytest 7/7); fixed ErrorBoundary import type (vitest 10/10) | `api/main.py`, `api/schemas.py`, `api/dependencies.py`, `tests/conftest.py`, `frontend/src/components/ErrorBoundary.tsx` |

## fix/checkpoint-compat
| 2026-03-13 | [x] | BUG: test.py loaded Conv2D model directly, crashing on legacy checkpoint — use _load_model_from_checkpoint + flatten legacy features in test loop | `test.py` |

## Session 2026-03-13 — LLM Evaluation Benchmark rebrand + Breakdown + Data

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-03-13 | [x] | BUG: Vite proxy pointed to port 8001, uvicorn runs on 8000 — all frontend API calls silently failed | `frontend/vite.config.ts` |
| 2026-03-13 | [x] | FEATURE: `/api/v1/predict/pair/breakdown` — sentence-level divergence analysis: alignment matrix, orphaned sentences, per-feature-group scores | `predict.py` (`predict_pair_breakdown`), `api/schemas.py` (`BreakdownResponse`), `api/main.py` |
| 2026-03-13 | [x] | FEATURE: `BreakdownPanel` frontend component — two-column sentence view colour-coded by alignment, divergence summary, feature score bars | `frontend/src/components/BreakdownPanel.tsx`, `frontend/src/components/PairScorer.tsx`, `frontend/src/services/api.ts`, `frontend/src/types/index.ts` |
| 2026-03-13 | [x] | FEATURE: Pipeline example test cases added to UI library — ex-1 (named), ex-2 (coref/pronoun) from `example.py` | `frontend/src/data/testCases.ts` |
| 2026-03-13 | [x] | FEATURE: Training data generated — 194 pairs across 14 domains (positive + hard negative + soft negative taxonomy) | `generate_data.py`, `data/train.json`, `data/validate.json`, `data/test.json` |
| 2026-03-13 | [x] | TRAINING: `python train.py` completed — best epoch 14, val acc 93.10%, test acc 90.0%, ROC AUC 0.9556 | `best_model.pth`, `training_reports/`, `test_reports/` |
| 2026-03-13 | [x] | FEATURE: Favicon — violet bullseye SVG matching brand colours | `frontend/public/favicon.svg`, `frontend/index.html` |
| 2026-03-13 | [x] | REBRAND: Renamed from "Text Similarity Tool" to "LLM Evaluation Benchmark" across all surfaces | `frontend/index.html`, `frontend/src/App.tsx`, `frontend/src/config/modes.ts`, `frontend/src/components/ComparisonModeSelector.tsx`, `frontend/src/components/ScoreGauge.tsx`, `api/main.py` |
| 2026-03-13 | [x] | DOCS: TASK.md, CLAUDE.md, AGENT.md brought up to date | all three doc files |

## Session 2026-03-14 — Refactor priorities 1–3

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-03-14 | [x] | FEATURE: Session retention — active tab, eval mode, and pair scorer draft (text1, text2, model names) persisted to localStorage via useLocalStorage hook; re-run from experiments overwrites stored draft before remount | `frontend/src/hooks/useLocalStorage.ts`, `frontend/src/App.tsx`, `frontend/src/components/PairScorer.tsx` |
| 2026-03-14 | [x] | BUG: train.py final print used Unicode arrow — UnicodeEncodeError on Windows cp1252; replaced with ASCII | `train.py` |
| 2026-03-14 | [x] | REFACTOR P1: Checkpoint manifest — `feature_registry.py` with `FEATURE_KEYS` (16 ordered keys), `build_manifest()`, `validate_manifest()`; `feature_map_to_tensor` now uses canonical key order; manifest embedded in all 3 checkpoint saves; `predict.py` validates manifest on load | `feature_registry.py`, `train.py`, `predict.py` |
| 2026-03-14 | [x] | REFACTOR P2: Config no side effects — HF `login()` deferred from module import to first local model load via `_ensure_hf_login()` | `Preprocess/coref/resolveEntity.py` |
| 2026-03-14 | [x] | REFACTOR P3: CI workflow — added `api-tests` job (pytest) and `tsc --noEmit` + vitest run to frontend job | `.github/workflows/ci.yml` |
| 2026-03-14 | [x] | BUG: PairScorer vitest test left stale localStorage state between cases — fixed with `localStorage.clear()` in beforeEach | `frontend/src/components/PairScorer.test.tsx` |

## Session 2026-03-22 — Backend refactor, per-mode models, docs

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-03-22 | [x] | FIX: Docker CD — remove empty ENV secrets (SecretsUsedInArgOrEnv lint), add `packages: write` permission to GHCR publish job | `.github/workflows/cd.yml`, `Dockerfile` — PR #29 |
| 2026-03-22 | [x] | FEATURE: Per-mode model routing — EvaluationMode literal, `get_predictor(mode)` with lru_cache(3), checkpoint resolution env var → `models/{mode}.pth` → fallback | `backend/api/schemas.py`, `backend/api/dependencies.py`, `backend/api/main.py` — PR #30 |
| 2026-03-22 | [x] | REFACTOR: Consolidate all Python backend code into `backend/` package — 27 files moved with `git mv`, all imports updated, sys.path hack removed | All `backend/**` files — PR #31 |
| 2026-03-22 | [x] | FEATURE: Per-mode training data — `generate_data.py` generates mode-specific splits under `data/{mode}/` (209 pairs each = 194 general + 15 hard negatives per mode) | `backend/generate_data.py`, `data/*/` |
| 2026-03-22 | [x] | FEATURE: `--mode` flag on `backend/train.py` and `backend/test.py` — trains/evaluates mode-specific checkpoints | `backend/train.py`, `backend/test.py` |
| 2026-03-22 | [x] | TESTS: Updated pytest conftest to patch `backend.api.main.get_predictor` directly (dependency_overrides no longer applies); 25 tests passing | `tests/conftest.py`, `tests/test_api.py` |
| 2026-03-22 | [x] | DOCS: Rewrote README.md, CLAUDE.md, AGENT.md, frontend/README.md for backend/ refactor and per-mode model routing | all four doc files — PR #32 |

## Session 2026-03-22 (continued) — Dataset expansion + monitoring + retrains

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-03-22 | [x] | FIX: CD Node.js 20 deprecation — bump docker/login-action@v3→@v4, docker/build-push-action@v5→@v6 | `.github/workflows/cd.yml` |
| 2026-03-22 | [x] | FIX: CI hashFiles() invalid in job-level if — removed condition from frontend job | `.github/workflows/ci.yml` |
| 2026-03-22 | [x] | FIX: CI Python 3.10 → 3.11 (numpy 2.3.x + scikit-learn 1.7.x require >=3.11) | `.github/workflows/ci.yml`, `pyproject.toml` |
| 2026-03-22 | [x] | FIX: flake8 F824 unused `global FEATURE_ORDER` removed from TextSimilarityDataset | `backend/train.py` |
| 2026-03-22 | [x] | FIX: smoke test imports updated for backend/ package layout | `.github/workflows/ci.yml` |
| 2026-03-22 | [x] | FIX: HTTP 500 `No module named 'Splitter'` — lazy imports in predict_pair_breakdown updated to backend.* | `backend/predict.py` |
| 2026-03-22 | [x] | FEATURE: Business strategy training data — 6 positives, 10 partial-overlap hard negatives, 3 soft negatives | `backend/generate_data.py` |
| 2026-03-22 | [x] | FEATURE: External dataset fetcher — STS-B, MNLI, QQP, QNLI, HaluEval QA via HuggingFace datasets | `backend/fetch_external_data.py`, `requirements.txt` |
| 2026-03-22 | [x] | FEATURE: MLflow + Prometheus monitoring — _Tracker class with late-connect replay, per-epoch metrics | `backend/train.py`, `docker-compose.yml`, `monitoring/` |
| 2026-03-22 | [x] | FEATURE: Batch feature precomputation — precompute_features.py deduplicates and pre-fills cache | `backend/precompute_features.py` |
| 2026-03-22 | [x] | FIX: split_txt resolve_coref=False by default — stops OpenAI API calls during training | `backend/Splitter/sentence_splitter.py`, `backend/predict.py` |
| 2026-03-22 | [x] | TRAINING: context-vs-generated retrain complete — 88.79% val acc @ epoch 16, early stop at 21 | `models/context-vs-generated.pth`, `training_reports/` |
| 2026-03-22 | [x] | TRAINING: reference-vs-generated retrain — 84.58% val acc @ epoch 7, early stop at 12 | `models/reference-vs-generated.pth`, `training_reports/` |
| 2026-03-22 | [x] | TRAINING: model-vs-model retrain — 84.11% val acc @ epoch 6, early stop at 11 | `models/model-vs-model.pth`, `training_reports/` |
| 2026-03-22 | [x] | FEATURE: MLflow Model Registry — best weights registered as silverbullet-{mode} after each run | `backend/train.py` |
| 2026-03-22 | [x] | FEATURE: MLflow test metric logging — test.py logs test_accuracy/roc_auc/avg_precision + report artifact | `backend/test.py` |
| 2026-03-31 | [x] | TESTING: Run python -m backend.test for all 3 modes — cvg 89.8% / rvg 80.6% / mvm 82.4% acc; bootstrap CIs + MC Dropout intervals computed | `backend/test.py`, `test_reports/` |
| 2026-03-22 | [x] | REFACTOR: Model file layout — models/{mode}/best.pth (active) + {ts}_best/{ts}_final archives in mode dir; dependencies.py falls back to legacy flat layout | `backend/train.py`, `backend/test.py`, `backend/api/dependencies.py`, `README.md`, `AGENT.md` |

## Pending
| 2026-03-15 | [x] | IMPROVEMENT: BCE → MSELoss on float labels for continuous faithfulness scoring | `train.py`, `test.py` |
| 2026-03-20 | [x] | IMPROVEMENT: Re-enable rate limiting — SlowAPIMiddleware (60/min global) + tighter limits on breakdown endpoints (20/min pair, 10/min batch) | `api/main.py`, `tests/test_api.py` |
| 2026-03-22 | [x] | IMPROVEMENT: Expand training dataset to 1 000+ pairs with adversarial/domain-balanced sampling | `generate_data.py`, `data/`, `fetch_external_data.py` |
| 2026-03-15 | [x] | IMPROVEMENT: Add `/api/v1/predict/batch/breakdown` parallel to batch predict endpoint | `api/main.py`, `predict.py`, `api/schemas.py`, `tests/` |

## Session 2026-03-22 — Sparse map scoring fix

**Root cause:** Short texts (3–4 sentences) produce a tiny populated region (e.g. 3×4) in the 64×64
feature maps. The CNN sees ~99.7% zeros — identical to two completely unrelated short texts. The
model can't distinguish "short but similar" from "short but unrelated", so it deflates scores for
short inputs.

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-03-22 | [x] | FIX P1 (Normalised pooling): `_apply_density_normalisation(tensor,n,m)` scales by `(64*64)/(n*m)`; TextSimilarityDataset always splits for n/m + normalises; cache stores raw; predict_pair_breakdown normalises manually | `backend/train.py`, `backend/predict.py`, `AGENT.md` — **delete ./cache/ and retrain all 3 modes** |
| 2026-03-22 | [x] | FIX P2 (Adaptive resize): resize_matrix() uses bilinear interpolation n×m→32×32; every cell carries signal; spatial_size=32 in CNN + manifest; n/m crop clamped in breakdown; P1 normalisation removed | `backend/Postprocess/__addpad.py`, all 5 extractors, `backend/model.py`, `backend/feature_registry.py`, `backend/train.py`, `backend/predict.py` — **delete cache, retrain** |
| 2026-04-02 | [x] | FIX P3 (Length conditioning): implemented — `use_length_cond=True/False` flag in TextSimilarityCNN, lengths stored in TextSimilarityDataset, threaded through train/test/predict; disabled by default after v5.0 experiment showed it destabilises training at dataset sizes ~1500; see v5.0 experiment entry | `backend/model.py`, `backend/train.py`, `backend/predict.py`, `backend/test.py` |

## Session 2026-03-22 — Lexical/LCS parallelisation + full retrain

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-03-22 | [x] | PERF: Lexical batch tokenisation — `sp_tokenize_batch()` encodes all sentences in one Rust call; `ThreadPoolExecutor` parallelises row computation | `backend/Features/Lexical/getLexicalWeights.py` |
| 2026-03-22 | [x] | PERF: LCS parallelisation — `ThreadPoolExecutor` parallelises row computation across phrase_list1 | `backend/Features/LCS/getLCSweights.py` |
| 2026-03-22 | [x] | TRAINING: Precompute features + retrain all 3 modes — cvg 87.38% @ ep4, rvg 86.45% @ ep12, mvm 85.98% @ ep5 | `backend/precompute_features.py`, `backend/train.py`, `models/*/` |

## Session 2026-03-22 — Batching optimisations

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-03-22 | [x] | PERF: Semantic batching — class-level `_embedding_cache` keyed by sentence; `__local__` encodes only unseen sentences in one `model.encode()` call per model, serves cached embeddings for repeat sentences; eliminates redundant encodes during precompute | `backend/Features/Semantic/__generate_semantic_features.py` |
| 2026-03-22 | [x] | PERF: GLiNER batching — `_batch_get_entities()` sends all sentences (both sides) in a single `batch_predict_entities()` call; fallback to per-sentence loop for older GLiNER versions | `backend/Features/EntityGroups/getOverlap.py` |
| 2026-04-01 | [x] | PERF: `predict_pair_breakdown` cache-first + parallel extractors — checks feature cache first (dict format); on hit skips all extractors; on miss runs all 5 in parallel via ThreadPoolExecutor; saves result to cache so predict_pair and breakdown share the same store | `backend/predict.py` |
| 2026-04-01 | [x] | FIX: Feature cache format — `TextSimilarityDataset` now saves feature_map dict `{key: list}` instead of stacked tensor; reader supports both formats (dict → reconstruct via feature_map_to_tensor, list → legacy path) | `backend/train.py` |
| 2026-04-01 | [x] | BUG: Breakdown coref resolver corrupts code inputs — GPT-4o-mini sent each line of a multi-line function as a "sentence" and returned verbose explanations as sentence text; fix: (1) removed resolve_coref=True from predict_pair_breakdown, (2) added _is_code() heuristic to split_txt to detect code and split on blank lines only instead of every newline | `backend/predict.py`, `backend/Splitter/sentence_splitter.py` |
| 2026-04-01 | [x] | BUG: HTTP 500 "Cannot copy out of meta tensor" — transformers 4.51.0 defaults to low_cpu_mem_usage=True on large models (roberta-large-mnli), initialising on meta device; .to(device) then fails; fix: pass low_cpu_mem_usage=False to from_pretrained; sentence-transformers 5.x same issue → pass device= to SentenceTransformer constructor | `backend/Features/NLI/getNLIweights.py`, `backend/Features/Semantic/__generate_semantic_features.py` |
| 2026-04-01 | [x] | FEATURE: Misalignment diagnostics — `_generate_misalignment_reasons()` produces ranked reasons (high/medium/low severity) from feature_scores + divergence data; 12 rules covering entailment conflict, semantic divergence, entity substitution, weak entailment, partial entity overlap, low lexical overlap, uncovered claims, unanchored content, structural mismatch, abstractive paraphrase risk, low sequential overlap; added to breakdown API response and rendered in BreakdownPanel | `backend/predict.py`, `backend/api/schemas.py`, `frontend/src/types/index.ts`, `frontend/src/components/BreakdownPanel.tsx` |
| 2026-04-01 | [x] | DOCS: Model-vs-model Playwright demo — two WebM recordings embedded in README (pair + drill-down, batch eval); videos committed to `docs/videos/`; PR #36 | `README.MD`, `docs/videos/`, `frontend/e2e/model-vs-model-demo.spec.ts`, `frontend/playwright.config.ts` |

## Session 2026-03-22 — Benchmark & Data Roadmap

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-03-31 | [x] | BENCHMARK: `backend/benchmark.py` — SilverBullet vs NLI-DeBERTa-v3-base and STS-RoBERTa-base on all 3 modes; full metrics + latency + failure cases | `backend/benchmark.py`, `benchmark_reports/` |
| 2026-03-31 | [x] | BENCHMARK: Failure-case analysis included in benchmark reports; SilverBullet outperforms on accuracy/MCC all modes; STS-RoBERTa-base has higher raw AUC on model-vs-model | `benchmark_reports/` |
| 2026-03-22 | [x] | DATA: Scout RAG hallucination examples for context-vs-generated — added HaluEval-Sum (document+summary) + HaluEval-Dial (knowledge+response) | `backend/fetch_external_data.py` |
| 2026-03-22 | [x] | DATA: Scout faithful paraphrase + abstractive summary pairs for reference-vs-generated — added PAWS (adversarial paraphrase pairs) | `backend/fetch_external_data.py` |
| 2026-03-22 | [x] | DATA: Scout real LLM-output pairs for model-vs-model — added PAWS + MRPC (sentence-level paraphrase equivalence) | `backend/fetch_external_data.py` |
| 2026-04-01 | [x] | FEATURE: Per-type entity maps — `EntityMismatch` (1 map) replaced by 14 per-type agreement maps; `ENTITY_TYPES` + `ENTITY_FEATURE_KEYS` moved to `feature_registry.py` as single source of truth; entity types expanded to cover named (person/org/location/product/event), legal/linguistic (law/language), temporal (date/time/duration), and quantitative (number/quantity/percentage/money); GLiNER now extracts all numeric/date entities, superseding the regex-based numeric feature | `backend/Features/EntityGroups/getOverlap.py`, `backend/feature_registry.py` |
| 2026-04-01 | [x] | FEATURE: BERTScore-style PREC/REC coverage maps — computed as max-pool over the cosine matrix already built in SemanticWeights; PREC[i,j] = best-match score for each text2 sentence (hallucination signal), REC[i,j] = best-match score for each text1 sentence (omission signal); produced for both mxbai and Qwen3 (+4 maps); soft-alignment step guarded to skip PREC_/REC_ keys | `backend/Features/Semantic/getSemanticWeights.py`, `backend/feature_registry.py` |
| 2026-04-01 | [x] | FEATURE: ROUGE-3 trigram recall map — added to LexicalWeights using existing `ngrams()` helper and tokenizer; computed in the same parallel row loop as other lexical features (+1 map) | `backend/Features/Lexical/getLexicalWeights.py`, `backend/feature_registry.py` |
| 2026-04-01 | [x] | INFRA: Delete ./cache/ and retrain all 3 modes — feature count changed 16→34 (v2.0→v3.0 manifest); existing checkpoints are incompatible | delete `./cache/`, `python -m backend.train --mode <each>` |
| 2026-04-01 | [x] | PERF: Semantic + Entity batch pre-encoding — `_prefill_semantic_cache()` / `_prefill_entity_cache()` collect all unique sentences upfront, issue one model call per model before the per-pair loop; added `EntityMatch._entity_cache` so repeated sentences are never re-run through GLiNER; NLI skipped (cross-encoder, needs pairs) | `backend/train.py`, `backend/Features/EntityGroups/getOverlap.py` |
| 2026-04-01 | [x] | IMPROVEMENT: Auto-start MLflow server if not reachable — `_ensure_mlflow_server(uri)` probes `/health`, spawns `mlflow server` subprocess if down, waits up to 30s | `backend/train.py` |
| 2026-04-01 | [x] | TRAINING: context-vs-generated retrain on expanded dataset (old 16-feature pipeline) — 80.84% val acc @ ep9, early stop ep14; below target, superseded by v3.0 feature set | `models/context-vs-generated/` |
| 2026-04-01 | [x] | TRAINING: context-vs-generated v3.0 (34 features) — 79.64% val acc @ ep8, early stop ep13; below baseline, see capacity fix below | `models/context-vs-generated/` |
| 2026-04-01 | [x] | FIX: CNN capacity + optimizer — hidden_dim 128→256, patience 5→8, Adam→AdamW (lr=0.0003, wd=1e-4), CosineAnnealingLR(T_max=50, eta_min=1e-6), drop_last=True on train loader | `backend/model.py`, `backend/train.py` |
| 2026-04-01 | [x] | TRAINING: All 3 modes v3.0 retrain with AdamW+cosine — cvg: best val 0.1498 ep9, rvg: best val 0.1225 ep12, mvm: best val 0.1445 ep8; ~79% accuracy ceiling consistent across modes | `models/*/` |
| 2026-04-01 | [x] | EVAL: Test set evaluation all 3 modes — cvg: 79.46% acc / 0.875 AUC / F1 0.801; rvg: 78.99% acc / 0.872 AUC / F1 0.800; mvm: 78.27% acc / 0.858 AUC / F1 0.808; ceiling is data quality/quantity, not architecture | `test_reports/` |
| 2026-04-02 | [x] | EXPERIMENT: Label smoothing (ε=0.05) + length conditioning (log n, log m → FC head) — v5.0 test: cvg 75.89% (−2%), rvg 79.35% (+0.7%), mvm 77.38% (−2.4%); mvm/cvg early-stopped at ep2-3, model destabilised by length scalars at ~1500 pairs; reverted to v4.0b checkpoints, kept code changes (use_length_cond flag supported but CLI defaults to False, label_smooth=0.0) | `backend/model.py`, `backend/train.py`, `backend/predict.py`, `backend/test.py` |
| 2026-04-02 | [ ] | IMPROVEMENT: Accuracy ceiling — next options: (1) per-mode data audit (cvg HaluEval label noise suspected), (2) more training data, (3) focal loss / class-weighted MSE to down-weight easy examples | `backend/train.py`, `data/` |

## Session 2026-04-02 — Feature ablation study (v4.0a → v4.0b)

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-04-02 | [x] | ABLATION: Run v3.0 (34-feature) ablation — `ablation_cluster.py` framework; 6293 pairs; 6-measure signal tiers; SOFT_ROW/SOFT_COL all NOISE/MARGINAL (p=0.47-0.80); entity_event/money/organization NOISE | `ablation_reports/experiments/20260401_201228_v3.0-all-modes/` |
| 2026-04-02 | [x] | FIX: `_feature_vector()` in ablation_cluster.py required exact key match — breaks when FEATURE_KEYS is pruned but cache has more keys; changed to subset match (only require FEATURE_KEYS ⊆ cache keys) | `backend/ablation_cluster.py` |
| 2026-04-02 | [x] | PRUNE v4.0a (34→30): Drop SOFT_ROW/SOFT_COL (4 features, p=0.47-0.80 confirmed noise); remove `__calc_soft_alignment__()` from SemanticWeights.getFeatureMap(); silence extra-key warning in feature_map_to_tensor; retrain all 3 modes | `backend/Features/Semantic/getSemanticWeights.py`, `backend/feature_registry.py`, `backend/train.py` |
| 2026-04-02 | [x] | TRAINING v4.0a (30 features): cvg val 79.64% ep10, rvg val 82.48% ep11 (+3.5%), mvm val 81.44% ep8 (+3.2%) — SOFT_ROW/SOFT_COL drop confirmed beneficial | `models/*/` |
| 2026-04-02 | [x] | ABLATION: Re-run on v4.0a (30-feature) — confirms 3 more DROP (entity_organization p=0.76, entity_event p=0.52, entity_money p=0.36); 4 more MARGINAL with p≥0.15 (entity_person, entity_date, entity_number, entity_quantity) | `ablation_reports/experiments/20260402_191507_v4.0a-all-modes/` |
| 2026-04-02 | [x] | PRUNE v4.0b (30→22): Drop 8 entity features with p≥0.09 (no Bonferroni significance): organization, event, person, money, date, number, quantity, language; keep 6: location, product, law, time, duration, percentage | `backend/feature_registry.py` |
| 2026-04-02 | [x] | TRAINING v4.0b (22 features): cvg val 78.44% ep14, rvg val 81.02% ep5, mvm val 81.74% ep14 | `models/*/` |
| 2026-04-02 | [x] | EVAL v4.0b test set: cvg 77.98% / rvg 78.62% / mvm 79.76% — vs v3.0 (79.46/78.99/78.27); mvm +1.5%, cvg/rvg within noise; 35% fewer features | `test_reports/` |
| 2026-04-02 | [x] | ABLATION: Re-run on v4.0b (22-feature) — ZERO DROP features; all 22 features ≥ MARGINAL; study converged | `ablation_reports/experiments/20260402_*_v4.0b-all-modes/` |

## Session 2026-04-02 — Data audit + model distribution

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-04-02 | [x] | DATA AUDIT: Identified cvg label noise — STS-B (similarity, not grounding) + MNLI (entailment, not grounding) account for 800/2231 pairs (36%); excluded from cvg assembly, kept for rvg/mvm where they are valid proxies | `backend/fetch_external_data.py` |
| 2026-04-02 | [x] | DATA: Increase HaluEval sampling to 700/source (from 400) to compensate for STS-B/MNLI removal; cvg now 2331 pairs (100% HaluEval + handcrafted) vs 2231 before; added `--halueval-max` flag | `backend/fetch_external_data.py`, `data/context-vs-generated/` |
| 2026-04-02 | [~] | TRAINING: Retrain cvg on clean HaluEval-only data — in progress | `models/context-vs-generated/best.pth` |
| 2026-04-02 | [x] | FEATURE: `backend/model_hub.py` — auto-download checkpoints from HuggingFace Hub when `SB_HF_REPO_ID` env var is set; wired into `api/dependencies.py`; no-op when env var unset or file present | `backend/model_hub.py`, `backend/api/dependencies.py` |

## Session 2026-04-03 — Ablation v4.1 + new data sources

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-04-03 | [x] | ABLATION: v4.1 pruning — drop 7 correlated-redundant features (cross-r ≥ 0.93): mxbai_cosine, REC_mxbai, Qwen_cosine, REC_Qwen, jaccard, cosine_lexical, rouge_unigram; 22→15 features; version bumped to 4.1 | `backend/feature_registry.py`, `backend/Features/Lexical/getLexicalWeights.py`, `backend/Features/Semantic/getSemanticWeights.py` — commit 74ffa19 |
| 2026-04-03 | [~] | TRAINING: v4.1 retrain all 3 modes (cache hit — no recompute needed; dict cache pre-selects from 22 available keys) | `models/*/best.pth` — cvg in progress |
| 2026-04-03 | [ ] | EVAL: Run python -m backend.test for all 3 modes after v4.1 training; compare vs v4.0b baseline (cvg 77.98% / rvg 78.62% / mvm 79.76%) | `test_reports/` |
| 2026-04-03 | [ ] | ABLATION: Re-run ablation_cluster on v4.1 15-feature set — tag v4.1-all-modes | `ablation_reports/experiments/` |
| 2026-04-03 | [x] | DATA SCOUTING: Identified 5 new data sources — RAGTruth (cvg, ~18k), FaithDial (cvg, ~50k), ANLI-R3 (rvg, ~45k), WiCE (rvg, ~8.8k), SummaC (cvg+rvg, ~1.6k); all on HuggingFace Hub | research |
| 2026-04-03 | [x] | DATA: Implement loaders for RAGTruth, FaithDial, ANLI-R3, WiCE, SummaC in fetch_external_data.py; field-name probing + graceful fallback for each | `backend/fetch_external_data.py` — commit 4305597 |
| 2026-04-03 | [ ] | DATA: Run python -m backend.fetch_external_data to pull new datasets and rebuild splits | `data/`, `data/external/` — requires re-run after v4.1 training complete |
| 2026-04-03 | [ ] | TRAINING: Retrain all 3 modes on expanded dataset (after fetch_external_data re-run) — delete ./cache/ first to recompute features for new pairs | `models/*/`, `./cache/` |

## Feature Roadmap — LLM-Jury Mode

> **Idea (from THOUGHTS.md):** Use a panel of LLMs as binary-question juries that measure the same dimensions
> as the CNN feature pipeline, but with reasoning + structured validation codes per judgment.
> User can toggle between CNN mode (fast, offline) and LLM-jury mode (interpretable, no training needed).
> Compare both to surface systematic disagreements (hard cases where CNN != LLM consensus).

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-04-06 | [ ] | DESIGN: Define jury question set — one binary question per feature cluster (NLI: "Does text2 logically contradict text1?"; Entity: "Does text2 introduce entities absent from text1?"; Semantic: "Is text2 semantically grounded in text1?"; Lexical: "Does text2 use the same key terminology as text1?"); each question returns answer + 1-sentence reasoning + a validation code (e.g. HALL-NUMERIC, ENT-SUBST, FACTUAL-OMIT, STRUCT-MISMATCH) | `backend/jury/questions.py` (new) |
| 2026-04-06 | [ ] | DESIGN: Validation code taxonomy — enumerated codes corresponding to the 12 misalignment reasons already in `_generate_misalignment_reasons()`; codes become machine-readable tags that enable downstream filtering and analysis | `backend/jury/codes.py` (new), `backend/predict.py` |
| 2026-04-06 | [ ] | IMPL: `backend/jury/llm_jury.py` — `LLMJury.evaluate(text1, text2, mode) -> JuryReport`; sends all questions in a single structured prompt (few-shot); parses binary answers + codes + reasoning from response; aggregates to score [0,1] via weighted majority | `backend/jury/llm_jury.py` (new) |
| 2026-04-06 | [ ] | IMPL: API endpoint `POST /api/v1/predict/pair/jury` — same request schema as `/pair`; returns `{score, votes: [{question, answer, code, reasoning}], consensus_codes: [...]}` | `backend/api/main.py`, `backend/api/schemas.py` |
| 2026-04-06 | [ ] | IMPL: CNN vs jury comparison — `backend/compare.py` runs both modes on a dataset and outputs disagreement cases sorted by |cnn_score - jury_score|; surfaces systematic CNN blind spots | `backend/compare.py` (new) |
| 2026-04-06 | [ ] | FRONTEND: Jury mode toggle in UI — switch between CNN score and jury breakdown; show per-question votes + reasoning + codes in a collapsible panel; codes rendered as colored chips | `frontend/src/components/JuryPanel.tsx` (new), `frontend/src/App.tsx` |

---

## Feature Roadmap — Relationship Extraction

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-04-01 | [ ] | FEATURE: Node-edge-node relationship extraction — use `knowledgator/gliner-relex-large-v1.0` (same `gliner` package, no new dependency); `model.inference(texts, labels=ENTITY_TYPES, relations=RELATION_TYPES, return_relations=True)` returns structured `{head.text, head.type, relation, tail.text, tail.type, score}` triples in a single forward pass; per-cell score = fraction of text1 sentence's triples that are grounded in text2 sentence (recall-style overlap on (head_type, relation, tail_type) with fuzzy entity text match) | new `backend/Features/Relations/getRelationWeights.py`, `backend/feature_registry.py` |
| 2026-04-01 | [ ] | DESIGN: Define `RELATION_TYPES` list in `feature_registry.py` — zero-shot so any verb-phrase label works; seed list: ["founded by", "located in", "employed by", "part of", "caused by", "treated with", "owned by", "acquired by", "published by", "created by"]; can be extended without retraining the gliner-relex model | `backend/feature_registry.py` |
| 2026-04-01 | [ ] | DEPENDENCY: Coreference resolution must be re-enabled before relationship extraction — raw pronouns produce useless triples ("He founded it" → no grounding value); coref resolves per-text before splitting (cheaper than cross-text coref, sufficient for triple grounding); same coref pass will also improve entity type maps and numeric maps since entity names become consistent across sentences | `backend/Splitter/sentence_splitter.py` `resolve_coref` flag, `backend/Preprocess/coref/resolveEntity.py` |
| 2026-04-01 | [ ] | DESIGN: Coref scope decision — per-text coref (resolve pronouns within each text independently before `split_txt`) is sufficient for triple extraction and entity matching; cross-text coref (resolve across both texts jointly) would additionally help surface shared entity references between text1 and text2 but requires a stronger model than the current GPT-4o-mini pronoun resolver; start with per-text | architecture decision |
| 2026-04-01 | [ ] | INFRA: Verify `gliner` version supports `model.inference(..., return_relations=True)` — gliner-relex API was stabilised in gliner≥0.2.26; current pin is 0.2.22; check if upgrading breaks `transformers==4.51.0` constraint (gliner-relex uses same GLiNER architecture so likely safe) | `requirements.txt`, `pyproject.toml` |

## Session 2026-04-03 — Data source research

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-04-03 | [~] | RESEARCH: Evaluate 10 public dataset candidates for expanding cvg/rvg/mvm training data | TASK.md (this entry) |

## Planned Refactor Roadmap
| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-03-13 | [ ] | Restructure into clear layers: `api/` transport, `app/services` + `domain`, `infrastructure/` (features, model loader, config, cache, logging), `cli/` utilities | create new package structure |
| 2026-03-13 | [ ] | Config bootstrap with no side effects; move HF/OpenAI login & model loading into explicit startup | `infrastructure/config/*`, `api/main.py` |
| 2026-03-13 | [ ] | Feature registry + checkpoint manifest (`version`, ordered features, `input_dim`, `model_hash`) and compatibility checker | `infrastructure/ml/*`, `train.py`, `predict.py`, `test.py` |
| 2026-03-13 | [ ] | Stateless extractors with injected resource factory; centralized padding/truncation + pre-flight guards | `Features/**`, `infrastructure/ml/tensor_utils.py` |
| 2026-03-13 | [ ] | API additions: `/ping`, `/metadata`, structured error mapper, re-enable rate limiting with compatible middleware | `api/main.py`, `api/schemas.py`, `api/middleware.py` |
| 2026-03-13 | [ ] | Artifact layout `artifacts/{run_id}/(model.pt, manifest.json, report.json)` + minimal inference bundle exporter | `train.py`, `predict.py`, `cli/` |
| 2026-03-13 | [ ] | Frontend cleanup: shared `src/lib/api.ts`, React Query state hooks, shared UI primitives in `src/ui/`, screens stay thin | `frontend/src/lib/*`, `frontend/src/ui/*`, `frontend/src/components/*` |
| 2026-03-13 | [ ] | Testing/CI: contract tests for metadata/compatibility, service tests with fake extractors, msw-backed frontend tests; CI runs lint+type+pytest+vitest | `tests/**`, `frontend/src/**`, CI config |
| 2026-03-13 | [ ] | Observability: structured JSON logs with request_id, metrics emitter, cache stats endpoint/CLI, optional experiment log (sqlite/JSONL) | `api/middleware.py`, `infrastructure/logging/*`, `cli/cache_inspect.py` |

### Modularity notes (guiding principles)
- Keep modules pure where possible; no network/model loads on import. All side-effects live in bootstrap/startup functions.
- Each extractor implements `FeatureExtractor` interface; registered via a small `feature_registry` so adding/removing features never touches API/training code.
- Services expose interfaces (`PredictionService`, `MetadataService`) consumed by API and CLI; adapters (FastAPI handlers, CLIs) own serialization only.
- Configuration resolved once and passed down; avoid `getVal()` globals—prefer dependency injection or constructor params.
- Model artifacts always accompanied by `manifest.json`; loaders validate version/feature list before use to prevent silent shape drift.
- Frontend uses shared API client + UI primitives; screens compose hooks (`useHealth`, `useMetadata`, `usePredict`) to keep concerns separated.

### What to modularize next & how
- **Config**: Replace `resources/getConfig.py` with `infrastructure/config` that loads env/.env via pydantic; pass config into services; remove hardcoded paths.
- **Feature pipeline**: Move tokenizer/model caching to `FeatureResources`; extract common tensor ops to `tensor_utils`; make each extractor a thin class with no globals.
- **Model loading**: Wrap checkpoint IO in `ModelLoader` (load, validate manifest, choose adapter); expose `load_for_api()` and `load_for_train_eval()`.
- **API dependencies**: Dependency providers return services, not raw predictors; handlers stay transport-only. Centralize error mapping in one module.
- **Caching**: Encapsulate feature cache behind `FeatureCache` interface; allow in-memory vs disk implementations; inject into services.
- **Logging/metrics**: Single module to build loggers and meters; middleware depends on interfaces, making it swappable/testable.
- **CLI/operations**: Small commands that call services (clear cache, inspect manifest, convert checkpoints) instead of duplicating logic.
- **Frontend state**: Shared hooks for API + React Query client; UI primitives for buttons/alerts/inputs to avoid Tailwind duplication; data parsing lives in hooks, not components.
