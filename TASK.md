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
| 2026-04-06 | [x] | LLM-as-jury evaluation path — 2 new endpoints + JuryEvaluator | `backend/jury/__init__.py`, `backend/jury/jury_evaluator.py`, `backend/api/schemas.py` (JuryQuestion/JuryResult/JuryRequest/JuryBatchRequest/JuryBatchResponse), `backend/api/main.py` (POST /api/v1/predict/jury/pair 20/min, POST /api/v1/predict/jury/batch 5/min) |
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

## Session 2026-04-06 — LLM-jury endpoints + v4.1→v4.4 feature recovery

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-04-06 | [x] | FEATURE: LLM-as-jury evaluation endpoints — `JuryEvaluator` (gpt-4o-mini, 6 binary questions across feature clusters, weighted score aggregation, validation codes HALL-NUMERIC/ENT-SUBST/NEG-FACT/OMIT-KEY/FAITHFUL); `POST /api/v1/predict/jury/pair` (20/min) + `/jury/batch` (5/min, max 10) | `backend/jury/__init__.py`, `backend/jury/jury_evaluator.py`, `backend/api/schemas.py`, `backend/api/main.py` |
| 2026-04-06 | [x] | ANALYSIS: v4.1 CVG regression investigation — per-mode label correlation analysis on 300 CVG pairs; `rouge` unigram was 3rd-ranked feature (r=-0.183) but dropped globally; cross-r with dice/rouge3 ≥ 0.93 was misleading — cross-r ≠ equal discriminative power per mode | `ablation_reports/`, `backend/feature_registry.py` |
| 2026-04-06 | [x] | PRUNE/RESTORE v4.2 (15→17): Restore `rouge` (CVG label-r=-0.183) + `REC_mxbai` (RVG=+0.388, MVM=+0.421, opposite sign to PREC_mxbai on CVG); confirmed REC_Qwen redundant (cross-r with PREC_Qwen=+0.999 on RVG) | `backend/feature_registry.py`, `backend/Features/Lexical/getLexicalWeights.py`, `backend/Features/Semantic/getSemanticWeights.py` |
| 2026-04-06 | [x] | FIX: train.py cache fallthrough — incomplete dict entries (missing keys after feature set change) now fall through to recompute instead of crashing; `_cache_entry_complete()` + `_missing_groups()` helpers added | `backend/train.py` |
| 2026-04-06 | [x] | FIX: Partial cache recompute — stale cache entries now run only missing extractor groups (lexical/semantic/nli/entity/lcs) instead of full pipeline; prevents semantic re-encoding when only a lexical feature is added; `_prefill_semantic_cache` / `_prefill_entity_cache` skip pairs whose relevant keys are already cached | `backend/train.py` |
| 2026-04-07 | [x] | PRUNE/RESTORE v4.3 (17→18): Restore `jaccard` (CVG label-r=-0.124, RVG=+0.213); cross-r=0.983 with dice was misleading same as rouge | `backend/feature_registry.py`, `backend/Features/Lexical/getLexicalWeights.py` |
| 2026-04-07 | [x] | PRUNE/RESTORE v4.4 (18→19): Restore `mxbai_cosine` raw pairwise map (CVG=+0.098, RVG=+0.352, MVM=+0.439); full n×m structure CNN can't learn from PREC/REC alone | `backend/feature_registry.py`, `backend/Features/Semantic/getSemanticWeights.py` |
| 2026-04-07 | [x] | EVAL v4.4 test set: CVG 74.64% / AUC 0.801, RVG 76.81% / AUC 0.868, MVM 83.04% / AUC 0.889 — MVM +5.7% vs v4.0b; RVG/CVG within noise of v4.0b with 3 fewer features | `test_reports/` |
| 2026-04-07 | [x] | ABLATION: Re-run ablation_cluster on v4.4 19-feature set — 1 DROP (entity_time NOISE), 14 REVIEW, 4 KEEP (entailment/contradiction STRONG, neutral/entity_percentage WEAK); entity_time to be pruned in v4.5 | `ablation_reports/experiments/20260407_174417_v4.4-all-modes/` |
| 2026-04-07 | [x] | PRUNE v4.5 attempted + reverted: entity_time removal cost CVG -3pt (training variance, not causal). entity_time retained. | `backend/feature_registry.py` |
| 2026-04-07 | [x] | v5.0 mode-specific feature baskets: CVG=13, RVG=15, MVM=14. ROC +0.028/+0.013/+0.008 vs v4.4. 0 DROPs in ablation. | `backend/feature_registry.py` (FEATURE_KEYS_BY_MODE), `backend/train.py`, `backend/predict.py`, `backend/test.py`, `backend/ablation_cluster.py` |
| 2026-04-07 | [x] | ANALYSIS: Jury vs CNN comparison (CVG, n=50): 62% agreement. CNN_MISS=10 (lexically anchored — misses faithful paraphrases); JURY_MISS=9 (jury fails on fragment/minimal answers). Real CNN failure: entity substitution (Piranha 3D cast). Key insight: CNN is lexically anchored, jury is semantically anchored — boundary-band hybrid (jury when CNN 0.3–0.7) could fix both. | `backend/jury/compare.py`, `jury_reports/` |
| 2026-04-07 | [x] | v5.1 factual groundedness — new CNN features: entity_value_prec/rec (fuzzy string match of GLiNER entity values across texts) + numeric_jaccard (Jaccard over normalised number sets $8B→8B, 25%→25.0%); wired into train.py + feature_registry.py; NumericGrounding extractor (pure-Python, regex+normaliser) | `backend/Features/Numeric/getNumericGrounding.py`, `backend/Features/EntityGroups/getOverlap.py`, `backend/feature_registry.py`, `backend/train.py` |
| 2026-04-07 | [x] | Jury v5.1 question update: Q2 lexical→omission_key_claim (higher_is_faithful=False); Q1 weight 1.0→0.5 (collinear with Q3 nli_entailment); Q7 numeric_hallucination added (w=1.0, inverted); HALL-NUMERIC reasoning guidance added to system prompt | `backend/jury/jury_evaluator.py` |
| 2026-04-07 | [x] | v5.2 per-type entity value comparison: `_per_type_value_overlap()` + `_FUZZY_THRESHOLDS` per type (location=0.85, product=0.80, date/time=0.90, duration=0.88, percentage=0.95); 12 new feature maps in `comparison_weights`; ENTITY_VALUE_TYPES + ENTITY_VALUE_KEYS in registry; FEATURE_KEYS_BY_MODE updated (CVG=21, RVG=26, MVM=23) | `backend/Features/EntityGroups/getOverlap.py`, `backend/feature_registry.py` |
| 2026-04-08 | [x] | DELETE ./cache/ and retrain CVG + RVG on v5.2 feature set — CVG: 21 features, best val 0.1747 @ ep10, early stop ep18, val acc 74.21%; RVG: 26 features, best val 0.1301 @ ep10, early stop ep18, val acc 83.94% | `./cache/`, `models/context-vs-generated/`, `models/reference-vs-generated/`, `train_cvg.log`, `train_rvg.log` |
| 2026-04-08 | [x] | RETRAIN MVM on v5.2 feature set (23 features) — best val loss 0.1539 @ ep13, early stop ep21, val acc 78.14% @ ep20 | `models/model-vs-model/20260408_103121_best.pth`, `train_mvm.log` |
| 2026-04-08 | [x] | EVAL CVG v5.2 test set: 70.66% acc / AUC 0.8063 / AUPRC 0.8277 / MCC 0.4228 — REGRESSION vs v5.0 (AUC 0.829); per-type entity value features (8 sparse maps) add noise: both-empty→1.0 for most HaluEval pairs | `test_reports/test_report_20260408_102442.json` |
| 2026-04-08 | [x] | EVAL RVG v5.2 test set: 78.72% acc / AUC 0.8703 / AUPRC 0.8471 / MCC 0.5759 — REGRESSION vs v5.0 (AUC 0.882); same sparse-feature noise issue | `test_reports/test_report_20260408_103632.json` |
| 2026-04-08 | [x] | EVAL MVM v5.2 test set: 80.90% acc / AUC 0.8906 / AUPRC 0.8553 / MCC 0.6196 — slight regression vs v5.0 (AUC 0.897); smallest gap of three modes | `test_reports/test_report_20260408_104228.json` |
| 2026-04-08 | [x] | ABLATION v5.2 all 3 modes — CVG: 1 DROP (entity_time_value_rec), 7 MARGINAL per-type; RVG: 0 DROPs, all per-type MARGINAL (ns Bonferroni); MVM: 2 DROPs (entity_time_prec/rec NOISE), entity_percentage_value_prec/rec KEEP (p<1e-6) | `ablation_reports/experiments/20260408_*_v5.2-*/` |
| 2026-04-08 | [x] | v5.3 feature baskets — CVG=17 (+entity_product_value_prec only, p=6e-4); RVG=18 (0 per-type, none pass Bonferroni); MVM=19 (+entity_percentage_value_prec/rec p<1e-6) | `backend/feature_registry.py` VERSION=5.3 |
| 2026-04-08 | [x] | RETRAIN all 3 modes on v5.3 — CVG: best val 0.1560 @ ep21, stop ep29, acc 79.08% (+5pt vs v5.2); RVG: best val 0.1316 @ ep10, stop ep18, acc 81.02%; MVM: best val 0.1524 @ ep2, stop ep10 | `models/*/`, `train_*.log` |
| 2026-04-08 | [x] | EVAL all 3 modes on v5.3 — CVG: 76.51% acc / AUC 0.852 (NEW BEST, +0.023 vs v5.0); RVG: 77.17% acc / AUC 0.872 (−0.010 vs v5.0, within noise); MVM: 80.95% acc / AUC 0.890 (−0.007 vs v5.0, within noise) | `test_reports/` |
| 2026-04-07 | [x] | DATA: Expand training set — RAGTruth (+400 CVG), ANLI-R3 (+400 RVG), WiCE (+349 RVG), MedHallu (+400 CVG), AporiaRAG (+400 CVG); CVG 2331→3531, RVG 1831→2580 | `backend/fetch_external_data.py` |
| 2026-04-07 | [ ] | DATA: Synthetic hallucination augmentation for CVG (paraphrase → inject hallucination via LLM, label=0); target CVG ROC > 0.86 | `backend/generate_data.py` |
| 2026-04-07 | [ ] | FEATURE: Score explainability narrative — post-process breakdown response through GPT-4o-mini to produce a single human-readable sentence; e.g. "Text 2 introduces a factual claim not grounded in Text 1 (entailment 0.12, contradiction 0.67)"; new endpoint or flag on existing breakdown | `backend/api/main.py`, `backend/api/schemas.py` |
| 2026-04-07 | [ ] | INSIGHT (LinkedIn): v5.0 architecture observation — same Conv2D topology for all modes despite NLI-dominant CVG vs semantic-dominant MVM; mode-specific architectures (deeper early stage for NLI modes, wider spatial RF for semantic modes) as next frontier | — |

## Session 2026-04-08/09 — Data expansion + NLI pair cache + expanded retraining

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-04-08 | [x] | Added cursor convention to global CLAUDE.md and TASK.md | `C:\Users\mehul\.claude\CLAUDE.md` |
| 2026-04-08 | [x] | PERF: NLI pair cache — `NLIWeights._pair_cache` keyed on (sent1, sent2); `_prefill_nli_cache()` in train.py batches all unique sentence pairs before training loop; RVG needed only 2303 new pairs scored (rest cached from CVG) | `backend/Features/NLI/getNLIweights.py`, `backend/train.py` |
| 2026-04-08 | [x] | DATA: Fixed 5 dataset loaders — RAGTruth (wandb/RAGTruth-processed), ANLI-R3 (facebook/anli, removed trust_remote_code), WiCE (jon-tow/wice, config=claim); FaithDial/SummaC unavailable (custom loading scripts deprecated in datasets>=3.0) | `backend/fetch_external_data.py` |
| 2026-04-08 | [x] | DATA: Added MedHallu loader (UTAustin-AIHealth/MedHallu, pqa_labeled+pqa_artificial configs; Knowledge+GroundTruth→1, Knowledge+HallucinatedAnswer→0) | `backend/fetch_external_data.py` |
| 2026-04-09 | [x] | DATA: Added AporiaRAG loader (aporia-ai/rag_hallucinations; context+answer+is_hallucination; 309/691 balanced to 200/200) | `backend/fetch_external_data.py` |
| 2026-04-09 | [x] | RETRAIN CVG on 2731 pairs (RAGTruth added) — best val 0.1583 @ ep20, stop ep28; test AUC 0.836 (−0.016 vs v5.3; MCC +0.062, more balanced) | `models/context-vs-generated/best.pth` |
| 2026-04-09 | [x] | RETRAIN RVG on 2580 pairs (ANLI-R3+WiCE added) — best val 0.1460 @ ep10, stop ep18, val acc 79.33% | `models/reference-vs-generated/best.pth` |
| 2026-04-09 | [x] | RETRAIN CVG on 3531 pairs (MedHallu+AporiaRAG added) — best val 0.1788 @ ep21, stop ep29 | `models/context-vs-generated/20260409_225448_best.pth` |
| 2026-04-09 | [x] | EVAL RVG on 2580-pair model — AUC 0.8734 (+0.001 vs v5.3 baseline 0.872), Acc 78.1%, MCC 0.5634 | `test_reports/test_report_20260409_003536.json` |
| 2026-04-11 | [x] | EVAL CVG on 3531-pair model — AUC 0.8529 (≈ v5.3 baseline), Acc 76.5%, MCC 0.530, AUPRC 0.869; balanced confusion (206/200) | `test_reports/test_report_20260411_093556.json` |
| 2026-04-09 | [x] | UI: Add MedHallu/medical + RAG hallucination test cases to frontend; update CVG mode description | `frontend/src/data/testCases.ts`, `frontend/src/config/modes.ts` |
| 2026-04-11 | [x] | COMMIT: checkpoints (CVG 3531-pair best.pth, RVG 2580-pair best.pth) + ablation logs + test reports — commit 0cbe1fd | `models/*/best.pth`, `test_reports/`, `*.log` |
| 2026-04-11 | [x] | RETRAIN MVM — best val 0.1480 @ ep10, early stop ep18 | `models/model-vs-model/20260411_132403_best.pth` |
| 2026-04-11 | [x] | EVAL MVM — AUC 0.8892 (+0.019 vs ~0.870 prev), Acc 81.9%, MCC 0.6445, F1 0.827 | `test_reports/test_report_20260411_133144.json` |
| 2026-04-11 | [x] | COMMIT MVM checkpoint + logs — commit 1930cb9 | `models/model-vs-model/best.pth`, `train_mvm_new.log`, `test_mvm_new.log` |
| 2026-04-11 | [x] | DATA: Adversarial augmentation — +200 hard negatives per mode (100 numeric-swap + 100 entity-swap) appended to train splits | `backend/augment_data.py`, `data/*/train.json` |
| 2026-04-11 | [ ] | RETRAIN all 3 modes on augmented data (v5.4 + adversarial pairs) — v5.4 retrain already in progress | `models/*/best.pth` |

## Session 2026-04-11 (continued) — v5.4 retrain cycle

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-04-11 | [x] | EVAL CVG v5.4 (18 features): AUC 0.8510, Acc 75.7%, MCC 0.539, AUPRC 0.864 | `test_reports/test_report_20260411_162606.md` |
| 2026-04-11 | [x] | DISCOVERY: bumjqc1oh background command trained RVG (18 feat) + MVM (19 feat) BEFORE entity_grounding_recall was added to their baskets — both are effectively v5.3 | `train_rvg.log`, `train_mvm.log` |
| 2026-04-11 | [~] | RETRAIN RVG (19 feat) + MVM (20 feat) with correct v5.4 feature baskets — running in background (bo2ichd9y) | `models/reference-vs-generated/best.pth`, `models/model-vs-model/best.pth` |
| 2026-04-11 | [ ] | EVAL RVG + MVM after retrain completes | `test_reports/` |
| 2026-04-11 | [ ] | COMMIT all 3 v5.4 checkpoints + test logs; update PR #40 with final v5.4 numbers | `models/*/best.pth` |

## Session 2026-04-11 (continued) — v5.5 relation triplet recall

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-04-11 | [x] | FEATURE: Relation triplet recall — `RelexGrounding` using `knowledgator/gliner-relex-large-v1.0` (zero-shot joint NER+RE); extracts (head, relation_type, tail) triplets; recall of text1 triplets in text2; key `relation_triplet_recall`; VERSION=5.5 | `backend/Features/Relations/getRelexWeights.py`, `backend/feature_registry.py`, `backend/train.py`, `backend/predict.py`, `backend/precompute_features.py` |
| 2026-04-11 | [~] | RETRAIN all 3 modes v5.5 — CVG=19, RVG=20, MVM=21 features; running in background (bqzcbqmgb); cache incremental (only relation_triplet_recall patch-computed) | `models/*/best.pth` |
| 2026-04-11 | [ ] | EVAL all 3 modes v5.5 after retrain; compare vs v5.4 | `test_reports/` |
| 2026-04-11 | [ ] | COMMIT v5.5 checkpoints + test logs; push; update PR #40 | `models/*/best.pth` |

## Session 2026-04-12 — SQLite cache + training pipeline fixes

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-04-12 | [x] | INFRA: Unified SQLite cache (cache/silverbullet.db) — replaces nli_pairs.json, entity_sentences.json, relex_triplets.json, embeddings/*.npz, cache/{md5}.json; WAL mode; thread-local connections; auto-migration | `backend/cache_db.py` (new), `backend/feature_cache.py`, `getNLIweights.py`, `getOverlap.py`, `getRelexWeights.py`, `__generate_semantic_features.py` |
| 2026-04-12 | [x] | FIX: Add `_prefill_relex_cache()` to train.py — without it, gliner-relex ran per-pair sequentially in the extraction loop | `backend/train.py` |
| 2026-04-12 | [x] | FIX: Prefill functions now persist to SQLite explicitly — semantic prefill calls `save_embedding_cache()`, NLI prefill calls `save_pair_cache()` per batch | `backend/train.py` |
| 2026-04-12 | [x] | FIX: Batched RE — `_extract_triplets` now uses `GLiNERRelationExtractor` for fully-batched NER+RE (two `model.run()` calls); per-sentence fallback preserved | `backend/Features/Relations/getRelexWeights.py` |
| 2026-04-12 | [x] | ADD: tqdm progress bars for relex prefill (`Relex prefill X/N`) and RE step (`Relex parse`); chunked prefill at 128 sentences/chunk for crash-safety | `backend/train.py`, `backend/Features/Relations/getRelexWeights.py` |
| 2026-04-12 | [x] | RETRAIN CVG v5.5 — completed 2026-04-12 02:41, 19 features, best val loss 0.1992 @ epoch 10 | `models/context-vs-generated/20260412_024151_best.pth` |

## Session 2026-04-13 — Prefill cache-miss root cause + frontend redesign

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-04-13 | [x] | FIX: All 4 prefill functions called load_*_cache() AFTER filtering → every sentence appeared new; added load before filter in `_prefill_semantic_cache`, `_prefill_entity_cache`, `_prefill_nli_cache`, `_prefill_relex_cache` | `backend/train.py` |
| 2026-04-13 | [x] | FIX: Relex cache key mismatch — `_extract_triplets` checked full text against truncated [:512] keys; fixed to `t[:_MAX_CHARS] not in _triplet_cache` | `backend/Features/Relations/getRelexWeights.py` |
| 2026-04-13 | [x] | UI: Frontend redesign — dark telemetry aesthetic; Syne + DM Mono + DM Sans; grid texture; CSS vars; dark overrides for all Tailwind classes | `frontend/src/index.css`, `frontend/src/App.tsx` |
| 2026-04-13 | [x] | UI: Admin tab — ModelCard (per-mode accent), TrainingPanel (live logs, start/stop), BenchmarkTable, CacheStats, CLI reference with click-to-copy | `frontend/src/components/AdminPanel.tsx` |
| 2026-04-13 | [x] | API: Training job management endpoints — POST /admin/train/{mode}, POST /admin/train/{mode}/stop, GET /admin/train/status, GET /admin/train/logs/{mode}?offset=N | `backend/api/main.py` |
| 2026-04-13 | [x] | TYPES: TrainingJobStatus, TrainingLogsResponse; services: startTraining, stopTraining, getTrainingStatus, getTrainingLogs | `frontend/src/types/index.ts`, `frontend/src/services/api.ts` |
| 2026-04-13 | [~] | RETRAIN RVG v5.5 — PID 8892 (.sbvenv); NLI prefill ~38/46 when session resumed; 20 features | `models/reference-vs-generated/best.pth`, `train_rvg.log` |
| 2026-04-13 | [ ] | RETRAIN MVM v5.5 — chain after RVG completes; 21 features | `models/model-vs-model/best.pth`, `train_mvm.log` |
| 2026-04-13 | [ ] | EVAL all 3 modes v5.5 | `test_reports/` |
| 2026-04-13 | [ ] | COMMIT v5.5 checkpoints + test logs; push; update PR #40 | `models/*/best.pth` |

| 2026-04-14 | [x] | STABILITY: patience 8→15, num_epochs 50→75, warmup_epochs=5; LinearLR warmup + SequentialLR + CosineAnnealingLR — prevents early stopping on val noise peaks | `backend/train.py` |
| 2026-04-14 | [ ] | STABILITY: Stratified val split by dataset source — high variance from mixing STS-B/MNLI/HaluEval in same val batches | `backend/fetch_external_data.py`, `backend/train.py` |
| 2026-04-14 | [ ] | STABILITY: Per-source loss weighting — normalize calibration differences across external dataset label distributions | `backend/train.py` |
| 2026-04-14 | [ ] | STABILITY: Augment MVM val set to ≥600 samples — only 334 val samples → 11 batches → very high gradient variance | `backend/fetch_external_data.py`, `data/model-vs-model/` |

| 2026-04-14 | [x] | FIX: UnboundLocalError on NLIWeights — moved memory cleanup to `_free_extractor_models()` method (separate scope, imports cannot shadow __init__ names) | `backend/train.py` |
| 2026-04-14 | [x] | FIX: Chainer false-positive abort — now checks last 500 bytes only, not full new_text | `backend/chain_train.py` |
| 2026-04-14 | [x] | FIX: Stray processes — launch via Python subprocess.Popen instead of bash &; PID files written for each step | `backend/chain_train.py` |
| 2026-04-14 | [x] | SWITCH: relex model large→base (`knowledgator/gliner-relex-base-v1.0`); batch_size=16; cleared triplet cache | `backend/Features/Relations/getRelexWeights.py` |

| 2026-04-23 | [x] | ABLATION: Drop relation_triplet_recall (Relex) — SHAP ~0 all modes, 98.6% trivial-1.0, gliner-relex-base extracts nothing; VERSION 5.6→5.7 | `backend/feature_registry.py`, `backend/train.py`, `backend/predict.py`, `backend/precompute_features.py` |
| 2026-04-23 | [x] | DATA: Add SVO-friendly datasets — FEVER (copenlu/fever_gold_evidence 1000), SNLI (1000), SciTail (1000); +2000 RVG pairs (+1000 CVG); fetch_external_data.py updated | `backend/fetch_external_data.py`, `data/*/` |
| 2026-04-23 | [x] | TRAIN v5.7: CVG → RVG → MVM chain complete | `models/*/best.pth` |
| 2026-04-23 | [x] | TEST v5.7: CVG 0.8053/71.98% (↓ vs v5.6), RVG 0.9001/81.54% (↑), MVM 0.8919/81.66% (↓) | `test_cvg_v57.log`, `test_rvg_v57.log`, `test_mvm_v57.log` |
| 2026-05-07 | [x] | SHAP v5.7: CVG top=entailment/contradiction/lcs_char; drop entity_product_value_prec. RVG top=entailment/contradiction/jaccard; drop entity_percentage (7e-7, constant). MVM top=entailment/contradiction/jaccard; drop entity_percentage (6.7e-7). SVO rank 9/11/14 across modes — keep. | `shap_reports/` |
| 2026-05-07 | [x] | FEATURE v5.8: EFG (External Factual Grounding) extractor — DeBERTa-v3-base-mnli-fever-anli; 3 maps: efg_supports, efg_refutes, efg_factual_delta; SQLite cache; _prefill_efg_cache in train.py; all 3 modes; drop entity_product_value_prec (CVG), entity_percentage (RVG+MVM) | `backend/Features/Factual/getFactualGrounding.py`, `backend/feature_registry.py`, `backend/train.py`, `backend/cache_db.py`, `backend/precompute_features.py`, `backend/predict.py` |
| 2026-05-07 | [~] | TRAIN v5.8: CVG (21F) → RVG (22F) → MVM (23F) chain; EFG prefill first run will take ~2h on CPU | `train_cvg_v58.log`, `train_rvg_v58.log`, `train_mvm_v58.log` |
| 2026-05-07 | [ ] | TEST v5.8: all 3 modes; compare vs v5.7 baseline (CVG 0.8053/71.98%, RVG 0.9001/81.54%, MVM 0.8919/81.66%) | `test_cvg_v58.log`, `test_rvg_v58.log`, `test_mvm_v58.log` |
<!-- CURSOR: 2026-05-07 — CVG v5.8 training started; EFG prefill running (DeBERTa-FEVER, slow first run); wait for CVG done, then RVG, then MVM; baseline: CVG 0.8053, RVG 0.9001, MVM 0.8919 -->
| 2026-04-22 | [x] | SHAP: shap_analysis.py written — GradientExplainer, per-feature mean|SHAP|, bar+beeswarm plots, cross-mode summary table; shap==0.51.0 installed | `backend/shap_analysis.py`, `requirements.txt` |
| 2026-04-22 | [ ] | SHAP: Run analysis — `python -m backend.shap_analysis --mode all`; outputs to `shap_reports/{mode}/` | `shap_reports/` |
| 2026-04-22 | [x] | TEST: CVG v5.6 — ROC-AUC 0.8203, acc 0.7414 (↑ from v5.5: 0.8010 / 0.7362) | `test_cvg_v56.log` |
| 2026-04-22 | [~] | TRAIN: RVG v5.6 — running; SVO prefill warm | `train_rvg_v56.log` |
| 2026-04-22 | [ ] | TRAIN: MVM v5.6 — queued after RVG | `train_mvm_v56.log` |
| 2026-04-22 | [ ] | TEST: RVG v5.6 — after retrain | `test_rvg_v56b.log` |
| 2026-04-22 | [ ] | TEST: MVM v5.6 — after retrain | `test_mvm_v56b.log` |
| 2026-04-22 | [ ] | ABLATION: run ablation on all 3 v5.6 checkpoints | `ablation_reports/` |
| 2026-04-16 | [x] | FIX: GLiNER prefill deadlock — replaced ThreadPoolExecutor (can't kill C-ext threads) with multiprocessing.Process + 120s watchdog; module-level _entity_prefill_worker + _relex_prefill_worker for Windows spawn compat | `backend/train.py` |
| 2026-04-16 | [x] | FIX: Jury model selection — added `model: str | None` to JuryRequest schema; wired to JuryEvaluator(model=body.model); fixed frontend types (answer boolean→string, model→model_used, weight→confidence) | `backend/api/schemas.py`, `backend/api/main.py`, `frontend/src/types/index.ts`, `frontend/src/services/api.ts`, `frontend/src/components/JuryScorer.tsx`, `frontend/src/components/JuryPanel.tsx` |
| 2026-04-16 | [x] | FIX: TruthfulQA dataset URL — updated to truthfulqa/truthful_qa; improved FactCC/FRANK/AggreFact fallback chains | `backend/fetch_external_data.py` |
| 2026-04-22 | [x] | CVG v5.6 train — completed, epoch 45 early stop; checkpoint 20260422_010340_best.pth | `train_cvg_v56.log` |
| 2026-04-16 | [x] | RVG v5.5 train — completed, epoch 30 early stop; checkpoint 20260415_180204_best.pth | `train_rvg.log` |
| 2026-04-15 | [x] | MVM v5.5 train — completed; checkpoint 20260415_232310_best.pth | `train_mvm_v55.log` |
| 2026-04-12 | [x] | BENCHMARK: Add validation benchmark loaders to fetch_external_data.py — SummEval, FactCC, FRANK, AggreFact; written to data/benchmarks/ (never merged into training) | `backend/fetch_external_data.py` |
| 2026-04-12 | [x] | BENCHMARK: New benchmark_eval.py — runs SilverBullet on held-out benchmarks; reports ROC-AUC, PR-AUC, Pearson r, Spearman ρ vs. human labels; saves to benchmark_reports/ | `backend/benchmark_eval.py` (new) |
| 2026-04-12 | [ ] | BENCHMARK: Fetch benchmark data — `python -m backend.fetch_external_data --force` to download SummEval/FactCC/FRANK/AggreFact into data/benchmarks/ | `data/benchmarks/` |
| 2026-04-12 | [ ] | BENCHMARK: Run first evaluation after v5.5 training completes — `python -m backend.benchmark_eval`; target: ROC-AUC ≥ 0.85, Pearson r ≥ 0.65 (beats SummaC baseline) | `benchmark_reports/` |
| 2026-04-12 | [ ] | PERF: Pre-split + cache sentence lists — `split_txt()` is called per-pair in the extraction loop (creates fresh EntityResolver each time = ~500ms/pair); fix: pre-split all texts once before the loop and store in a dict; eliminate coref-resolver overhead at training time | `backend/train.py`, `backend/Splitter/sentence_splitter.py` |

## Session 2026-04-10 — Feature pattern analysis

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-04-10 | [x] | ANALYSIS: Feature pattern study — qualitative failure taxonomy + case analysis done; run_analysis.py written for quantitative tables | `analysis_reports/run_analysis.py`, `analysis_reports/feature_analysis_report.md`, `analysis_reports/feature_analysis_report.json` |
| 2026-04-10 | [ ] | ANALYSIS (quantitative): Run `python analysis_reports/run_analysis.py` to generate full per-feature mean/std/delta/Cohen's d tables (requires Bash permission; script is ready) | `analysis_reports/run_analysis.py` |

## Session 2026-04-10 — Entity grounding recall feature (v5.4)

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-04-10 | [~] | FEATURE: Entity grounding recall — `RelationGrounding` extractor using existing GLiNER model; recall of text1 entities in text2; key `entity_grounding_recall`; v5.4 registry; wired into train/predict/precompute | `backend/Features/Relations/__init__.py`, `backend/Features/Relations/getRelationWeights.py`, `backend/feature_registry.py` (VERSION=5.4, FEATURE_KEYS+all 3 mode baskets), `backend/train.py` (import+instantiation+_missing_groups+full-compute+patch-compute), `backend/predict.py` (predict_pair_breakdown extractor list), `backend/precompute_features.py` |
| 2026-04-10 | [ ] | VERIFY: Run `python -c "from backend.Features.Relations.getRelationWeights import RelationGrounding; print('OK')"` + `python -c "from backend.feature_registry import FEATURE_KEYS; print(FEATURE_KEYS)"` to confirm clean import and registry |
| 2026-04-10 | [ ] | COMMIT: `git add backend/Features/Relations/ backend/feature_registry.py backend/train.py backend/predict.py backend/precompute_features.py` then commit + push |
| 2026-04-10 | [ ] | RETRAIN: delete ./cache/ and retrain all 3 modes — feature count changed (v5.3→v5.4), old checkpoints incompatible |

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

## Feature Roadmap — v6 Architecture Ideas (2026-04-15)

> Three directions that came out of reasoning about what the current architecture misses.
> None require retraining from scratch — each is additive to the existing inference pipeline.

### v6-A: Min-alignment readout (catch partial hallucinations)

> **Problem:** A generated text that's 90% faithful but has one fabricated fact scores ~0.9.
> In production, "90% faithful with one critical lie" is a failure, not a success.
> **Fix:** Surface `min_alignment` (minimum cell value in the cross-attention matrix) alongside
> `avg_alignment`. If any generated sentence can't be grounded to any source sentence,
> flag it regardless of the aggregate score. Two numbers instead of one.

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-04-15 | [x] | IMPL: Add `min_alignment` to breakdown response — find the minimum value in the sentence-pair alignment matrix; return alongside `probability` in all breakdown endpoints | `backend/predict.py`, `backend/api/schemas.py` |
| 2026-04-15 | [x] | IMPL: Add `min_alignment_pair` to breakdown — (i, j) indices of the weakest sentence pair, so the caller knows exactly which generated sentence is least supported | `backend/predict.py`, `backend/api/schemas.py` |
| 2026-04-15 | [x] | FRONTEND: Render min-alignment indicator in BreakdownPanel — highlight the weakest cell in the alignment heatmap; show `min_alignment` score with a "weakest link" label | `frontend/src/components/BreakdownPanel.tsx` |

### v6-B: Hallucination type classifier (explain *how*, not just *where*)

> **Problem:** Sentence-level localization tells you *which* sentence is wrong, not *why*.
> The per-cell feature values already encode the answer — NLI contradiction → negation;
> high cosine + low entity overlap → entity substitution; triplet predicate mismatch → predicate flip;
> low everything → topic drift. It just isn't surfaced.
> **Fix:** 4-class classifier over the 16 feature values at each divergent cell.
> Training labels are implicit in HaluEval + RAGTruth annotations.

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-04-15 | [ ] | DESIGN: Define hallucination type taxonomy — 4 classes: `NEGATION` (NLI contradiction high), `ENTITY_SUBST` (cosine high + entity overlap low), `PREDICATE_FLIP` (triplet subjects match, predicates differ), `TOPIC_DRIFT` (all signals low); define thresholds per class | `backend/Features/` (new `hallucination_type.py`) |
| 2026-04-15 | [ ] | DATA: Extract per-cell labels from HaluEval + RAGTruth — use annotator comments to assign type labels to divergent sentence pairs; build small labeled dataset (~500 pairs) | `data/hallucination_types/` (new) |
| 2026-04-15 | [ ] | IMPL: Train tiny 4-class classifier (16 features → 4 classes) on labeled cell data; pickle alongside main model | `backend/hallucination_type_classifier.py` (new) |
| 2026-04-15 | [ ] | INTEGRATE: Add `hallucination_type` field to breakdown response for divergent sentences — run classifier on the (i,j) cell of each divergent sentence; return type + confidence | `backend/predict.py`, `backend/api/schemas.py` |

### v6-D: SVO extractor alongside Relex — ablation comparison

> **Observation:** `gliner-relex-base` is O(n²) per sentence — 8-12h prefill for CVG,
> 90s watchdog fires constantly. spaCy SVO via `token.dep_` runs in milliseconds.
> **Goal:** Add SVO as a second relation feature (`svo_triplet_recall`) alongside the
> existing `relation_triplet_recall` (Relex), retrain, and compare per-feature contribution
> via ablation. If SVO adds independent signal → keep both. If SVO substitutes Relex signal
> with negligible AUC loss → swap. Data decides.

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-04-21 | [x] | IMPL: `backend/Features/Relations/getSVOWeights.py` — spaCy `en_core_web_sm` SVO extractor; key `svo_triplet_recall`; same `getFeatureMap()` interface as `getRelexWeights.py` | `backend/Features/Relations/getSVOWeights.py` (new) |
| 2026-04-21 | [x] | REGISTER: Add `svo_triplet_recall` to `FEATURE_KEYS` and all 3 mode baskets in `feature_registry.py`; bumped VERSION to 5.6 | `backend/feature_registry.py` |
| 2026-04-21 | [x] | WIRE: SVOGrounding added to `_missing_groups` (new "svo" group), TextSimilarityDataset, patch path, full compute, cleanup in `train.py`; extractors list in `predict.py` and `precompute_features.py` | `backend/train.py`, `backend/predict.py`, `backend/precompute_features.py` |
| 2026-04-21 | [~] | TRAIN: CVG v5.6 training running — feature extraction 86%/4158 pairs; Relex prefill stalled at 160/1090 (watchdog fired), fallback to per-pair Relex; ~2-3h to epoch start | `train_cvg_v56.log` |
| 2026-04-21 | [x] | TEST: RVG v5.5 — Acc 80.52% [77.93, 83.10] / AUC 0.8936 [0.873, 0.913] / AUPRC 0.8802 / MCC 0.611 | `test_rvg_v56.log`, `test_reports/` |
| 2026-04-21 | [x] | TEST: MVM v5.5 — Acc 82.95% [80.15, 85.62] / AUC 0.8943 [0.870, 0.917] / AUPRC 0.8793 / MCC 0.660 | `test_mvm_v56.log`, `test_reports/` |
| 2026-04-21 | [ ] | ABLATION: Retrain all 3 modes with both features; run ablation dropping each independently — compare AUC(with Relex, without SVO) vs AUC(with SVO, without Relex) vs AUC(both) | `ablation_reports/` |

### v6-C: Adversarial hard-negative generation (fix the data, not the architecture)

> **Problem:** The model has seen too few adversarial pairs — high semantic similarity, low faithfulness.
> Ablation shows NLI entailment + relation triplets carry the most signal on hard cases,
> but semantic similarity is *adversarial* (near-1 cosine for logical opposites).
> The model can learn to down-weight semantic sim on hard negatives — but only if it's seen enough.
> **Fix:** Targeted adversarial pair generation for each of the 4 failure types.

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-04-15 | [ ] | IMPL: `backend/generate_adversarial.py` — for each label=1 pair in training data, generate one adversarial label=0 variant per failure type: (1) negate a key verb; (2) swap one entity; (3) invert a relation predicate; (4) substitute an adjacent topic sentence | `backend/generate_adversarial.py` (new) |
| 2026-04-15 | [ ] | IMPL: Use GPT-4o-mini for adversarial transforms — prompt: "Modify this sentence to change exactly [entity/predicate/negation] while keeping all other text identical"; validate with NLI that entailment score drops | `backend/generate_adversarial.py` |
| 2026-04-15 | [ ] | DATA: Generate ~200 adversarial pairs per mode per failure type (→ ~800 new pairs per mode); merge into training splits; retrain and compare AUC on hard-negative subset specifically | `data/*/train.json`, `backend/train.py` |
| 2026-04-15 | [ ] | EVAL: After retrain, specifically measure performance on original hard-negative subset (pairs where score was 0.35–0.65 before adversarial training) — the aggregate AUC may barely move while the hard-case accuracy jumps significantly | `backend/test.py`, `test_reports/` |

---

## Feature Roadmap — Active Learning Loop

> **Idea:** Run the trained model on a large unlabeled corpus, find pairs where it's most uncertain
> (probability 0.4–0.6), label them cheaply with the LLM jury, add to training data.
> Active learning with the jury as the oracle — no new data collection, mine the model's own uncertainty.

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-04-11 | [ ] | DESIGN: Define unlabeled corpus source — candidates: HuggingFace datasets (RAGTruth unannotated, MS-MARCO passages, OpenAssistant conversations); aim for 5k–10k pairs per mode | research |
| 2026-04-11 | [ ] | IMPL: `backend/active_learn.py` — run `SimilarityPredictor.predict_batch()` on corpus, filter to uncertainty window (0.35–0.65), output candidate pairs JSON | `backend/active_learn.py` (new) |
| 2026-04-11 | [ ] | IMPL: Auto-label uncertain pairs with `JuryEvaluator.evaluate_batch()` — jury score ≥ 0.6 → label=1, ≤ 0.4 → label=0, else discard | `backend/active_learn.py`, `backend/jury/jury_evaluator.py` |
| 2026-04-11 | [ ] | INTEGRATE: Merge jury-labeled pairs into mode-specific train splits; retrain and compare AUC vs baseline | `data/*/train.json`, `backend/train.py` |

---

## Feature Roadmap — Structured Failure Codes on Every Inference

> **Idea:** Every production inference returns not just a score but a machine-readable diagnosis:
> `HALL-NUMERIC`, `ENT-SUBST`, `OMIT-KEY`, `FAITHFUL` etc. Downstream systems can route differently —
> numeric errors to a fact-checker, entity substitutions to a knowledge graph lookup,
> omissions to a retrieval re-run. SilverBullet is ~3 endpoints away from being this.

| Date | Status | Task | Files / Notes |
|------|--------|------|---------------|
| 2026-04-11 | [ ] | DESIGN: Define failure code taxonomy — extend existing jury diagnostic codes (`HALL-NUMERIC`, `ENT-SUBST`, `NEG-FACT`, `OMIT-KEY`, `FAITHFUL`) to cover all feature clusters; each code maps to a CNN feature group and a jury question | `backend/jury/codes.py` (new) |
| 2026-04-11 | [ ] | IMPL: Attach codes to breakdown endpoint — `POST /api/v1/predict/pair/breakdown` response gains `failure_codes: list[str]` field derived from feature scores (e.g. low `numeric_jaccard` → `HALL-NUMERIC`, low `entity_grounding_recall` → `ENT-SUBST`) | `backend/predict.py`, `backend/api/schemas.py`, `backend/api/main.py` |
| 2026-04-11 | [ ] | IMPL: Jury endpoint also returns codes — `POST /api/v1/predict/pair/jury` response already has `reasoning` strings; extract codes from reasoning using the jury's own diagnostic code convention | `backend/jury/jury_evaluator.py`, `backend/api/schemas.py` |
| 2026-04-11 | [ ] | FRONTEND: Render codes as colored chips in breakdown panel — `HALL-NUMERIC` red, `ENT-SUBST` orange, `OMIT-KEY` yellow, `FAITHFUL` green | `frontend/src/components/BreakdownPanel.tsx` |

---

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
