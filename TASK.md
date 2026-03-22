# SilverBullet — Task Log

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

## Pending
| 2026-03-15 | [x] | IMPROVEMENT: BCE → MSELoss on float labels for continuous faithfulness scoring | `train.py`, `test.py` |
| 2026-03-20 | [x] | IMPROVEMENT: Re-enable rate limiting — SlowAPIMiddleware (60/min global) + tighter limits on breakdown endpoints (20/min pair, 10/min batch) | `api/main.py`, `tests/test_api.py` |
| 2026-03-13 | [ ] | IMPROVEMENT: Expand training dataset to 1 000+ pairs with adversarial/domain-balanced sampling | `generate_data.py`, `data/` |
| 2026-03-15 | [x] | IMPROVEMENT: Add `/api/v1/predict/batch/breakdown` parallel to batch predict endpoint | `api/main.py`, `predict.py`, `api/schemas.py`, `tests/` |

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
