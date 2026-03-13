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
| 2026-03-13 | [~] | TRAINING: `python train.py` running in background on new dataset | `best_model.pth` (will be updated on completion) |
| 2026-03-13 | [x] | FEATURE: Favicon — violet bullseye SVG matching brand colours | `frontend/public/favicon.svg`, `frontend/index.html` |
| 2026-03-13 | [x] | REBRAND: Renamed from "Text Similarity Tool" to "LLM Evaluation Benchmark" across all surfaces | `frontend/index.html`, `frontend/src/App.tsx`, `frontend/src/config/modes.ts`, `frontend/src/components/ComparisonModeSelector.tsx`, `frontend/src/components/ScoreGauge.tsx`, `api/main.py` |
| 2026-03-13 | [x] | DOCS: TASK.md, CLAUDE.md, AGENT.md brought up to date | all three doc files |

## Pending
| 2026-03-11 | [ ] | IMPROVEMENT: BCE → MSELoss on float labels for continuous faithfulness scoring | `model.py`, `train.py`, `data/*.json` |
| 2026-03-12 | [ ] | IMPROVEMENT: Re-enable per-endpoint rate limiting via SlowAPIMiddleware | `api/main.py` |
| 2026-03-13 | [ ] | IMPROVEMENT: Expand training dataset to 1 000+ pairs with adversarial/domain-balanced sampling | `generate_data.py`, `data/` |
| 2026-03-13 | [ ] | IMPROVEMENT: Add `/api/v1/predict/batch/breakdown` parallel to batch predict endpoint | `api/main.py`, `predict.py` |
