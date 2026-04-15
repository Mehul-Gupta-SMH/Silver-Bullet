"""SilverBullet FastAPI application factory."""

from __future__ import annotations

import glob
import json
import os
import sys
import time
import threading
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

# ---------------------------------------------------------------------------
# In-memory training job state
# ---------------------------------------------------------------------------

class _TrainingJob:
    __slots__ = ("status", "lines", "proc", "returncode", "started_at", "ended_at")

    def __init__(self) -> None:
        self.status: str = "idle"          # idle | running | done | error
        self.lines: deque[str] = deque(maxlen=2000)
        self.proc = None
        self.returncode: int | None = None
        self.started_at: float | None = None
        self.ended_at: float | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "returncode": self.returncode,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "lines": list(self.lines),
        }


_TRAIN_JOBS: dict[str, _TrainingJob] = {}


def _run_training_thread(mode: str, root: Path) -> None:
    """Background thread: run training, capture output, update job state."""
    job = _TRAIN_JOBS[mode]
    job.status = "running"
    job.started_at = time.time()
    job.lines.clear()
    try:
        import subprocess
        proc = subprocess.Popen(
            [sys.executable, "-m", "backend.train", "--mode", mode],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(root),
        )
        job.proc = proc
        for raw in proc.stdout:  # type: ignore[union-attr]
            line = raw.rstrip("\n\r")
            # Skip tqdm progress-bar lines (contain carriage returns or it/s)
            if "\r" in raw or ("it/s" in line and "%" in line):
                continue
            if line:
                job.lines.append(line)
        proc.wait()
        job.returncode = proc.returncode
        job.status = "done" if proc.returncode == 0 else "error"
    except Exception as exc:
        job.status = "error"
        job.lines.append(f"[launcher] ERROR: {exc}")
    finally:
        job.proc = None
        job.ended_at = time.time()

from backend.api.dependencies import _MODE_ENV, get_predictor
from backend.api.middleware import LoggingMiddleware, RequestIDMiddleware
from backend.api.schemas import (
    BatchBreakdownRequest,
    BatchBreakdownResponse,
    BatchRequest,
    BatchResponse,
    BreakdownResponse,
    HealthResponse,
    JuryBatchRequest,
    JuryBatchResponse,
    JuryRequest,
    JuryResult,
    PairRequest,
    PairResponse,
)


# ---------------------------------------------------------------------------
# Lifespan — startup model check
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Fail fast on startup if MODEL_PATH is explicitly set but the file is missing."""
    model_env = os.environ.get("MODEL_PATH")
    if model_env:
        model_path = Path(model_env)
        if not model_path.exists():
            raise RuntimeError(
                f"MODEL_PATH={model_env!r} does not exist or is not readable. "
                "Ensure the file path points to a valid checkpoint."
            )
    yield


# ---------------------------------------------------------------------------
# Rate limiter
#   Global default: 60/minute (all prediction endpoints)
#   Breakdown endpoints override to a tighter limit (expensive — reruns pipeline)
# ---------------------------------------------------------------------------

def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SilverBullet — LLM Evaluation API",
    description=(
        "Real-time LLM evaluation benchmark. "
        "Given two texts, returns a faithfulness / agreement score in **[0, 1]** "
        "using a Conv2D model trained on 16 multi-signal feature maps "
        "(semantic, lexical, NLI, entity, LCS).\n\n"
        "**Use cases:** hallucination detection, model-vs-model agreement, RAG groundedness.\n\n"
        "Interactive docs: `/api/v1/docs` · ReDoc: `/api/v1/redoc`"
    ),
    version="1.0.0",
    contact={
        "name": "SilverBullet",
        "url": "https://github.com/Mehul-Gupta-SMH/Silver-Bullet",
    },
    openapi_tags=[
        {"name": "health", "description": "Service health and readiness checks"},
        {"name": "prediction", "description": "LLM evaluation scoring endpoints"},
    ],
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(getattr(request, "state", None), "request_id", None)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "request_id": request_id},
    )


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

_default_origins = ["http://localhost:5173"]
_extra = os.environ.get("ALLOWED_ORIGINS", "")
_allowed_origins = _default_origins + [o.strip() for o in _extra.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestIDMiddleware)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check",
    response_description="Service status and model load state",
)
async def health() -> HealthResponse:
    """Return service status and per-mode model load state. No authentication required."""
    models: dict[str, bool] = {}
    for mode in _MODE_ENV:
        try:
            get_predictor(mode)
            models[mode] = True
        except Exception:
            models[mode] = False
    return HealthResponse(status="ok", model_loaded=any(models.values()), models=models)


@app.get("/api/v1/admin/status", tags=["admin"], summary="Admin dashboard status")
async def admin_status() -> dict:
    """Return extended status for the admin dashboard: per-mode checkpoint info,
    latest benchmark results, latest training metrics, and cache statistics."""
    _ROOT = Path(__file__).parent.parent.parent

    # --- per-mode checkpoint info ---
    mode_info: dict[str, dict] = {}
    for mode in _MODE_ENV:
        ckpt_path = Path(_ROOT / "models" / mode / "best.pth")
        loaded = False
        try:
            get_predictor(mode)
            loaded = True
        except Exception:
            pass
        info: dict = {"loaded": loaded, "checkpoint": None}
        if ckpt_path.exists():
            stat = ckpt_path.stat()
            info["checkpoint"] = {
                "path": str(ckpt_path.relative_to(_ROOT)),
                "size_mb": round(stat.st_size / 1_048_576, 2),
                "modified_ts": stat.st_mtime,
            }
        mode_info[mode] = info

    # --- latest benchmark report ---
    benchmark_results: list[dict] = []
    report_dir = _ROOT / "benchmark_reports"
    if report_dir.exists():
        reports = sorted(report_dir.glob("*.json"), reverse=True)
        if reports:
            try:
                data = json.loads(reports[0].read_text(encoding="utf-8"))
                benchmark_results = data.get("results", [])
            except Exception:
                pass

    # --- latest training report per mode ---
    training_summaries: dict[str, dict] = {}
    training_dir = _ROOT / "training_reports"
    if training_dir.exists():
        for mode in _MODE_ENV:
            mode_slug = mode.replace("-", "_")
            # match both old and new naming conventions
            candidates = sorted(
                list(training_dir.glob(f"*{mode}*.json")) +
                list(training_dir.glob(f"*{mode_slug}*.json")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for c in candidates:
                if "kfold" in c.name:
                    continue
                try:
                    data = json.loads(c.read_text(encoding="utf-8"))
                    metrics = data.get("training_metrics", {})
                    training_summaries[mode] = {
                        "report_file": c.name,
                        "best_val_loss": metrics.get("best_val_loss"),
                        "best_epoch": metrics.get("best_epoch"),
                        "total_epochs": metrics.get("total_epochs"),
                        "timestamp": c.stat().st_mtime,
                    }
                    break
                except Exception:
                    continue

    # --- cache statistics ---
    cache_stats: dict = {}
    try:
        from backend.cache_db import CacheDB
        db = CacheDB.get()
        cache_stats = db.stats() if hasattr(db, "stats") else {}
    except Exception:
        pass

    return {
        "models": mode_info,
        "benchmark_results": benchmark_results,
        "training_summaries": training_summaries,
        "cache_stats": cache_stats,
        "server_time": time.time(),
    }


@app.post("/api/v1/admin/train/{mode}", tags=["admin"], summary="Trigger model training")
async def start_training(mode: str) -> dict:
    """Spawn a training subprocess for *mode* in the background.
    Returns immediately; poll /admin/train/status for progress."""
    _ROOT = Path(__file__).parent.parent.parent
    if mode not in _MODE_ENV:
        raise HTTPException(status_code=400, detail=f"Unknown mode '{mode}'. Valid: {list(_MODE_ENV)}")

    if mode not in _TRAIN_JOBS:
        _TRAIN_JOBS[mode] = _TrainingJob()

    job = _TRAIN_JOBS[mode]
    if job.status == "running":
        return {"started": False, "reason": "already_running", "mode": mode}

    t = threading.Thread(target=_run_training_thread, args=(mode, _ROOT), daemon=True)
    t.start()
    return {"started": True, "mode": mode}


@app.post("/api/v1/admin/train/{mode}/stop", tags=["admin"], summary="Stop a running training job")
async def stop_training(mode: str) -> dict:
    """Send SIGTERM to the training process for *mode*, if running."""
    import signal as _signal
    job = _TRAIN_JOBS.get(mode)
    if not job or job.status != "running" or job.proc is None:
        return {"stopped": False, "reason": "not_running"}
    try:
        job.proc.send_signal(_signal.SIGTERM)
    except Exception:
        job.proc.kill()
    return {"stopped": True, "mode": mode}


@app.get("/api/v1/admin/train/status", tags=["admin"], summary="Training job statuses")
async def training_status() -> dict:
    """Return current status + last log lines for all training jobs."""
    return {
        mode: {
            "status": j.status,
            "returncode": j.returncode,
            "started_at": j.started_at,
            "ended_at": j.ended_at,
            "line_count": len(j.lines),
        }
        for mode, j in _TRAIN_JOBS.items()
    }


@app.get("/api/v1/admin/train/logs/{mode}", tags=["admin"], summary="Training log lines")
async def training_logs(mode: str, offset: int = 0) -> dict:
    """Return log lines for *mode* starting at *offset*."""
    job = _TRAIN_JOBS.get(mode)
    if not job:
        return {"mode": mode, "offset": 0, "lines": [], "status": "idle"}
    lines = list(job.lines)
    return {
        "mode": mode,
        "status": job.status,
        "offset": len(lines),
        "lines": lines[offset:],
    }


@app.post(
    "/api/v1/predict/pair",
    response_model=PairResponse,
    tags=["prediction"],
    summary="Evaluate a single LLM output",
    response_description="Faithfulness / agreement score in [0, 1] with binary verdict",
    responses={422: {"description": "Validation error — text empty or exceeds 10 000 characters"}},
)
async def predict_pair(
    request: Request,
    body: PairRequest,
) -> PairResponse:
    """Score the similarity between two texts using the model for the requested evaluation mode."""
    predictor = get_predictor(body.mode)
    result = predictor.predict_pair(body.text1, body.text2)
    return PairResponse(**result)


@app.post(
    "/api/v1/predict/pair/breakdown",
    response_model=BreakdownResponse,
    tags=["prediction"],
    summary="Sentence-level divergence analysis for a text pair",
    response_description=(
        "Per-sentence alignment matrix, divergence indices, and feature-group scores. "
        "More expensive than /predict/pair — runs the full feature pipeline without cache."
    ),
    responses={422: {"description": "Validation error — text empty or exceeds 10 000 characters"}},
)
@limiter.limit("20/minute")
async def predict_pair_breakdown(
    request: Request,
    body: PairRequest,
) -> BreakdownResponse:
    """Return a full divergence breakdown: sentence splits, alignment matrix,
    divergent sentence indices, and per-feature-group similarity scores."""
    predictor = get_predictor(body.mode)
    result = predictor.predict_pair_breakdown(body.text1, body.text2)
    return BreakdownResponse(**result)


@app.post(
    "/api/v1/predict/batch",
    response_model=BatchResponse,
    tags=["prediction"],
    summary="Evaluate a batch of LLM outputs",
    response_description="List of faithfulness / agreement scores, one per input pair",
    responses={422: {"description": "Validation error — batch exceeds 100 pairs or pairs are malformed"}},
)
async def predict_batch(
    request: Request,
    body: BatchRequest,
) -> BatchResponse:
    """Score a list of text pairs using the model for the requested evaluation mode. Max 100 pairs."""
    predictor = get_predictor(body.mode)
    results = predictor.predict_batch(body.pairs)
    return BatchResponse(results=[PairResponse(**r) for r in results])


@app.post(
    "/api/v1/predict/batch/breakdown",
    response_model=BatchBreakdownResponse,
    tags=["prediction"],
    summary="Sentence-level divergence analysis for a batch of text pairs",
    response_description=(
        "Per-sentence alignment matrix, divergence indices, and feature-group scores for each pair. "
        "Expensive — reruns the full feature pipeline per pair. Max 10 pairs."
    ),
    responses={422: {"description": "Validation error — batch exceeds 10 pairs or pairs are malformed"}},
)
@limiter.limit("10/minute")
async def predict_batch_breakdown(
    request: Request,
    body: BatchBreakdownRequest,
) -> BatchBreakdownResponse:
    """Return full divergence breakdowns for up to 10 text pairs using the mode-specific model."""
    predictor = get_predictor(body.mode)
    results = predictor.predict_batch_breakdown(body.pairs)
    return BatchBreakdownResponse(results=[BreakdownResponse(**r) for r in results])


# ---------------------------------------------------------------------------
# Jury (LLM-as-judge) endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/predict/jury/pair",
    response_model=JuryResult,
    tags=["prediction"],
    summary="LLM-as-jury evaluation for a single text pair",
    response_description=(
        "Aggregated faithfulness score [0, 1] with per-question LLM breakdown. "
        "Uses an LLM (default: gpt-4o-mini) instead of the CNN model."
    ),
    responses={
        422: {"description": "Validation error — text empty or exceeds 10 000 characters"},
        502: {"description": "LLM call failed or returned malformed JSON"},
        503: {"description": "OpenAI API key not configured"},
    },
)
@limiter.limit("20/minute")
async def predict_jury_pair(
    request: Request,
    body: JuryRequest,
) -> JuryResult:
    """Evaluate a text pair using a structured LLM jury.

    Instead of the CNN feature pipeline, this endpoint sends the pair to an LLM
    (configurable via SB_JURY_MODEL, default gpt-4o-mini) with a battery of binary
    yes/no questions mirroring the CNN's feature clusters.  Each question receives
    an answer, confidence, and one-sentence reasoning.  The final score is a weighted
    mean of per-question faithfulness contributions.
    """
    from fastapi import HTTPException
    from backend.jury.jury_evaluator import JuryEvaluator

    try:
        evaluator = JuryEvaluator()
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail="Jury mode requires an OpenAI API key (SB_OPENAI_TOKEN)",
        ) from exc

    try:
        return evaluator.evaluate(body.text1, body.text2, body.mode)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post(
    "/api/v1/predict/jury/batch",
    response_model=JuryBatchResponse,
    tags=["prediction"],
    summary="LLM-as-jury evaluation for a batch of text pairs",
    response_description=(
        "List of jury evaluation results, one per input pair. "
        "Each pair makes one LLM call — max 10 pairs per request."
    ),
    responses={
        422: {"description": "Validation error — batch exceeds 10 pairs or pairs are malformed"},
        502: {"description": "LLM call failed or returned malformed JSON"},
        503: {"description": "OpenAI API key not configured"},
    },
)
@limiter.limit("5/minute")
async def predict_jury_batch(
    request: Request,
    body: JuryBatchRequest,
) -> JuryBatchResponse:
    """Evaluate up to 10 text pairs using a structured LLM jury (sequential LLM calls)."""
    from fastapi import HTTPException
    from backend.jury.jury_evaluator import JuryEvaluator

    try:
        evaluator = JuryEvaluator()
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail="Jury mode requires an OpenAI API key (SB_OPENAI_TOKEN)",
        ) from exc

    try:
        results = evaluator.evaluate_batch(
            [(p.text1, p.text2, p.mode) for p in body.pairs]
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return JuryBatchResponse(results=results)
