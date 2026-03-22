# SilverBullet — Frontend

React + TypeScript + Vite UI for the SilverBullet LLM Evaluation Benchmark.

## Stack

- **React 18** with TypeScript
- **Vite** dev server with HMR
- **Tailwind CSS** for styling
- **Vitest** + jsdom for component tests

## Development

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

The dev server proxies `/api` requests to `http://localhost:8000` (the FastAPI backend). Start the backend first:

```bash
uvicorn backend.api.main:app --reload
```

## Build

```bash
npm run build        # outputs to dist/
npm run preview      # serve the production build locally
```

## Tests

```bash
npm run test         # vitest (jsdom)
```

## Project Structure

```
frontend/
├── src/
│   ├── App.tsx                     # Shell: header, tabs (Single / Batch / Experiments)
│   ├── config/
│   │   └── modes.ts                # 3 evaluation modes with interpret() + labels
│   ├── types/
│   │   └── index.ts                # ComparisonMode, PredictionResult, BreakdownResult…
│   ├── services/
│   │   └── api.ts                  # predictPair(), predictBatch(), healthCheck() — all send mode
│   ├── hooks/
│   │   └── useLocalStorage.ts      # Persist draft inputs across page refreshes
│   ├── components/
│   │   ├── PairScorer.tsx          # Single eval — textareas, score, Drill Down
│   │   ├── BreakdownPanel.tsx      # Sentence colour map + feature score bars
│   │   ├── BatchScorer.tsx         # Batch eval — JSON upload + distribution chart
│   │   ├── ScoreGauge.tsx          # Score bar with prediction badge
│   │   ├── ComparisonModeSelector.tsx  # Mode picker (3 modes)
│   │   ├── FeaturePanel.tsx        # Collapsible feature family reference
│   │   ├── ExperimentsPanel.tsx    # Saved experiments with re-run
│   │   ├── TestCasePanel.tsx       # Pre-built test case library (17 pairs, 2 batch)
│   │   ├── ModelConfig.tsx         # Model / source name inputs
│   │   └── SaveExperimentForm.tsx  # Name + notes form
│   └── data/
│       └── testCases.ts            # 17 pair + 2 batch test cases
├── public/
│   └── favicon.svg                 # Violet bullseye SVG
├── index.html
└── vite.config.ts                  # Proxy /api → localhost:8000; vitest config
```

## Evaluation Modes

The UI exposes three modes that map directly to the backend API `mode` field:

| Mode ID | Label | text1 | text2 |
|---------|-------|-------|-------|
| `context-vs-generated` | Context vs Generated | Source document / RAG chunk | LLM answer |
| `reference-vs-generated` | Reference vs Generated | Ground-truth reference | Generated answer |
| `model-vs-model` | Model vs Model | Model A output | Model B output |

Switching mode changes the field labels, placeholders, interpretation thresholds, and — via the API — the model used for scoring.

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_URL` | `/api/v1` | Base URL for backend API calls |

Copy `.env.example` to `.env` and adjust if your backend runs on a different port.
