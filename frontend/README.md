# SilverBullet Frontend

React + TypeScript + Vite UI for the SilverBullet LLM Evaluation Benchmark.

## Stack

- React 19 with TypeScript
- Vite dev server with HMR
- Tailwind CSS for styling
- Vitest + jsdom for component tests
- Playwright for end-to-end coverage and recordable demo walkthroughs

## Development

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

The dev server proxies `/api` requests to `http://localhost:8000`, so start the FastAPI backend first:

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
npm run test                 # vitest (jsdom)
npm run test:e2e             # Playwright end-to-end coverage
npm run demo:videos          # record all demo walkthrough videos
npm run demo:videos:headed   # same suite, with the browser visible
```

Before the first Playwright run, install the browser bundle:

```bash
npx playwright install chromium
```

## Demo Videos

The Playwright demo suite records long-form walkthroughs with live backend inference and saves each clip under `frontend/test-results/`.

Included recordings:

- `pair-analysis-demo` - long single-eval flow with source labels, full inference, and drill-down opened
- `batch-analysis-demo` - long batch flow with model labels, baseline selection, charts, and results table
- `experiments-rerun-demo` - saved experiment flow with note editing, expanded details, and rerun
- `full-product-showcase-demo` - one longer end-to-end recording that moves through pair, batch, and experiments in one video

Run just the showcase clip with:

```bash
npx playwright test e2e/demo-videos.spec.ts --grep "full-product-showcase-demo"
```

## Project Structure

```text
frontend/
|-- e2e/
|   |-- demo-videos.spec.ts         # Playwright demo flows and long-form showcase
|   `-- fixtures/demo-batch.json    # Batch upload fixture for demos
|-- public/
|   `-- favicon.svg                 # App icon
|-- src/
|   |-- components/
|   |-- config/
|   |-- data/
|   |-- hooks/
|   |-- services/
|   `-- types/
|-- index.html
|-- package.json
|-- playwright.config.ts            # Video recording + local Vite webServer
`-- vite.config.ts                  # Frontend build/test config
```

## Evaluation Modes

The UI exposes three modes that map directly to the backend API `mode` field:

| Mode ID | Label | text1 | text2 |
| --- | --- | --- | --- |
| `context-vs-generated` | Context vs Generated | Source document / RAG chunk | LLM answer |
| `reference-vs-generated` | Reference vs Generated | Ground-truth reference | Generated answer |
| `model-vs-model` | Model vs Model | Model A output | Model B output |

Switching modes changes field labels, placeholders, interpretation thresholds, and the backend model used for scoring.

## Environment

| Variable | Default | Description |
| --- | --- | --- |
| `VITE_API_URL` | `/api/v1` | Base URL for backend API calls |

Copy `.env.example` to `.env` and adjust it if the backend runs on a different port.
