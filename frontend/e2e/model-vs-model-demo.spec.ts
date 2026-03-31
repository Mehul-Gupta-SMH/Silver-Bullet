/**
 * Model-vs-Model demo recording
 *
 * Records a complete walkthrough of the model-vs-model evaluation workflow:
 *   1. Mode selection and model labelling
 *   2. High-agreement pair  → score → verify green result
 *   3. Divergent pair       → score → drill-down → walk every section:
 *        • Sentence alignment heat-map
 *        • Feature score bars (semantic / NLI / entity / LCS / lexical)
 *        • Misalignment diagnostics cards (severity-coded)
 *   4. Save experiment
 *   5. Batch evaluation of all three fixture pairs
 *
 * Run:  npm run demo:mvm            (headless, records video)
 *       npm run demo:mvm:headed     (headed, slower pauses)
 */

import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test, type Locator, type Page } from '@playwright/test';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const FIXTURE = path.join(__dirname, 'fixtures', 'mvm-pairs.json');

// ---------------------------------------------------------------------------
// Timing constants — increase PACE_FACTOR env var to slow down for headed runs
// ---------------------------------------------------------------------------
const PACE = Number(process.env.PACE_FACTOR ?? 1);
const T = {
  char:   18  * PACE,   // ms per character when typing
  short:  600 * PACE,
  medium: 1_000 * PACE,
  long:   1_800 * PACE,
  read:   2_500 * PACE,  // pause so viewer can read a section
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
async function wait(ms: number) {
  await new Promise((r) => setTimeout(r, ms));
}

async function typeText(locator: Locator, text: string) {
  await locator.click();
  await locator.clear();
  await locator.pressSequentially(text, { delay: T.char });
}

async function resetPage(page: Page) {
  await page.addInitScript(() => localStorage.clear());
  // networkidle waits until no requests for 500 ms — necessary with the
  // Vite preview server so the bundle is fully loaded before clicking.
  await page.goto('/', { waitUntil: 'networkidle' });
  await expect(page.getByRole('heading', { name: /SilverBullet/i })).toBeVisible();
  await wait(T.medium);
}

// ---------------------------------------------------------------------------
// CONTENT — realistic LLM outputs for the demo
// ---------------------------------------------------------------------------

// Pair A: both models say essentially the same thing — expect HIGH agreement
const PAIR_AGREE = {
  modelA: 'GPT-4o',
  modelB: 'Claude 3.5 Sonnet',
  text1: [
    'Python is a high-level, interpreted programming language known for its clean syntax and readability.',
    'It was created by Guido van Rossum and first released in 1991.',
    'Python supports multiple paradigms — object-oriented, procedural, and functional programming.',
    'Its rich standard library and active ecosystem make it the default choice for data science, web development, and automation.',
  ].join(' '),
  text2: [
    'Python is a widely-used, high-level language celebrated for its readable, expressive syntax.',
    'Guido van Rossum developed it, with the first version appearing in 1991.',
    'The language accommodates object-oriented, functional, and procedural styles.',
    'A rich ecosystem of third-party packages makes it a top pick for data science, scripting, and backend development.',
  ].join(' '),
};

// Pair B: Model B hallucinates facts — expect DIVERGENCE + misalignment diagnostics
const PAIR_DIVERGE = {
  modelA: 'GPT-4o',
  modelB: 'Gemini 1.5 Pro',
  text1: [
    'The Eiffel Tower was completed in 1889 and stands 330 metres tall.',
    'It was designed by Gustave Eiffel as the entrance arch for the 1889 World\'s Fair held in Paris.',
    'At completion it was the world\'s tallest man-made structure, a title it held for 41 years until the Chrysler Building surpassed it in 1930.',
  ].join(' '),
  text2: [
    'The Eiffel Tower was constructed in 1892 and reaches approximately 400 metres in height.',
    'It was designed by the architect Jules Verne to commemorate the centennial of the French Revolution.',
    'The tower was originally painted blue and later repainted to its familiar brown colour during the 1950s.',
  ].join(' '),
};

// ---------------------------------------------------------------------------
// TEST 1 — Single pair: agreement → divergence → drill-down
// ---------------------------------------------------------------------------
test.describe('Model-vs-Model demo', () => {
  test.setTimeout(600_000);

  test('mvm-pair-drilldown-demo', async ({ page }) => {
    await resetPage(page);

    // ── Step 1: Select Model vs Model mode ──────────────────────────────────
    await test.step('Select Model vs Model mode', async () => {
      await page.getByRole('button', { name: /Model vs Model/i }).click();
      await wait(T.long);
    });

    // ── Step 2: Label the models ─────────────────────────────────────────────
    await test.step('Label Model A and Model B', async () => {
      await typeText(page.getByPlaceholder(/e\.g\. GPT-4o/i).first(), PAIR_AGREE.modelA);
      await wait(T.short);
      // Set GPT-4o as baseline
      await page.getByRole('button', { name: /Set baseline/i }).first().click();
      await wait(T.short);
      await typeText(page.getByPlaceholder(/e\.g\. Claude 3\.5 Sonnet/i), PAIR_AGREE.modelB);
      await wait(T.medium);
    });

    // ── Step 3: Enter high-agreement pair ───────────────────────────────────
    await test.step('Enter high-agreement Python pair', async () => {
      await typeText(page.getByPlaceholder(/Paste the output from Model A/i), PAIR_AGREE.text1);
      await wait(T.medium);
      await typeText(page.getByPlaceholder(/Paste the output from Model B/i), PAIR_AGREE.text2);
      await wait(T.long);
    });

    // ── Step 4: Score the pair ───────────────────────────────────────────────
    await test.step('Score high-agreement pair', async () => {
      await page.getByRole('button', { name: /Analyse Pair/i }).click();
      await expect(page.getByText(/Evaluation Score/i)).toBeVisible({ timeout: 90_000 });
      await wait(T.read);
      // Verify result is green / strong agreement
      await expect(page.getByText(/Strong agreement|Partial agreement/i)).toBeVisible();
      await wait(T.read);
    });

    // ── Step 5: Switch to divergent pair ────────────────────────────────────
    await test.step('Switch models — enter divergent Eiffel Tower pair', async () => {
      // Update model labels
      await typeText(page.getByPlaceholder(/e\.g\. GPT-4o/i).first(), PAIR_DIVERGE.modelA);
      await wait(T.short);
      await typeText(page.getByPlaceholder(/e\.g\. Claude 3\.5 Sonnet/i), PAIR_DIVERGE.modelB);
      await wait(T.short);

      // Replace text
      await typeText(page.getByPlaceholder(/Paste the output from Model A/i), PAIR_DIVERGE.text1);
      await wait(T.medium);
      await typeText(page.getByPlaceholder(/Paste the output from Model B/i), PAIR_DIVERGE.text2);
      await wait(T.long);
    });

    // ── Step 6: Score divergent pair ────────────────────────────────────────
    await test.step('Score divergent pair — expect low score', async () => {
      await page.getByRole('button', { name: /Analyse Pair/i }).click();
      await expect(page.getByText(/Evaluation Score/i)).toBeVisible({ timeout: 90_000 });
      await wait(T.read);
      await expect(page.getByText(/Significant divergence|Partial agreement/i)).toBeVisible();
      await wait(T.read);
    });

    // ── Step 7: Open drill-down ──────────────────────────────────────────────
    await test.step('Open Drill Down panel', async () => {
      await page.getByRole('button', { name: /Drill Down/i }).click();
      await expect(page.getByText(/Divergence Analysis/i)).toBeVisible({ timeout: 90_000 });
      await wait(T.long);
    });

    // ── Step 8: Walk through sentence alignment section ─────────────────────
    await test.step('Sentence alignment — highlight divergent sentences', async () => {
      // Scroll to sentence columns so viewer can read them
      await page.getByText(/Text 1 —/i).scrollIntoViewIfNeeded();
      await wait(T.read);
      await page.getByText(/Text 2 —/i).scrollIntoViewIfNeeded();
      await wait(T.read);
    });

    // ── Step 9: Walk through feature score bars ──────────────────────────────
    await test.step('Feature scores — review per-signal bars', async () => {
      await page.getByText(/What drove the score/i).scrollIntoViewIfNeeded();
      await wait(T.read);

      // Verify each feature group is visible
      await expect(page.getByText(/Semantic \(mxbai\)/i)).toBeVisible();
      await expect(page.getByText(/NLI Entailment/i)).toBeVisible();
      await expect(page.getByText(/Entity Match/i)).toBeVisible();
      await expect(page.getByText(/LCS Token/i)).toBeVisible();
      await expect(page.getByText(/Lexical/i).first()).toBeVisible();
      await wait(T.read);
    });

    // ── Step 10: Walk through misalignment diagnostics ──────────────────────
    await test.step('Misalignment diagnostics — severity-coded reasons', async () => {
      await page.getByText(/Misalignment Diagnostics/i).scrollIntoViewIfNeeded();
      await wait(T.long);

      // At least one diagnostic card should be present
      const highCard   = page.locator('text=HIGH').first();
      const mediumCard = page.locator('text=MEDIUM').first();
      const hasHigh   = await highCard.isVisible().catch(() => false);
      const hasMedium = await mediumCard.isVisible().catch(() => false);
      expect(hasHigh || hasMedium).toBeTruthy();

      await wait(T.read);

      // Entity Substitution and/or Entailment Conflict should fire for this pair
      const entityCard     = page.getByText(/Entity Substitution|Partial Entity/i).first();
      const entailmentCard = page.getByText(/Entailment Conflict|Weak Entailment/i).first();
      const hasEntity     = await entityCard.isVisible().catch(() => false);
      const hasEntailment = await entailmentCard.isVisible().catch(() => false);
      expect(hasEntity || hasEntailment).toBeTruthy();

      await wait(T.read);
    });

    // ── Step 11: Save experiment ─────────────────────────────────────────────
    await test.step('Save experiment', async () => {
      // Scroll back up to find the save form
      await page.getByPlaceholder(/Experiment name/i).scrollIntoViewIfNeeded();
      await wait(T.short);
      await typeText(
        page.getByPlaceholder(/Experiment name/i),
        'MvM — Eiffel Tower hallucination test',
      );
      await wait(T.short);
      await typeText(
        page.getByPlaceholder(/Notes \(optional\)/i),
        'Gemini 1.5 Pro hallucinated year (1892), height (400m), and architect (Jules Verne). ' +
        'Entity Substitution and Entailment Conflict flagged at HIGH severity.',
      );
      await wait(T.medium);
      await page.getByRole('button', { name: /Save Experiment/i }).click();
      await expect(page.getByText(/Saved to Experiments/i)).toBeVisible();
      await wait(T.read);
    });
  });

  // ── TEST 2 — Batch evaluation of all three fixture pairs ────────────────
  test('mvm-batch-demo', async ({ page }) => {
    await resetPage(page);

    await test.step('Select Model vs Model mode', async () => {
      await page.getByRole('button', { name: /Model vs Model/i }).click();
      await wait(T.long);
    });

    await test.step('Label baseline model', async () => {
      await typeText(page.getByPlaceholder(/e\.g\. GPT-4o/i).first(), 'GPT-4o');
      await wait(T.short);
      await page.getByRole('button', { name: /Set baseline/i }).first().click();
      await wait(T.short);
      await typeText(page.getByPlaceholder(/e\.g\. Claude 3\.5 Sonnet/i), 'Various challengers');
      await wait(T.medium);
    });

    await test.step('Switch to Batch Eval tab', async () => {
      await page.getByRole('button', { name: /Batch Eval/i }).click();
      await expect(page.getByText(/Upload JSON File/i)).toBeVisible();
      await wait(T.medium);
    });

    await test.step('Upload fixture — 3 model-vs-model pairs', async () => {
      await page.locator('input[type="file"]').setInputFiles(FIXTURE);
      await expect(page.getByText(/3 pairs ready/i)).toBeVisible();
      await wait(T.long);
    });

    await test.step('Run batch scoring', async () => {
      await page.getByRole('button', { name: /Analyse Batch \(3\)/i }).click();
      await expect(page.getByText(/Score Distribution/i)).toBeVisible({ timeout: 180_000 });
      await expect(page.getByText(/All Results/i)).toBeVisible({ timeout: 180_000 });
      await wait(T.read);
    });

    await test.step('Review score distribution chart', async () => {
      await page.getByText(/Score Distribution/i).scrollIntoViewIfNeeded();
      await wait(T.read);
    });

    await test.step('Review individual results table', async () => {
      await page.getByText(/All Results/i).scrollIntoViewIfNeeded();
      await wait(T.read);
    });

    await test.step('Save batch experiment', async () => {
      await page.getByPlaceholder(/Experiment name/i).scrollIntoViewIfNeeded();
      await wait(T.short);
      await typeText(
        page.getByPlaceholder(/Experiment name/i),
        'MvM batch — Python / Eiffel / Transformers',
      );
      await wait(T.short);
      await typeText(
        page.getByPlaceholder(/Notes \(optional\)/i),
        'Three prompt types: factual agreement (Python), hallucination (Eiffel), and technical paraphrase (Transformers).',
      );
      await wait(T.medium);
      await page.getByRole('button', { name: /Save Experiment/i }).click();
      await expect(page.getByText(/Saved to Experiments/i)).toBeVisible();
      await wait(T.read);
    });
  });
});
