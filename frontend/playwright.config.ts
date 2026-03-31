import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,          // run demos sequentially — each is a long recording
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: [
    ['list'],
    ['html', { outputFolder: 'playwright-report', open: 'never' }],
  ],
  outputDir: 'demo-recordings',  // WebM videos land here

  use: {
    // Backend API must be running on port 8000 before starting the demo:
    //   uvicorn backend.api.main:app --reload   (from repo root)
    baseURL: 'http://127.0.0.1:4173',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    viewport: { width: 1440, height: 900 },
    video: {
      mode: 'on',
      size: { width: 1440, height: 900 },
    },
    actionTimeout:    60_000,
    navigationTimeout: 30_000,
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  webServer: {
    // Preview serves the compiled bundle — no HMR, no repeated reloads.
    // Run "npm run build" once before recording, then reuse the build.
    command: 'npx vite preview --host 127.0.0.1 --port 4173',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
