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
    baseURL: 'http://127.0.0.1:5173',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    viewport: { width: 1440, height: 900 },
    video: {
      mode: 'on',
      size: { width: 1440, height: 900 },
    },
    // Slow down all actions so recordings are readable
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
    // npx vite is more reliable than "npm run dev --" on Windows PowerShell
    command: 'npx vite --host 127.0.0.1 --port 5173',
    url: 'http://127.0.0.1:5173',
    reuseExistingServer: true,   // reuse if already running (e.g. from another terminal)
    timeout: 120_000,
  },
});
