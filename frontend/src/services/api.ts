import type { PredictionResult, BatchResponse, BreakdownResult, HealthResponse, ComparisonMode } from '../types';

const BASE_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? '/api/v1';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const healthCheck = (): Promise<HealthResponse> =>
  request('/health');

export const predictPair = (
  text1: string,
  text2: string,
  mode: ComparisonMode,
): Promise<PredictionResult> =>
  request('/predict/pair', {
    method: 'POST',
    body: JSON.stringify({ text1, text2, mode }),
  });

export const predictPairBreakdown = (
  text1: string,
  text2: string,
  mode: ComparisonMode,
): Promise<BreakdownResult> =>
  request('/predict/pair/breakdown', {
    method: 'POST',
    body: JSON.stringify({ text1, text2, mode }),
  });

export const predictBatch = (
  pairs: Array<[string, string]>,
  mode: ComparisonMode,
): Promise<BatchResponse> =>
  request('/predict/batch', {
    method: 'POST',
    body: JSON.stringify({ pairs, mode }),
  });
