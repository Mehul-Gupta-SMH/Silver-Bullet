import type { PredictionResult, BatchResponse, HealthResponse } from '../types';

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

export const predictPair = (text1: string, text2: string): Promise<PredictionResult> =>
  request('/predict/pair', {
    method: 'POST',
    body: JSON.stringify({ text1, text2 }),
  });

export const predictBatch = (pairs: Array<[string, string]>): Promise<BatchResponse> =>
  request('/predict/batch', {
    method: 'POST',
    body: JSON.stringify({ pairs }),
  });
