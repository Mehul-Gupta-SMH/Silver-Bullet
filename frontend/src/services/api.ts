import type { PredictionResult, BatchResponse, BreakdownResult, HealthResponse, ComparisonMode, JuryResult, AdminStatus, TrainingJobStatus, TrainingLogsResponse } from '../types';

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

export const predictJuryPair = (
  text1: string,
  text2: string,
  mode: ComparisonMode,
): Promise<JuryResult> =>
  request('/predict/jury/pair', {
    method: 'POST',
    body: JSON.stringify({ text1, text2, mode }),
  });

export const getAdminStatus = (): Promise<AdminStatus> =>
  request('/admin/status');

export const startTraining = (mode: string): Promise<{ started: boolean; reason?: string; mode: string }> =>
  request(`/admin/train/${mode}`, { method: 'POST' });

export const stopTraining = (mode: string): Promise<{ stopped: boolean; reason?: string }> =>
  request(`/admin/train/${mode}/stop`, { method: 'POST' });

export const getTrainingStatus = (): Promise<Record<string, TrainingJobStatus>> =>
  request('/admin/train/status');

export const getTrainingLogs = (mode: string, offset = 0): Promise<TrainingLogsResponse> =>
  request(`/admin/train/logs/${mode}?offset=${offset}`);
