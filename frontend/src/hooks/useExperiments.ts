import { useState, useEffect, useCallback } from 'react';
import type { ComparisonMode, PredictionResult } from '../types';
import type { TextMeta } from '../components/ModelConfig';

export interface BatchStats {
  total: number;
  mean: number;
  median: number;
  std: number;
  similar: number;
  different: number;
}

export interface ExperimentRecord {
  id: string;
  savedAt: string;
  name: string;
  notes: string;
  mode: ComparisonMode;
  modelMeta: TextMeta;
  type: 'pair' | 'batch';
  // pair
  text1?: string;
  text2?: string;
  result?: { prediction: number; probability: number };
  // batch
  fileName?: string;
  batchStats?: BatchStats;
  results?: PredictionResult[];
}

const STORAGE_KEY = 'sb_experiments';
const MAX_RECORDS = 100;

function load(): ExperimentRecord[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as ExperimentRecord[]) : [];
  } catch {
    return [];
  }
}

function save(records: ExperimentRecord[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
  } catch {
    /* storage full — silently ignore */
  }
}

function computeBatchStats(results: PredictionResult[]): BatchStats {
  const total = results.length;
  const similar = results.filter((r) => r.prediction === 1).length;
  const probs = results.map((r) => r.probability);
  const mean = probs.reduce((s, p) => s + p, 0) / total;
  const sorted = [...probs].sort((a, b) => a - b);
  const median = sorted[Math.floor(total / 2)] ?? 0;
  const std = Math.sqrt(probs.reduce((s, p) => s + (p - mean) ** 2, 0) / total);
  return { total, mean, median, std, similar, different: total - similar };
}

export function useExperiments() {
  const [experiments, setExperiments] = useState<ExperimentRecord[]>(load);

  // Keep localStorage in sync
  useEffect(() => {
    save(experiments);
  }, [experiments]);

  const savePairExperiment = useCallback(
    (opts: {
      name: string;
      notes: string;
      mode: ComparisonMode;
      modelMeta: TextMeta;
      text1: string;
      text2: string;
      result: { prediction: number; probability: number };
    }) => {
      const record: ExperimentRecord = {
        id: `exp-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        savedAt: new Date().toISOString(),
        type: 'pair',
        ...opts,
      };
      setExperiments((prev) => [record, ...prev].slice(0, MAX_RECORDS));
      return record.id;
    },
    [],
  );

  const saveBatchExperiment = useCallback(
    (opts: {
      name: string;
      notes: string;
      mode: ComparisonMode;
      modelMeta: TextMeta;
      fileName: string;
      results: PredictionResult[];
    }) => {
      const record: ExperimentRecord = {
        id: `exp-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        savedAt: new Date().toISOString(),
        type: 'batch',
        batchStats: computeBatchStats(opts.results),
        ...opts,
      };
      setExperiments((prev) => [record, ...prev].slice(0, MAX_RECORDS));
      return record.id;
    },
    [],
  );

  const deleteExperiment = useCallback((id: string) => {
    setExperiments((prev) => prev.filter((e) => e.id !== id));
  }, []);

  const updateNotes = useCallback((id: string, notes: string) => {
    setExperiments((prev) =>
      prev.map((e) => (e.id === id ? { ...e, notes } : e)),
    );
  }, []);

  const clearAll = useCallback(() => setExperiments([]), []);

  const exportAll = useCallback(() => {
    const blob = new Blob(
      [JSON.stringify({ exported_at: new Date().toISOString(), version: '1.0', experiments }, null, 2)],
      { type: 'application/json' },
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `silverbullet-experiments-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [experiments]);

  return {
    experiments,
    savePairExperiment,
    saveBatchExperiment,
    deleteExperiment,
    updateNotes,
    clearAll,
    exportAll,
  };
}
