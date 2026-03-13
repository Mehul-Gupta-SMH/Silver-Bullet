import { useState, useRef } from 'react';
import { predictBatch } from '../services/api';
import { ResultsTable } from './ResultsTable';
import { BatchDistribution } from './BatchDistribution';
import { getModeConfig } from '../config/modes';
import type { ComparisonMode, PredictionResult } from '../types';

interface JsonRecord {
  text1: string;
  text2: string;
}

interface Props {
  mode: ComparisonMode;
}

function parseJson(raw: string): Array<[string, string]> {
  const parsed = JSON.parse(raw) as unknown;
  let records: JsonRecord[];
  if (Array.isArray(parsed)) {
    records = parsed as JsonRecord[];
  } else if (parsed && typeof parsed === 'object' && 'data' in parsed) {
    records = (parsed as { data: JsonRecord[] }).data;
  } else {
    throw new Error('Expected a JSON array or {"data": [...]} object');
  }
  return records.map((r, i) => {
    if (!r.text1 || !r.text2) throw new Error(`Record ${i + 1} is missing text1 or text2`);
    return [r.text1, r.text2];
  });
}

export function BatchScorer({ mode }: Props) {
  const [pairs, setPairs] = useState<Array<[string, string]>>([]);
  const [fileName, setFileName] = useState<string | null>(null);
  const [results, setResults] = useState<PredictionResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const cfg = getModeConfig(mode);

  const loadFile = (file: File) => {
    setError(null);
    setResults([]);
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const parsed = parseJson(e.target?.result as string);
        if (parsed.length === 0) throw new Error('No records found in file');
        if (parsed.length > 100) throw new Error(`Max 100 pairs per batch (got ${parsed.length})`);
        setPairs(parsed);
        setFileName(file.name);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to parse file');
        setPairs([]);
        setFileName(null);
      }
    };
    reader.readAsText(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) loadFile(file);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) loadFile(file);
  };

  const handleBatch = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await predictBatch(pairs);
      setResults(
        res.results.map((r, i) => ({
          ...r,
          text1: pairs[i]?.[0] ?? '',
          text2: pairs[i]?.[1] ?? '',
        })),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-5">
      {/* Drop zone */}
      <div>
        <label className="block text-sm font-semibold text-slate-700 mb-2">Upload JSON File</label>
        <div
          onDrop={handleDrop}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onClick={() => fileRef.current?.click()}
          className={`relative border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-150 ${
            dragOver
              ? 'border-violet-400 bg-violet-50'
              : 'border-slate-200 bg-white hover:border-violet-300 hover:bg-slate-50/50'
          }`}
        >
          <input
            ref={fileRef}
            type="file"
            accept=".json"
            onChange={handleFileInput}
            className="hidden"
          />
          {fileName ? (
            <div className="space-y-1.5">
              <div className="text-4xl">📄</div>
              <div className="font-semibold text-slate-800">{fileName}</div>
              <div className="text-sm text-slate-500">
                {pairs.length} pairs ready · Click to replace
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              <div className="text-4xl">📁</div>
              <div className="font-semibold text-slate-700">Drop a JSON file or click to browse</div>
              <div className="text-sm text-slate-400 space-y-1">
                <div>Max 100 pairs per batch</div>
                <div className="flex items-center justify-center gap-2 flex-wrap">
                  <code className="bg-slate-100 text-slate-600 px-2 py-0.5 rounded text-xs">
                    {'[{"text1":"…","text2":"…"}]'}
                  </code>
                  <span className="text-slate-300">or</span>
                  <code className="bg-slate-100 text-slate-600 px-2 py-0.5 rounded text-xs">
                    {'{"data":[{"text1":"…","text2":"…"}]}'}
                  </code>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <button
        onClick={handleBatch}
        disabled={loading || pairs.length === 0}
        className="flex items-center gap-2.5 px-6 py-2.5 bg-violet-600 text-white rounded-xl font-semibold text-sm hover:bg-violet-700 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-150 shadow-sm shadow-violet-200"
      >
        {loading && (
          <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
        )}
        {loading
          ? `Analysing ${pairs.length} pairs…`
          : `Analyse Batch${pairs.length > 0 ? ` (${pairs.length})` : ''}`}
      </button>

      {error && (
        <div className="flex items-start gap-2 rounded-xl bg-red-50 border border-red-200 text-red-700 px-4 py-3 text-sm">
          <span className="flex-1">{error}</span>
          <button
            className="text-red-400 hover:text-red-600 font-bold text-base leading-none"
            onClick={() => setError(null)}
            aria-label="Dismiss error"
          >
            ×
          </button>
        </div>
      )}

      {results.length > 0 && (
        <>
          <BatchDistribution results={results} cfg={cfg} />
          <ResultsTable results={results} cfg={cfg} />
        </>
      )}
    </div>
  );
}
