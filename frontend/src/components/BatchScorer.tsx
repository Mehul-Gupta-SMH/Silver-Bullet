import { useState, useRef } from 'react';
import { predictBatch } from '../services/api';
import { ResultsTable } from './ResultsTable';
import { BatchDistribution } from './BatchDistribution';
import { ModelConfig } from './ModelConfig';
import { TestCasePanel } from './TestCasePanel';
import { SaveExperimentForm } from './SaveExperimentForm';
import { getModeConfig } from '../config/modes';
import type { TextMeta } from './ModelConfig';
import type { BatchTestCase } from '../data/testCases';
import type { ComparisonMode, PredictionResult } from '../types';

interface JsonRecord {
  text1: string;
  text2: string;
}

interface Props {
  mode: ComparisonMode;
  onSave: (opts: {
    name: string;
    notes: string;
    mode: ComparisonMode;
    modelMeta: TextMeta;
    fileName: string;
    results: PredictionResult[];
  }) => void;
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

export function BatchScorer({ mode, onSave }: Props) {
  const [pairs, setPairs] = useState<Array<[string, string]>>([]);
  const [fileName, setFileName] = useState<string | null>(null);
  const [meta, setMeta] = useState<TextMeta>({ name1: '', name2: '', baseline: null });
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

  const handleLoadTestCase = (tc: BatchTestCase) => {
    const loaded: Array<[string, string]> = tc.pairs.map((p) => [p.text1, p.text2]);
    setPairs(loaded);
    setFileName(`${tc.title} (example)`);
    setMeta({ name1: tc.name1 ?? '', name2: tc.name2 ?? '', baseline: null });
    setResults([]);
    setError(null);
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
      const res = await predictBatch(pairs, mode);
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

  const resolvedCfg = {
    ...cfg,
    text1Label: meta.name1
      ? mode === 'model-vs-model' && meta.baseline === '1'
        ? `${meta.name1} (baseline)`
        : meta.name1
      : cfg.text1Label,
    text2Label: meta.name2
      ? mode === 'model-vs-model' && meta.baseline === '2'
        ? `${meta.name2} (baseline)`
        : meta.name2
      : cfg.text2Label,
  };

  const defaultExpName = `Batch · ${cfg.label} · ${new Date().toLocaleDateString()}`;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Test cases */}
      <TestCasePanel scope="batch" mode={mode} onLoad={handleLoadTestCase} />

      {/* Model labels */}
      <ModelConfig
        mode={mode}
        meta={meta}
        onChange={setMeta}
        label1={cfg.text1Label}
        label2={cfg.text2Label}
      />

      {/* Drop zone */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <div style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 10,
          color: 'var(--text-3)',
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
        }}>
          Upload JSON File
        </div>
        <div
          onDrop={handleDrop}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onClick={() => fileRef.current?.click()}
          style={{
            border: `2px dashed ${dragOver ? 'var(--accent)' : 'var(--border-2)'}`,
            borderRadius: 12,
            padding: '36px 24px',
            textAlign: 'center',
            cursor: 'pointer',
            background: dragOver ? 'var(--accent-dim)' : 'var(--bg-3)',
            transition: 'border-color 0.15s, background 0.15s',
          }}
          onMouseOver={e => { if (!dragOver) (e.currentTarget as HTMLElement).style.borderColor = 'rgba(0,229,204,0.35)'; }}
          onMouseOut={e => { if (!dragOver) (e.currentTarget as HTMLElement).style.borderColor = 'var(--border-2)'; }}
        >
          <input
            ref={fileRef}
            type="file"
            accept=".json"
            onChange={handleFileInput}
            style={{ display: 'none' }}
          />
          {fileName ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
              <div style={{ fontSize: 32 }}>📄</div>
              <div style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 13,
                fontWeight: 500,
                color: 'var(--text-1)',
              }}>
                {fileName}
              </div>
              <div style={{
                fontFamily: 'var(--font-body)',
                fontSize: 12,
                color: 'var(--text-3)',
              }}>
                {pairs.length} pairs ready · Click to replace
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
              <div style={{ fontSize: 32, opacity: 0.5 }}>📁</div>
              <div style={{
                fontFamily: 'var(--font-body)',
                fontSize: 13,
                fontWeight: 500,
                color: 'var(--text-2)',
              }}>
                Drop a JSON file or click to browse
              </div>
              <div style={{
                fontFamily: 'var(--font-body)',
                fontSize: 12,
                color: 'var(--text-3)',
                display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4,
              }}>
                <div>Max 100 pairs per batch</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
                  <code style={{
                    background: 'var(--bg-4)',
                    color: 'var(--accent)',
                    padding: '2px 7px',
                    borderRadius: 4,
                    fontFamily: 'var(--font-mono)',
                    fontSize: 10,
                    border: '1px solid var(--border)',
                  }}>
                    {'[{"text1":"…","text2":"…"}]'}
                  </code>
                  <span style={{ color: 'var(--text-3)', fontSize: 11 }}>or</span>
                  <code style={{
                    background: 'var(--bg-4)',
                    color: 'var(--accent)',
                    padding: '2px 7px',
                    borderRadius: 4,
                    fontFamily: 'var(--font-mono)',
                    fontSize: 10,
                    border: '1px solid var(--border)',
                  }}>
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
        className="sb-btn-primary"
        style={{ alignSelf: 'flex-start' }}
      >
        {loading && (
          <span style={{
            width: 14, height: 14,
            border: '2px solid rgba(4,6,7,0.3)',
            borderTopColor: '#040607',
            borderRadius: '50%',
            display: 'inline-block',
            animation: 'spin 0.7s linear infinite',
          }} />
        )}
        {loading
          ? `Analysing ${pairs.length} pairs…`
          : `Analyse Batch${pairs.length > 0 ? ` (${pairs.length})` : ''}`}
      </button>

      {error && (
        <div style={{
          display: 'flex', alignItems: 'flex-start', gap: 8,
          background: 'rgba(239,68,68,0.08)',
          border: '1px solid rgba(239,68,68,0.25)',
          borderRadius: 10,
          padding: '10px 14px',
          fontFamily: 'var(--font-body)',
          fontSize: 13,
          color: '#F87171',
        }}>
          <span style={{ flex: 1 }}>{error}</span>
          <button
            onClick={() => setError(null)}
            style={{
              background: 'none', border: 'none', cursor: 'pointer',
              color: '#F87171', fontSize: 16, lineHeight: 1, padding: 0,
              opacity: 0.6,
            }}
            aria-label="Dismiss"
          >×</button>
        </div>
      )}

      {results.length > 0 && (
        <>
          <BatchDistribution results={results} cfg={resolvedCfg} />
          <ResultsTable results={results} cfg={resolvedCfg} />
          <SaveExperimentForm
            defaultName={defaultExpName}
            onSave={(name, notes) =>
              onSave({
                name,
                notes,
                mode,
                modelMeta: meta,
                fileName: fileName ?? 'unknown',
                results,
              })
            }
          />
        </>
      )}
    </div>
  );
}
