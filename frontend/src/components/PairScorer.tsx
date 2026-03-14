import { useState } from 'react';
import { predictPair, predictPairBreakdown } from '../services/api';
import { useLocalStorage } from '../hooks/useLocalStorage';
import { ScoreGauge } from './ScoreGauge';
import { BreakdownPanel } from './BreakdownPanel';
import { ModelConfig } from './ModelConfig';
import { TestCasePanel } from './TestCasePanel';
import { SaveExperimentForm } from './SaveExperimentForm';
import { getModeConfig } from '../config/modes';
import type { TextMeta } from './ModelConfig';
import type { PairTestCase } from '../data/testCases';
import type { BreakdownResult, ComparisonMode, PredictionResult } from '../types';

export interface PairInitData {
  text1: string;
  text2: string;
  meta: TextMeta;
}

interface Props {
  mode: ComparisonMode;
  initData?: PairInitData;
  onSave: (opts: {
    name: string;
    notes: string;
    mode: ComparisonMode;
    modelMeta: TextMeta;
    text1: string;
    text2: string;
    result: { prediction: number; probability: number };
  }) => void;
}

const interpBg = { green: 'bg-emerald-50 border-emerald-200', yellow: 'bg-amber-50 border-amber-200', red: 'bg-red-50 border-red-200' } as const;
const interpHeading = { green: 'text-emerald-700', yellow: 'text-amber-700', red: 'text-red-700' } as const;
const interpBody = { green: 'text-emerald-600', yellow: 'text-amber-600', red: 'text-red-600' } as const;
const interpDivider = { green: 'border-emerald-200', yellow: 'border-amber-200', red: 'border-red-200' } as const;
const interpMuted = { green: 'text-emerald-500', yellow: 'text-amber-500', red: 'text-red-500' } as const;

export function PairScorer({ mode, initData, onSave }: Props) {
  // When initData is provided (re-run from experiments), use it directly and persist it.
  // Otherwise restore from localStorage so the draft survives page refreshes.
  const [text1, setText1] = useLocalStorage<string>(
    'sb_pair_text1',
    initData?.text1 ?? '',
  );
  const [text2, setText2] = useLocalStorage<string>(
    'sb_pair_text2',
    initData?.text2 ?? '',
  );
  const [meta, setMeta] = useLocalStorage<TextMeta>(
    'sb_pair_meta',
    initData?.meta ?? { name1: '', name2: '', baseline: null },
  );

  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [breakdown, setBreakdown] = useState<BreakdownResult | null>(null);
  const [breakdownLoading, setBreakdownLoading] = useState(false);
  const [breakdownError, setBreakdownError] = useState<string | null>(null);
  const [showBreakdown, setShowBreakdown] = useState(false);

  const cfg = getModeConfig(mode);
  const label1 = meta.name1 || cfg.text1Label;
  const label2 = meta.name2 || cfg.text2Label;

  const handleScore = async () => {
    if (!text1.trim() || !text2.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setBreakdown(null);
    setShowBreakdown(false);
    try {
      setResult(await predictPair(text1, text2));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const handleDrillDown = async () => {
    if (showBreakdown) {
      setShowBreakdown(false);
      return;
    }
    if (breakdown) {
      setShowBreakdown(true);
      return;
    }
    setBreakdownLoading(true);
    setBreakdownError(null);
    try {
      const bd = await predictPairBreakdown(text1, text2);
      setBreakdown(bd);
      setShowBreakdown(true);
    } catch (e) {
      setBreakdownError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setBreakdownLoading(false);
    }
  };

  const handleLoadTestCase = (tc: PairTestCase) => {
    setText1(tc.text1);
    setText2(tc.text2);
    setMeta({
      name1: tc.name1 ?? '',
      name2: tc.name2 ?? '',
      baseline: null,
    });
    setResult(null);
    setError(null);
    setBreakdown(null);
    setShowBreakdown(false);
  };

  const interpretation = result ? cfg.interpret(result.probability) : null;
  const comparisonLabel =
    mode === 'model-vs-model' && (meta.name1 || meta.name2)
      ? `${meta.baseline === '1' ? `${label1} (baseline)` : label1} vs ${meta.baseline === '2' ? `${label2} (baseline)` : label2}`
      : cfg.label;

  const defaultExpName = `${cfg.label} · ${new Date().toLocaleDateString()}`;

  return (
    <div className="space-y-5">
      {/* Test case examples */}
      <TestCasePanel scope="pair" mode={mode} onLoad={handleLoadTestCase} />

      {/* Model / source name config */}
      <ModelConfig
        mode={mode}
        meta={meta}
        onChange={setMeta}
        label1={cfg.text1Label}
        label2={cfg.text2Label}
      />

      {/* Text inputs */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-1.5">
          <label className="block text-sm font-semibold text-slate-700">{label1}</label>
          <textarea
            className="w-full h-44 p-3.5 border border-slate-200 rounded-xl text-sm resize-none focus:outline-none focus:ring-2 focus:ring-violet-400 focus:border-transparent placeholder:text-slate-300 font-mono leading-relaxed transition-shadow"
            placeholder={cfg.text1Placeholder}
            value={text1}
            onChange={(e) => setText1(e.target.value)}
          />
          <div className="text-xs text-slate-400 text-right tabular-nums">
            {text1.length.toLocaleString()} / 10,000
          </div>
        </div>

        <div className="space-y-1.5">
          <label className="block text-sm font-semibold text-slate-700">{label2}</label>
          <textarea
            className="w-full h-44 p-3.5 border border-slate-200 rounded-xl text-sm resize-none focus:outline-none focus:ring-2 focus:ring-violet-400 focus:border-transparent placeholder:text-slate-300 font-mono leading-relaxed transition-shadow"
            placeholder={cfg.text2Placeholder}
            value={text2}
            onChange={(e) => setText2(e.target.value)}
          />
          <div className="text-xs text-slate-400 text-right tabular-nums">
            {text2.length.toLocaleString()} / 10,000
          </div>
        </div>
      </div>

      <button
        onClick={handleScore}
        disabled={loading || !text1.trim() || !text2.trim()}
        className="flex items-center gap-2.5 px-6 py-2.5 bg-violet-600 text-white rounded-xl font-semibold text-sm hover:bg-violet-700 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-150 shadow-sm shadow-violet-200"
      >
        {loading && (
          <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
        )}
        {loading ? 'Analysing…' : 'Analyse Pair'}
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

      {result && interpretation && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <ScoreGauge probability={result.probability} prediction={result.prediction} />
            <div className={`rounded-2xl border p-5 flex flex-col justify-center ${interpBg[interpretation.color]}`}>
              <div className={`text-xl font-bold mb-2 ${interpHeading[interpretation.color]}`}>
                {interpretation.headline}
              </div>
              <p className={`text-sm leading-relaxed ${interpBody[interpretation.color]}`}>
                {interpretation.detail}
              </p>
              <div className={`mt-4 pt-3 border-t ${interpDivider[interpretation.color]}`}>
                <span className={`text-xs font-mono font-semibold uppercase tracking-wide ${interpMuted[interpretation.color]}`}>
                  {comparisonLabel} · Score {result.probability.toFixed(3)}
                </span>
              </div>
            </div>
          </div>

          {/* Drill-down trigger */}
          <div className="flex items-center gap-3">
            <button
              onClick={handleDrillDown}
              disabled={breakdownLoading}
              className="flex items-center gap-2 px-4 py-2 rounded-xl border border-violet-300 bg-white text-violet-700 text-sm font-semibold hover:bg-violet-50 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-150"
            >
              {breakdownLoading ? (
                <span className="w-3.5 h-3.5 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
              ) : (
                <span className="text-base">{showBreakdown ? '▲' : '▼'}</span>
              )}
              {breakdownLoading ? 'Running deep analysis…' : showBreakdown ? 'Hide Breakdown' : 'Drill Down — Impact & Divergence'}
            </button>
            {breakdownError && (
              <span className="text-xs text-red-600">{breakdownError}</span>
            )}
          </div>

          {/* Breakdown panel */}
          {showBreakdown && breakdown && (
            <BreakdownPanel breakdown={breakdown} />
          )}

          {/* Save to experiments */}
          <SaveExperimentForm
            defaultName={defaultExpName}
            onSave={(name, notes) =>
              onSave({ name, notes, mode, modelMeta: meta, text1, text2, result })
            }
          />
        </>
      )}
    </div>
  );
}
