import { useState } from 'react';
import { predictPair } from '../services/api';
import { ScoreGauge } from './ScoreGauge';
import { getModeConfig } from '../config/modes';
import type { ComparisonMode, PredictionResult } from '../types';

interface Props {
  mode: ComparisonMode;
}

const interpBg = { green: 'bg-emerald-50 border-emerald-200', yellow: 'bg-amber-50 border-amber-200', red: 'bg-red-50 border-red-200' } as const;
const interpHeading = { green: 'text-emerald-700', yellow: 'text-amber-700', red: 'text-red-700' } as const;
const interpBody = { green: 'text-emerald-600', yellow: 'text-amber-600', red: 'text-red-600' } as const;
const interpDivider = { green: 'border-emerald-200', yellow: 'border-amber-200', red: 'border-red-200' } as const;
const interpMuted = { green: 'text-emerald-500', yellow: 'text-amber-500', red: 'text-red-500' } as const;

export function PairScorer({ mode }: Props) {
  const [text1, setText1] = useState('');
  const [text2, setText2] = useState('');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const cfg = getModeConfig(mode);

  const handleScore = async () => {
    if (!text1.trim() || !text2.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await predictPair(text1, text2));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const interpretation = result ? cfg.interpret(result.probability) : null;

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-1.5">
          <label className="block text-sm font-semibold text-slate-700">{cfg.text1Label}</label>
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
          <label className="block text-sm font-semibold text-slate-700">{cfg.text2Label}</label>
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
                {cfg.label} · Score {result.probability.toFixed(3)}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
