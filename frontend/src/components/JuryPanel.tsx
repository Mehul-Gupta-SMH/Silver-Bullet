import { useState } from 'react';
import type { JuryResult } from '../types';

interface Props {
  result: JuryResult;
}

export function JuryPanel({ result }: Props) {
  const [expanded, setExpanded] = useState(false);
  const pct = Math.round(result.score * 100);
  const isFaithful = result.verdict === 'faithful';
  const modelLabel = result.model_used;

  const verdictColor  = isFaithful ? 'text-emerald-600' : 'text-red-500';
  const barColor      = isFaithful ? 'bg-emerald-500'   : 'bg-red-500';
  const trackColor    = isFaithful ? 'bg-emerald-100'   : 'bg-red-100';
  const badgeBg       = isFaithful ? 'bg-emerald-500'   : 'bg-red-500';

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-slate-600">LLM Jury Score</span>
          {modelLabel && (
            <span className="text-[10px] text-slate-400 bg-slate-100 border border-slate-200 px-2 py-0.5 rounded-full font-mono">
              {modelLabel}
            </span>
          )}
        </div>
        <span className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold text-white ${badgeBg}`}>
          <span className="w-1.5 h-1.5 rounded-full bg-white opacity-80" />
          {isFaithful ? 'Faithful' : 'Hallucinated'}
        </span>
      </div>

      {/* Score */}
      <div className={`text-5xl font-black tabular-nums ${verdictColor}`}>
        {result.score.toFixed(3)}
      </div>

      {/* Bar */}
      <div className={`w-full ${trackColor} rounded-full h-3`}>
        <div
          className={`h-3 rounded-full transition-all duration-700 ease-out ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="flex justify-between mt-1 px-0.5">
        {['0', '0.25', '0.5', '0.75', '1'].map(v => (
          <span key={v} className="text-[10px] text-slate-400 font-mono">{v}</span>
        ))}
      </div>

      {/* Questions breakdown toggle */}
      {result.questions?.length > 0 && (
        <div>
          <button
            onClick={() => setExpanded(e => !e)}
            className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-700 transition-colors"
          >
            <span className={`transition-transform duration-200 ${expanded ? 'rotate-90' : ''}`}>▶</span>
            {result.questions.length} jury questions
          </button>

          {expanded && (
            <div className="mt-3 space-y-2">
              {result.questions.map((q, i) => (
                <div
                  key={i}
                  className={`rounded-xl border p-3 text-xs ${
                    q.answer === 'yes'
                      ? 'border-emerald-200 bg-emerald-50'
                      : 'border-red-200 bg-red-50'
                  }`}
                >
                  <div className="flex items-start justify-between gap-3 mb-1.5">
                    <span className="font-medium text-slate-700 leading-snug">{q.question}</span>
                    <div className="flex items-center gap-1.5 shrink-0">
                      <span className={`font-bold ${q.answer === 'yes' ? 'text-emerald-600' : 'text-red-500'}`}>
                        {q.answer === 'yes' ? 'YES' : 'NO'}
                      </span>
                      <span className="text-slate-400 font-mono">conf={q.confidence.toFixed(2)}</span>
                    </div>
                  </div>
                  {q.reasoning && (
                    <p className="text-slate-500 italic leading-snug">{q.reasoning}</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
