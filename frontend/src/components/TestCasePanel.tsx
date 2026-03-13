import { useState } from 'react';
import { getTestCases, OUTCOME_LABEL, OUTCOME_COLOR } from '../data/testCases';
import type { PairTestCase, BatchTestCase } from '../data/testCases';
import type { ComparisonMode } from '../types';

// ── Pair scorer callback ──────────────────────────────────────────────────────
interface PairLoadProps {
  scope: 'pair';
  mode: ComparisonMode;
  onLoad: (tc: PairTestCase) => void;
}

// ── Batch scorer callback ─────────────────────────────────────────────────────
interface BatchLoadProps {
  scope: 'batch';
  mode: ComparisonMode;
  onLoad: (tc: BatchTestCase) => void;
}

type Props = PairLoadProps | BatchLoadProps;

const TAG_COLORS: Record<string, string> = {
  hallucination: 'bg-red-50 text-red-600',
  grounded: 'bg-emerald-50 text-emerald-600',
  faithful: 'bg-emerald-50 text-emerald-600',
  agreement: 'bg-blue-50 text-blue-600',
  divergence: 'bg-rose-50 text-rose-600',
  code: 'bg-violet-50 text-violet-700',
  batch: 'bg-slate-100 text-slate-600',
};

const tagClass = (tag: string) =>
  TAG_COLORS[tag] ?? 'bg-slate-100 text-slate-500';

export function TestCasePanel(props: Props) {
  const [open, setOpen] = useState(false);
  const cases = getTestCases(props.mode, props.scope);

  if (cases.length === 0) return null;

  return (
    <div className="bg-white rounded-2xl border border-slate-200 overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-5 py-3.5 hover:bg-slate-50 transition-colors"
      >
        <div className="flex items-center gap-2.5">
          <span className="text-sm font-semibold text-slate-700">📖 Example Test Cases</span>
          <span className="text-xs bg-violet-100 text-violet-700 px-2 py-0.5 rounded-full font-medium">
            {cases.length} examples
          </span>
        </div>
        <span
          className={`text-slate-400 text-xs transition-transform duration-200 select-none ${
            open ? 'rotate-180' : ''
          }`}
        >
          ▼
        </span>
      </button>

      {open && (
        <div className="border-t border-slate-100 p-4 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
          {cases.map((tc) => (
            <div
              key={tc.id}
              className="rounded-xl border border-slate-100 bg-slate-50/60 p-3.5 flex flex-col gap-2.5"
            >
              {/* Header */}
              <div className="flex items-start gap-2">
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-slate-800 leading-tight">{tc.title}</p>
                  <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">{tc.description}</p>
                </div>
              </div>

              {/* Expected outcome badge */}
              <span
                className={`self-start text-xs font-semibold px-2 py-0.5 rounded-full ${OUTCOME_COLOR[tc.expectedOutcome]}`}
              >
                {OUTCOME_LABEL[tc.expectedOutcome]}
              </span>

              {/* Tags */}
              <div className="flex flex-wrap gap-1">
                {tc.tags.map((t) => (
                  <span key={t} className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${tagClass(t)}`}>
                    {t}
                  </span>
                ))}
              </div>

              {/* Model names if present */}
              {(tc.name1 ?? tc.name2) && (
                <div className="flex items-center gap-1.5 text-xs text-slate-500">
                  <span className="font-mono bg-white border border-slate-200 px-1.5 py-0.5 rounded text-[11px]">
                    {tc.name1 ?? '—'}
                  </span>
                  <span className="text-slate-300">vs</span>
                  <span className="font-mono bg-white border border-slate-200 px-1.5 py-0.5 rounded text-[11px]">
                    {tc.name2 ?? '—'}
                  </span>
                </div>
              )}

              {/* Load button */}
              <button
                onClick={() => {
                  if (props.scope === 'pair') {
                    props.onLoad(tc as PairTestCase);
                  } else {
                    props.onLoad(tc as BatchTestCase);
                  }
                  setOpen(false);
                }}
                className="mt-auto w-full py-1.5 text-xs font-semibold bg-violet-600 text-white rounded-lg hover:bg-violet-700 transition-colors"
              >
                Load Example →
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
