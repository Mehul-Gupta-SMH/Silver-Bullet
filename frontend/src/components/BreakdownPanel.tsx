import type { BreakdownResult, MisalignmentReason } from '../types';

interface Props {
  breakdown: BreakdownResult;
}

const THRESH_HIGH = 0.7;
const THRESH_MID  = 0.4;

function alignScore(score: number) {
  if (score >= THRESH_HIGH) return { label: 'Aligned',  bg: 'bg-emerald-50', border: 'border-emerald-300', text: 'text-emerald-700', dot: 'bg-emerald-500' };
  if (score >= THRESH_MID)  return { label: 'Partial',  bg: 'bg-amber-50',   border: 'border-amber-300',   text: 'text-amber-700',   dot: 'bg-amber-400' };
  return                           { label: 'Divergent', bg: 'bg-red-50',     border: 'border-red-300',     text: 'text-red-700',     dot: 'bg-red-500' };
}

function barColor(score: number) {
  if (score >= THRESH_HIGH) return 'bg-emerald-500';
  if (score >= THRESH_MID)  return 'bg-amber-400';
  return 'bg-red-500';
}

const SEVERITY_STYLES: Record<MisalignmentReason['severity'], {
  border: string; bg: string; badge: string; badgeText: string; icon: string;
}> = {
  high:   { border: 'border-red-200',    bg: 'bg-red-50',    badge: 'bg-red-100',    badgeText: 'text-red-700',    icon: '⚠' },
  medium: { border: 'border-amber-200',  bg: 'bg-amber-50',  badge: 'bg-amber-100',  badgeText: 'text-amber-700',  icon: '◆' },
  low:    { border: 'border-slate-200',  bg: 'bg-slate-50',  badge: 'bg-slate-100',  badgeText: 'text-slate-600',  icon: '▸' },
};

export function BreakdownPanel({ breakdown }: Props) {
  const {
    sentences1, sentences2,
    alignment,
    divergent_in_1, divergent_in_2,
    feature_scores,
    misalignment_reasons = [],
  } = breakdown;

  const n = sentences1.length;
  const m = sentences2.length;

  // Best alignment score for each sentence
  const maxRow = alignment.length > 0
    ? alignment.map(row => Math.max(...row))
    : [];
  const maxCol = m > 0 && alignment.length > 0
    ? Array.from({ length: m }, (_, j) => Math.max(...alignment.map(r => r[j])))
    : [];

  const hasDivergence = divergent_in_1.length > 0 || divergent_in_2.length > 0;

  return (
    <div className="space-y-6 rounded-2xl border border-violet-100 bg-violet-50/40 p-5">

      {/* Header */}
      <div className="flex items-center gap-2.5">
        <span className="text-violet-700 text-base font-bold">Divergence Analysis</span>
        {hasDivergence ? (
          <span className="px-2 py-0.5 rounded-full bg-red-100 text-red-700 text-xs font-semibold">
            {divergent_in_1.length + divergent_in_2.length} orphaned sentence{divergent_in_1.length + divergent_in_2.length !== 1 ? 's' : ''}
          </span>
        ) : (
          <span className="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-700 text-xs font-semibold">
            Full coverage
          </span>
        )}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 text-xs text-slate-500">
        {[
          { dot: 'bg-emerald-500', label: `Aligned (≥ ${THRESH_HIGH})` },
          { dot: 'bg-amber-400',   label: `Partial (${THRESH_MID}–${THRESH_HIGH})` },
          { dot: 'bg-red-500',     label: `Divergent (< ${THRESH_MID})` },
        ].map(({ dot, label }) => (
          <span key={label} className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${dot}`} />
            {label}
          </span>
        ))}
      </div>

      {/* Sentence columns */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Text 1 */}
        <div className="space-y-2">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
            Text 1 — {n} sentence{n !== 1 ? 's' : ''}
          </p>
          {sentences1.length === 0 && (
            <p className="text-sm text-slate-400 italic">No sentences detected.</p>
          )}
          {sentences1.map((s, i) => {
            const score = maxRow[i] ?? 0;
            const style = alignScore(score);
            return (
              <div
                key={i}
                className={`rounded-xl border p-3 ${style.bg} ${style.border} space-y-1.5`}
              >
                <p className="text-sm leading-relaxed text-slate-800">{s}</p>
                <div className="flex items-center gap-1.5">
                  <span className={`w-1.5 h-1.5 rounded-full ${style.dot}`} />
                  <span className={`text-xs font-semibold ${style.text}`}>
                    {style.label} · best match {score.toFixed(2)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Text 2 */}
        <div className="space-y-2">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
            Text 2 — {m} sentence{m !== 1 ? 's' : ''}
          </p>
          {sentences2.length === 0 && (
            <p className="text-sm text-slate-400 italic">No sentences detected.</p>
          )}
          {sentences2.map((s, j) => {
            const score = maxCol[j] ?? 0;
            const style = alignScore(score);
            return (
              <div
                key={j}
                className={`rounded-xl border p-3 ${style.bg} ${style.border} space-y-1.5`}
              >
                <p className="text-sm leading-relaxed text-slate-800">{s}</p>
                <div className="flex items-center gap-1.5">
                  <span className={`w-1.5 h-1.5 rounded-full ${style.dot}`} />
                  <span className={`text-xs font-semibold ${style.text}`}>
                    {style.label} · best match {score.toFixed(2)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Divergence summary */}
      {hasDivergence && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 space-y-2">
          <p className="text-sm font-semibold text-red-700">Points of Divergence</p>
          {divergent_in_1.length > 0 && (
            <div className="space-y-1">
              <p className="text-xs text-red-500 font-medium">
                {divergent_in_1.length} sentence{divergent_in_1.length !== 1 ? 's' : ''} in Text 1 have no strong counterpart in Text 2:
              </p>
              {divergent_in_1.map(i => (
                <p key={i} className="text-xs font-mono text-red-700 bg-red-100 rounded px-2 py-1">
                  [{i + 1}] {sentences1[i]}
                </p>
              ))}
            </div>
          )}
          {divergent_in_2.length > 0 && (
            <div className="space-y-1 mt-2">
              <p className="text-xs text-red-500 font-medium">
                {divergent_in_2.length} sentence{divergent_in_2.length !== 1 ? 's' : ''} in Text 2 have no strong counterpart in Text 1:
              </p>
              {divergent_in_2.map(j => (
                <p key={j} className="text-xs font-mono text-red-700 bg-red-100 rounded px-2 py-1">
                  [{j + 1}] {sentences2[j]}
                </p>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Feature scores */}
      <div className="space-y-2">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
          What drove the score
        </p>
        <div className="space-y-2">
          {Object.entries(feature_scores).map(([name, score]) => (
            <div key={name} className="flex items-center gap-3">
              <span className="text-xs text-slate-600 w-36 shrink-0 font-medium">{name}</span>
              <div className="flex-1 h-2.5 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${barColor(score)}`}
                  style={{ width: `${Math.round(score * 100)}%` }}
                />
              </div>
              <span className="text-xs font-mono text-slate-500 w-10 text-right tabular-nums">
                {score.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Misalignment reasons */}
      {misalignment_reasons.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
            Misalignment Diagnostics
          </p>
          <div className="space-y-2">
            {misalignment_reasons.map((reason, idx) => {
              const s = SEVERITY_STYLES[reason.severity];
              return (
                <div
                  key={idx}
                  className={`rounded-xl border p-3 space-y-1 ${s.border} ${s.bg}`}
                >
                  <div className="flex items-center gap-2">
                    <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${s.badge} ${s.badgeText}`}>
                      {s.icon} {reason.severity.toUpperCase()}
                    </span>
                    <span className="text-sm font-semibold text-slate-800">{reason.label}</span>
                    <span className="ml-auto text-xs text-slate-400 shrink-0">{reason.signal}</span>
                  </div>
                  <p className="text-xs text-slate-600 leading-relaxed">{reason.description}</p>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {misalignment_reasons.length === 0 && (
        <div className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3">
          <p className="text-xs font-semibold text-emerald-700">No misalignment signals detected</p>
          <p className="text-xs text-emerald-600 mt-0.5">
            All feature signals are within expected alignment thresholds.
          </p>
        </div>
      )}
    </div>
  );
}
