import type { PredictionResult, ModeConfig } from '../types';

const truncate = (s: string, n = 70) => (s.length > n ? s.slice(0, n) + '…' : s);

interface Props {
  results: PredictionResult[];
  cfg: ModeConfig;
}

export function ResultsTable({ results, cfg }: Props) {
  return (
    <div className="bg-white rounded-2xl border border-slate-200 overflow-hidden">
      <div className="px-5 py-3 border-b border-slate-100 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-700">All Results</h3>
        <span className="text-xs text-slate-400 font-medium">{results.length} pairs</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-50 border-b border-slate-100">
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500 uppercase tracking-wide w-10">
                #
              </th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500 uppercase tracking-wide">
                {cfg.text1Label}
              </th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500 uppercase tracking-wide">
                {cfg.text2Label}
              </th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500 uppercase tracking-wide w-32">
                Score
              </th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-slate-500 uppercase tracking-wide">
                Verdict
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-50">
            {results.map((r, i) => {
              const interp = cfg.interpret(r.probability);
              const badgeClass =
                interp.color === 'green'
                  ? 'bg-emerald-100 text-emerald-700'
                  : interp.color === 'yellow'
                  ? 'bg-amber-100 text-amber-700'
                  : 'bg-red-100 text-red-700';
              const barClass =
                interp.color === 'green'
                  ? 'bg-emerald-500'
                  : interp.color === 'yellow'
                  ? 'bg-amber-400'
                  : 'bg-red-500';
              return (
                <tr key={i} className="hover:bg-slate-50/60 transition-colors">
                  <td className="px-4 py-3 text-slate-400 text-xs tabular-nums">{i + 1}</td>
                  <td className="px-4 py-3 text-slate-600 font-mono text-xs max-w-xs">
                    {truncate(r.text1 ?? '')}
                  </td>
                  <td className="px-4 py-3 text-slate-600 font-mono text-xs max-w-xs">
                    {truncate(r.text2 ?? '')}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="w-16 bg-slate-100 rounded-full h-1.5 flex-shrink-0">
                        <div
                          className={`h-1.5 rounded-full ${barClass}`}
                          style={{ width: `${r.probability * 100}%` }}
                        />
                      </div>
                      <span className="font-mono font-semibold text-slate-800 text-xs tabular-nums">
                        {r.probability.toFixed(3)}
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${badgeClass}`}>
                      {interp.headline}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
