import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import type { PredictionResult, ModeConfig } from '../types';

interface Props {
  results: PredictionResult[];
  cfg: ModeConfig;
}

const PIE_COLORS = ['#10b981', '#ef4444'];

function buildHistogram(results: PredictionResult[]) {
  const bins = Array.from({ length: 10 }, (_, i) => ({
    range: `${(i * 0.1).toFixed(1)}–${((i + 1) * 0.1).toFixed(1)}`,
    count: 0,
  }));
  for (const r of results) {
    const idx = Math.min(Math.floor(r.probability * 10), 9);
    bins[idx].count++;
  }
  return bins;
}

export function BatchDistribution({ results, cfg }: Props) {
  const similar = results.filter((r) => r.prediction === 1).length;
  const different = results.length - similar;
  const avg = results.reduce((s, r) => s + r.probability, 0) / results.length;
  const sorted = [...results].map((r) => r.probability).sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)] ?? 0;
  const variance = results.reduce((s, r) => s + (r.probability - avg) ** 2, 0) / results.length;
  const std = Math.sqrt(variance);

  const histogram = buildHistogram(results);
  const pieData = [
    { name: cfg.interpret(0.8).headline, value: similar },
    { name: cfg.interpret(0.1).headline, value: different },
  ];

  const stats = [
    { label: 'Total Pairs', value: String(results.length) },
    { label: 'Mean Score', value: avg.toFixed(3) },
    { label: 'Median Score', value: median.toFixed(3) },
    { label: 'Std Dev', value: std.toFixed(3) },
  ];

  return (
    <div className="space-y-4">
      {/* Stats row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {stats.map((s) => (
          <div
            key={s.label}
            className="bg-white rounded-2xl border border-slate-200 p-4 text-center"
          >
            <div className="text-2xl font-black text-slate-900 tabular-nums">{s.value}</div>
            <div className="text-xs text-slate-500 mt-0.5 font-medium">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Histogram — 2/3 width */}
        <div className="md:col-span-2 bg-white rounded-2xl border border-slate-200 p-5">
          <h3 className="text-sm font-semibold text-slate-700 mb-4">Score Distribution</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={histogram} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
              <XAxis
                dataKey="range"
                tick={{ fontSize: 9, fill: '#94a3b8' }}
                interval={0}
                angle={-35}
                textAnchor="end"
                height={40}
              />
              <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} allowDecimals={false} />
              <Tooltip
                contentStyle={{ fontSize: 12, borderRadius: 10, border: '1px solid #e2e8f0' }}
                formatter={(v) => [v ?? 0, 'Pairs']}
              />
              <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Donut chart */}
        <div className="bg-white rounded-2xl border border-slate-200 p-5">
          <h3 className="text-sm font-semibold text-slate-700 mb-4">Outcome Split</h3>
          <ResponsiveContainer width="100%" height={150}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={42}
                outerRadius={65}
                paddingAngle={3}
                dataKey="value"
                startAngle={90}
                endAngle={-270}
              >
                {pieData.map((_, i) => (
                  <Cell key={i} fill={PIE_COLORS[i]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ fontSize: 12, borderRadius: 10 }}
                formatter={(v, name) => [v ?? 0, name]}
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex flex-col gap-1.5 mt-1">
            {pieData.map((d, i) => (
              <div key={i} className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <span
                    className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                    style={{ backgroundColor: PIE_COLORS[i] }}
                  />
                  <span className="text-xs text-slate-600">{d.name}</span>
                </div>
                <span className="text-xs font-semibold text-slate-800 tabular-nums">
                  {d.value}{' '}
                  <span className="text-slate-400 font-normal">
                    ({results.length > 0 ? Math.round((d.value / results.length) * 100) : 0}%)
                  </span>
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
