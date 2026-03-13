interface Props {
  probability: number;
  prediction: number;
}

export function ScoreGauge({ probability, prediction }: Props) {
  const pct = Math.round(probability * 100);
  const isHigh = probability >= 0.7;
  const isMid = probability >= 0.4;

  const barColor = isHigh ? 'bg-emerald-500' : isMid ? 'bg-amber-400' : 'bg-red-500';
  const trackColor = isHigh ? 'bg-emerald-100' : isMid ? 'bg-amber-100' : 'bg-red-100';
  const textColor = isHigh ? 'text-emerald-600' : isMid ? 'text-amber-600' : 'text-red-600';
  const badgeBg = prediction === 1 ? 'bg-emerald-500' : 'bg-red-500';

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5">
      <div className="flex items-center justify-between mb-4">
        <span className="text-sm font-semibold text-slate-600">Evaluation Score</span>
        <span
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold text-white ${badgeBg}`}
        >
          <span className="w-1.5 h-1.5 rounded-full bg-white opacity-80" />
          {prediction === 1 ? 'Similar' : 'Different'}
        </span>
      </div>

      <div className={`text-5xl font-black mb-5 tabular-nums ${textColor}`}>
        {probability.toFixed(3)}
      </div>

      <div className={`w-full ${trackColor} rounded-full h-3`}>
        <div
          className={`h-3 rounded-full transition-all duration-700 ease-out ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>

      <div className="flex justify-between mt-1.5 px-0.5">
        {['0', '0.25', '0.5', '0.75', '1'].map((v) => (
          <span key={v} className="text-[10px] text-slate-400 font-mono">
            {v}
          </span>
        ))}
      </div>
    </div>
  );
}
