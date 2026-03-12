interface ScoreGaugeProps {
  probability: number;
  prediction: number;
}

export function ScoreGauge({ probability, prediction }: ScoreGaugeProps) {
  const pct = Math.round(probability * 100);
  const barColor =
    probability >= 0.7 ? 'bg-green-500' : probability >= 0.4 ? 'bg-yellow-400' : 'bg-red-500';

  return (
    <div className="mt-4 p-4 border rounded-lg bg-white shadow-sm">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-600">Similarity Score</span>
        <span
          className={`px-2 py-0.5 rounded-full text-xs font-semibold text-white ${
            prediction === 1 ? 'bg-green-600' : 'bg-red-600'
          }`}
        >
          {prediction === 1 ? 'Similar' : 'Different'}
        </span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-4">
        <div
          className={`h-4 rounded-full transition-all duration-500 ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="mt-1 text-right text-lg font-bold text-gray-800">{probability.toFixed(3)}</p>
    </div>
  );
}
