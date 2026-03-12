import type { PredictionResult } from '../types';

const truncate = (s: string, n = 60) => (s.length > n ? s.slice(0, n) + '…' : s);

export function ResultsTable({ results }: { results: PredictionResult[] }) {
  return (
    <div className="overflow-x-auto mt-4">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="bg-gray-100">
            {['#', 'Text 1', 'Text 2', 'Score', 'Prediction'].map((h) => (
              <th key={h} className="p-2 text-left border">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => (
            <tr key={i} className="hover:bg-gray-50">
              <td className="p-2 border text-gray-500">{i + 1}</td>
              <td className="p-2 border font-mono">{truncate(r.text1)}</td>
              <td className="p-2 border font-mono">{truncate(r.text2)}</td>
              <td className="p-2 border font-bold">{r.probability.toFixed(3)}</td>
              <td className="p-2 border">
                <span
                  className={`px-2 py-0.5 rounded-full text-xs font-semibold text-white ${
                    r.prediction === 1 ? 'bg-green-600' : 'bg-red-600'
                  }`}
                >
                  {r.prediction === 1 ? 'Similar' : 'Different'}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
