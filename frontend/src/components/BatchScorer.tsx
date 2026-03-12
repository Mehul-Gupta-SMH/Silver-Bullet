import { useState } from 'react';
import { predictBatch } from '../services/api';
import { ResultsTable } from './ResultsTable';
import type { PredictionResult } from '../types';

const parsePairs = (raw: string): Array<[string, string]> =>
  raw
    .split('\n')
    .map((line) => line.split('|||').map((s) => s.trim()))
    .filter((parts): parts is [string, string] => parts.length === 2 && !!parts[0] && !!parts[1]);

export function BatchScorer() {
  const [input, setInput] = useState('');
  const [results, setResults] = useState<PredictionResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleBatch = async () => {
    const pairs = parsePairs(input);
    if (pairs.length === 0) {
      setError('No valid pairs found. Use format: text1 ||| text2');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await predictBatch(pairs);
      setResults(res.results);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const pairCount = parsePairs(input).length;

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-500">
        One pair per line:{' '}
        <code className="bg-gray-100 px-1 rounded">text1 ||| text2</code>
      </p>
      <textarea
        className="w-full h-48 p-3 border rounded-lg font-mono text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-400"
        placeholder={"The cat sat on the mat. ||| A cat was sitting on the mat.\nThe sky is blue. ||| The ocean is vast."}
        value={input}
        onChange={(e) => setInput(e.target.value)}
      />
      <button
        onClick={handleBatch}
        disabled={loading || pairCount === 0}
        className="flex items-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {loading && (
          <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
        )}
        {loading ? `Scoring ${pairCount} pairs…` : `Score Batch${pairCount > 0 ? ` (${pairCount})` : ''}`}
      </button>
      {error && (
        <div className="flex items-start gap-2 rounded-lg bg-red-50 border border-red-200 text-red-700 px-4 py-3 text-sm">
          <span className="flex-1">{error}</span>
          <button
            className="text-red-500 hover:text-red-700 font-bold leading-none"
            onClick={() => setError(null)}
            aria-label="Dismiss error"
          >
            ×
          </button>
        </div>
      )}
      {results.length > 0 && <ResultsTable results={results} />}
    </div>
  );
}
