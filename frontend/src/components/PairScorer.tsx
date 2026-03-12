import { useState } from 'react';
import { predictPair } from '../services/api';
import { ScoreGauge } from './ScoreGauge';
import type { PredictionResult } from '../types';

export function PairScorer() {
  const [text1, setText1] = useState('');
  const [text2, setText2] = useState('');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleScore = async () => {
    if (!text1.trim() || !text2.trim()) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await predictPair(text1, text2));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Text 1</label>
          <textarea
            className="w-full h-40 p-3 border rounded-lg font-mono text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-400"
            placeholder="Enter first text..."
            value={text1}
            onChange={(e) => setText1(e.target.value)}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Text 2</label>
          <textarea
            className="w-full h-40 p-3 border rounded-lg font-mono text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-400"
            placeholder="Enter second text..."
            value={text2}
            onChange={(e) => setText2(e.target.value)}
          />
        </div>
      </div>

      <button
        onClick={handleScore}
        disabled={loading || !text1.trim() || !text2.trim()}
        className="flex items-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {loading && (
          <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
        )}
        {loading ? 'Scoring…' : 'Score'}
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
      {result && <ScoreGauge probability={result.probability} prediction={result.prediction} />}
    </div>
  );
}
