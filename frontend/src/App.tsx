import { useState } from 'react';
import { PairScorer } from './components/PairScorer';
import { BatchScorer } from './components/BatchScorer';
import { ErrorBoundary } from './components/ErrorBoundary';
import './index.css';

type Tab = 'pair' | 'batch';

function App() {
  const [tab, setTab] = useState<Tab>('pair');

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b shadow-sm px-6 py-4">
        <h1 className="text-2xl font-bold text-gray-900">
          Silver<span className="text-blue-600">Bullet</span>
        </h1>
        <p className="text-sm text-gray-500 mt-0.5">Text similarity &amp; faithfulness scorer</p>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-8">
        <div className="flex space-x-1 mb-6 border-b">
          {(['pair', 'batch'] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-4 py-2 text-sm font-medium capitalize border-b-2 transition-colors ${
                tab === t
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {t === 'pair' ? 'Single Pair' : 'Batch Scoring'}
            </button>
          ))}
        </div>

        {tab === 'pair' ? (
          <ErrorBoundary>
            <PairScorer />
          </ErrorBoundary>
        ) : (
          <ErrorBoundary>
            <BatchScorer />
          </ErrorBoundary>
        )}
      </main>
    </div>
  );
}

export default App;
