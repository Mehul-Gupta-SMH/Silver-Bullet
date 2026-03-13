import { useState } from 'react';
import { PairScorer } from './components/PairScorer';
import { BatchScorer } from './components/BatchScorer';
import { ErrorBoundary } from './components/ErrorBoundary';
import { ComparisonModeSelector } from './components/ComparisonModeSelector';
import { FeaturePanel } from './components/FeaturePanel';
import type { ComparisonMode } from './types';
import './index.css';

type Tab = 'pair' | 'batch';

const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: 'pair', label: 'Single Pair', icon: '⚡' },
  { id: 'batch', label: 'Batch Scoring', icon: '📦' },
];

function App() {
  const [tab, setTab] = useState<Tab>('pair');
  const [mode, setMode] = useState<ComparisonMode>('reference-vs-generated');

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10 backdrop-blur-sm bg-white/95">
        <div className="max-w-6xl mx-auto px-6 py-3.5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-violet-600 flex items-center justify-center text-white font-black text-sm shadow-sm shadow-violet-200">
              SB
            </div>
            <div>
              <h1 className="text-base font-bold text-slate-900 leading-none">
                Silver<span className="text-violet-600">Bullet</span>
              </h1>
              <p className="text-[11px] text-slate-400 mt-0.5 leading-none">
                Text similarity &amp; faithfulness scorer
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-1.5 text-xs text-slate-500 bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-full">
              <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
              API live
            </div>
            <div className="flex items-center gap-1 text-xs text-slate-500">
              <span className="hidden sm:inline text-slate-400">5 signal families</span>
              <span className="text-slate-300 hidden sm:inline">·</span>
              <span className="hidden sm:inline text-slate-400">16 feature maps</span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8 space-y-6">
        {/* Mode selector */}
        <ComparisonModeSelector selected={mode} onChange={setMode} />

        {/* Feature panel (collapsible) */}
        <FeaturePanel />

        {/* Tab bar */}
        <div className="flex gap-1 border-b border-slate-200">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium border-b-2 -mb-px transition-colors duration-150 ${
                tab === t.id
                  ? 'border-violet-600 text-violet-700'
                  : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
              }`}
            >
              <span>{t.icon}</span>
              {t.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div>
          {tab === 'pair' ? (
            <ErrorBoundary>
              <PairScorer mode={mode} />
            </ErrorBoundary>
          ) : (
            <ErrorBoundary>
              <BatchScorer mode={mode} />
            </ErrorBoundary>
          )}
        </div>
      </main>

      <footer className="border-t border-slate-200 mt-16 py-6">
        <div className="max-w-6xl mx-auto px-6 flex items-center justify-between">
          <span className="text-xs text-slate-400">
            SilverBullet · Learned text similarity via Conv2D on multi-signal 64×64 feature maps
          </span>
          <a
            href="http://localhost:8000/api/v1/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-violet-500 hover:text-violet-700 transition-colors"
          >
            API docs →
          </a>
        </div>
      </footer>
    </div>
  );
}

export default App;
