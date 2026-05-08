import { useState, useEffect } from 'react';
import { PairScorer } from './components/PairScorer';
import type { PairInitData } from './components/PairScorer';
import { BatchScorer } from './components/BatchScorer';
import { ErrorBoundary } from './components/ErrorBoundary';
import { ComparisonModeSelector } from './components/ComparisonModeSelector';
import { FeaturePanel } from './components/FeaturePanel';
import { ExperimentsPanel } from './components/ExperimentsPanel';
import { AdminPanel } from './components/AdminPanel';
import { JuryScorer } from './components/JuryScorer';
import { useExperiments } from './hooks/useExperiments';
import { useLocalStorage } from './hooks/useLocalStorage';
import { healthCheck } from './services/api';
import type { ComparisonMode } from './types';
import './index.css';

type Tab = 'pair' | 'batch' | 'jury' | 'experiments' | 'admin';

const TABS: { id: Tab; label: string; icon: string; mono?: boolean }[] = [
  { id: 'pair',        label: 'Single Eval',   icon: '⚡' },
  { id: 'batch',       label: 'Batch Eval',    icon: '▦' },
  { id: 'jury',        label: 'LLM Jury',      icon: '⚖' },
  { id: 'experiments', label: 'Experiments',   icon: '◈' },
  { id: 'admin',       label: 'Admin',         icon: '⬡', mono: true },
];

function App() {
  const [tab, setTab] = useLocalStorage<Tab>('sb_tab', 'pair');
  const [mode, setMode] = useLocalStorage<ComparisonMode>('sb_mode', 'reference-vs-generated');
  const [theme, setTheme] = useLocalStorage<'dark' | 'light'>('sb_theme', 'dark');
  const [pairInit, setPairInit] = useState<PairInitData | undefined>(undefined);
  const [pairKey, setPairKey] = useState(0);
  const [apiAlive, setApiAlive] = useState<boolean | null>(null);

  // Sync theme to <html data-theme>
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const {
    experiments,
    savePairExperiment,
    saveBatchExperiment,
    deleteExperiment,
    updateNotes,
    clearAll,
    exportAll,
  } = useExperiments();

  // Health check
  useEffect(() => {
    healthCheck()
      .then(() => setApiAlive(true))
      .catch(() => setApiAlive(false));
    const id = setInterval(() => {
      healthCheck().then(() => setApiAlive(true)).catch(() => setApiAlive(false));
    }, 30_000);
    return () => clearInterval(id);
  }, []);

  const handleRerun = (data: {
    mode: ComparisonMode;
    text1: string;
    text2: string;
    name1: string;
    name2: string;
    baseline: '1' | '2' | null;
  }) => {
    localStorage.setItem('sb_pair_text1', JSON.stringify(data.text1));
    localStorage.setItem('sb_pair_text2', JSON.stringify(data.text2));
    localStorage.setItem('sb_pair_meta', JSON.stringify({ name1: data.name1, name2: data.name2, baseline: data.baseline }));
    setMode(data.mode);
    setPairInit({
      text1: data.text1,
      text2: data.text2,
      meta: { name1: data.name1, name2: data.name2, baseline: data.baseline },
    });
    setPairKey((k) => k + 1);
    setTab('pair');
  };

  const expCount = experiments.length;

  return (
    <div style={{ position: 'relative', minHeight: '100vh', zIndex: 1 }}>

      {/* ── Header ─────────────────────────────────────────────── */}
      <header style={{
        position: 'sticky',
        top: 0,
        zIndex: 50,
        background: theme === 'dark' ? 'rgba(8,9,14,0.92)' : 'rgba(242,243,248,0.92)',
        backdropFilter: 'blur(16px)',
        borderBottom: '1px solid var(--border)',
      }}>
        <div style={{
          maxWidth: 1140,
          margin: '0 auto',
          padding: '0 24px',
          height: 56,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          {/* Wordmark */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{
              width: 32, height: 32,
              borderRadius: 8,
              background: 'var(--accent-dim)',
              border: '1px solid rgba(0 229 204 / 0.3)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              fontWeight: 500,
              color: 'var(--accent)',
              letterSpacing: '0.06em',
            }}>SB</div>
            <div>
              <div style={{
                fontFamily: 'var(--font-display)',
                fontSize: 16,
                fontWeight: 700,
                color: 'var(--text-1)',
                letterSpacing: '-0.01em',
                lineHeight: 1,
              }}>
                Silver<span style={{ color: 'var(--accent)' }}>Bullet</span>
              </div>
              <div style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 10,
                color: 'var(--text-3)',
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                marginTop: 2,
              }}>LLM Evaluation</div>
            </div>
          </div>

          {/* Right side status */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            {expCount > 0 && (
              <button
                onClick={() => setTab('experiments')}
                style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  padding: '4px 10px',
                  background: 'var(--accent-dim)',
                  border: '1px solid rgba(0 196 173 / 0.2)',
                  borderRadius: 6,
                  fontFamily: 'var(--font-mono)',
                  fontSize: 10,
                  color: 'var(--accent)',
                  cursor: 'pointer',
                  letterSpacing: '0.06em',
                  textTransform: 'uppercase',
                }}
              >
                ◈ {expCount} exp{expCount !== 1 ? 's' : ''}
              </button>
            )}

            {/* Theme toggle */}
            <button
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                width: 30, height: 30,
                background: 'var(--bg-3)',
                border: '1px solid var(--border-2)',
                borderRadius: 6,
                cursor: 'pointer',
                fontSize: 13,
                transition: 'background 0.15s, border-color 0.15s',
                flexShrink: 0,
              }}
              onMouseOver={e => (e.currentTarget.style.borderColor = 'var(--border-hover)')}
              onMouseOut={e => (e.currentTarget.style.borderColor = 'var(--border-2)')}
            >
              {theme === 'dark' ? '☀' : '◐'}
            </button>
            <div style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '4px 10px',
              background: 'var(--bg-2)',
              border: '1px solid var(--border)',
              borderRadius: 6,
              fontFamily: 'var(--font-mono)',
              fontSize: 10,
              color: apiAlive === null ? 'var(--text-3)' : apiAlive ? 'var(--green)' : 'var(--red)',
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
            }}>
              <span style={{
                width: 6, height: 6,
                borderRadius: '50%',
                background: apiAlive === null ? 'var(--text-3)' : apiAlive ? 'var(--green)' : 'var(--red)',
                display: 'inline-block',
                boxShadow: apiAlive ? '0 0 4px var(--green)' : 'none',
                animation: apiAlive ? 'sb-pulse-accent 2s ease-in-out infinite' : 'none',
              }} />
              {apiAlive === null ? 'checking' : apiAlive ? 'api live' : 'offline'}
            </div>
          </div>
        </div>
      </header>

      {/* ── Main ───────────────────────────────────────────────── */}
      <main style={{ maxWidth: 1140, margin: '0 auto', padding: '32px 24px', position: 'relative', zIndex: 1 }}>

        {/* Mode selector + features — hide on admin/experiments/jury */}
        {tab !== 'experiments' && tab !== 'admin' && (
          <div className="sb-fade-up sb-content-area" style={{ marginBottom: 28 }}>
            <ComparisonModeSelector selected={mode} onChange={setMode} />
          </div>
        )}
        {tab !== 'experiments' && tab !== 'admin' && tab !== 'jury' && (
          <div className="sb-fade-up sb-content-area" style={{ marginBottom: 28, animationDelay: '40ms' }}>
            <FeaturePanel mode={mode} />
          </div>
        )}

        {/* ── Tab bar ──────────────────────────────────────────── */}
        <div style={{
          display: 'flex',
          borderBottom: '1px solid var(--border)',
          marginBottom: 28,
          gap: 0,
        }}>
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`sb-tab ${tab === t.id ? 'active' : ''}`}
              style={{ fontFamily: t.mono ? 'var(--font-mono)' : undefined }}
            >
              <span style={{ marginRight: 6, opacity: 0.7 }}>{t.icon}</span>
              {t.label}
              {t.id === 'experiments' && expCount > 0 && (
                <span style={{
                  marginLeft: 6,
                  background: tab === 'experiments' ? 'var(--accent-dim)' : 'var(--bg-4)',
                  color: tab === 'experiments' ? 'var(--accent)' : 'var(--text-3)',
                  fontFamily: 'var(--font-mono)',
                  fontSize: 10,
                  fontWeight: 500,
                  padding: '1px 6px',
                  borderRadius: 4,
                }}>
                  {expCount}
                </span>
              )}
            </button>
          ))}
        </div>

        {/* ── Content ──────────────────────────────────────────── */}
        <div className="sb-fade-up sb-content-area">
          {tab === 'pair' && (
            <ErrorBoundary>
              <PairScorer key={pairKey} mode={mode} initData={pairInit} onSave={savePairExperiment} />
            </ErrorBoundary>
          )}
          {tab === 'batch' && (
            <ErrorBoundary>
              <BatchScorer mode={mode} onSave={saveBatchExperiment} />
            </ErrorBoundary>
          )}
          {tab === 'jury' && (
            <ErrorBoundary>
              <JuryScorer mode={mode} />
            </ErrorBoundary>
          )}
          {tab === 'experiments' && (
            <ExperimentsPanel
              experiments={experiments}
              onDelete={deleteExperiment}
              onUpdateNotes={updateNotes}
              onClearAll={clearAll}
              onExportAll={exportAll}
              onRerun={handleRerun}
            />
          )}
          {tab === 'admin' && (
            <ErrorBoundary>
              <AdminPanel />
            </ErrorBoundary>
          )}
        </div>
      </main>

      {/* ── Footer ─────────────────────────────────────────────── */}
      <footer style={{
        borderTop: '1px solid var(--border)',
        marginTop: 64,
        padding: '20px 24px',
      }}>
        <div style={{
          maxWidth: 1140,
          margin: '0 auto',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--text-3)',
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
          }}>
            SilverBullet · Conv2D · 21–23 signal maps · 9 families · v5.8
          </span>
          <a
            href="http://localhost:8000/api/v1/docs"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 10,
              color: 'var(--accent)',
              textDecoration: 'none',
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
              opacity: 0.7,
              transition: 'opacity 0.15s',
            }}
            onMouseOver={e => (e.currentTarget.style.opacity = '1')}
            onMouseOut={e => (e.currentTarget.style.opacity = '0.7')}
          >
            API docs →
          </a>
        </div>
      </footer>
    </div>
  );
}

export default App;
