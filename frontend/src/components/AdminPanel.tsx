import { useState, useEffect, useCallback, useRef } from 'react';
import { getAdminStatus, startTraining, stopTraining, getTrainingLogs, getTrainingStatus } from '../services/api';
import type { AdminStatus, BenchmarkResult, TrainingSummary, ModeAdminInfo, TrainingJobStatus } from '../types';

const MODES = ['context-vs-generated', 'reference-vs-generated', 'model-vs-model'] as const;
type Mode = typeof MODES[number];

const MODE_SHORT: Record<string, string> = {
  'context-vs-generated':   'CVG',
  'reference-vs-generated': 'RVG',
  'model-vs-model':         'MVM',
};
const MODE_LABEL: Record<string, string> = {
  'context-vs-generated':   'Context vs Generated',
  'reference-vs-generated': 'Reference vs Generated',
  'model-vs-model':         'Model vs Model',
};
const MODE_COLOR: Record<string, string> = {
  'context-vs-generated':   'var(--cvg)',
  'reference-vs-generated': 'var(--rvg)',
  'model-vs-model':         'var(--mvm)',
};
const MODE_DIM: Record<string, string> = {
  'context-vs-generated':   'rgba(139 92 246 / 0.12)',
  'reference-vs-generated': 'rgba(16 185 129 / 0.1)',
  'model-vs-model':         'rgba(59 130 246 / 0.1)',
};

function fmt(n: number | null | undefined, d = 4): string {
  if (n == null || isNaN(n)) return '—';
  return n.toFixed(d);
}
function timeAgo(ts: number | null | undefined): string {
  if (!ts) return '—';
  const diff = Date.now() / 1000 - ts;
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}
function elapsed(start: number | null, end: number | null): string {
  if (!start) return '—';
  const sec = Math.round(((end ?? Date.now() / 1000) - start));
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60), s = sec % 60;
  if (m < 60) return `${m}m ${s}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

// ── Classify log line for colour
function logClass(line: string): string {
  if (/error|exception|fail|traceback/i.test(line)) return 'log-error';
  if (/warning|warn/i.test(line)) return 'log-warn';
  if (/training complete|early stopping|best model|done/i.test(line)) return 'log-done';
  if (/epoch \d+/i.test(line)) return 'log-epoch';
  if (/^\[/.test(line)) return '';
  return 'log-dim';
}

// ─────────────────────────────────────────────────────────────────────
// ModelCard
// ─────────────────────────────────────────────────────────────────────
function ModelCard({ mode, info, summary }: {
  mode: string;
  info: ModeAdminInfo;
  summary?: TrainingSummary;
}) {
  const color = MODE_COLOR[mode] ?? 'var(--text-2)';
  const dim   = MODE_DIM[mode]   ?? 'var(--bg-3)';

  return (
    <div style={{
      background: dim,
      border: `1px solid ${info.loaded ? color : 'var(--border)'}`,
      borderRadius: 12,
      padding: '14px 16px',
      position: 'relative',
      overflow: 'hidden',
      transition: 'border-color 0.2s',
    }}>
      {/* Subtle corner glow when loaded */}
      {info.loaded && (
        <div style={{
          position: 'absolute', top: 0, right: 0,
          width: 80, height: 80,
          background: `radial-gradient(circle at top right, ${color}20, transparent 70%)`,
          pointerEvents: 'none',
        }} />
      )}

      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 12 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2 }}>
            <span style={{
              fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 500,
              color: color, letterSpacing: '0.04em',
            }}>
              {MODE_SHORT[mode] ?? mode}
            </span>
            <span style={{
              width: 6, height: 6, borderRadius: '50%',
              background: info.loaded ? 'var(--green)' : 'var(--text-3)',
              display: 'inline-block',
              boxShadow: info.loaded ? '0 0 6px var(--green)' : 'none',
              animation: info.loaded ? 'sb-pulse-accent 2.5s ease-in-out infinite' : 'none',
            }} />
          </div>
          <div style={{ fontFamily: 'var(--font-body)', fontSize: 11, color: 'var(--text-3)' }}>
            {MODE_LABEL[mode] ?? mode}
          </div>
        </div>

        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: 9, fontWeight: 500,
          padding: '3px 7px', borderRadius: 4,
          background: info.loaded ? 'rgba(16 185 129 / 0.15)' : 'var(--bg-4)',
          color: info.loaded ? 'var(--green)' : 'var(--text-3)',
          border: `1px solid ${info.loaded ? 'rgba(16 185 129 / 0.3)' : 'var(--border)'}`,
          letterSpacing: '0.08em', textTransform: 'uppercase',
        }}>
          {info.loaded ? 'LIVE' : 'COLD'}
        </span>
      </div>

      {info.checkpoint ? (
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: 10, display: 'flex', flexDirection: 'column', gap: 5 }}>
          {([
            ['Size',       `${info.checkpoint.size_mb} MB`],
            ['Updated',    timeAgo(info.checkpoint.modified_ts)],
            summary?.best_val_loss != null ? ['Best loss', fmt(summary.best_val_loss)] : null,
            summary?.best_epoch != null ? ['Stopped', `ep ${summary.best_epoch}/${summary.total_epochs}`] : null,
          ] as ([string, string] | null)[]).filter((x): x is [string, string] => x !== null).map(([k, v]) => (
            <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11 }}>
              <span style={{ color: 'var(--text-3)' }}>{k}</span>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-2)', fontSize: 11 }}>{v}</span>
            </div>
          ))}
        </div>
      ) : (
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: 10, fontSize: 11, color: 'var(--text-3)', fontStyle: 'italic' }}>
          No checkpoint — train first
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// TrainingPanel — live log + controls for one mode
// ─────────────────────────────────────────────────────────────────────
function TrainingPanel({ mode, initialStatus }: {
  mode: Mode;
  initialStatus?: TrainingJobStatus;
}) {
  const [jobStatus, setJobStatus] = useState<TrainingJobStatus | null>(initialStatus ?? null);
  const [lines, setLines] = useState<string[]>([]);
  const [offset, setOffset] = useState(0);
  const [busy, setBusy] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);
  const color = MODE_COLOR[mode];

  const isRunning = jobStatus?.status === 'running';

  // Poll logs while running
  useEffect(() => {
    if (!isRunning) return;
    const poll = async () => {
      try {
        const res = await getTrainingLogs(mode, offset);
        if (res.lines.length > 0) {
          setLines(prev => [...prev, ...res.lines]);
          setOffset(res.offset);
        }
        setJobStatus(prev => prev ? { ...prev, status: res.status as TrainingJobStatus['status'] } : null);
      } catch { /* ignore */ }
    };
    const id = setInterval(poll, 1500);
    poll();
    return () => clearInterval(id);
  }, [mode, isRunning, offset]);

  // Poll status when done/error to grab final lines
  useEffect(() => {
    if (isRunning) return;
    const fetchFinal = async () => {
      try {
        const res = await getTrainingLogs(mode, 0);
        setLines(res.lines);
        setOffset(res.offset);
        setJobStatus(prev => prev ? { ...prev, status: res.status as TrainingJobStatus['status'] } : null);
      } catch { /* ignore */ }
    };
    fetchFinal();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-scroll
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [lines]);

  const handleStart = async () => {
    setBusy(true);
    setLines([]);
    setOffset(0);
    try {
      const r = await startTraining(mode);
      if (r.started) {
        setJobStatus({ status: 'running', returncode: null, started_at: Date.now() / 1000, ended_at: null, line_count: 0 });
      } else {
        alert(`Could not start: ${r.reason ?? 'unknown'}`);
      }
    } catch (e) {
      alert(e instanceof Error ? e.message : 'Failed to start training');
    } finally {
      setBusy(false);
    }
  };

  const handleStop = async () => {
    setBusy(true);
    try {
      await stopTraining(mode);
      setJobStatus(prev => prev ? { ...prev, status: 'idle' } : null);
    } catch { /* ignore */ } finally {
      setBusy(false);
    }
  };

  const statusColor = {
    idle:    'var(--text-3)',
    running: 'var(--amber)',
    done:    'var(--green)',
    error:   'var(--red)',
  }[jobStatus?.status ?? 'idle'];

  const statusLabel = {
    idle:    'idle',
    running: 'running',
    done:    'done',
    error:   'error',
  }[jobStatus?.status ?? 'idle'];

  return (
    <div style={{
      background: 'var(--bg-2)',
      border: `1px solid var(--border)`,
      borderRadius: 12,
      overflow: 'hidden',
    }}>
      {/* Header bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '10px 14px',
        borderBottom: '1px solid var(--border)',
        background: 'var(--bg-3)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 500, color }}>
            {MODE_SHORT[mode]}
          </span>
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: 9,
            color: statusColor,
            padding: '2px 7px', borderRadius: 4,
            background: `${statusColor}18`,
            border: `1px solid ${statusColor}30`,
            textTransform: 'uppercase', letterSpacing: '0.08em',
          }}>
            {statusLabel}
          </span>
          {jobStatus?.started_at && (
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-3)' }}>
              {elapsed(jobStatus.started_at, jobStatus.ended_at ?? null)}
            </span>
          )}
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          {isRunning ? (
            <button className="sb-btn-danger" disabled={busy} onClick={handleStop}>
              ◼ Stop
            </button>
          ) : (
            <button className="sb-btn-primary" disabled={busy} onClick={handleStart}
              style={{ padding: '6px 14px', fontSize: 11 }}
            >
              {busy ? (
                <span style={{
                  width: 10, height: 10, borderRadius: '50%',
                  border: '2px solid #040607',
                  borderTopColor: 'transparent',
                  display: 'inline-block',
                  animation: 'spin 0.6s linear infinite',
                }} />
              ) : '▶'}
              {busy ? 'Starting…' : 'Train'}
            </button>
          )}
        </div>
      </div>

      {/* Log terminal */}
      <div ref={logRef} className="sb-log" style={{ maxHeight: 280, borderRadius: 0, border: 'none' }}>
        {lines.length === 0 ? (
          <span className="log-dim">
            {isRunning ? 'waiting for output…' : 'no log — click Train to start'}
          </span>
        ) : (
          lines.map((line, i) => (
            <div key={i} className={logClass(line)}>{line}</div>
          ))
        )}
        {isRunning && (
          <span style={{ display: 'inline-block' }}>
            <span className="sb-cursor" style={{ display: 'inline-block', width: 7, height: 13, background: 'var(--accent)', verticalAlign: 'text-bottom', marginLeft: 2 }} />
          </span>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// BenchmarkTable
// ─────────────────────────────────────────────────────────────────────
function BenchmarkTable({ results }: { results: BenchmarkResult[] }) {
  if (results.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: '32px 16px', color: 'var(--text-3)', fontSize: 12 }}>
        No benchmark data —
        {' '}<code style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-2)', background: 'var(--bg-4)', padding: '2px 6px', borderRadius: 4 }}>
          python -m backend.benchmark_eval
        </code>
      </div>
    );
  }

  const cols = ['Benchmark', 'n', 'ROC-AUC', 'PR-AUC', 'Acc', 'Pearson r', 'Spearman ρ'];

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border)' }}>
            {cols.map(h => (
              <th key={h} style={{
                textAlign: 'left', padding: '8px 12px',
                fontFamily: 'var(--font-mono)', fontSize: 9, fontWeight: 500,
                color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.1em',
              }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {results.map((r) => {
            const auc = r.roc_auc ?? 0;
            const aucColor = auc >= 0.8 ? 'var(--green)' : auc >= 0.65 ? 'var(--amber)' : 'var(--red)';
            return (
              <tr key={r.benchmark} style={{ borderBottom: '1px solid var(--border)' }}
                onMouseOver={e => (e.currentTarget.style.background = 'var(--bg-3)')}
                onMouseOut={e => (e.currentTarget.style.background = '')}
              >
                <td style={{ padding: '9px 12px', fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 500, color: 'var(--text-1)' }}>{r.benchmark}</td>
                <td style={{ padding: '9px 12px', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-3)' }}>{r.n}</td>
                <td style={{ padding: '9px 12px', fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 500, color: aucColor }}>{fmt(r.roc_auc)}</td>
                <td style={{ padding: '9px 12px', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-2)' }}>{fmt(r.pr_auc)}</td>
                <td style={{ padding: '9px 12px', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-2)' }}>
                  {r.accuracy != null ? `${(r.accuracy * 100).toFixed(1)}%` : '—'}
                </td>
                <td style={{ padding: '9px 12px', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-2)' }}>
                  {fmt(r.pearson_r_human ?? r.pearson_r_binary)}
                </td>
                <td style={{ padding: '9px 12px', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-2)' }}>
                  {fmt(r.spearman_r_human ?? r.spearman_r_binary)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// CacheStats
// ─────────────────────────────────────────────────────────────────────
function CacheStats({ stats }: { stats: AdminStatus['cache_stats'] }) {
  const entries = Object.entries(stats.table_counts ?? {});
  const total   = entries.reduce((s, [, v]) => s + v, 0);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', gap: 8 }}>
      {entries.map(([table, count]) => (
        <div key={table} style={{
          background: 'var(--bg-3)', border: '1px solid var(--border)',
          borderRadius: 8, padding: '10px 12px',
        }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 4 }}>{table}</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 20, fontWeight: 500, color: 'var(--text-1)', lineHeight: 1 }}>{count.toLocaleString()}</div>
        </div>
      ))}
      <div style={{ background: 'var(--bg-3)', border: '1px solid var(--border)', borderRadius: 8, padding: '10px 12px' }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 4 }}>db size</div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 20, fontWeight: 500, color: 'var(--text-1)', lineHeight: 1 }}>{stats.db_size_mb} MB</div>
      </div>
      <div style={{ background: 'var(--accent-dim)', border: '1px solid rgba(0 229 204 / 0.2)', borderRadius: 8, padding: '10px 12px' }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--accent)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 4 }}>total</div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 20, fontWeight: 500, color: 'var(--accent)', lineHeight: 1 }}>{total.toLocaleString()}</div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// SectionLabel
// ─────────────────────────────────────────────────────────────────────
function SectionLabel({ icon, title }: { icon: string; title: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-3)' }}>{icon}</span>
      <span style={{
        fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 500,
        color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.12em',
      }}>{title}</span>
      <div style={{ flex: 1, height: 1, background: 'var(--border)', marginLeft: 4 }} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// AdminPanel — main export
// ─────────────────────────────────────────────────────────────────────
export function AdminPanel() {
  const [status, setStatus] = useState<AdminStatus | null>(null);
  const [jobStatuses, setJobStatuses] = useState<Record<string, TrainingJobStatus>>({});
  const [error, setError]   = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
  const [activeTrainMode, setActiveTrainMode] = useState<Mode | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [s, jobs] = await Promise.all([
        getAdminStatus(),
        getTrainingStatus().catch(() => ({})),
      ]);
      setStatus(s);
      setJobStatuses(jobs);
      setLastRefresh(new Date());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch status');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    const id = setInterval(refresh, 30_000);
    return () => clearInterval(id);
  }, [refresh]);

  return (
    <div className="sb-stagger" style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>

      {/* ── Top bar ─────────────────────────────────────────── */}
      <div className="sb-fade-up" style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '10px 16px',
        background: 'var(--bg-2)',
        border: '1px solid var(--border)',
        borderRadius: 10,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{
            width: 7, height: 7, borderRadius: '50%',
            background: 'var(--accent)',
            display: 'inline-block',
            animation: 'sb-pulse-accent 2s ease-in-out infinite',
          }} />
          <span style={{
            fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 500,
            color: 'var(--text-2)', letterSpacing: '0.1em', textTransform: 'uppercase',
          }}>Admin Console</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-3)' }}>
            {lastRefresh.toLocaleTimeString()}
          </span>
          <button className="sb-btn-ghost" onClick={refresh} disabled={loading}>
            {loading ? '…' : '↺'} refresh
          </button>
        </div>
      </div>

      {error && (
        <div style={{
          background: 'rgba(239 68 68 / 0.1)', border: '1px solid rgba(239 68 68 / 0.3)',
          borderRadius: 8, padding: '10px 14px',
          fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--red)',
        }}>
          ⚠ {error}
        </div>
      )}

      {/* ── Model Status ────────────────────────────────────── */}
      <div className="sb-fade-up">
        <SectionLabel icon="⬡" title="Model Status" />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
          {status
            ? MODES.map(m => (
                <ModelCard key={m} mode={m} info={status.models[m] ?? { loaded: false, checkpoint: null }}
                  summary={status.training_summaries?.[m]} />
              ))
            : [1,2,3].map(i => (
                <div key={i} style={{ height: 140, borderRadius: 12, background: 'var(--bg-3)', border: '1px solid var(--border)', animation: 'sb-pulse-accent 1.5s ease-in-out infinite' }} />
              ))
          }
        </div>
      </div>

      {/* ── Training ────────────────────────────────────────── */}
      <div className="sb-fade-up">
        <SectionLabel icon="▶" title="Training" />

        {/* Mode picker */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
          {MODES.map(m => {
            const isActive = activeTrainMode === m;
            const color = MODE_COLOR[m];
            const isRunning = jobStatuses[m]?.status === 'running';
            return (
              <button
                key={m}
                onClick={() => setActiveTrainMode(isActive ? null : m)}
                style={{
                  padding: '7px 14px',
                  borderRadius: 8,
                  border: `1px solid ${isActive ? color : 'var(--border)'}`,
                  background: isActive ? `${color}18` : 'var(--bg-3)',
                  color: isActive ? color : 'var(--text-2)',
                  fontFamily: 'var(--font-mono)',
                  fontSize: 11, fontWeight: 500,
                  cursor: 'pointer',
                  transition: 'all 0.15s',
                  display: 'flex', alignItems: 'center', gap: 7,
                  letterSpacing: '0.04em',
                }}
              >
                {isRunning && (
                  <span style={{
                    width: 6, height: 6, borderRadius: '50%',
                    background: 'var(--amber)',
                    animation: 'sb-pulse-accent 1s ease-in-out infinite',
                    display: 'inline-block',
                  }} />
                )}
                {MODE_SHORT[m]}
                {isActive ? ' ▲' : ' ▼'}
              </button>
            );
          })}
        </div>

        {/* Expanded training panel */}
        {activeTrainMode && (
          <div className="sb-fade-up">
            <TrainingPanel
              key={activeTrainMode}
              mode={activeTrainMode}
              initialStatus={jobStatuses[activeTrainMode]}
            />
          </div>
        )}

        {!activeTrainMode && (
          <div style={{
            padding: '16px 14px',
            background: 'var(--bg-2)', border: '1px solid var(--border)',
            borderRadius: 10, fontFamily: 'var(--font-mono)', fontSize: 11,
            color: 'var(--text-3)', textAlign: 'center',
          }}>
            Select a mode above to open its training console
          </div>
        )}
      </div>

      {/* ── Benchmark Results ──────────────────────────────── */}
      <div className="sb-fade-up">
        <SectionLabel icon="◈" title="Benchmark Evaluation" />
        <div style={{ background: 'var(--bg-2)', border: '1px solid var(--border)', borderRadius: 10, overflow: 'hidden' }}>
          {status
            ? <BenchmarkTable results={status.benchmark_results} />
            : <div style={{ height: 80, animation: 'sb-pulse-accent 1.5s ease-in-out infinite', background: 'var(--bg-3)', borderRadius: 10 }} />
          }
        </div>
      </div>

      {/* ── Cache Stats ────────────────────────────────────── */}
      <div className="sb-fade-up">
        <SectionLabel icon="▦" title="SQLite Cache" />
        {status
          ? <CacheStats stats={status.cache_stats} />
          : <div style={{ height: 60, animation: 'sb-pulse-accent 1.5s ease-in-out infinite', background: 'var(--bg-3)', borderRadius: 10 }} />
        }
      </div>

      {/* ── CLI Reference ──────────────────────────────────── */}
      <div className="sb-fade-up">
        <SectionLabel icon="$" title="CLI Reference" />
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {[
            ['Run benchmark eval',           'python -m backend.benchmark_eval'],
            ['Train CVG',                    'python -m backend.train --mode context-vs-generated'],
            ['Train RVG',                    'python -m backend.train --mode reference-vs-generated'],
            ['Train MVM',                    'python -m backend.train --mode model-vs-model'],
            ['K-fold CV (CVG, k=5)',          'python -m backend.kfold_train --mode context-vs-generated --k 5'],
            ['Fetch external data',          'python -m backend.fetch_external_data --force'],
            ['Precompute features (CVG)',     'python -m backend.precompute_features --mode context-vs-generated'],
          ].map(([label, cmd]) => (
            <div key={cmd} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{ fontFamily: 'var(--font-body)', fontSize: 11, color: 'var(--text-3)', width: 160, flexShrink: 0 }}>{label}</span>
              <code
                onClick={() => navigator.clipboard?.writeText(cmd)}
                title="Click to copy"
                style={{
                  fontFamily: 'var(--font-mono)', fontSize: 11,
                  color: 'var(--accent)', background: 'var(--bg-3)',
                  border: '1px solid var(--border)',
                  borderRadius: 5, padding: '4px 10px',
                  flex: 1, cursor: 'pointer',
                  transition: 'border-color 0.15s',
                  userSelect: 'all',
                }}
                onMouseOver={e => ((e.target as HTMLElement).style.borderColor = 'rgba(0 229 204 / 0.3)')}
                onMouseOut={e => ((e.target as HTMLElement).style.borderColor = 'var(--border)')}
              >
                {cmd}
              </code>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
