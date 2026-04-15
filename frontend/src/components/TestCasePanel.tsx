import { useState } from 'react';
import { getTestCases } from '../data/testCases';
import type { PairTestCase, BatchTestCase } from '../data/testCases';
import type { ComparisonMode } from '../types';

interface PairLoadProps {
  scope: 'pair';
  mode: ComparisonMode;
  onLoad: (tc: PairTestCase) => void;
}
interface BatchLoadProps {
  scope: 'batch';
  mode: ComparisonMode;
  onLoad: (tc: BatchTestCase) => void;
}
type Props = PairLoadProps | BatchLoadProps;

// Outcome badge — inline-style color maps
const OUTCOME_LABEL: Record<string, string> = {
  high:   'Expected: High ≥ 0.7',
  medium: 'Expected: Medium 0.4–0.7',
  low:    'Expected: Low < 0.4',
  mixed:  'Mixed distribution',
};
const OUTCOME_COLOR: Record<string, string> = {
  high:   '#34D399',
  medium: '#F59E0B',
  low:    '#F87171',
  mixed:  'var(--text-3)',
};
const OUTCOME_BG: Record<string, string> = {
  high:   'rgba(52,211,153,0.1)',
  medium: 'rgba(245,158,11,0.1)',
  low:    'rgba(248,113,113,0.1)',
  mixed:  'var(--bg-4)',
};
const OUTCOME_BORDER: Record<string, string> = {
  high:   'rgba(52,211,153,0.25)',
  medium: 'rgba(245,158,11,0.25)',
  low:    'rgba(248,113,113,0.25)',
  mixed:  'var(--border-2)',
};

// Tag pill colors
const TAG_COLOR: Record<string, { color: string; bg: string }> = {
  hallucination: { color: '#F87171', bg: 'rgba(248,113,113,0.10)' },
  grounded:      { color: '#34D399', bg: 'rgba(52,211,153,0.10)' },
  faithful:      { color: '#34D399', bg: 'rgba(52,211,153,0.10)' },
  agreement:     { color: '#60A5FA', bg: 'rgba(96,165,250,0.10)' },
  divergence:    { color: '#F87171', bg: 'rgba(248,113,113,0.10)' },
  code:          { color: 'var(--cvg)', bg: 'rgba(139,92,246,0.10)' },
  batch:         { color: 'var(--text-3)', bg: 'var(--bg-4)' },
};
const defaultTag = { color: 'var(--text-3)', bg: 'var(--bg-4)' };

export function TestCasePanel(props: Props) {
  const [open, setOpen] = useState(false);
  const cases = getTestCases(props.mode, props.scope);

  if (cases.length === 0) return null;

  return (
    <div style={{
      background: 'var(--bg-2)',
      border: '1px solid var(--border)',
      borderRadius: 12,
      overflow: 'hidden',
    }}>
      {/* ── Collapse header ────────────────────────────────────── */}
      <button
        onClick={() => setOpen(v => !v)}
        style={{
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '11px 16px',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          transition: 'background 0.12s',
        }}
        onMouseOver={e => (e.currentTarget.style.background = 'var(--bg-3)')}
        onMouseOut={e => (e.currentTarget.style.background = 'none')}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--text-2)',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
          }}>
            Example Test Cases
          </span>
          <span style={{
            padding: '1px 7px',
            borderRadius: 99,
            background: 'var(--accent-dim)',
            border: '1px solid rgba(0 196 173 / 0.2)',
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--accent)',
            letterSpacing: '0.04em',
          }}>
            {cases.length}
          </span>
        </div>
        <span style={{
          color: 'var(--text-3)',
          fontSize: 10,
          display: 'inline-block',
          transform: open ? 'rotate(180deg)' : 'rotate(0deg)',
          transition: 'transform 0.2s',
          userSelect: 'none',
        }}>
          ▼
        </span>
      </button>

      {/* ── Cards ──────────────────────────────────────────────── */}
      {open && (
        <div style={{
          borderTop: '1px solid var(--border)',
          padding: 14,
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
          gap: 10,
        }}>
          {cases.map((tc) => {
            const outColor  = OUTCOME_COLOR[tc.expectedOutcome]  ?? 'var(--text-3)';
            const outBg     = OUTCOME_BG[tc.expectedOutcome]     ?? 'var(--bg-4)';
            const outBorder = OUTCOME_BORDER[tc.expectedOutcome] ?? 'var(--border-2)';

            return (
              <div
                key={tc.id}
                style={{
                  background: 'var(--bg-3)',
                  border: '1px solid var(--border)',
                  borderRadius: 10,
                  padding: '12px 13px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 9,
                }}
              >
                {/* Title + description */}
                <div>
                  <div style={{
                    fontFamily: 'var(--font-body)',
                    fontSize: 12,
                    fontWeight: 600,
                    color: 'var(--text-1)',
                    lineHeight: 1.35,
                    marginBottom: 3,
                  }}>
                    {tc.title}
                  </div>
                  <div style={{
                    fontFamily: 'var(--font-body)',
                    fontSize: 11,
                    color: 'var(--text-3)',
                    lineHeight: 1.5,
                  }}>
                    {tc.description}
                  </div>
                </div>

                {/* Outcome badge */}
                <span style={{
                  alignSelf: 'flex-start',
                  padding: '2px 8px',
                  borderRadius: 99,
                  background: outBg,
                  border: `1px solid ${outBorder}`,
                  fontFamily: 'var(--font-mono)',
                  fontSize: 10,
                  color: outColor,
                  letterSpacing: '0.04em',
                }}>
                  {OUTCOME_LABEL[tc.expectedOutcome] ?? tc.expectedOutcome}
                </span>

                {/* Tags */}
                {tc.tags.length > 0 && (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {tc.tags.map(tag => {
                      const t = TAG_COLOR[tag] ?? defaultTag;
                      return (
                        <span key={tag} style={{
                          padding: '1px 6px',
                          borderRadius: 4,
                          background: t.bg,
                          fontFamily: 'var(--font-mono)',
                          fontSize: 10,
                          color: t.color,
                          letterSpacing: '0.03em',
                        }}>
                          {tag}
                        </span>
                      );
                    })}
                  </div>
                )}

                {/* Model name chips */}
                {(tc.name1 ?? tc.name2) && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                    <span style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: 10,
                      color: 'var(--text-2)',
                      background: 'var(--bg-4)',
                      border: '1px solid var(--border-2)',
                      padding: '1px 6px',
                      borderRadius: 4,
                    }}>
                      {tc.name1 ?? '—'}
                    </span>
                    <span style={{ color: 'var(--text-3)', fontSize: 10 }}>vs</span>
                    <span style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: 10,
                      color: 'var(--text-2)',
                      background: 'var(--bg-4)',
                      border: '1px solid var(--border-2)',
                      padding: '1px 6px',
                      borderRadius: 4,
                    }}>
                      {tc.name2 ?? '—'}
                    </span>
                  </div>
                )}

                {/* Load button */}
                <button
                  onClick={() => {
                    if (props.scope === 'pair') props.onLoad(tc as PairTestCase);
                    else props.onLoad(tc as BatchTestCase);
                    setOpen(false);
                  }}
                  className="sb-btn-ghost"
                  style={{
                    marginTop: 'auto',
                    width: '100%',
                    justifyContent: 'center',
                    fontSize: 10,
                    padding: '6px 10px',
                  }}
                >
                  Load Example →
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
