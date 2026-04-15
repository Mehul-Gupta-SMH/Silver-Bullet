import type { BreakdownResult, MisalignmentReason } from '../types';

interface Props {
  breakdown: BreakdownResult;
}

const THRESH_HIGH = 0.7;
const THRESH_MID  = 0.4;

type AlignTier = 'high' | 'mid' | 'low';

function alignTier(score: number): AlignTier {
  if (score >= THRESH_HIGH) return 'high';
  if (score >= THRESH_MID)  return 'mid';
  return 'low';
}

const ALIGN_COLOR: Record<AlignTier, string> = {
  high: '#34D399',
  mid:  '#F59E0B',
  low:  '#F87171',
};

const ALIGN_BG: Record<AlignTier, string> = {
  high: 'rgba(52,211,153,0.07)',
  mid:  'rgba(245,158,11,0.07)',
  low:  'rgba(248,113,113,0.07)',
};

const ALIGN_BORDER: Record<AlignTier, string> = {
  high: 'rgba(52,211,153,0.22)',
  mid:  'rgba(245,158,11,0.22)',
  low:  'rgba(248,113,113,0.22)',
};

const ALIGN_LABEL: Record<AlignTier, string> = {
  high: 'Aligned',
  mid:  'Partial',
  low:  'Divergent',
};

const SEVERITY_COLOR: Record<MisalignmentReason['severity'], string> = {
  high:   '#F87171',
  medium: '#F59E0B',
  low:    '#60A5FA',
};
const SEVERITY_BG: Record<MisalignmentReason['severity'], string> = {
  high:   'rgba(248,113,113,0.07)',
  medium: 'rgba(245,158,11,0.07)',
  low:    'rgba(96,165,250,0.07)',
};
const SEVERITY_BORDER: Record<MisalignmentReason['severity'], string> = {
  high:   'rgba(248,113,113,0.22)',
  medium: 'rgba(245,158,11,0.22)',
  low:    'rgba(96,165,250,0.22)',
};
const SEVERITY_ICON: Record<MisalignmentReason['severity'], string> = {
  high: '⚠',
  medium: '◆',
  low: '▸',
};

export function BreakdownPanel({ breakdown }: Props) {
  const {
    sentences1, sentences2,
    alignment,
    divergent_in_1, divergent_in_2,
    feature_scores,
    misalignment_reasons = [],
  } = breakdown;

  const n = sentences1.length;
  const m = sentences2.length;

  const maxRow = alignment.length > 0
    ? alignment.map(row => Math.max(...row))
    : [];
  const maxCol = m > 0 && alignment.length > 0
    ? Array.from({ length: m }, (_, j) => Math.max(...alignment.map(r => r[j])))
    : [];

  const hasDivergence = divergent_in_1.length > 0 || divergent_in_2.length > 0;
  const orphanCount = divergent_in_1.length + divergent_in_2.length;

  return (
    <div style={{
      background: 'var(--bg-2)',
      border: '1px solid var(--border)',
      borderRadius: 14,
      padding: '20px 22px',
      display: 'flex',
      flexDirection: 'column',
      gap: 22,
    }}>

      {/* ── Header ─────────────────────────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 10,
          color: 'var(--text-3)',
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
        }}>
          Divergence Analysis
        </span>

        {hasDivergence ? (
          <span style={{
            padding: '2px 8px',
            borderRadius: 99,
            background: 'rgba(248,113,113,0.1)',
            border: '1px solid rgba(248,113,113,0.25)',
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: '#F87171',
            letterSpacing: '0.06em',
          }}>
            {orphanCount} orphan{orphanCount !== 1 ? 's' : ''}
          </span>
        ) : (
          <span style={{
            padding: '2px 8px',
            borderRadius: 99,
            background: 'rgba(52,211,153,0.1)',
            border: '1px solid rgba(52,211,153,0.25)',
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: '#34D399',
            letterSpacing: '0.06em',
          }}>
            full coverage
          </span>
        )}
      </div>

      {/* ── Legend ─────────────────────────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        {(['high', 'mid', 'low'] as AlignTier[]).map((tier) => (
          <span key={tier} style={{
            display: 'flex', alignItems: 'center', gap: 5,
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--text-3)',
            letterSpacing: '0.05em',
          }}>
            <span style={{
              width: 7, height: 7, borderRadius: '50%',
              background: ALIGN_COLOR[tier],
              display: 'inline-block',
              boxShadow: `0 0 4px ${ALIGN_COLOR[tier]}`,
            }} />
            {ALIGN_LABEL[tier]}
          </span>
        ))}
      </div>

      {/* ── Sentence columns ────────────────────────────────────── */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2, 1fr)',
        gap: 16,
      }}>
        {/* Text 1 */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--text-3)',
            textTransform: 'uppercase',
            letterSpacing: '0.1em',
            marginBottom: 2,
          }}>
            Text 1 — {n} sentence{n !== 1 ? 's' : ''}
          </div>
          {sentences1.length === 0 && (
            <p style={{ fontFamily: 'var(--font-body)', fontSize: 12, color: 'var(--text-3)', fontStyle: 'italic' }}>
              No sentences detected.
            </p>
          )}
          {sentences1.map((s, i) => {
            const tier = alignTier(maxRow[i] ?? 0);
            const score = maxRow[i] ?? 0;
            return (
              <div key={i} style={{
                background: ALIGN_BG[tier],
                border: `1px solid ${ALIGN_BORDER[tier]}`,
                borderRadius: 10,
                padding: '10px 12px',
                display: 'flex', flexDirection: 'column', gap: 7,
              }}>
                <p style={{
                  fontFamily: 'var(--font-body)',
                  fontSize: 12,
                  lineHeight: 1.6,
                  color: 'var(--text-1)',
                  margin: 0,
                }}>
                  {s}
                </p>
                <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                  <span style={{
                    width: 5, height: 5, borderRadius: '50%',
                    background: ALIGN_COLOR[tier],
                    display: 'inline-block',
                    flexShrink: 0,
                  }} />
                  <span style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: 10,
                    color: ALIGN_COLOR[tier],
                    letterSpacing: '0.04em',
                  }}>
                    {ALIGN_LABEL[tier]} · {score.toFixed(2)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Text 2 */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--text-3)',
            textTransform: 'uppercase',
            letterSpacing: '0.1em',
            marginBottom: 2,
          }}>
            Text 2 — {m} sentence{m !== 1 ? 's' : ''}
          </div>
          {sentences2.length === 0 && (
            <p style={{ fontFamily: 'var(--font-body)', fontSize: 12, color: 'var(--text-3)', fontStyle: 'italic' }}>
              No sentences detected.
            </p>
          )}
          {sentences2.map((s, j) => {
            const tier = alignTier(maxCol[j] ?? 0);
            const score = maxCol[j] ?? 0;
            return (
              <div key={j} style={{
                background: ALIGN_BG[tier],
                border: `1px solid ${ALIGN_BORDER[tier]}`,
                borderRadius: 10,
                padding: '10px 12px',
                display: 'flex', flexDirection: 'column', gap: 7,
              }}>
                <p style={{
                  fontFamily: 'var(--font-body)',
                  fontSize: 12,
                  lineHeight: 1.6,
                  color: 'var(--text-1)',
                  margin: 0,
                }}>
                  {s}
                </p>
                <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                  <span style={{
                    width: 5, height: 5, borderRadius: '50%',
                    background: ALIGN_COLOR[tier],
                    display: 'inline-block',
                    flexShrink: 0,
                  }} />
                  <span style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: 10,
                    color: ALIGN_COLOR[tier],
                    letterSpacing: '0.04em',
                  }}>
                    {ALIGN_LABEL[tier]} · {score.toFixed(2)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Divergence summary ──────────────────────────────────── */}
      {hasDivergence && (
        <div style={{
          background: 'rgba(248,113,113,0.06)',
          border: '1px solid rgba(248,113,113,0.2)',
          borderRadius: 10,
          padding: '12px 14px',
          display: 'flex', flexDirection: 'column', gap: 10,
        }}>
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: '#F87171',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
          }}>
            Points of Divergence
          </div>

          {divergent_in_1.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <div style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 10,
                color: 'rgba(248,113,113,0.7)',
                letterSpacing: '0.04em',
              }}>
                {divergent_in_1.length} sentence{divergent_in_1.length !== 1 ? 's' : ''} in Text 1 unmatched:
              </div>
              {divergent_in_1.map(i => (
                <div key={i} style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 11,
                  color: '#F87171',
                  background: 'rgba(248,113,113,0.07)',
                  border: '1px solid rgba(248,113,113,0.15)',
                  borderRadius: 6,
                  padding: '4px 8px',
                }}>
                  <span style={{ opacity: 0.5 }}>[{i + 1}]</span> {sentences1[i]}
                </div>
              ))}
            </div>
          )}

          {divergent_in_2.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <div style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 10,
                color: 'rgba(248,113,113,0.7)',
                letterSpacing: '0.04em',
              }}>
                {divergent_in_2.length} sentence{divergent_in_2.length !== 1 ? 's' : ''} in Text 2 unmatched:
              </div>
              {divergent_in_2.map(j => (
                <div key={j} style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 11,
                  color: '#F87171',
                  background: 'rgba(248,113,113,0.07)',
                  border: '1px solid rgba(248,113,113,0.15)',
                  borderRadius: 6,
                  padding: '4px 8px',
                }}>
                  <span style={{ opacity: 0.5 }}>[{j + 1}]</span> {sentences2[j]}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Feature scores ──────────────────────────────────────── */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 10,
          color: 'var(--text-3)',
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
        }}>
          Signal breakdown
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
          {Object.entries(feature_scores).map(([name, score]) => {
            const tier = alignTier(score);
            return (
              <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 10,
                  color: 'var(--text-2)',
                  width: 140,
                  flexShrink: 0,
                  letterSpacing: '0.02em',
                }}>
                  {name}
                </span>
                <div style={{
                  flex: 1,
                  height: 4,
                  background: 'var(--bg-4)',
                  borderRadius: 99,
                  overflow: 'hidden',
                }}>
                  <div style={{
                    height: '100%',
                    width: `${Math.round(score * 100)}%`,
                    background: ALIGN_COLOR[tier],
                    borderRadius: 99,
                    transition: 'width 0.6s cubic-bezier(0.16, 1, 0.3, 1)',
                    boxShadow: score > 0.3 ? `0 0 6px ${ALIGN_COLOR[tier]}60` : 'none',
                  }} />
                </div>
                <span style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 10,
                  color: ALIGN_COLOR[tier],
                  width: 36,
                  textAlign: 'right',
                  flexShrink: 0,
                  letterSpacing: '0.02em',
                }}>
                  {score.toFixed(2)}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Misalignment reasons ────────────────────────────────── */}
      {misalignment_reasons.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--text-3)',
            textTransform: 'uppercase',
            letterSpacing: '0.1em',
          }}>
            Misalignment Diagnostics
          </div>
          {misalignment_reasons.map((reason, idx) => (
            <div key={idx} style={{
              background: SEVERITY_BG[reason.severity],
              border: `1px solid ${SEVERITY_BORDER[reason.severity]}`,
              borderRadius: 10,
              padding: '10px 12px',
              display: 'flex', flexDirection: 'column', gap: 5,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 10,
                  color: SEVERITY_COLOR[reason.severity],
                  letterSpacing: '0.06em',
                }}>
                  {SEVERITY_ICON[reason.severity]} {reason.severity.toUpperCase()}
                </span>
                <span style={{
                  fontFamily: 'var(--font-body)',
                  fontSize: 12,
                  fontWeight: 600,
                  color: 'var(--text-1)',
                }}>
                  {reason.label}
                </span>
                <span style={{
                  marginLeft: 'auto',
                  fontFamily: 'var(--font-mono)',
                  fontSize: 10,
                  color: 'var(--text-3)',
                  flexShrink: 0,
                }}>
                  {reason.signal}
                </span>
              </div>
              <p style={{
                fontFamily: 'var(--font-body)',
                fontSize: 12,
                color: 'var(--text-2)',
                lineHeight: 1.55,
                margin: 0,
              }}>
                {reason.description}
              </p>
            </div>
          ))}
        </div>
      )}

      {misalignment_reasons.length === 0 && (
        <div style={{
          background: 'rgba(52,211,153,0.06)',
          border: '1px solid rgba(52,211,153,0.2)',
          borderRadius: 10,
          padding: '10px 14px',
        }}>
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: '#34D399',
            letterSpacing: '0.06em',
          }}>
            No misalignment signals detected
          </div>
          <div style={{
            fontFamily: 'var(--font-body)',
            fontSize: 12,
            color: 'rgba(52,211,153,0.6)',
            marginTop: 4,
          }}>
            All feature signals are within expected alignment thresholds.
          </div>
        </div>
      )}
    </div>
  );
}
