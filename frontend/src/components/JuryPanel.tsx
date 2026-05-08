import { useState } from 'react';
import type { JuryResult } from '../types';

interface Props {
  result: JuryResult;
}

export function JuryPanel({ result }: Props) {
  const [expanded, setExpanded] = useState(false);
  const pct = Math.round(result.score * 100);
  const isFaithful = result.verdict === 'faithful';

  const verdictColor  = isFaithful ? '#34D399' : '#F87171';
  const verdictBg     = isFaithful ? 'rgba(52,211,153,0.08)' : 'rgba(248,113,113,0.08)';
  const verdictBorder = isFaithful ? 'rgba(52,211,153,0.22)' : 'rgba(248,113,113,0.22)';
  const barColor      = isFaithful ? '#34D399' : '#F87171';

  return (
    <div style={{
      background: 'var(--bg-2)',
      border: '1px solid var(--border)',
      borderRadius: 14,
      padding: '18px 20px',
      display: 'flex',
      flexDirection: 'column',
      gap: 14,
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--text-3)',
            textTransform: 'uppercase',
            letterSpacing: '0.1em',
          }}>
            LLM Jury Score
          </span>
          {result.model_used && (
            <span style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 9,
              color: 'var(--text-3)',
              background: 'var(--bg-4)',
              border: '1px solid var(--border-2)',
              padding: '1px 6px',
              borderRadius: 4,
              letterSpacing: '0.02em',
            }}>
              {result.model_used}
            </span>
          )}
        </div>
        <span style={{
          display: 'flex',
          alignItems: 'center',
          gap: 5,
          padding: '3px 10px',
          borderRadius: 99,
          background: verdictBg,
          border: `1px solid ${verdictBorder}`,
          fontFamily: 'var(--font-mono)',
          fontSize: 10,
          color: verdictColor,
          letterSpacing: '0.06em',
          textTransform: 'uppercase',
        }}>
          <span style={{
            width: 5, height: 5,
            borderRadius: '50%',
            background: verdictColor,
            display: 'inline-block',
          }} />
          {isFaithful ? 'Faithful' : 'Hallucinated'}
        </span>
      </div>

      {/* Score */}
      <div style={{
        fontFamily: 'var(--font-display)',
        fontSize: 44,
        fontWeight: 800,
        color: verdictColor,
        letterSpacing: '-0.02em',
        lineHeight: 1,
      }}>
        {result.score.toFixed(3)}
      </div>

      {/* Bar */}
      <div>
        <div style={{
          width: '100%',
          height: 6,
          background: 'var(--bg-4)',
          borderRadius: 99,
          overflow: 'hidden',
        }}>
          <div style={{
            width: `${pct}%`,
            height: '100%',
            background: barColor,
            borderRadius: 99,
            boxShadow: `0 0 8px ${barColor}60`,
            transition: 'width 0.7s cubic-bezier(0.16, 1, 0.3, 1)',
          }} />
        </div>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginTop: 4,
        }}>
          {['0', '0.25', '0.5', '0.75', '1'].map(v => (
            <span key={v} style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 9,
              color: 'var(--text-3)',
              letterSpacing: '0.04em',
            }}>
              {v}
            </span>
          ))}
        </div>
      </div>

      {/* Questions breakdown */}
      {result.questions?.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <button
            onClick={() => setExpanded(e => !e)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: 0,
              fontFamily: 'var(--font-mono)',
              fontSize: 10,
              color: 'var(--text-3)',
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
              transition: 'color 0.15s',
            }}
            onMouseOver={e => (e.currentTarget.style.color = 'var(--text-2)')}
            onMouseOut={e => (e.currentTarget.style.color = 'var(--text-3)')}
          >
            <span style={{
              display: 'inline-block',
              transform: expanded ? 'rotate(90deg)' : 'rotate(0deg)',
              transition: 'transform 0.2s',
              fontSize: 8,
            }}>▶</span>
            {result.questions.length} jury questions
          </button>

          {expanded && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {result.questions.map((q, i) => {
                const isYes = q.answer === 'yes';
                const qColor  = isYes ? '#34D399' : '#F87171';
                const qBg     = isYes ? 'rgba(52,211,153,0.06)' : 'rgba(248,113,113,0.06)';
                const qBorder = isYes ? 'rgba(52,211,153,0.18)' : 'rgba(248,113,113,0.18)';
                return (
                  <div key={i} style={{
                    background: qBg,
                    border: `1px solid ${qBorder}`,
                    borderRadius: 10,
                    padding: '10px 12px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 5,
                  }}>
                    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 10 }}>
                      <span style={{
                        fontFamily: 'var(--font-body)',
                        fontSize: 12,
                        fontWeight: 500,
                        color: 'var(--text-1)',
                        lineHeight: 1.5,
                      }}>
                        {q.question}
                      </span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
                        <span style={{
                          fontFamily: 'var(--font-mono)',
                          fontSize: 10,
                          fontWeight: 700,
                          color: qColor,
                          letterSpacing: '0.06em',
                        }}>
                          {isYes ? 'YES' : 'NO'}
                        </span>
                        <span style={{
                          fontFamily: 'var(--font-mono)',
                          fontSize: 9,
                          color: 'var(--text-3)',
                        }}>
                          conf={q.confidence.toFixed(2)}
                        </span>
                      </div>
                    </div>
                    {q.reasoning && (
                      <p style={{
                        fontFamily: 'var(--font-body)',
                        fontSize: 11,
                        color: 'var(--text-2)',
                        lineHeight: 1.55,
                        fontStyle: 'italic',
                        margin: 0,
                      }}>
                        {q.reasoning}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
