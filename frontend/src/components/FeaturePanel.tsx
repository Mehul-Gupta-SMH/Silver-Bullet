import { useState } from 'react';

const TOTAL_MAPS = 16;

const FEATURES = [
  {
    name: 'Semantic',
    abbr: 'SEM',
    maps: 6,
    color: '#A78BFA',
    bg: 'rgba(167,139,250,0.08)',
    border: 'rgba(167,139,250,0.2)',
    models: ['mxbai-embed-large-v1', 'Qwen3-Embedding-0.6B'],
    signals: ['Cosine sim', 'Soft row align', 'Soft col align'],
    description: 'Two embedding models vote to reduce single-model bias. Dense vectors capture meaning beyond surface form.',
  },
  {
    name: 'Lexical',
    abbr: 'LEX',
    maps: 4,
    color: '#60A5FA',
    bg: 'rgba(96,165,250,0.08)',
    border: 'rgba(96,165,250,0.2)',
    models: ['SentencePiece tokenizer'],
    signals: ['Jaccard', 'Dice', 'Cosine', 'ROUGE'],
    description: 'Token-overlap statistics with zero model inference overhead.',
  },
  {
    name: 'NLI',
    abbr: 'NLI',
    maps: 3,
    color: '#FCD34D',
    bg: 'rgba(252,211,77,0.08)',
    border: 'rgba(252,211,77,0.2)',
    models: ['roberta-large-mnli'],
    signals: ['Entailment', 'Neutral', 'Contradiction'],
    description: 'Directional logical inference — the backbone of faithfulness scoring.',
  },
  {
    name: 'Entity',
    abbr: 'ENT',
    maps: 1,
    color: '#F87171',
    bg: 'rgba(248,113,113,0.08)',
    border: 'rgba(248,113,113,0.2)',
    models: ['modern-gliner-bi-base-v1.0'],
    signals: ['NER mismatch'],
    description: 'Named entity overlap catches factual grounding — same people, places, numbers?',
  },
  {
    name: 'LCS',
    abbr: 'LCS',
    maps: 2,
    color: '#34D399',
    bg: 'rgba(52,211,153,0.08)',
    border: 'rgba(52,211,153,0.2)',
    models: ['Built-in DP'],
    signals: ['LCS token', 'LCS char'],
    description: 'Longest Common Subsequence at both token and character levels. Zero cost, structural signal.',
  },
] as const;

export function FeaturePanel() {
  const [open, setOpen] = useState(false);
  const [hovered, setHovered] = useState<string | null>(null);

  return (
    <div style={{
      background: 'var(--bg-2)',
      border: '1px solid var(--border)',
      borderRadius: 12,
      overflow: 'hidden',
    }}>

      {/* ── Collapse header ──────────────────────────────────────── */}
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
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--text-2)',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
          }}>
            Signal Architecture
          </span>

          {/* Mini tape preview — always visible */}
          <div style={{
            display: 'flex',
            height: 6,
            width: 80,
            borderRadius: 99,
            overflow: 'hidden',
            gap: 1,
          }}>
            {FEATURES.map(f => (
              <div key={f.abbr} style={{
                flex: f.maps,
                background: f.color,
                opacity: 0.7,
              }} />
            ))}
          </div>

          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--text-3)',
            letterSpacing: '0.04em',
          }}>
            {TOTAL_MAPS} maps · 5 families
          </span>
        </div>

        <span style={{
          color: 'var(--text-3)',
          fontSize: 10,
          display: 'inline-block',
          transform: open ? 'rotate(180deg)' : 'rotate(0deg)',
          transition: 'transform 0.2s',
          userSelect: 'none',
        }}>▼</span>
      </button>

      {open && (
        <div style={{ borderTop: '1px solid var(--border)', padding: '16px 16px 18px' }}>

          {/* ── Proportional signal tape ──────────────────────────── */}
          <div style={{ marginBottom: 16 }}>
            <div style={{
              display: 'flex',
              height: 20,
              borderRadius: 6,
              overflow: 'hidden',
              gap: 2,
            }}>
              {FEATURES.map(f => {
                const isHov = hovered === f.abbr;
                return (
                  <div
                    key={f.abbr}
                    onMouseOver={() => setHovered(f.abbr)}
                    onMouseOut={() => setHovered(null)}
                    style={{
                      flex: f.maps,
                      background: f.color,
                      opacity: hovered === null ? 0.75 : isHov ? 1 : 0.25,
                      transition: 'opacity 0.15s, flex 0.2s',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      cursor: 'default',
                      borderRadius: 3,
                    }}
                  >
                    {(f.maps / TOTAL_MAPS > 0.12) && (
                      <span style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: 9,
                        fontWeight: 600,
                        color: 'rgba(0,0,0,0.55)',
                        letterSpacing: '0.06em',
                        pointerEvents: 'none',
                      }}>
                        {f.abbr}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Map count labels */}
            <div style={{ display: 'flex', marginTop: 4, gap: 2 }}>
              {FEATURES.map(f => (
                <div key={f.abbr} style={{
                  flex: f.maps,
                  display: 'flex',
                  justifyContent: 'center',
                }}>
                  <span style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: 9,
                    color: hovered === f.abbr ? f.color : 'var(--text-3)',
                    transition: 'color 0.15s',
                    letterSpacing: '0.04em',
                  }}>
                    {f.maps}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* ── Feature cards ─────────────────────────────────────── */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(5, 1fr)',
            gap: 8,
          }}>
            {FEATURES.map(f => {
              const isHov = hovered === f.abbr;
              return (
                <div
                  key={f.abbr}
                  onMouseOver={() => setHovered(f.abbr)}
                  onMouseOut={() => setHovered(null)}
                  style={{
                    background: isHov ? f.bg : 'var(--bg-3)',
                    border: `1px solid ${isHov ? f.border : 'var(--border)'}`,
                    borderRadius: 9,
                    overflow: 'hidden',
                    transition: 'background 0.15s, border-color 0.15s',
                    display: 'flex',
                    flexDirection: 'column',
                  }}
                >
                  {/* Color top bar */}
                  <div style={{
                    height: 3,
                    background: f.color,
                    opacity: isHov ? 1 : 0.5,
                    transition: 'opacity 0.15s',
                  }} />

                  <div style={{ padding: '10px 11px', display: 'flex', flexDirection: 'column', gap: 8, flex: 1 }}>

                    {/* Name + map count */}
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <span style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: 11,
                        fontWeight: 600,
                        color: isHov ? f.color : 'var(--text-1)',
                        letterSpacing: '0.04em',
                        transition: 'color 0.15s',
                      }}>
                        {f.name}
                      </span>
                      <span style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: 9,
                        color: 'var(--text-3)',
                        letterSpacing: '0.04em',
                      }}>
                        {f.maps}×
                      </span>
                    </div>

                    {/* Signals */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                      {f.signals.map(s => (
                        <div key={s} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                          <span style={{
                            width: 4, height: 4,
                            borderRadius: '50%',
                            background: f.color,
                            opacity: 0.6,
                            flexShrink: 0,
                          }} />
                          <span style={{
                            fontFamily: 'var(--font-mono)',
                            fontSize: 10,
                            color: 'var(--text-2)',
                            letterSpacing: '0.02em',
                          }}>
                            {s}
                          </span>
                        </div>
                      ))}
                    </div>

                    {/* Description */}
                    <p style={{
                      fontFamily: 'var(--font-body)',
                      fontSize: 11,
                      color: 'var(--text-3)',
                      lineHeight: 1.55,
                      margin: 0,
                      flex: 1,
                    }}>
                      {f.description}
                    </p>

                    {/* Model chips */}
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
                      {f.models.map(m => (
                        <span key={m} style={{
                          fontFamily: 'var(--font-mono)',
                          fontSize: 9,
                          color: isHov ? f.color : 'var(--text-3)',
                          background: isHov ? f.bg : 'var(--bg-4)',
                          border: `1px solid ${isHov ? f.border : 'var(--border-2)'}`,
                          padding: '1px 5px',
                          borderRadius: 3,
                          letterSpacing: '0.02em',
                          transition: 'all 0.15s',
                          wordBreak: 'break-all',
                        }}>
                          {m}
                        </span>
                      ))}
                    </div>

                  </div>
                </div>
              );
            })}
          </div>

          {/* ── Pipeline caption ──────────────────────────────────── */}
          <div style={{
            marginTop: 12,
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            justifyContent: 'center',
          }}>
            {['Text Pair', '→', 'Feature Extraction', '→', `${TOTAL_MAPS} × [64×64]`, '→', 'Conv2D Head', '→', 'Score [0,1]'].map((seg, i) => (
              <span key={i} style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 9,
                color: i % 2 === 0 ? 'var(--text-3)' : 'var(--border-2)',
                letterSpacing: '0.05em',
                textTransform: i % 2 === 0 ? 'uppercase' : undefined,
              }}>
                {seg}
              </span>
            ))}
          </div>

        </div>
      )}
    </div>
  );
}
