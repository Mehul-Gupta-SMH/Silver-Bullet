import { useState, useRef, useEffect } from 'react';
import type { ComparisonMode, JuryResult } from '../types';
import { predictJuryPair } from '../services/api';

// ─── Juror model catalogue ────────────────────────────────────────────────────

interface JurorModel {
  id: string;
  label: string;
  vendor: 'openai' | 'anthropic' | 'google';
  tier: 'fast' | 'balanced' | 'powerful';
  tagline: string;
}

const JUROR_MODELS: JurorModel[] = [
  { id: 'gpt-4o-mini',      label: 'GPT-4o mini',    vendor: 'openai',    tier: 'fast',     tagline: 'Quick deliberation' },
  { id: 'gpt-4o',           label: 'GPT-4o',         vendor: 'openai',    tier: 'powerful', tagline: 'Deep reasoning' },
  { id: 'gpt-4-turbo',      label: 'GPT-4 Turbo',    vendor: 'openai',    tier: 'balanced', tagline: 'Balanced judgment' },
  { id: 'claude-3-5-haiku-20241022', label: 'Claude Haiku', vendor: 'anthropic', tier: 'fast',     tagline: 'Efficient verdict' },
  { id: 'claude-3-5-sonnet-20241022', label: 'Claude Sonnet', vendor: 'anthropic', tier: 'balanced', tagline: 'Nuanced review' },
  { id: 'claude-opus-4-5',   label: 'Claude Opus',   vendor: 'anthropic', tier: 'powerful', tagline: 'Supreme judgment' },
];

const VENDOR_COLORS = {
  openai:    { dot: '#10A37F', bg: 'rgba(16,163,127,0.10)', border: 'rgba(16,163,127,0.25)' },
  anthropic: { dot: '#D4694B', bg: 'rgba(212,105,75,0.10)',  border: 'rgba(212,105,75,0.25)'  },
  google:    { dot: '#4285F4', bg: 'rgba(66,133,244,0.10)', border: 'rgba(66,133,244,0.25)' },
};

const TIER_LABELS = { fast: '⚡ Fast', balanced: '⚖ Balanced', powerful: '◆ Powerful' };

const MODE_CONFIG: Record<ComparisonMode, { label: string; t1: string; t2: string; color: string }> = {
  'context-vs-generated':   { label: 'Context vs Generated', t1: 'Source Context / RAG Chunk', t2: 'LLM-Generated Answer', color: 'var(--cvg)' },
  'reference-vs-generated': { label: 'Reference vs Generated', t1: 'Ground-Truth Reference',    t2: 'Generated Answer',    color: 'var(--rvg)' },
  'model-vs-model':         { label: 'Model vs Model',         t1: 'Model A Output',             t2: 'Model B Output',      color: 'var(--mvm)' },
};

// ─── Sub-components ───────────────────────────────────────────────────────────

function VendorDot({ vendor }: { vendor: JurorModel['vendor'] }) {
  const c = VENDOR_COLORS[vendor];
  return (
    <span style={{
      display: 'inline-block',
      width: 7, height: 7,
      borderRadius: '50%',
      background: c.dot,
      boxShadow: `0 0 6px ${c.dot}`,
      flexShrink: 0,
    }} />
  );
}

function DeliberationSpinner() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 20, padding: '40px 0' }}>
      <div style={{ position: 'relative', width: 64, height: 64 }}>
        {/* Scales SVG */}
        <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"
          style={{ width: 64, height: 64, animation: 'juryPulse 2s ease-in-out infinite' }}>
          <line x1="32" y1="8" x2="32" y2="56" stroke="var(--amber)" strokeWidth="2" strokeLinecap="round"/>
          <line x1="12" y1="16" x2="52" y2="16" stroke="var(--amber)" strokeWidth="2" strokeLinecap="round"/>
          {/* Left pan */}
          <line x1="16" y1="16" x2="12" y2="30" stroke="var(--amber)" strokeWidth="1.5"/>
          <line x1="20" y1="16" x2="16" y2="30" stroke="var(--amber)" strokeWidth="1.5"/>
          <path d="M10 30 Q14 36 18 30" stroke="var(--amber)" strokeWidth="1.5" fill="none" strokeLinecap="round"/>
          {/* Right pan */}
          <line x1="48" y1="16" x2="44" y2="30" stroke="var(--amber)" strokeWidth="1.5"/>
          <line x1="52" y1="16" x2="48" y2="30" stroke="var(--amber)" strokeWidth="1.5"/>
          <path d="M42 30 Q46 36 50 30" stroke="var(--amber)" strokeWidth="1.5" fill="none" strokeLinecap="round"/>
          {/* Top orb */}
          <circle cx="32" cy="8" r="4" fill="var(--amber)" opacity="0.9"/>
        </svg>
        {/* Glow */}
        <div style={{
          position: 'absolute', inset: 0, borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(245,158,11,0.15) 0%, transparent 70%)',
          animation: 'juryGlow 2s ease-in-out infinite',
        }}/>
      </div>
      <div style={{ textAlign: 'center' }}>
        <div style={{
          fontFamily: 'var(--font-display)',
          fontSize: 15,
          fontWeight: 600,
          color: 'var(--amber)',
          letterSpacing: '0.06em',
          textTransform: 'uppercase',
        }}>
          Jury Deliberating
        </div>
        <div style={{ display: 'flex', gap: 5, justifyContent: 'center', marginTop: 8 }}>
          {[0, 1, 2].map(i => (
            <span key={i} style={{
              width: 5, height: 5, borderRadius: '50%',
              background: 'var(--amber)',
              display: 'inline-block',
              animation: `juryDot 1.4s ease-in-out ${i * 0.2}s infinite`,
              opacity: 0.3,
            }}/>
          ))}
        </div>
        <div style={{
          marginTop: 10,
          fontFamily: 'var(--font-mono)',
          fontSize: 11,
          color: 'var(--text-3)',
          letterSpacing: '0.04em',
        }}>Consulting the panel of jurors…</div>
      </div>
    </div>
  );
}

function VerdictCard({ result, model }: { result: JuryResult; model: string }) {
  const [revealed, setRevealed] = useState(false);
  const [questionReveal, setQuestionReveal] = useState<boolean[]>([]);

  useEffect(() => {
    // Staggered reveal
    const t0 = setTimeout(() => setRevealed(true), 100);
    const timers = result.questions.map((_, i) =>
      setTimeout(() => setQuestionReveal(prev => {
        const next = [...prev];
        next[i] = true;
        return next;
      }), 500 + i * 120)
    );
    return () => { clearTimeout(t0); timers.forEach(clearTimeout); };
  }, [result.questions.length]);

  const pct = Math.round(result.score * 100);
  const isFaithful = result.verdict === 'faithful';
  const verdictColor = isFaithful ? 'var(--green)' : 'var(--red)';
  const verdictLabel = isFaithful ? 'FAITHFUL' : 'HALLUCINATED';

  return (
    <div style={{
      opacity: revealed ? 1 : 0,
      transform: revealed ? 'translateY(0)' : 'translateY(12px)',
      transition: 'opacity 0.5s ease, transform 0.5s ease',
    }}>
      {/* Verdict banner */}
      <div style={{
        background: 'var(--bg-2)',
        border: `1px solid ${verdictColor}`,
        borderRadius: 14,
        padding: '28px 28px 24px',
        marginBottom: 16,
        position: 'relative',
        overflow: 'hidden',
      }}>
        {/* Ambient glow */}
        <div style={{
          position: 'absolute', top: 0, left: 0, right: 0, height: 2,
          background: `linear-gradient(90deg, transparent, ${verdictColor}, transparent)`,
          opacity: 0.6,
        }}/>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 16 }}>
          {/* Score */}
          <div>
            <div style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 10,
              color: 'var(--text-3)',
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
              marginBottom: 6,
            }}>Jury Score</div>
            <div style={{
              fontFamily: 'var(--font-display)',
              fontSize: 56,
              fontWeight: 800,
              color: verdictColor,
              lineHeight: 1,
              letterSpacing: '-0.03em',
            }}>{result.score.toFixed(3)}</div>
            {/* Score bar */}
            <div style={{
              width: 200,
              height: 3,
              background: 'var(--bg-4)',
              borderRadius: 2,
              marginTop: 10,
              position: 'relative',
              overflow: 'hidden',
            }}>
              <div style={{
                position: 'absolute', top: 0, left: 0, bottom: 0,
                width: `${pct}%`,
                background: verdictColor,
                borderRadius: 2,
                transition: 'width 0.8s cubic-bezier(0.25,0.46,0.45,0.94)',
                boxShadow: `0 0 8px ${verdictColor}`,
              }}/>
            </div>
          </div>

          {/* Verdict stamp */}
          <div style={{
            border: `2px solid ${verdictColor}`,
            borderRadius: 8,
            padding: '10px 18px',
            transform: 'rotate(-1.5deg)',
            boxShadow: `0 0 20px ${verdictColor}22`,
          }}>
            <div style={{
              fontFamily: 'var(--font-display)',
              fontSize: 20,
              fontWeight: 800,
              color: verdictColor,
              letterSpacing: '0.14em',
              textTransform: 'uppercase',
            }}>{verdictLabel}</div>
            <div style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 9,
              color: 'var(--text-3)',
              letterSpacing: '0.1em',
              marginTop: 3,
              textTransform: 'uppercase',
            }}>Jury Verdict · {model}</div>
          </div>
        </div>
      </div>

      {/* Per-question cards */}
      {result.questions.length > 0 && (
        <div>
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--text-3)',
            letterSpacing: '0.12em',
            textTransform: 'uppercase',
            marginBottom: 10,
          }}>Deliberation — {result.questions.length} Questions</div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {result.questions.map((q, i) => {
              const visible = questionReveal[i];
              const qColor = q.answer ? 'var(--green)' : 'var(--red)';
              const qBg = q.answer
                ? 'rgba(16,185,129,0.06)'
                : 'rgba(239,68,68,0.06)';
              const qBorder = q.answer
                ? 'rgba(16,185,129,0.2)'
                : 'rgba(239,68,68,0.2)';
              return (
                <div
                  key={i}
                  style={{
                    background: qBg,
                    border: `1px solid ${qBorder}`,
                    borderRadius: 10,
                    padding: '12px 14px',
                    opacity: visible ? 1 : 0,
                    transform: visible ? 'translateX(0)' : 'translateX(-8px)',
                    transition: 'opacity 0.3s ease, transform 0.3s ease',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12 }}>
                    <div style={{ flex: 1 }}>
                      <div style={{
                        fontSize: 12,
                        fontWeight: 500,
                        color: 'var(--text-1)',
                        lineHeight: 1.4,
                        marginBottom: q.reasoning ? 6 : 0,
                      }}>{q.question}</div>
                      {q.reasoning && (
                        <div style={{
                          fontSize: 11,
                          color: 'var(--text-2)',
                          fontStyle: 'italic',
                          lineHeight: 1.5,
                        }}>{q.reasoning}</div>
                      )}
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4, flexShrink: 0 }}>
                      <span style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: 11,
                        fontWeight: 700,
                        color: qColor,
                        letterSpacing: '0.06em',
                      }}>{q.answer ? 'YES' : 'NO'}</span>
                      <span style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: 9,
                        color: 'var(--text-3)',
                      }}>w={q.weight.toFixed(1)} · {(q.weighted_score).toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Main JuryScorer ──────────────────────────────────────────────────────────

interface Props {
  mode: ComparisonMode;
}

export function JuryScorer({ mode }: Props) {
  const [selectedModel, setSelectedModel] = useState<string>('gpt-4o-mini');
  const [text1, setText1] = useState('');
  const [text2, setText2] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<JuryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  const modeConf = MODE_CONFIG[mode];
  const selectedJuror = JUROR_MODELS.find(m => m.id === selectedModel) ?? JUROR_MODELS[0];
  const canSubmit = text1.trim().length > 0 && text2.trim().length > 0 && !loading;

  const handleSubmit = async () => {
    if (!canSubmit) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const res = await predictJuryPair(text1.trim(), text2.trim(), mode, selectedModel);
      setResult(res);
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Jury evaluation failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* ── Juror Selection ───────────────────────────────────────── */}
      <section style={{
        background: 'var(--bg-2)',
        border: '1px solid var(--border)',
        borderRadius: 14,
        padding: '20px 20px 16px',
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          marginBottom: 14,
        }}>
          {/* Scales icon */}
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <line x1="8" y1="2" x2="8" y2="14" stroke="var(--amber)" strokeWidth="1.5" strokeLinecap="round"/>
            <line x1="3" y1="4" x2="13" y2="4" stroke="var(--amber)" strokeWidth="1.5" strokeLinecap="round"/>
            <line x1="4" y1="4" x2="3" y2="8" stroke="var(--amber)" strokeWidth="1.2"/>
            <line x1="5" y1="4" x2="4" y2="8" stroke="var(--amber)" strokeWidth="1.2"/>
            <path d="M2.5 8 Q4 10 5.5 8" stroke="var(--amber)" strokeWidth="1.2" fill="none" strokeLinecap="round"/>
            <line x1="12" y1="4" x2="11" y2="8" stroke="var(--amber)" strokeWidth="1.2"/>
            <line x1="13" y1="4" x2="12" y2="8" stroke="var(--amber)" strokeWidth="1.2"/>
            <path d="M10.5 8 Q12 10 13.5 8" stroke="var(--amber)" strokeWidth="1.2" fill="none" strokeLinecap="round"/>
          </svg>
          <span style={{
            fontFamily: 'var(--font-display)',
            fontSize: 13,
            fontWeight: 600,
            color: 'var(--text-1)',
            letterSpacing: '0.01em',
          }}>Select Juror Model</span>
          <span style={{
            marginLeft: 'auto',
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--text-3)',
            letterSpacing: '0.08em',
          }}>PANEL OF {JUROR_MODELS.length}</span>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(170px, 1fr))',
          gap: 8,
        }}>
          {JUROR_MODELS.map(juror => {
            const isSelected = juror.id === selectedModel;
            const vc = VENDOR_COLORS[juror.vendor];
            return (
              <button
                key={juror.id}
                onClick={() => { setSelectedModel(juror.id); setResult(null); }}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 6,
                  padding: '10px 12px',
                  background: isSelected ? vc.bg : 'var(--bg-3)',
                  border: `1px solid ${isSelected ? vc.border : 'var(--border)'}`,
                  borderRadius: 10,
                  cursor: 'pointer',
                  textAlign: 'left',
                  transition: 'all 0.15s ease',
                  outline: 'none',
                  boxShadow: isSelected ? `0 0 0 1px ${vc.border}` : 'none',
                }}
                onMouseEnter={e => {
                  if (!isSelected) (e.currentTarget as HTMLButtonElement).style.borderColor = 'var(--border-2)';
                }}
                onMouseLeave={e => {
                  if (!isSelected) (e.currentTarget as HTMLButtonElement).style.borderColor = 'var(--border)';
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                  <VendorDot vendor={juror.vendor} />
                  <span style={{
                    fontFamily: 'var(--font-display)',
                    fontSize: 12,
                    fontWeight: 600,
                    color: isSelected ? 'var(--text-1)' : 'var(--text-2)',
                    lineHeight: 1,
                  }}>{juror.label}</span>
                </div>
                <div style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 9,
                  color: isSelected ? vc.dot : 'var(--text-3)',
                  letterSpacing: '0.06em',
                  textTransform: 'uppercase',
                }}>{TIER_LABELS[juror.tier]}</div>
                <div style={{
                  fontSize: 10,
                  color: 'var(--text-3)',
                  lineHeight: 1.3,
                }}>{juror.tagline}</div>
              </button>
            );
          })}
        </div>

        {/* Selected juror summary */}
        <div style={{
          marginTop: 12,
          padding: '8px 12px',
          background: 'var(--bg-3)',
          borderRadius: 8,
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}>
          <VendorDot vendor={selectedJuror.vendor} />
          <span style={{ fontSize: 11, color: 'var(--text-2)' }}>
            <span style={{ color: 'var(--text-1)', fontWeight: 500 }}>{selectedJuror.label}</span>
            {' '}will evaluate using structured yes/no questions across faithfulness dimensions
          </span>
        </div>
      </section>

      {/* ── Text inputs ───────────────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        {([
          { key: 't1', label: modeConf.t1, value: text1, set: setText1 },
          { key: 't2', label: modeConf.t2, value: text2, set: setText2 },
        ] as const).map(({ key, label, value, set }) => (
          <div key={key} style={{
            background: 'var(--bg-2)',
            border: '1px solid var(--border)',
            borderRadius: 14,
            padding: '16px 16px 12px',
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
          }}>
            <div style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 10,
              color: 'var(--text-3)',
              letterSpacing: '0.10em',
              textTransform: 'uppercase',
            }}>{label}</div>
            <textarea
              value={value}
              onChange={e => { set(e.target.value); setResult(null); }}
              placeholder={`Enter ${label.toLowerCase()}…`}
              rows={6}
              style={{
                width: '100%',
                background: 'var(--bg-3)',
                border: '1px solid var(--border)',
                borderRadius: 8,
                padding: '10px 12px',
                color: 'var(--text-1)',
                fontFamily: 'var(--font-body)',
                fontSize: 13,
                lineHeight: 1.55,
                resize: 'vertical',
                outline: 'none',
                transition: 'border-color 0.15s',
              }}
              onFocus={e => { e.target.style.borderColor = 'var(--border-2)'; }}
              onBlur={e => { e.target.style.borderColor = 'var(--border)'; }}
            />
            <div style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 9,
              color: value.length > 8000 ? 'var(--red)' : 'var(--text-3)',
              textAlign: 'right',
            }}>{value.length.toLocaleString()} / 10,000</div>
          </div>
        ))}
      </div>

      {/* ── Convene button ───────────────────────────────────────── */}
      <div style={{ display: 'flex', justifyContent: 'center' }}>
        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            padding: '13px 32px',
            background: canSubmit
              ? 'linear-gradient(135deg, var(--amber) 0%, #D97706 100%)'
              : 'var(--bg-4)',
            border: 'none',
            borderRadius: 10,
            cursor: canSubmit ? 'pointer' : 'not-allowed',
            color: canSubmit ? '#000' : 'var(--text-3)',
            fontFamily: 'var(--font-display)',
            fontSize: 14,
            fontWeight: 700,
            letterSpacing: '0.04em',
            textTransform: 'uppercase',
            transition: 'all 0.2s ease',
            boxShadow: canSubmit ? '0 4px 20px rgba(245,158,11,0.35)' : 'none',
          }}
          onMouseEnter={e => {
            if (canSubmit) {
              (e.currentTarget as HTMLButtonElement).style.transform = 'translateY(-1px)';
              (e.currentTarget as HTMLButtonElement).style.boxShadow = '0 6px 28px rgba(245,158,11,0.45)';
            }
          }}
          onMouseLeave={e => {
            (e.currentTarget as HTMLButtonElement).style.transform = 'translateY(0)';
            (e.currentTarget as HTMLButtonElement).style.boxShadow = canSubmit ? '0 4px 20px rgba(245,158,11,0.35)' : 'none';
          }}
        >
          {/* Gavel icon */}
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <rect x="6" y="2" width="7" height="4" rx="1.5" fill="currentColor" transform="rotate(45 6 2)"/>
            <line x1="4" y1="8" x2="2" y2="14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
          Convene Jury
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            opacity: 0.7,
            fontWeight: 400,
            textTransform: 'none',
            letterSpacing: 0,
          }}>· {selectedJuror.label}</span>
        </button>
      </div>

      {/* ── Deliberation / Result ─────────────────────────────────── */}
      <div ref={resultRef}>
        {loading && <DeliberationSpinner />}

        {error && !loading && (
          <div style={{
            background: 'rgba(239,68,68,0.08)',
            border: '1px solid rgba(239,68,68,0.2)',
            borderRadius: 12,
            padding: '16px 20px',
            display: 'flex',
            gap: 12,
            alignItems: 'flex-start',
          }}>
            <span style={{ fontSize: 18, flexShrink: 0 }}>⚠</span>
            <div>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--red)', marginBottom: 4 }}>
                Jury could not reach a verdict
              </div>
              <div style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 11,
                color: 'var(--text-2)',
                wordBreak: 'break-word',
              }}>{error}</div>
              <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-3)' }}>
                Ensure <code style={{ fontFamily: 'var(--font-mono)', background: 'var(--bg-4)', padding: '1px 4px', borderRadius: 3 }}>SB_OPENAI_TOKEN</code> is configured in the backend.
              </div>
            </div>
          </div>
        )}

        {result && !loading && (
          <VerdictCard result={result} model={selectedJuror.label} />
        )}
      </div>
    </div>
  );
}
