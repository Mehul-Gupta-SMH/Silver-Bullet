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
  // OpenAI
  { id: 'gpt-4o-mini',            label: 'GPT-4o mini',      vendor: 'openai',    tier: 'fast',     tagline: 'Quick deliberation' },
  { id: 'gpt-4o',                 label: 'GPT-4o',           vendor: 'openai',    tier: 'balanced', tagline: 'Strong reasoning' },
  { id: 'o4-mini',                label: 'o4-mini',          vendor: 'openai',    tier: 'powerful', tagline: 'Deep chain-of-thought' },
  // Anthropic
  { id: 'claude-haiku-4-5',       label: 'Claude Haiku 4.5', vendor: 'anthropic', tier: 'fast',     tagline: 'Efficient verdict' },
  { id: 'claude-sonnet-4-5',      label: 'Claude Sonnet 4.5',vendor: 'anthropic', tier: 'balanced', tagline: 'Nuanced review' },
  { id: 'claude-opus-4-5',        label: 'Claude Opus 4.5',  vendor: 'anthropic', tier: 'powerful', tagline: 'Supreme judgment' },
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

// ─── Example cases ───────────────────────────────────────────────────────────

type ExpectedVerdict = 'faithful' | 'hallucinated' | 'ambiguous';

interface JuryExample {
  id: string;
  mode: ComparisonMode;
  title: string;
  description: string;
  expected: ExpectedVerdict;
  tags: string[];
  text1: string;
  text2: string;
}

const JURY_EXAMPLES: JuryExample[] = [
  // ── Context vs Generated ─────────────────────────────────────────────
  {
    id: 'cvg-faithful',
    mode: 'context-vs-generated',
    title: 'Earnings Report — Grounded Summary',
    description: 'LLM accurately summarises all key figures from the source.',
    expected: 'faithful',
    tags: ['finance', 'summary'],
    text1: `Acme Corp reported Q3 2024 revenue of $4.2 billion, up 12% year-over-year. Net income was $380 million, representing a net margin of 9.0%. The company added 1.2 million new subscribers, bringing total subscribers to 18.7 million. Operating cash flow was $610 million. Management raised full-year guidance to $16.5–16.8 billion.`,
    text2: `Acme Corp posted strong Q3 2024 results with $4.2B in revenue, a 12% annual increase. Net income reached $380M (9% margin), and the subscriber base grew by 1.2M to 18.7M total. Free cash flow came in at $610M, prompting management to lift full-year guidance to the $16.5–16.8B range.`,
  },
  {
    id: 'cvg-hallucinated',
    mode: 'context-vs-generated',
    title: 'Drug Trial — Fabricated Efficacy',
    description: 'LLM inverts the trial result — source shows no benefit, answer claims success.',
    expected: 'hallucinated',
    tags: ['medical', 'hallucination', 'negation'],
    text1: `The Phase III trial of compound XR-77 enrolled 2,400 patients over 18 months. The primary endpoint — reduction in cardiovascular events — was not met. The treatment group showed a 4.1% event rate versus 3.9% in placebo (p=0.61). The trial was discontinued after interim analysis showed no benefit.`,
    text2: `The Phase III trial of XR-77 demonstrated a statistically significant reduction in cardiovascular events. The treatment group achieved a 4.1% event rate compared to 3.9% in placebo, confirming the drug's efficacy. Regulatory submission is planned for Q1 2025.`,
  },
  {
    id: 'cvg-ambiguous',
    mode: 'context-vs-generated',
    title: 'Policy Memo — Partial Coverage',
    description: 'Answer is faithful for half the content but omits a critical constraint.',
    expected: 'ambiguous',
    tags: ['policy', 'omission'],
    text1: `Effective January 1, 2025, remote work is approved for roles classified as Tier-2 or above, subject to manager approval and a minimum of two in-office days per week. Employees in client-facing roles are excluded. The policy applies only to domestic locations.`,
    text2: `Starting January 2025, Tier-2 and above employees may work remotely with manager approval, with a requirement of at least two days per week in the office.`,
  },
  // ── Reference vs Generated ───────────────────────────────────────────
  {
    id: 'rvg-faithful',
    mode: 'reference-vs-generated',
    title: 'Climate Abstract — Faithful Paraphrase',
    description: 'Generated abstract preserves all key claims from the reference.',
    expected: 'faithful',
    tags: ['science', 'paraphrase'],
    text1: `Global mean surface temperature increased by 1.1°C above pre-industrial levels as of 2022. Arctic warming is occurring at four times the global average rate. Sea level has risen 20 cm since 1900, with the rate accelerating to 3.7 mm/year over the past decade. Extreme weather events have increased in frequency and intensity.`,
    text2: `As of 2022, Earth's average surface temperature is 1.1°C warmer than pre-industrial baselines. The Arctic is heating four times faster than the global mean. Since 1900, sea levels are up 20 cm, now rising at 3.7 mm/year — an accelerating trend — while extreme weather events grow more frequent and severe.`,
  },
  {
    id: 'rvg-hallucinated',
    mode: 'reference-vs-generated',
    title: 'Legal Ruling — Wrong Precedent',
    description: 'Generated text cites the correct case name but inverts the ruling.',
    expected: 'hallucinated',
    tags: ['legal', 'entity-subst', 'hallucination'],
    text1: `In Brown v. Board of Education (1954), the Supreme Court unanimously ruled that racial segregation in public schools was unconstitutional, overturning the "separate but equal" doctrine established in Plessy v. Ferguson (1896).`,
    text2: `In Brown v. Board of Education (1954), the Supreme Court ruled 5-4 that while racial segregation in public schools was undesirable, it did not violate the Constitution, affirming the separate but equal doctrine from Plessy v. Ferguson.`,
  },
  // ── Model vs Model ───────────────────────────────────────────────────
  {
    id: 'mvm-agreement',
    mode: 'model-vs-model',
    title: 'Sorting Algorithms — Strong Agreement',
    description: 'Two models explain quicksort with the same key facts.',
    expected: 'faithful',
    tags: ['cs', 'algorithms', 'agreement'],
    text1: `Quicksort is a divide-and-conquer sorting algorithm with average-case time complexity of O(n log n). It selects a pivot element, partitions the array into elements less than and greater than the pivot, and recursively sorts each partition. Worst-case complexity is O(n²), occurring when the pivot is always the smallest or largest element.`,
    text2: `Quicksort works by picking a pivot and partitioning the input into two groups — values below and above the pivot — then recursively sorting each group. Average time complexity is O(n log n), making it fast in practice. However, poor pivot selection (e.g., always choosing the minimum) degrades performance to O(n²).`,
  },
  {
    id: 'mvm-divergence',
    mode: 'model-vs-model',
    title: 'Monetary Policy — Opposing Conclusions',
    description: 'Two models reach opposite recommendations on rate policy.',
    expected: 'hallucinated',
    tags: ['economics', 'divergence', 'opinion'],
    text1: `Given persistently elevated inflation at 3.8% and a strong labor market with unemployment at 3.7%, the Fed should maintain a restrictive stance and hold rates at current levels. Cutting prematurely risks re-igniting inflation expectations that took considerable effort to anchor.`,
    text2: `With inflation declining toward the 2% target and real rates well into restrictive territory, the Fed has room to begin cutting rates. Maintaining elevated rates unnecessarily could trigger a labor market deterioration, and the cost of waiting too long outweighs the risk of a modest easing.`,
  },
];

const EXPECTED_CONFIG: Record<ExpectedVerdict, { label: string; color: string; bg: string; border: string }> = {
  faithful:    { label: 'Faithful',    color: 'var(--green)',  bg: 'rgba(16,185,129,0.08)', border: 'rgba(16,185,129,0.22)' },
  hallucinated:{ label: 'Hallucinated',color: 'var(--red)',    bg: 'rgba(239,68,68,0.08)',  border: 'rgba(239,68,68,0.22)'  },
  ambiguous:   { label: 'Ambiguous',   color: 'var(--amber)',  bg: 'rgba(245,158,11,0.08)', border: 'rgba(245,158,11,0.22)' },
};

// ─── Sub-components ───────────────────────────────────────────────────────────

function ExamplesBar({
  mode,
  onLoad,
}: {
  mode: ComparisonMode;
  onLoad: (text1: string, text2: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const cases = JURY_EXAMPLES.filter(e => e.mode === mode);
  if (cases.length === 0) return null;

  return (
    <div style={{
      background: 'var(--bg-2)',
      border: '1px solid var(--border)',
      borderRadius: 12,
      overflow: 'hidden',
    }}>
      {/* Toggle header */}
      <button
        onClick={() => setOpen(v => !v)}
        style={{
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '10px 16px',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          color: 'var(--text-2)',
        }}
      >
        <span style={{
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: 18,
          height: 18,
          borderRadius: 4,
          background: 'var(--bg-4)',
          fontSize: 9,
          fontFamily: 'var(--font-mono)',
          color: 'var(--amber)',
          transition: 'transform 0.2s',
          transform: open ? 'rotate(90deg)' : 'none',
          flexShrink: 0,
        }}>▶</span>
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 10,
          letterSpacing: '0.10em',
          textTransform: 'uppercase',
          color: 'var(--text-3)',
        }}>Example Cases</span>
        <span style={{
          marginLeft: 'auto',
          fontFamily: 'var(--font-mono)',
          fontSize: 9,
          color: 'var(--text-3)',
          background: 'var(--bg-4)',
          padding: '2px 6px',
          borderRadius: 4,
        }}>{cases.length}</span>
      </button>

      {open && (
        <div style={{
          borderTop: '1px solid var(--border)',
          padding: '12px 14px',
          display: 'flex',
          flexDirection: 'column',
          gap: 8,
        }}>
          {cases.map(ex => {
            const ec = EXPECTED_CONFIG[ex.expected];
            return (
              <button
                key={ex.id}
                onClick={() => { onLoad(ex.text1, ex.text2); setOpen(false); }}
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 12,
                  padding: '10px 12px',
                  background: 'var(--bg-3)',
                  border: '1px solid var(--border)',
                  borderRadius: 9,
                  cursor: 'pointer',
                  textAlign: 'left',
                  transition: 'border-color 0.15s, background 0.15s',
                  width: '100%',
                }}
                onMouseEnter={e => {
                  (e.currentTarget as HTMLButtonElement).style.borderColor = ec.border;
                  (e.currentTarget as HTMLButtonElement).style.background = ec.bg;
                }}
                onMouseLeave={e => {
                  (e.currentTarget as HTMLButtonElement).style.borderColor = 'var(--border)';
                  (e.currentTarget as HTMLButtonElement).style.background = 'var(--bg-3)';
                }}
              >
                {/* Expected verdict dot */}
                <div style={{
                  width: 8, height: 8,
                  borderRadius: '50%',
                  background: ec.color,
                  flexShrink: 0,
                  marginTop: 5,
                  boxShadow: `0 0 6px ${ec.color}`,
                }} />

                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3, flexWrap: 'wrap' }}>
                    <span style={{
                      fontFamily: 'var(--font-display)',
                      fontSize: 12,
                      fontWeight: 600,
                      color: 'var(--text-1)',
                    }}>{ex.title}</span>
                    <span style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: 9,
                      color: ec.color,
                      background: ec.bg,
                      border: `1px solid ${ec.border}`,
                      padding: '1px 6px',
                      borderRadius: 4,
                      letterSpacing: '0.06em',
                      textTransform: 'uppercase',
                      flexShrink: 0,
                    }}>→ {ec.label}</span>
                  </div>
                  <div style={{
                    fontSize: 11,
                    color: 'var(--text-3)',
                    lineHeight: 1.4,
                    marginBottom: 6,
                  }}>{ex.description}</div>
                  <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
                    {ex.tags.map(tag => (
                      <span key={tag} style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: 9,
                        color: 'var(--text-3)',
                        background: 'var(--bg-4)',
                        padding: '1px 5px',
                        borderRadius: 3,
                        letterSpacing: '0.04em',
                      }}>{tag}</span>
                    ))}
                  </div>
                </div>

                {/* Load arrow */}
                <span style={{
                  fontSize: 14,
                  color: 'var(--text-3)',
                  flexShrink: 0,
                  marginTop: 2,
                }}>→</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

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
              const isYes = q.answer === 'yes';
              const qColor = isYes ? 'var(--green)' : 'var(--red)';
              const qBg = isYes
                ? 'rgba(16,185,129,0.06)'
                : 'rgba(239,68,68,0.06)';
              const qBorder = isYes
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
                      }}>{isYes ? 'YES' : 'NO'}</span>
                      <span style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: 9,
                        color: 'var(--text-3)',
                      }}>conf={q.confidence.toFixed(2)}</span>
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

      {/* ── Example cases ────────────────────────────────────────── */}
      <ExamplesBar
        mode={mode}
        onLoad={(t1, t2) => { setText1(t1); setText2(t2); setResult(null); }}
      />

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
