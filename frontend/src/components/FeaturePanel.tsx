import { useState } from 'react';
import type { ComparisonMode } from '../types';

interface FeatureGroup {
  name: string;
  abbr: string;
  maps: number;
  color: string;
  bg: string;
  border: string;
  models: string[];
  signals: string[];
  description: string;
}

const BASE_GROUPS = {
  lexical: {
    name: 'Lexical', abbr: 'LEX', maps: 4,
    color: '#60A5FA', bg: 'rgba(96,165,250,0.08)', border: 'rgba(96,165,250,0.2)',
    models: ['SentencePiece tokenizer'],
    signals: ['Jaccard', 'Dice', 'ROUGE-3', 'ROUGE'],
    description: 'Token-overlap statistics at multiple granularities. Zero model inference cost.',
  },
  semantic: {
    name: 'Semantic', abbr: 'SEM', maps: 4,
    color: '#A78BFA', bg: 'rgba(167,139,250,0.08)', border: 'rgba(167,139,250,0.2)',
    models: ['mxbai-embed-large-v1', 'Qwen3-Embedding-0.6B'],
    signals: ['mxbai Cosine', 'mxbai Precision', 'mxbai Recall', 'Qwen3 Precision'],
    description: 'Two embedding models with precision/recall asymmetry — reduces single-model bias.',
  },
  nli: {
    name: 'NLI', abbr: 'NLI', maps: 3,
    color: '#FCD34D', bg: 'rgba(252,211,77,0.08)', border: 'rgba(252,211,77,0.2)',
    models: ['roberta-large-mnli'],
    signals: ['Entailment', 'Neutral', 'Contradiction'],
    description: 'Directional logical inference — the backbone of faithfulness scoring.',
  },
  lcs: {
    name: 'LCS', abbr: 'LCS', maps: 2,
    color: '#34D399', bg: 'rgba(52,211,153,0.08)', border: 'rgba(52,211,153,0.2)',
    models: ['Built-in DP'],
    signals: ['LCS Token', 'LCS Char'],
    description: 'Longest Common Subsequence at token and character levels. Structural signal, zero cost.',
  },
  numeric: {
    name: 'Numeric', abbr: 'NUM', maps: 1,
    color: '#FB923C', bg: 'rgba(251,146,60,0.08)', border: 'rgba(251,146,60,0.2)',
    models: ['Regex extractor'],
    signals: ['Numeric Jaccard'],
    description: 'Numeric value overlap — flags hallucinated quantities even when surrounding text matches.',
  },
  grounding: {
    name: 'Grounding', abbr: 'GND', maps: 1,
    color: '#F87171', bg: 'rgba(248,113,113,0.08)', border: 'rgba(248,113,113,0.2)',
    models: ['modern-gliner-bi-base-v1.0'],
    signals: ['Entity Grounding Recall'],
    description: 'Fraction of source entities covered in the generated text — measures factual anchoring.',
  },
  triplet: {
    name: 'Triplet', abbr: 'TRP', maps: 1,
    color: '#2DD4BF', bg: 'rgba(45,212,191,0.08)', border: 'rgba(45,212,191,0.2)',
    models: ['gliner-relex-base-v1.0'],
    signals: ['Relation Triplet Recall'],
    description: 'Subject-predicate-object relation recall — catches predicate flips and argument swaps.',
  },
} satisfies Record<string, FeatureGroup>;

// Per-mode Entity group — signals and map counts differ across modes
const ENTITY_BY_MODE: Record<ComparisonMode, FeatureGroup> = {
  'context-vs-generated': {
    name: 'Entity', abbr: 'ENT', maps: 3,
    color: '#E879F9', bg: 'rgba(232,121,249,0.08)', border: 'rgba(232,121,249,0.2)',
    models: ['modern-gliner-bi-base-v1.0'],
    signals: ['Value Precision', 'Value Recall', 'Product Value Prec'],
    description: 'Named entity overlap with value-level scoring. Product entity precision catches context-specific factual errors.',
  },
  'reference-vs-generated': {
    name: 'Entity', abbr: 'ENT', maps: 4,
    color: '#E879F9', bg: 'rgba(232,121,249,0.08)', border: 'rgba(232,121,249,0.2)',
    models: ['modern-gliner-bi-base-v1.0'],
    signals: ['Value Precision', 'Value Recall', 'Product Count', 'Percentage Count'],
    description: 'Entity overlap with type-count signals for products and percentages — critical for reference faithfulness.',
  },
  'model-vs-model': {
    name: 'Entity', abbr: 'ENT', maps: 5,
    color: '#E879F9', bg: 'rgba(232,121,249,0.08)', border: 'rgba(232,121,249,0.2)',
    models: ['modern-gliner-bi-base-v1.0'],
    signals: ['Value Precision', 'Value Recall', 'Pct Count', 'Pct Value Prec', 'Pct Value Rec'],
    description: 'Percentage entity signals dominate model agreement. Tracks both presence and value-level parity.',
  },
};

const FEATURES_BY_MODE: Record<ComparisonMode, FeatureGroup[]> = {
  'context-vs-generated': [
    BASE_GROUPS.lexical,
    BASE_GROUPS.semantic,
    BASE_GROUPS.nli,
    ENTITY_BY_MODE['context-vs-generated'],
    BASE_GROUPS.lcs,
    BASE_GROUPS.numeric,
    BASE_GROUPS.grounding,
    BASE_GROUPS.triplet,
  ],
  'reference-vs-generated': [
    BASE_GROUPS.lexical,
    BASE_GROUPS.semantic,
    BASE_GROUPS.nli,
    ENTITY_BY_MODE['reference-vs-generated'],
    BASE_GROUPS.lcs,
    BASE_GROUPS.numeric,
    BASE_GROUPS.grounding,
    BASE_GROUPS.triplet,
  ],
  'model-vs-model': [
    BASE_GROUPS.lexical,
    BASE_GROUPS.semantic,
    BASE_GROUPS.nli,
    ENTITY_BY_MODE['model-vs-model'],
    BASE_GROUPS.lcs,
    BASE_GROUPS.numeric,
    BASE_GROUPS.grounding,
    BASE_GROUPS.triplet,
  ],
};

const SPATIAL_SIZE = 32;

interface FeaturePanelProps {
  mode: ComparisonMode;
}

export function FeaturePanel({ mode }: FeaturePanelProps) {
  const [open, setOpen] = useState(false);
  const [hovered, setHovered] = useState<string | null>(null);

  const features = FEATURES_BY_MODE[mode];
  const totalMaps = features.reduce((sum, f) => sum + f.maps, 0);
  const numFamilies = features.length;

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
            {features.map(f => (
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
            {totalMaps} maps · {numFamilies} families
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
              {features.map(f => {
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
                    {(f.maps / totalMaps > 0.10) && (
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
              {features.map(f => (
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
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: 8,
          }}>
            {features.map(f => {
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
            {['Text Pair', '→', 'Feature Extraction', '→', `${totalMaps} × [${SPATIAL_SIZE}×${SPATIAL_SIZE}]`, '→', 'Conv2D Head', '→', 'Score [0,1]'].map((seg, i) => (
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
