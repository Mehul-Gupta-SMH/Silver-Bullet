import { useState } from 'react';
import { predictPair, predictPairBreakdown } from '../services/api';
import { useLocalStorage } from '../hooks/useLocalStorage';
import { ScoreGauge } from './ScoreGauge';
import { BreakdownPanel } from './BreakdownPanel';
import { ModelConfig } from './ModelConfig';
import { TestCasePanel } from './TestCasePanel';
import { SaveExperimentForm } from './SaveExperimentForm';
import { getModeConfig } from '../config/modes';
import type { TextMeta } from './ModelConfig';
import type { PairTestCase } from '../data/testCases';
import type { BreakdownResult, ComparisonMode, PredictionResult } from '../types';

export interface PairInitData {
  text1: string;
  text2: string;
  meta: TextMeta;
}

interface Props {
  mode: ComparisonMode;
  initData?: PairInitData;
  onSave: (opts: {
    name: string;
    notes: string;
    mode: ComparisonMode;
    modelMeta: TextMeta;
    text1: string;
    text2: string;
    result: { prediction: number; probability: number };
  }) => void;
}

const INTERP_COLOR  = { green: '#34D399', yellow: '#F59E0B', red: '#F87171' } as const;
const INTERP_BG     = { green: 'rgba(52,211,153,0.07)',  yellow: 'rgba(245,158,11,0.07)',  red: 'rgba(248,113,113,0.07)'  } as const;
const INTERP_BORDER = { green: 'rgba(52,211,153,0.22)',  yellow: 'rgba(245,158,11,0.22)',  red: 'rgba(248,113,113,0.22)'  } as const;

export function PairScorer({ mode, initData, onSave }: Props) {
  // When initData is provided (re-run from experiments), use it directly and persist it.
  // Otherwise restore from localStorage so the draft survives page refreshes.
  const [text1, setText1] = useLocalStorage<string>(
    'sb_pair_text1',
    initData?.text1 ?? '',
  );
  const [text2, setText2] = useLocalStorage<string>(
    'sb_pair_text2',
    initData?.text2 ?? '',
  );
  const [meta, setMeta] = useLocalStorage<TextMeta>(
    'sb_pair_meta',
    initData?.meta ?? { name1: '', name2: '', baseline: null },
  );

  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [breakdown, setBreakdown] = useState<BreakdownResult | null>(null);
  const [breakdownLoading, setBreakdownLoading] = useState(false);
  const [breakdownError, setBreakdownError] = useState<string | null>(null);
  const [showBreakdown, setShowBreakdown] = useState(false);

  const cfg = getModeConfig(mode);
  const label1 = meta.name1 || cfg.text1Label;
  const label2 = meta.name2 || cfg.text2Label;

  const handleScore = async () => {
    if (!text1.trim() || !text2.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setBreakdown(null);
    setShowBreakdown(false);
    try {
      setResult(await predictPair(text1, text2, mode));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const handleDrillDown = async () => {
    if (showBreakdown) {
      setShowBreakdown(false);
      return;
    }
    if (breakdown) {
      setShowBreakdown(true);
      return;
    }
    setBreakdownLoading(true);
    setBreakdownError(null);
    try {
      const bd = await predictPairBreakdown(text1, text2, mode);
      setBreakdown(bd);
      setShowBreakdown(true);
    } catch (e) {
      setBreakdownError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setBreakdownLoading(false);
    }
  };

  const handleLoadTestCase = (tc: PairTestCase) => {
    setText1(tc.text1);
    setText2(tc.text2);
    setMeta({
      name1: tc.name1 ?? '',
      name2: tc.name2 ?? '',
      baseline: null,
    });
    setResult(null);
    setError(null);
    setBreakdown(null);
    setShowBreakdown(false);
  };

  const interpretation = result ? cfg.interpret(result.probability) : null;
  const comparisonLabel =
    mode === 'model-vs-model' && (meta.name1 || meta.name2)
      ? `${meta.baseline === '1' ? `${label1} (baseline)` : label1} vs ${meta.baseline === '2' ? `${label2} (baseline)` : label2}`
      : cfg.label;

  const defaultExpName = `${cfg.label} · ${new Date().toLocaleDateString()}`;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Test case examples */}
      <TestCasePanel scope="pair" mode={mode} onLoad={handleLoadTestCase} />

      {/* Model / source name config */}
      <ModelConfig
        mode={mode}
        meta={meta}
        onChange={setMeta}
        label1={cfg.text1Label}
        label2={cfg.text2Label}
      />

      {/* Text inputs */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <label style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--text-3)',
            textTransform: 'uppercase',
            letterSpacing: '0.1em',
          }}>
            {label1}
          </label>
          <textarea
            style={{
              width: '100%',
              height: 176,
              padding: '12px 14px',
              background: 'var(--bg-3)',
              border: '1px solid var(--border-2)',
              borderRadius: 10,
              color: 'var(--text-1)',
              fontFamily: 'var(--font-mono)',
              fontSize: 12,
              lineHeight: 1.65,
              resize: 'none',
              outline: 'none',
              transition: 'border-color 0.15s, box-shadow 0.15s',
              boxSizing: 'border-box',
            }}
            placeholder={cfg.text1Placeholder}
            value={text1}
            onChange={(e) => setText1(e.target.value)}
            onFocus={e => { e.target.style.borderColor = 'var(--accent)'; e.target.style.boxShadow = '0 0 0 3px var(--accent-dim)'; }}
            onBlur={e => { e.target.style.borderColor = 'var(--border-2)'; e.target.style.boxShadow = 'none'; }}
          />
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--text-3)',
            textAlign: 'right',
            letterSpacing: '0.04em',
          }}>
            {text1.length.toLocaleString()} / 10,000
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <label style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--text-3)',
            textTransform: 'uppercase',
            letterSpacing: '0.1em',
          }}>
            {label2}
          </label>
          <textarea
            style={{
              width: '100%',
              height: 176,
              padding: '12px 14px',
              background: 'var(--bg-3)',
              border: '1px solid var(--border-2)',
              borderRadius: 10,
              color: 'var(--text-1)',
              fontFamily: 'var(--font-mono)',
              fontSize: 12,
              lineHeight: 1.65,
              resize: 'none',
              outline: 'none',
              transition: 'border-color 0.15s, box-shadow 0.15s',
              boxSizing: 'border-box',
            }}
            placeholder={cfg.text2Placeholder}
            value={text2}
            onChange={(e) => setText2(e.target.value)}
            onFocus={e => { e.target.style.borderColor = 'var(--accent)'; e.target.style.boxShadow = '0 0 0 3px var(--accent-dim)'; }}
            onBlur={e => { e.target.style.borderColor = 'var(--border-2)'; e.target.style.boxShadow = 'none'; }}
          />
          <div style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--text-3)',
            textAlign: 'right',
            letterSpacing: '0.04em',
          }}>
            {text2.length.toLocaleString()} / 10,000
          </div>
        </div>
      </div>

      <button
        onClick={handleScore}
        disabled={loading || !text1.trim() || !text2.trim()}
        className="sb-btn-primary"
        style={{ alignSelf: 'flex-start' }}
      >
        {loading && (
          <span style={{
            width: 14, height: 14,
            border: '2px solid rgba(4,6,7,0.3)',
            borderTopColor: '#040607',
            borderRadius: '50%',
            display: 'inline-block',
            animation: 'spin 0.7s linear infinite',
          }} />
        )}
        {loading ? 'Analysing…' : 'Analyse Pair'}
      </button>

      {error && (
        <div style={{
          display: 'flex', alignItems: 'flex-start', gap: 8,
          background: 'rgba(239,68,68,0.08)',
          border: '1px solid rgba(239,68,68,0.25)',
          borderRadius: 10,
          padding: '10px 14px',
          fontFamily: 'var(--font-body)',
          fontSize: 13,
          color: '#F87171',
        }}>
          <span style={{ flex: 1 }}>{error}</span>
          <button
            onClick={() => setError(null)}
            style={{
              background: 'none', border: 'none', cursor: 'pointer',
              color: '#F87171', fontSize: 16, lineHeight: 1, padding: 0,
              opacity: 0.6,
            }}
            aria-label="Dismiss"
          >×</button>
        </div>
      )}

      {result && interpretation && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: 16, alignItems: 'stretch' }}>
            <ScoreGauge probability={result.probability} prediction={result.prediction} />
            <div style={{
              background: INTERP_BG[interpretation.color],
              border: `1px solid ${INTERP_BORDER[interpretation.color]}`,
              borderRadius: 12,
              padding: '20px 22px',
              display: 'flex', flexDirection: 'column', justifyContent: 'center',
              gap: 8,
            }}>
              <div style={{
                fontFamily: 'var(--font-display)',
                fontSize: 20,
                fontWeight: 700,
                color: INTERP_COLOR[interpretation.color],
                letterSpacing: '-0.01em',
                lineHeight: 1.2,
              }}>
                {interpretation.headline}
              </div>
              <p style={{
                fontFamily: 'var(--font-body)',
                fontSize: 13,
                lineHeight: 1.65,
                color: 'var(--text-2)',
                margin: 0,
              }}>
                {interpretation.detail}
              </p>
              <div style={{
                marginTop: 8,
                paddingTop: 10,
                borderTop: `1px solid ${INTERP_BORDER[interpretation.color]}`,
              }}>
                <span style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 10,
                  color: INTERP_COLOR[interpretation.color],
                  textTransform: 'uppercase',
                  letterSpacing: '0.08em',
                  opacity: 0.75,
                }}>
                  {comparisonLabel} · {result.probability.toFixed(3)}
                </span>
              </div>
            </div>
          </div>

          {/* Drill-down trigger */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <button
              onClick={handleDrillDown}
              disabled={breakdownLoading}
              className="sb-btn-ghost"
            >
              {breakdownLoading ? (
                <span style={{
                  width: 12, height: 12,
                  border: '1.5px solid var(--text-3)',
                  borderTopColor: 'var(--text-2)',
                  borderRadius: '50%',
                  display: 'inline-block',
                  animation: 'spin 0.7s linear infinite',
                }} />
              ) : (
                <span style={{ fontSize: 11 }}>{showBreakdown ? '▲' : '▼'}</span>
              )}
              {breakdownLoading ? 'Running deep analysis…' : showBreakdown ? 'Hide Breakdown' : 'Drill Down — Impact & Divergence'}
            </button>
            {breakdownError && (
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#F87171' }}>
                {breakdownError}
              </span>
            )}
          </div>

          {/* Breakdown panel */}
          {showBreakdown && breakdown && (
            <BreakdownPanel breakdown={breakdown} />
          )}

          {/* Save to experiments */}
          <SaveExperimentForm
            defaultName={defaultExpName}
            onSave={(name, notes) =>
              onSave({ name, notes, mode, modelMeta: meta, text1, text2, result })
            }
          />
        </>
      )}
    </div>
  );
}
