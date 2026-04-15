import { MODES } from '../config/modes';
import type { ComparisonMode } from '../types';

interface Props {
  selected: ComparisonMode;
  onChange: (mode: ComparisonMode) => void;
}

const accent: Record<ComparisonMode, { color: string; dim: string; border: string }> = {
  'model-vs-model':         { color: 'var(--mvm)', dim: 'rgba(59 130 246 / 0.1)',  border: 'rgba(59 130 246 / 0.4)' },
  'reference-vs-generated': { color: 'var(--rvg)', dim: 'rgba(16 185 129 / 0.1)', border: 'rgba(16 185 129 / 0.4)' },
  'context-vs-generated':   { color: 'var(--cvg)', dim: 'rgba(139 92 246 / 0.1)', border: 'rgba(139 92 246 / 0.4)' },
};

export function ComparisonModeSelector({ selected, onChange }: Props) {
  return (
    <div>
      <div style={{
        fontFamily: 'var(--font-mono)',
        fontSize: 9,
        fontWeight: 500,
        color: 'var(--text-3)',
        textTransform: 'uppercase',
        letterSpacing: '0.12em',
        marginBottom: 12,
      }}>
        Evaluation Mode
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
        {MODES.map((mode) => {
          const isSelected = selected === mode.id;
          const a = accent[mode.id];
          return (
            <button
              key={mode.id}
              onClick={() => onChange(mode.id)}
              style={{
                textAlign: 'left',
                padding: '14px 16px',
                borderRadius: 12,
                border: `1px solid ${isSelected ? a.border : 'var(--border)'}`,
                background: isSelected ? a.dim : 'var(--bg-2)',
                cursor: 'pointer',
                transition: 'all 0.15s',
                outline: isSelected ? `none` : 'none',
                boxShadow: 'none',
              }}
              onMouseOver={e => {
                if (!isSelected) {
                  (e.currentTarget as HTMLElement).style.borderColor = 'var(--border-2)';
                  (e.currentTarget as HTMLElement).style.background = 'var(--bg-3)';
                }
              }}
              onMouseOut={e => {
                if (!isSelected) {
                  (e.currentTarget as HTMLElement).style.borderColor = 'var(--border)';
                  (e.currentTarget as HTMLElement).style.background = 'var(--bg-2)';
                }
              }}
            >
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
                <span style={{ fontSize: 20, lineHeight: 1, marginTop: 1 }}>{mode.emoji}</span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3 }}>
                    <span style={{
                      fontFamily: 'var(--font-body)',
                      fontSize: 13,
                      fontWeight: 600,
                      color: isSelected ? a.color : 'var(--text-1)',
                      letterSpacing: '-0.01em',
                    }}>
                      {mode.label}
                    </span>
                    {isSelected && (
                      <span style={{
                        width: 5, height: 5, borderRadius: '50%',
                        background: a.color,
                        flexShrink: 0,
                        opacity: 0.8,
                      }} />
                    )}
                  </div>
                  <p style={{
                    fontFamily: 'var(--font-body)',
                    fontSize: 11,
                    fontWeight: 500,
                    color: 'var(--text-2)',
                    margin: 0,
                    marginBottom: 4,
                  }}>
                    {mode.tagline}
                  </p>
                  <p style={{
                    fontFamily: 'var(--font-body)',
                    fontSize: 11,
                    color: 'var(--text-3)',
                    margin: 0,
                    lineHeight: 1.5,
                  }}>
                    {mode.description}
                  </p>
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
