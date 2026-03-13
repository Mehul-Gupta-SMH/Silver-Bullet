import { MODES } from '../config/modes';
import type { ComparisonMode } from '../types';

interface Props {
  selected: ComparisonMode;
  onChange: (mode: ComparisonMode) => void;
}

const accent: Record<ComparisonMode, { ring: string; bg: string; dot: string }> = {
  'model-vs-model': { ring: 'ring-blue-400', bg: 'bg-blue-50', dot: 'bg-blue-500' },
  'reference-vs-generated': { ring: 'ring-emerald-400', bg: 'bg-emerald-50', dot: 'bg-emerald-500' },
  'context-vs-generated': { ring: 'ring-violet-400', bg: 'bg-violet-50', dot: 'bg-violet-500' },
};

export function ComparisonModeSelector({ selected, onChange }: Props) {
  return (
    <div>
      <p className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-3">
        Evaluation Mode
      </p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {MODES.map((mode) => {
          const isSelected = selected === mode.id;
          const a = accent[mode.id];
          return (
            <button
              key={mode.id}
              onClick={() => onChange(mode.id)}
              className={`text-left p-4 rounded-2xl border-2 transition-all duration-150 ${
                isSelected
                  ? `ring-2 ${a.ring} ring-offset-1 border-transparent ${a.bg}`
                  : 'border-slate-200 bg-white hover:border-slate-300 hover:shadow-sm'
              }`}
            >
              <div className="flex items-start gap-3">
                <span className="text-2xl leading-none mt-0.5">{mode.emoji}</span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="font-semibold text-slate-900 text-sm">{mode.label}</span>
                    {isSelected && (
                      <span className={`w-2 h-2 rounded-full flex-shrink-0 ${a.dot}`} />
                    )}
                  </div>
                  <p className="text-xs text-slate-500 font-medium">{mode.tagline}</p>
                  <p className="text-xs text-slate-400 mt-1 leading-relaxed">{mode.description}</p>
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
