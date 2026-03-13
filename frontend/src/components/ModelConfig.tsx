import type { ComparisonMode } from '../types';

export interface TextMeta {
  name1: string;
  name2: string;
  baseline: '1' | '2' | null;
}

interface Props {
  mode: ComparisonMode;
  meta: TextMeta;
  onChange: (meta: TextMeta) => void;
  label1: string;
  label2: string;
}

const PRESET_MODELS = [
  'GPT-4o',
  'GPT-4o mini',
  'Claude 3.5 Sonnet',
  'Claude 3.5 Haiku',
  'Claude 3 Opus',
  'Gemini 1.5 Pro',
  'Gemini 1.5 Flash',
  'Llama 3.1 70B',
  'Llama 3.1 8B',
  'Mistral Large',
  'Qwen2.5 72B',
];

function NameInput({
  value,
  onChange,
  placeholder,
  isBaseline,
  onSetBaseline,
  showBaseline,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
  isBaseline: boolean;
  onSetBaseline: () => void;
  showBaseline: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <div className="relative flex-1">
        <input
          type="text"
          list="model-presets"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-400 focus:border-transparent placeholder:text-slate-300"
        />
        <datalist id="model-presets">
          {PRESET_MODELS.map((m) => (
            <option key={m} value={m} />
          ))}
        </datalist>
      </div>
      {showBaseline && (
        <button
          type="button"
          onClick={onSetBaseline}
          title="Set as baseline"
          className={`flex-shrink-0 px-3 py-2 rounded-lg text-xs font-semibold border transition-all duration-150 ${
            isBaseline
              ? 'bg-violet-600 text-white border-violet-600 shadow-sm shadow-violet-200'
              : 'bg-white text-slate-500 border-slate-200 hover:border-violet-300 hover:text-violet-600'
          }`}
        >
          {isBaseline ? '★ Baseline' : '☆ Set baseline'}
        </button>
      )}
    </div>
  );
}

export function ModelConfig({ mode, meta, onChange, label1, label2 }: Props) {
  const isModelMode = mode === 'model-vs-model';
  const showBaseline = isModelMode;

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-4">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xs font-semibold text-slate-500 uppercase tracking-widest">
          {isModelMode ? 'Model Names' : 'Source Labels'}
        </span>
        <span className="text-xs text-slate-400">(optional)</span>
        {isModelMode && meta.baseline && (
          <span className="ml-auto text-xs bg-violet-100 text-violet-700 px-2 py-0.5 rounded-full font-medium">
            {meta.baseline === '1'
              ? (meta.name1 || label1) + ' is baseline'
              : (meta.name2 || label2) + ' is baseline'}
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div className="space-y-1">
          <span className="text-xs text-slate-500 font-medium">{label1}</span>
          <NameInput
            value={meta.name1}
            onChange={(v) => onChange({ ...meta, name1: v })}
            placeholder={isModelMode ? 'e.g. GPT-4o' : `Label for ${label1}`}
            isBaseline={meta.baseline === '1'}
            onSetBaseline={() =>
              onChange({ ...meta, baseline: meta.baseline === '1' ? null : '1' })
            }
            showBaseline={showBaseline}
          />
        </div>
        <div className="space-y-1">
          <span className="text-xs text-slate-500 font-medium">{label2}</span>
          <NameInput
            value={meta.name2}
            onChange={(v) => onChange({ ...meta, name2: v })}
            placeholder={isModelMode ? 'e.g. Claude 3.5 Sonnet' : `Label for ${label2}`}
            isBaseline={meta.baseline === '2'}
            onSetBaseline={() =>
              onChange({ ...meta, baseline: meta.baseline === '2' ? null : '2' })
            }
            showBaseline={showBaseline}
          />
        </div>
      </div>

      {isModelMode && meta.baseline && (
        <p className="mt-2.5 text-xs text-slate-400 leading-relaxed">
          The baseline model sets the reference point. Scores reflect how closely the compared
          model aligns with the baseline output.
        </p>
      )}
    </div>
  );
}
