import { useState } from 'react';

const FEATURES = [
  {
    name: 'Semantic',
    icon: '🧠',
    pill: 'bg-violet-100 text-violet-700',
    count: '6 maps',
    models: ['mxbai-embed-large-v1', 'Qwen3-Embedding-0.6B'],
    signals: ['Cosine similarity', 'Soft row alignment', 'Soft column alignment'],
    description:
      'Dense vector embeddings capture deep semantic meaning. Two models vote to reduce single-model bias.',
  },
  {
    name: 'Lexical',
    icon: '📝',
    pill: 'bg-blue-100 text-blue-700',
    count: '4 maps',
    models: ['SentencePiece tokenizer'],
    signals: ['Jaccard', 'Dice', 'Cosine (token)', 'ROUGE'],
    description:
      'Token-level overlap statistics measure surface-form similarity without any model inference.',
  },
  {
    name: 'NLI',
    icon: '🔍',
    pill: 'bg-amber-100 text-amber-700',
    count: '3 maps',
    models: ['roberta-large-mnli'],
    signals: ['Entailment', 'Neutral', 'Contradiction'],
    description:
      'Natural Language Inference captures directional logical relationships — crucial for faithfulness.',
  },
  {
    name: 'Entity',
    icon: '🏷️',
    pill: 'bg-rose-100 text-rose-700',
    count: '1 map',
    models: ['modern-gliner-bi-base-v1.0'],
    signals: ['Named entity mismatch'],
    description:
      'Named entity overlap detects factual grounding — are the same people, places, and numbers present?',
  },
  {
    name: 'LCS',
    icon: '🔗',
    pill: 'bg-emerald-100 text-emerald-700',
    count: '2 maps',
    models: ['Built-in (no model)'],
    signals: ['LCS token', 'LCS character'],
    description:
      'Longest Common Subsequence at token and character levels captures structural similarity with zero cost.',
  },
] as const;

export function FeaturePanel() {
  const [open, setOpen] = useState(false);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-5 py-3.5 hover:bg-slate-50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="text-sm font-semibold text-slate-700">Signal Features</span>
          <span className="text-xs bg-slate-100 text-slate-500 px-2.5 py-0.5 rounded-full font-medium">
            5 families · 16 maps · 2 embedding models · 3 ML models
          </span>
        </div>
        <span
          className={`text-slate-400 text-xs transition-transform duration-200 select-none ${
            open ? 'rotate-180' : ''
          }`}
        >
          ▼
        </span>
      </button>

      {open && (
        <div className="border-t border-slate-100 p-5 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {FEATURES.map((f) => (
            <div
              key={f.name}
              className="rounded-xl border border-slate-100 bg-slate-50/50 p-3.5 space-y-2.5"
            >
              <div className="flex items-center gap-2">
                <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${f.pill}`}>
                  {f.icon} {f.name}
                </span>
                <span className="text-xs text-slate-400 font-medium">{f.count}</span>
              </div>

              <ul className="space-y-1">
                {f.signals.map((s) => (
                  <li key={s} className="text-xs text-slate-600 flex items-center gap-1.5">
                    <span className="w-1 h-1 bg-slate-300 rounded-full flex-shrink-0" />
                    {s}
                  </li>
                ))}
              </ul>

              <p className="text-xs text-slate-400 leading-relaxed">{f.description}</p>

              <div className="flex flex-wrap gap-1">
                {f.models.map((m) => (
                  <span
                    key={m}
                    className="inline-block text-[10px] bg-white border border-slate-200 text-slate-500 px-1.5 py-0.5 rounded font-mono"
                  >
                    {m}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
