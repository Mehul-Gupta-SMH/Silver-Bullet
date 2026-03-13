import { useState } from 'react';
import { getModeConfig } from '../config/modes';
import type { ExperimentRecord } from '../hooks/useExperiments';
import type { ComparisonMode } from '../types';

interface RerunData {
  mode: ComparisonMode;
  text1: string;
  text2: string;
  name1: string;
  name2: string;
  baseline: '1' | '2' | null;
}

interface Props {
  experiments: ExperimentRecord[];
  onDelete: (id: string) => void;
  onUpdateNotes: (id: string, notes: string) => void;
  onClearAll: () => void;
  onExportAll: () => void;
  onRerun: (data: RerunData) => void;
}

const MODE_FILTERS: { id: ComparisonMode | 'all'; label: string; emoji: string }[] = [
  { id: 'all', label: 'All', emoji: '🔬' },
  { id: 'model-vs-model', label: 'Model vs Model', emoji: '🤖' },
  { id: 'reference-vs-generated', label: 'Reference vs Generated', emoji: '📋' },
  { id: 'context-vs-generated', label: 'Context vs Generated', emoji: '📚' },
];

function ScoreBadge({ prob, prediction }: { prob: number; prediction: number }) {
  const isHigh = prob >= 0.7;
  const isMid = prob >= 0.4;
  const bg = isHigh ? 'bg-emerald-500' : isMid ? 'bg-amber-400' : 'bg-red-500';
  const text = prediction === 1 ? 'Similar' : 'Different';
  return (
    <div className="flex items-center gap-2">
      <div className="w-20 bg-slate-100 rounded-full h-1.5">
        <div
          className={`h-1.5 rounded-full ${bg}`}
          style={{ width: `${prob * 100}%` }}
        />
      </div>
      <span className="text-xs font-mono font-semibold text-slate-700 tabular-nums">
        {prob.toFixed(3)}
      </span>
      <span
        className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-full text-white ${bg}`}
      >
        {text}
      </span>
    </div>
  );
}

function ExperimentCard({
  exp,
  onDelete,
  onUpdateNotes,
  onRerun,
}: {
  exp: ExperimentRecord;
  onDelete: () => void;
  onUpdateNotes: (notes: string) => void;
  onRerun: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [editingNotes, setEditingNotes] = useState(false);
  const [notesVal, setNotesVal] = useState(exp.notes);
  const cfg = getModeConfig(exp.mode);

  const date = new Date(exp.savedAt);
  const dateStr = date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

  const label1 = exp.modelMeta.name1 || cfg.text1Label;
  const label2 = exp.modelMeta.name2 || cfg.text2Label;

  return (
    <div className="bg-white rounded-2xl border border-slate-200 overflow-hidden">
      {/* Header row */}
      <div
        className="flex items-start gap-3 px-4 py-3.5 cursor-pointer hover:bg-slate-50/50 transition-colors"
        onClick={() => setExpanded((v) => !v)}
      >
        <span className="text-xl mt-0.5 flex-shrink-0">{cfg.emoji ?? '🔬'}</span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-semibold text-slate-900 text-sm">{exp.name}</span>
            <span className="text-[10px] bg-slate-100 text-slate-500 px-2 py-0.5 rounded-full font-medium">
              {exp.type === 'batch' ? `batch · ${exp.batchStats?.total ?? 0}` : 'pair'}
            </span>
          </div>
          <div className="flex items-center gap-3 mt-1 flex-wrap">
            <span className="text-xs text-slate-400">{dateStr}</span>
            {exp.type === 'pair' && exp.result && (
              <ScoreBadge prob={exp.result.probability} prediction={exp.result.prediction} />
            )}
            {exp.type === 'batch' && exp.batchStats && (
              <span className="text-xs text-slate-500">
                Mean {exp.batchStats.mean.toFixed(3)} · {exp.batchStats.similar}/{exp.batchStats.total} similar
              </span>
            )}
          </div>
        </div>
        <span className={`text-slate-400 text-xs flex-shrink-0 mt-1 transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}>
          ▼
        </span>
      </div>

      {/* Expanded body */}
      {expanded && (
        <div className="border-t border-slate-100 px-4 py-4 space-y-4">
          {/* Mode + labels */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-widest">
              {cfg.label}
            </span>
            {(exp.modelMeta.name1 || exp.modelMeta.name2) && (
              <>
                <span className="text-slate-300">·</span>
                <span className="text-xs font-mono bg-slate-100 px-1.5 py-0.5 rounded">
                  {exp.modelMeta.baseline === '1' ? `${exp.modelMeta.name1 || label1} (baseline)` : (exp.modelMeta.name1 || label1)}
                </span>
                <span className="text-xs text-slate-400">vs</span>
                <span className="text-xs font-mono bg-slate-100 px-1.5 py-0.5 rounded">
                  {exp.modelMeta.baseline === '2' ? `${exp.modelMeta.name2 || label2} (baseline)` : (exp.modelMeta.name2 || label2)}
                </span>
              </>
            )}
          </div>

          {/* Pair texts */}
          {exp.type === 'pair' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="space-y-1">
                <p className="text-xs font-semibold text-slate-500">{label1}</p>
                <p className="text-xs text-slate-600 font-mono bg-slate-50 border border-slate-100 rounded-lg p-2.5 max-h-28 overflow-y-auto leading-relaxed whitespace-pre-wrap">
                  {exp.text1}
                </p>
              </div>
              <div className="space-y-1">
                <p className="text-xs font-semibold text-slate-500">{label2}</p>
                <p className="text-xs text-slate-600 font-mono bg-slate-50 border border-slate-100 rounded-lg p-2.5 max-h-28 overflow-y-auto leading-relaxed whitespace-pre-wrap">
                  {exp.text2}
                </p>
              </div>
            </div>
          )}

          {/* Batch stats */}
          {exp.type === 'batch' && exp.batchStats && (
            <div className="grid grid-cols-3 gap-2">
              {[
                { label: 'Mean', val: exp.batchStats.mean.toFixed(3) },
                { label: 'Median', val: exp.batchStats.median.toFixed(3) },
                { label: 'Std Dev', val: exp.batchStats.std.toFixed(3) },
                { label: 'Similar', val: String(exp.batchStats.similar) },
                { label: 'Different', val: String(exp.batchStats.different) },
                { label: 'Total', val: String(exp.batchStats.total) },
              ].map((s) => (
                <div key={s.label} className="bg-slate-50 rounded-lg p-2 text-center">
                  <div className="text-base font-bold text-slate-800 tabular-nums">{s.val}</div>
                  <div className="text-[10px] text-slate-500">{s.label}</div>
                </div>
              ))}
            </div>
          )}

          {/* Notes */}
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <p className="text-xs font-semibold text-slate-500">Notes</p>
              {!editingNotes && (
                <button
                  onClick={(e) => { e.stopPropagation(); setEditingNotes(true); }}
                  className="text-xs text-violet-500 hover:text-violet-700"
                >
                  {exp.notes ? 'Edit' : '+ Add'}
                </button>
              )}
            </div>
            {editingNotes ? (
              <div className="flex gap-2">
                <textarea
                  value={notesVal}
                  onChange={(e) => setNotesVal(e.target.value)}
                  className="flex-1 text-xs border border-slate-200 rounded-lg p-2 resize-none h-16 focus:outline-none focus:ring-2 focus:ring-violet-400"
                  placeholder="Add notes…"
                />
                <div className="flex flex-col gap-1.5">
                  <button
                    onClick={(e) => { e.stopPropagation(); onUpdateNotes(notesVal); setEditingNotes(false); }}
                    className="px-2 py-1 bg-violet-600 text-white text-xs rounded-lg hover:bg-violet-700"
                  >
                    Save
                  </button>
                  <button
                    onClick={(e) => { e.stopPropagation(); setNotesVal(exp.notes); setEditingNotes(false); }}
                    className="px-2 py-1 bg-slate-100 text-slate-600 text-xs rounded-lg hover:bg-slate-200"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              exp.notes ? (
                <p className="text-xs text-slate-600 bg-slate-50 rounded-lg p-2 italic">{exp.notes}</p>
              ) : (
                <p className="text-xs text-slate-400 italic">No notes</p>
              )
            )}
          </div>

          {/* Actions */}
          <div className="flex items-center gap-2 pt-1">
            {exp.type === 'pair' && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onRerun();
                }}
                className="px-3 py-1.5 text-xs font-semibold bg-violet-600 text-white rounded-lg hover:bg-violet-700 transition-colors"
              >
                ↺ Re-run
              </button>
            )}
            <button
              onClick={(e) => { e.stopPropagation(); onDelete(); }}
              className="px-3 py-1.5 text-xs font-semibold bg-red-50 text-red-600 border border-red-200 rounded-lg hover:bg-red-100 transition-colors"
            >
              Delete
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export function ExperimentsPanel({
  experiments,
  onDelete,
  onUpdateNotes,
  onClearAll,
  onExportAll,
  onRerun,
}: Props) {
  const [filter, setFilter] = useState<ComparisonMode | 'all'>('all');

  const filtered =
    filter === 'all' ? experiments : experiments.filter((e) => e.mode === filter);

  return (
    <div className="space-y-5">
      {/* Toolbar */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex gap-1.5 flex-wrap">
          {MODE_FILTERS.map((f) => (
            <button
              key={f.id}
              onClick={() => setFilter(f.id)}
              className={`flex items-center gap-1 px-3 py-1.5 rounded-full text-xs font-semibold border transition-all ${
                filter === f.id
                  ? 'bg-violet-600 text-white border-violet-600'
                  : 'bg-white text-slate-600 border-slate-200 hover:border-slate-300'
              }`}
            >
              <span>{f.emoji}</span>
              {f.label}
              {f.id !== 'all' && (
                <span className={`ml-0.5 tabular-nums ${filter === f.id ? 'opacity-80' : 'text-slate-400'}`}>
                  ({experiments.filter((e) => e.mode === f.id).length})
                </span>
              )}
            </button>
          ))}
        </div>
        <div className="flex gap-2">
          {experiments.length > 0 && (
            <>
              <button
                onClick={onExportAll}
                className="px-3 py-1.5 text-xs font-semibold bg-white border border-slate-200 text-slate-600 rounded-lg hover:border-slate-300 transition-colors"
              >
                ↓ Export JSON
              </button>
              <button
                onClick={() => { if (confirm('Clear all experiments?')) onClearAll(); }}
                className="px-3 py-1.5 text-xs font-semibold bg-red-50 border border-red-200 text-red-600 rounded-lg hover:bg-red-100 transition-colors"
              >
                Clear All
              </button>
            </>
          )}
        </div>
      </div>

      {/* Empty state */}
      {filtered.length === 0 && (
        <div className="text-center py-20 space-y-3">
          <div className="text-5xl">🧪</div>
          <p className="text-slate-500 font-medium">No experiments saved yet</p>
          <p className="text-sm text-slate-400">
            Run a pair analysis or batch job, then click{' '}
            <span className="font-mono bg-slate-100 px-1 rounded">Save Experiment</span> to track it here.
          </p>
        </div>
      )}

      {/* List */}
      <div className="space-y-3">
        {filtered.map((exp) => (
          <ExperimentCard
            key={exp.id}
            exp={exp}
            onDelete={() => onDelete(exp.id)}
            onUpdateNotes={(notes) => onUpdateNotes(exp.id, notes)}
            onRerun={() =>
              onRerun({
                mode: exp.mode,
                text1: exp.text1 ?? '',
                text2: exp.text2 ?? '',
                name1: exp.modelMeta.name1,
                name2: exp.modelMeta.name2,
                baseline: exp.modelMeta.baseline,
              })
            }
          />
        ))}
      </div>
    </div>
  );
}
