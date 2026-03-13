import { useState } from 'react';

interface Props {
  defaultName: string;
  onSave: (name: string, notes: string) => void;
}

export function SaveExperimentForm({ defaultName, onSave }: Props) {
  const [name, setName] = useState(defaultName);
  const [notes, setNotes] = useState('');
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    if (!name.trim()) return;
    onSave(name.trim(), notes.trim());
    setSaved(true);
  };

  if (saved) {
    return (
      <div className="flex items-center gap-2 text-emerald-700 bg-emerald-50 border border-emerald-200 rounded-xl px-4 py-2.5 text-sm font-medium">
        <span>✓</span>
        <span>Saved to Experiments</span>
      </div>
    );
  }

  return (
    <div className="bg-slate-50 border border-slate-200 rounded-2xl p-4 space-y-3">
      <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest">
        Save to Experiments
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Experiment name"
          className="px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-400 bg-white"
        />
        <input
          type="text"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Notes (optional)"
          className="px-3 py-2 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-violet-400 bg-white"
        />
      </div>
      <button
        onClick={handleSave}
        disabled={!name.trim()}
        className="flex items-center gap-2 px-4 py-2 bg-slate-800 text-white text-sm font-semibold rounded-lg hover:bg-slate-900 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
      >
        🧪 Save Experiment
      </button>
    </div>
  );
}
