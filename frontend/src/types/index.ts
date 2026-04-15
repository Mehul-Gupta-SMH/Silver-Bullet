export type ComparisonMode = 'model-vs-model' | 'reference-vs-generated' | 'context-vs-generated';

export interface InterpretationResult {
  headline: string;
  detail: string;
  color: 'green' | 'yellow' | 'red';
}

export interface ModeConfig {
  id: ComparisonMode;
  label: string;
  emoji: string;
  tagline: string;
  description: string;
  text1Label: string;
  text2Label: string;
  text1Placeholder: string;
  text2Placeholder: string;
  interpret: (probability: number) => InterpretationResult;
}

export interface PredictionResult {
  prediction: number;
  probability: number;
  text1?: string;
  text2?: string;
}

export interface MisalignmentReason {
  label: string;
  description: string;
  severity: 'high' | 'medium' | 'low';
  signal: string;
}

export interface BreakdownResult {
  prediction: number;
  probability: number;
  sentences1: string[];
  sentences2: string[];
  alignment: number[][];
  divergent_in_1: number[];
  divergent_in_2: number[];
  min_alignment: number;
  min_alignment_pair: number[];
  feature_scores: Record<string, number>;
  misalignment_reasons: MisalignmentReason[];
}

export interface BatchResponse {
  results: { prediction: number; probability: number }[];
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  models: Record<ComparisonMode, boolean>;
}

export interface JuryQuestion {
  question: string;
  answer: boolean;
  reasoning: string;
  weight: number;
  weighted_score: number;
}

export interface JuryResult {
  score: number;
  verdict: 'faithful' | 'hallucinated';
  questions: JuryQuestion[];
  model?: string;
}

export interface CheckpointInfo {
  path: string;
  size_mb: number;
  modified_ts: number;
}

export interface ModeAdminInfo {
  loaded: boolean;
  checkpoint: CheckpointInfo | null;
}

export interface TrainingSummary {
  report_file: string;
  best_val_loss: number | null;
  best_epoch: number | null;
  total_epochs: number | null;
  timestamp: number;
}

export interface BenchmarkResult {
  benchmark: string;
  mode: string;
  n: number;
  roc_auc?: number;
  pr_auc?: number;
  accuracy?: number;
  pearson_r_human?: number;
  spearman_r_human?: number;
  pearson_r_binary?: number;
  spearman_r_binary?: number;
}

export interface CacheStats {
  table_counts: Record<string, number>;
  db_size_mb: number;
}

export interface AdminStatus {
  models: Record<string, ModeAdminInfo>;
  benchmark_results: BenchmarkResult[];
  training_summaries: Record<string, TrainingSummary>;
  cache_stats: CacheStats;
  server_time: number;
}

export interface TrainingJobStatus {
  status: 'idle' | 'running' | 'done' | 'error';
  returncode: number | null;
  started_at: number | null;
  ended_at: number | null;
  line_count: number;
}

export interface TrainingLogsResponse {
  mode: string;
  status: string;
  offset: number;
  lines: string[];
}
