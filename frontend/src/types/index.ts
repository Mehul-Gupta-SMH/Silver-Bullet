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

export interface BreakdownResult {
  prediction: number;
  probability: number;
  sentences1: string[];
  sentences2: string[];
  alignment: number[][];
  divergent_in_1: number[];
  divergent_in_2: number[];
  feature_scores: Record<string, number>;
}

export interface BatchResponse {
  results: { prediction: number; probability: number }[];
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  models: Record<ComparisonMode, boolean>;
}
