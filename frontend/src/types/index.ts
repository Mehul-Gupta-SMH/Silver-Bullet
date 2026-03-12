export interface PairRequest {
  text1: string;
  text2: string;
}

export interface PredictionResult {
  prediction: number;
  probability: number;
  text1: string;
  text2: string;
}

export interface BatchResponse {
  results: PredictionResult[];
  count: number;
}

export interface HealthResponse {
  status: string;
}
