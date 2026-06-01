const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export type HealthResponse = {
  status: string;
  checkpoint_exists: boolean;
  metrics_exists: boolean;
  device: string;
};

export type ProbabilityItem = {
  class_name: string;
  probability: number;
};

export type PredictionResponse = {
  predicted_class: string;
  predicted_index: number;
  confidence: number;
  probabilities: ProbabilityItem[];
  gradcam_image: string;
  model_name: string;
  image_size: number;
  device: string;
};

export type MetricsResponse = {
  available: boolean;
  model_name?: string;
  image_size?: number;
  classes?: string[];
  test?: {
    accuracy?: number;
    macro_f1?: number;
    loss?: number;
    confusion_matrix?: number[][];
    per_class?: Record<string, { precision: number; recall: number; f1: number; support: number }>;
  };
};

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // Keep the HTTP status text when the body is not JSON.
    }
    throw new Error(detail || "请求失败");
  }
  return response.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  return readJson<HealthResponse>(await fetch(`${API_BASE_URL}/health`));
}

export async function getMetrics(): Promise<MetricsResponse> {
  return readJson<MetricsResponse>(await fetch(`${API_BASE_URL}/metrics`));
}

export async function predictImage(file: File): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file);
  return readJson<PredictionResponse>(
    await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: form,
    }),
  );
}
