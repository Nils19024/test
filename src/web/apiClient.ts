import type { TrainingConfig } from "../ml/train";
import type { TrainingStatus } from "../server/trainingManager";

async function post<T>(url: string, body?: unknown): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export const api = {
  startTraining(config: Partial<TrainingConfig>) {
    return post<TrainingStatus>("/api/training/start", config);
  },
  stopTraining() {
    return post<TrainingStatus>("/api/training/stop");
  },
  async trainingStatus() {
    const response = await fetch("/api/training/status");
    return response.json() as Promise<TrainingStatus>;
  },
  async diagnostics() {
    const response = await fetch("/api/training/diagnostics");
    if (!response.ok) throw new Error(await response.text());
    return response.json();
  },
  benchmark() {
    return post<unknown>("/api/benchmark/start");
  },
  exportModel() {
    return post<{ path: string }>("/api/models/export-browser");
  }
};
