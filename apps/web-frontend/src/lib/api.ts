import type {
  DatasetProfile,
  JobResponse,
  OverviewPayload,
  RunSummary,
  CheckpointItem,
  AgentAnalyzeResponse,
  AgentRequirements,
  AgentSessionResponse,
  FloorplanRecord,
  ConfigJobCreate,
  EvalJobCreate,
  EvaluationReportDetail,
  EvaluationReportSummary,
  ProjectStatus,
  TrainJobCreate,
  TrainingCurvePoint,
  TransformJobCreate,
  DatasetUploadResult,
  ServingModel,
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000/api";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    throw new Error(`${path} returned ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function fetchOverview(): Promise<OverviewPayload> {
  return apiFetch<OverviewPayload>("/overview");
}

export async function fetchDatasets(): Promise<DatasetProfile[]> {
  const data = await apiFetch<{ items: DatasetProfile[] }>("/datasets");
  return data.items;
}

export async function fetchProjectStatus(): Promise<ProjectStatus> {
  return apiFetch<ProjectStatus>("/project/status");
}

export async function createDataset(body: {
  name: string;
  class_names: string[];
  train?: number;
  val?: number;
  test?: number;
}): Promise<DatasetProfile> {
  return apiFetch<DatasetProfile>("/datasets", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function uploadDatasetArchive(file: File, datasetName: string): Promise<DatasetUploadResult> {
  const form = new FormData();
  form.append("dataset_name", datasetName);
  form.append("file", file);
  return apiFetch<DatasetUploadResult>("/datasets/upload", { method: "POST", body: form });
}

export async function updateDataset(
  name: string,
  body: {
    class_names?: string[];
    train?: number;
    val?: number;
    test?: number;
  }
): Promise<DatasetProfile> {
  return apiFetch<DatasetProfile>(`/datasets/${name}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function deleteDataset(name: string): Promise<{ success: boolean; name: string }> {
  return apiFetch<{ success: boolean; name: string }>(`/datasets/${name}`, {
    method: "DELETE",
  });
}

export async function fetchTrainingRuns(): Promise<RunSummary[]> {
  const data = await apiFetch<{ items: RunSummary[] }>("/training/runs");
  return data.items;
}

export async function fetchCheckpoints(): Promise<CheckpointItem[]> {
  const data = await apiFetch<{ items: CheckpointItem[] }>("/checkpoints");
  return data.items;
}

export async function fetchJobs(): Promise<JobResponse[]> {
  return apiFetch<JobResponse[]>("/jobs");
}

export async function fetchJob(jobId: string): Promise<JobResponse> {
  return apiFetch<JobResponse>(`/jobs/${jobId}`);
}

export async function fetchTrainingCurves(): Promise<Array<{
  run_id: string; dataset: string; task: string; model: string;
  epochs: number; status: string; metric: number | null;
  best_checkpoint: string | null; last_checkpoint: string | null;
  curve: TrainingCurvePoint[];
}>> {
  return apiFetch("/training/curves/latest");
}

export async function fetchEvaluationReports(): Promise<EvaluationReportSummary[]> {
  const data = await apiFetch<{ items: EvaluationReportSummary[] }>("/evaluation/reports");
  return data.items;
}

export async function fetchEvaluationReport(reportId: string): Promise<EvaluationReportDetail> {
  return apiFetch<EvaluationReportDetail>(`/evaluation/reports/${reportId}`);
}

export async function fetchServingModels(): Promise<ServingModel[]> {
  const data = await apiFetch<{ items: ServingModel[] }>("/models/serving");
  return data.items;
}

export async function submitTrainJob(body: TrainJobCreate): Promise<JobResponse> {
  return apiFetch<JobResponse>("/jobs/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function submitInitJob(): Promise<JobResponse> {
  return apiFetch<JobResponse>("/jobs/init", { method: "POST" });
}

export async function submitTransformJob(body: TransformJobCreate): Promise<JobResponse> {
  return apiFetch<JobResponse>("/jobs/transform", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function submitValidateJob(body: { dataset: string; task_type?: string }): Promise<JobResponse> {
  return apiFetch<JobResponse>("/jobs/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function submitConfigJob(body: ConfigJobCreate): Promise<JobResponse> {
  return apiFetch<JobResponse>("/jobs/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function submitEvaluateJob(body: EvalJobCreate): Promise<JobResponse> {
  return apiFetch<JobResponse>("/jobs/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function cancelJob(jobId: string): Promise<JobResponse> {
  return apiFetch<JobResponse>(`/jobs/${jobId}`, { method: "DELETE" });
}

export async function uploadFloorplan(file: File): Promise<{
  success: boolean; filename: string; path: string; message: string;
}> {
  const form = new FormData();
  form.append("file", file);
  return apiFetch("/upload/floorplan", { method: "POST", body: form });
}

export async function analyzeFloorplan(
  file: File,
  requirements?: Partial<AgentRequirements>,
  modelName?: string
): Promise<AgentAnalyzeResponse> {
  const form = new FormData();
  form.append("file", file);
  if (requirements) {
    form.append("requirements", JSON.stringify(requirements));
  }
  if (modelName) {
    form.append("model_name", modelName);
  }
  return apiFetch<AgentAnalyzeResponse>("/analyze", { method: "POST", body: form });
}

export async function fetchAgentSession(sessionId: string): Promise<AgentSessionResponse> {
  return apiFetch<AgentSessionResponse>(`/session/${sessionId}`);
}

export async function sendAgentChat(sessionId: string, message: string): Promise<{ response: string }> {
  const form = new FormData();
  form.append("message", message);
  return apiFetch<{ response: string }>(`/chat/${sessionId}`, { method: "POST", body: form });
}

export async function streamAgentChat(
  sessionId: string,
  message: string,
  onToken: (token: string) => void
): Promise<void> {
  const form = new FormData();
  form.append("message", message);
  const res = await fetch(`${API_BASE}/chat/${sessionId}/stream`, { method: "POST", body: form });
  if (!res.ok) {
    throw new Error(`/chat/${sessionId}/stream returned ${res.status}`);
  }
  if (!res.body) {
    throw new Error("当前浏览器不支持流式读取。");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (!data) continue;
      if (data === "[DONE]") return;
      try {
        const payload = JSON.parse(data) as { token?: string };
        if (payload.token) onToken(payload.token);
      } catch {
        onToken(data);
      }
    }
  }
}

export async function fetchFloorplans(): Promise<FloorplanRecord[]> {
  const data = await apiFetch<{ items: FloorplanRecord[] }>("/floorplans");
  return data.items;
}

export async function fetchFloorplan(recordId: string): Promise<FloorplanRecord> {
  return apiFetch<FloorplanRecord>(`/floorplans/${recordId}`);
}

export async function deleteFloorplan(recordId: string): Promise<{ success: boolean; record_id: string }> {
  return apiFetch<{ success: boolean; record_id: string }>(`/floorplans/${recordId}`, {
    method: "DELETE",
  });
}
