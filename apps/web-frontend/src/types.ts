export type StageStatus = "done" | "running" | "ready" | "warning";

export interface PipelineStage {
  key: string;
  title: string;
  status: StageStatus;
  note: string;
}

export interface DatasetItem {
  name: string;
  task: string;
  yaml_path: string;
  data_root: string;
  splits: Record<string, number>;
  class_names: string[];
  coverage: number;
  status: string;
}

export interface RunSummary {
  run_id: string;
  dataset: string;
  task: string;
  model: string;
  epochs: number;
  status: "completed" | "running" | "failed";
  project_dir: string;
  best_checkpoint?: string | null;
  last_checkpoint?: string | null;
  metric?: number | null;
  started_at: string;
}

export interface CheckpointItem {
  path: string;
  name: string;
  dataset: string;
  task: string;
  created_at: string;
  kind: "best" | "last";
}

export interface Region {
  name: string;
  color: string;
  area_ratio: number;
  note: string;
}

export interface InferenceResult {
  dataset: string;
  image_name: string;
  image_path: string;
  mask_path: string;
  confidence: number;
  regions: Region[];
  summary: string;
}

export interface AgentAdvice {
  title: string;
  description: string;
  priority: "high" | "medium" | "low";
}

export interface AgentReport {
  report_id: string;
  dataset: string;
  scene_type: string;
  spaces: string[];
  advice: AgentAdvice[];
  circulation: string;
  summary: string;
  export_path: string;
}

export interface AgentYoloRoom {
  id: number;
  confidence: number;
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  };
  area_ratio: number;
  polygon?: number[][];
}

export interface AgentRequirements {
  family_size: string;
  has_pet: boolean;
  pet_type: string;
  style: string;
  budget: string;
  priorities: string[];
  notes: string;
}

export interface AgentAnalyzeResponse {
  record_id: string;
  session_id: string;
  image_size: {
    width: number;
    height: number;
  };
  visualization: string;
  yolo_rooms: AgentYoloRoom[];
  requirements?: Partial<AgentRequirements>;
  model?: {
    name: string;
    path: string;
    label?: string;
  };
  status: string;
}

export interface ServingModel {
  name: string;
  path: string;
  size_bytes: number;
  updated_at: string;
  is_default: boolean;
  label: string;
}

export interface FloorplanRecord {
  record_id: string;
  filename: string;
  session_id: string;
  created_at: string;
  updated_at: string;
  agent_status: "analyzing" | "done" | "error" | string;
  agent_error?: string | null;
  analysis?: Record<string, unknown> | null;
  image_size: {
    width: number;
    height: number;
  };
  rooms: AgentYoloRoom[];
  room_count: number;
  original_path: string;
  visualization_path: string;
  visualization: string;
  summary: string;
}

export interface AgentSessionResponse {
  session_id: string;
  status: "analyzing" | "done" | "error" | "unknown";
  analyses: Record<string, unknown>[];
  messages: Record<string, unknown>[];
  reasoning_steps: Record<string, unknown>[];
  analysis?: Record<string, unknown> | null;
  image_size: {
    width: number;
    height: number;
  };
  visualization: string;
  yolo_rooms: AgentYoloRoom[];
  requirements?: Partial<AgentRequirements>;
  error?: string | null;
}

export interface TrainingPoint {
  epoch: number;
  miou: number;
  loss: number;
}

export type TrainingCurvePoint = TrainingPoint;

export interface ValidationSummary {
  overall_severity: string;
  counts_by_severity: Record<string, number>;
  exit_code: number;
}

export interface ConfigTraceItem {
  source: string;
  value: string;
}

export interface OverviewPayload {
  generated_at: string;
  project_name: string;
  pipeline: PipelineStage[];
  datasets: DatasetItem[];
  runs: RunSummary[];
  checkpoints: CheckpointItem[];
  inference: InferenceResult;
  agent: AgentReport;
  metrics: {
    dataset_count: number;
    checkpoint_count: number;
    run_count: number;
    validated_dataset: string;
    training_curve: TrainingPoint[];
    validation_summary: ValidationSummary;
    config_trace: ConfigTraceItem[];
  };
}

export type JobStatus = "pending" | "running" | "completed" | "failed" | "cancelled";

export interface JobResponse {
  job_id: string;
  task: string;
  command: string[];
  status: JobStatus;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  return_code: number | null;
  progress_percent: number;
  pid: number | null;
  log_path: string | null;
  stdout_tail: string;
  stderr_tail: string;
  error: string;
  result: Record<string, unknown> | null;
}

export interface TrainJobCreate {
  model?: string;
  data?: string;
  epochs?: number;
  batch?: number;
  device?: string;
  project?: string;
  name?: string;
  task_type?: string;
  resume?: boolean;
}

export interface ValidateJobCreate {
  dataset: string;
  task_type?: string;
}

export interface TransformJobCreate {
  dataset: string;
  format: "pascal_voc" | "coco" | "yolo";
  task_type?: string;
}

export interface ConfigJobCreate {
  task: "train" | "val" | "infer";
  force?: boolean;
}

export interface EvalJobCreate {
  model: string;
  data: string;
  device?: string;
  batch?: number;
  imgsz?: number;
  split?: string;
  task_type?: string;
}

export interface ProjectStatusItem {
  name: string;
  path: string;
  kind?: string;
  updated_at?: string;
}

export interface ProjectStatus {
  root: string;
  raw_datasets: ProjectStatusItem[];
  processed_datasets: ProjectStatusItem[];
  dataset_configs: ProjectStatusItem[];
  runtime_configs: ProjectStatusItem[];
  checkpoints: CheckpointItem[];
}

export interface DetectedDatasetFormat {
  format: "coco" | "yolo" | "pascal_voc" | "unknown";
  confidence: "high" | "medium" | "low" | string;
  reasons: string[];
  candidates?: Array<{ format: string; reasons: string[] }>;
}

export interface DatasetUploadResult {
  dataset_name: string;
  raw_path: string;
  file_count: number;
  dir_count: number;
  total_bytes: number;
  detected_format: DetectedDatasetFormat;
  next_step: string;
}

export interface EvaluationAsset {
  name: string;
  path: string;
  url: string;
  updated_at: string;
}

export interface EvaluationMetrics {
  map50: number | null;
  map50_95: number | null;
  precision: number | null;
  recall: number | null;
  fitness: number | null;
}

export interface EvaluationReportSummary {
  report_id: string;
  name: string;
  path: string;
  updated_at: string;
  metrics: EvaluationMetrics;
  asset_count: number;
}

export interface EvaluationReportDetail extends EvaluationReportSummary {
  assets: {
    confusion: EvaluationAsset[];
    curves: EvaluationAsset[];
    samples: EvaluationAsset[];
  };
}

export interface DatasetProfile {
  name: string;
  task: string;
  yaml_path: string;
  data_root: string;
  splits: Record<string, number>;
  class_names: string[];
  coverage: number;
  status: string;
}
