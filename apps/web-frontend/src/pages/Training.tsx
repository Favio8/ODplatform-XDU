import { useEffect, useState, useCallback } from "react";
import type { EChartsOption } from "echarts";
import { EChart } from "../components/EChart";
import {
  fetchCheckpoints,
  fetchDatasets,
  fetchEvaluationReport,
  fetchEvaluationReports,
  fetchJobs,
  fetchPretrainedModels,
  fetchTrainingCurves,
  submitEvaluateJob,
  submitTrainJob,
  cancelJob,
} from "../lib/api";
import { stripAnsi } from "../lib/terminal";
import type { CheckpointItem, DatasetProfile, EvaluationAsset, EvaluationReportDetail, EvaluationReportSummary, JobResponse, ServingModel } from "../types";

const API_ORIGIN = (import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000/api").replace(/\/api\/?$/, "");

const STATUS_CONFIG: Record<string, { label: string; badge: string; icon: string }> = {
  pending:   { label: "等待中",   badge: "bg-[var(--parchment)] text-[var(--warm-gray)] border-[var(--border)]", icon: "fa-clock" },
  running:   { label: "进行中",   badge: "bg-[var(--terracotta-pale)] text-[var(--terracotta)] border-[var(--terracotta-light)]", icon: "fa-spinner fa-spin" },
  completed: { label: "已完成",  badge: "bg-[var(--sage-pale)] text-[var(--sage)] border-[var(--sage-light)]", icon: "fa-check" },
  failed:    { label: "失败",    badge: "bg-red-50 text-red-500 border border-red-200", icon: "fa-xmark" },
  cancelled:  { label: "已取消",  badge: "bg-amber-50 text-amber-600 border border-amber-200", icon: "fa-ban" },
};

const TASK_LABELS: Record<string, string> = {
  train: "训练",
  evaluate: "评估",
  validate: "质检",
  transform: "转换",
  config: "配置",
  init: "初始化",
  infer: "推理",
};

function formatTrainingModelLoadError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error ?? "");
  if (message.includes("/models/pretrained returned 404")) {
    return "当前 web-backend 尚未加载 /models/pretrained 接口，请重启后端后刷新页面。";
  }
  if (message) {
    return `加载预训练权重失败：${message}`;
  }
  return "加载预训练权重失败，请检查 web-backend 是否正常运行。";
}

function summarizeHistoryResult(job: JobResponse): { text: string; tone: "default" | "success" | "danger" } {
  if (job.status === "failed" || job.status === "cancelled") {
    return {
      text: stripAnsi(job.error || job.stderr_tail || job.stdout_tail).slice(0, 260) || "任务执行失败。",
      tone: "danger",
    };
  }

  const result = job.result ?? {};

  if (job.task === "train" && job.status === "completed") {
    const best = typeof result.best_checkpoint === "string" ? result.best_checkpoint.split(/[\\/]/).pop() : null;
    const runDir = typeof result.run_dir === "string" ? result.run_dir.split(/[\\/]/).pop() : null;
    return {
      text: [runDir ? `训练完成 · ${runDir}` : "训练完成", best ? `best: ${best}` : "", "可在上方 Checkpoint 管理与评估报告中继续查看结果。"]
        .filter(Boolean)
        .join(" · "),
      tone: "success",
    };
  }

  if (job.task === "validate" && job.status === "completed") {
    const report = result.report as {
      overall_severity?: string;
      counts_by_severity?: Record<string, number>;
      snapshot?: { total_images?: number };
    } | undefined;
    if (report) {
      const counts = report.counts_by_severity ?? {};
      return {
        text: `质检${report.overall_severity === "PASS" ? "通过" : "完成"} · PASS ${counts.PASS ?? 0} · WARNING ${counts.WARNING ?? 0} · ERROR ${counts.ERROR ?? 0} · 图像 ${report.snapshot?.total_images ?? "-" } 张`,
        tone: "success",
      };
    }
  }

  if (job.task === "transform" && job.status === "completed") {
    return { text: "数据转换已完成，可继续执行数据质检与配置生成。", tone: "success" };
  }

  if (job.task === "evaluate" && job.status === "completed") {
    const outputDir = typeof result.output_dir === "string" ? result.output_dir.split(/[\\/]/).pop() : null;
    return {
      text: outputDir ? `模型评估完成 · 输出目录 ${outputDir}` : "模型评估完成，可在评估报告区域查看图表。",
      tone: "success",
    };
  }

  if (job.task === "config" && job.status === "completed") {
    const count = Array.isArray(result.configs) ? result.configs.length : 0;
    return { text: count > 0 ? `运行配置已生成 · ${count} 个文件` : "运行配置已生成。", tone: "success" };
  }

  return {
    text: stripAnsi(job.stdout_tail || job.stderr_tail).slice(0, 260) || "任务已完成。",
    tone: "default",
  };
}

function findCurveRunIdByCheckpoint(
  checkpointPath: string,
  curves: Array<{ run_id: string; best_checkpoint: string | null; last_checkpoint: string | null }>
): string | null {
  const normalizedCheckpointPath = checkpointPath.replace(/\\/g, "/").toLowerCase();
  const direct = curves.find((item) =>
    item.best_checkpoint?.replace(/\\/g, "/").toLowerCase() === normalizedCheckpointPath ||
    item.last_checkpoint?.replace(/\\/g, "/").toLowerCase() === normalizedCheckpointPath
  );
  if (direct) return direct.run_id;

  const fileName = checkpointPath.split(/[\\/]/).pop()?.toLowerCase() ?? "";
  if (!fileName) return null;

  const inferredRunId = inferRunIdFromCheckpointFileName(fileName);
  if (inferredRunId) {
    const exactRun = curves.find((item) => item.run_id.toLowerCase() === inferredRunId);
    if (exactRun) return exactRun.run_id;
  }

  const loose = curves.find((item) => fileName.startsWith(`${item.run_id.toLowerCase()}-`) || fileName.includes(item.run_id.toLowerCase()));
  return loose?.run_id ?? null;
}

function inferRunIdFromCheckpointFileName(fileName: string): string | null {
  const stem = fileName.replace(/\.pt$/i, "");
  const trimmedKind = stem.replace(/-(best|last)$/i, "");
  const match = trimmedKind.match(/^(.*?)-\d{8}-\d{6}-.+$/);
  if (match?.[1]) {
    return match[1].toLowerCase();
  }
  return null;
}

function JobDetail({ job }: { job: JobResponse }) {
  const summary = summarizeHistoryResult(job);
  return (
    <div className="space-y-2">
      {job.command.length > 0 && (
        <p className="font-mono text-[11px] text-[var(--warm-gray)] break-all bg-[var(--ivory)] rounded-lg px-3 py-2 leading-5">
          {job.command.join(" ")}
        </p>
      )}
      {summary.text && (
        <p
          className={`text-[11px] line-clamp-3 break-words leading-5 ${
            summary.tone === "danger"
              ? "text-red-500"
              : summary.tone === "success"
                ? "text-[var(--sage)]"
                : "text-[var(--mid-gray)]"
          }`}
        >
          {summary.text}
        </p>
      )}
    </div>
  );
}

function summarizeTrainingLog(job: JobResponse): string {
  const stdout = stripAnsi(job.stdout_tail);
  const stderr = stripAnsi(job.stderr_tail);
  const combined = [stderr, stdout].filter(Boolean).join("\n");
  const lines = combined
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  return lines.slice(-8).join("\n");
}

function LiveTrainingPanel({
  job,
  onCancel,
}: {
  job: JobResponse;
  onCancel: (jobId: string) => void;
}) {
  const cfg = STATUS_CONFIG[job.status] ?? STATUS_CONFIG.pending;
  const logText = summarizeTrainingLog(job);

  return (
    <div className="mt-6 rounded-3xl border border-[var(--terracotta-light)] bg-[var(--terracotta-pale)]/35 p-5">
      <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-wider text-[var(--terracotta)] mb-1">实时训练进度</p>
          <h3 className="text-base font-bold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>
            {job.task} · {cfg.label}
          </h3>
          <p className="mt-1 text-xs text-[var(--mid-gray)] font-mono break-all">{job.command.join(" ")}</p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`px-3 py-1 rounded-full text-xs font-medium border ${cfg.badge}`}>{cfg.label}</span>
          {(job.status === "pending" || job.status === "running") && (
            <button
              type="button"
              onClick={() => onCancel(job.job_id)}
              className="px-3 py-1.5 rounded-xl text-xs text-red-500 border border-red-200 hover:bg-red-50 transition-all"
            >
              取消任务
            </button>
          )}
        </div>
      </div>

      <div className="mt-4 flex items-center gap-3">
        <div className="flex-1 h-2 rounded-full bg-white/70 overflow-hidden">
          <div
            className="h-full rounded-full bg-[var(--terracotta)] transition-all duration-500"
            style={{ width: `${Math.max(job.progress_percent || 0, job.status === "running" ? 8 : 0)}%` }}
          />
        </div>
        <span className="text-sm font-semibold text-[var(--terracotta)] min-w-12 text-right">
          {job.progress_percent || 0}%
        </span>
      </div>

      <div className="mt-4 grid grid-cols-1 lg:grid-cols-3 gap-3 text-xs">
        <div className="rounded-2xl bg-[var(--warm-white)] border border-[var(--border)] p-3">
          <p className="text-[var(--mid-gray)] uppercase tracking-wider">任务 ID</p>
          <p className="mt-2 font-mono text-[var(--charcoal)] break-all">{job.job_id}</p>
        </div>
        <div className="rounded-2xl bg-[var(--warm-white)] border border-[var(--border)] p-3">
          <p className="text-[var(--mid-gray)] uppercase tracking-wider">后台日志目录</p>
          <p className="mt-2 font-mono text-[var(--charcoal)] break-all">{job.log_path ?? "暂无"}</p>
        </div>
        <div className="rounded-2xl bg-[var(--warm-white)] border border-[var(--border)] p-3">
          <p className="text-[var(--mid-gray)] uppercase tracking-wider">说明</p>
          <p className="mt-2 text-[var(--warm-gray)] leading-6">
            Web 训练任务会在后台进程中运行，因此不会自动附着到你当前 IDE 终端。
          </p>
        </div>
      </div>

      <div className="mt-4 rounded-2xl bg-[var(--charcoal)] text-[#f3ede5] border border-[rgba(255,255,255,0.08)] overflow-hidden">
        <div className="px-4 py-2 border-b border-[rgba(255,255,255,0.08)] text-xs uppercase tracking-wider text-[#dcc9ba]">
          实时日志预览
        </div>
        <pre className="px-4 py-3 text-[11px] leading-5 whitespace-pre-wrap break-words max-h-64 overflow-auto font-mono">
          {logText || "训练任务已提交，正在等待首批日志输出..."}
        </pre>
      </div>
    </div>
  );
}

function LiveEvaluatePanel({
  job,
  onCancel,
}: {
  job: JobResponse;
  onCancel: (jobId: string) => void;
}) {
  const cfg = STATUS_CONFIG[job.status] ?? STATUS_CONFIG.pending;
  const logText = summarizeTrainingLog(job);

  return (
    <div className="mt-6 rounded-3xl border border-[var(--sage-light)] bg-[var(--sage-pale)]/35 p-5">
      <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-wider text-[var(--sage)] mb-1">实时评估进度</p>
          <h3 className="text-base font-bold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>
            {job.task} · {cfg.label}
          </h3>
          <p className="mt-1 text-xs text-[var(--mid-gray)] font-mono break-all">{job.command.join(" ")}</p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`px-3 py-1 rounded-full text-xs font-medium border ${cfg.badge}`}>{cfg.label}</span>
          {(job.status === "pending" || job.status === "running") && (
            <button
              type="button"
              onClick={() => onCancel(job.job_id)}
              className="px-3 py-1.5 rounded-xl text-xs text-red-500 border border-red-200 hover:bg-red-50 transition-all"
            >
              取消任务
            </button>
          )}
        </div>
      </div>

      <div className="mt-4 flex items-center gap-3">
        <div className="flex-1 h-2 rounded-full bg-white/70 overflow-hidden">
          <div
            className="h-full rounded-full bg-[var(--sage)] transition-all duration-500"
            style={{ width: `${Math.max(job.progress_percent || 0, job.status === "running" ? 8 : 0)}%` }}
          />
        </div>
        <span className="text-sm font-semibold text-[var(--sage)] min-w-12 text-right">
          {job.progress_percent || 0}%
        </span>
      </div>

      <div className="mt-4 grid grid-cols-1 lg:grid-cols-3 gap-3 text-xs">
        <div className="rounded-2xl bg-[var(--warm-white)] border border-[var(--border)] p-3">
          <p className="text-[var(--mid-gray)] uppercase tracking-wider">任务 ID</p>
          <p className="mt-2 font-mono text-[var(--charcoal)] break-all">{job.job_id}</p>
        </div>
        <div className="rounded-2xl bg-[var(--warm-white)] border border-[var(--border)] p-3">
          <p className="text-[var(--mid-gray)] uppercase tracking-wider">后台日志目录</p>
          <p className="mt-2 font-mono text-[var(--charcoal)] break-all">{job.log_path ?? "暂无"}</p>
        </div>
        <div className="rounded-2xl bg-[var(--warm-white)] border border-[var(--border)] p-3">
          <p className="text-[var(--mid-gray)] uppercase tracking-wider">说明</p>
          <p className="mt-2 text-[var(--warm-gray)] leading-6">
            评估任务同样在 Web 后台进程中运行，完成后会自动出现在评估报告区域。
          </p>
        </div>
      </div>

      <div className="mt-4 rounded-2xl bg-[var(--charcoal)] text-[#f3ede5] border border-[rgba(255,255,255,0.08)] overflow-hidden">
        <div className="px-4 py-2 border-b border-[rgba(255,255,255,0.08)] text-xs uppercase tracking-wider text-[#d8e3d2]">
          实时日志预览
        </div>
        <pre className="px-4 py-3 text-[11px] leading-5 whitespace-pre-wrap break-words max-h-64 overflow-auto font-mono">
          {logText || "评估任务已提交，正在等待首批日志输出..."}
        </pre>
      </div>
    </div>
  );
}

function formatMetric(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "暂无";
  return value <= 1 ? `${(value * 100).toFixed(1)}%` : value.toFixed(4);
}

function assetUrl(asset: EvaluationAsset) {
  if (asset.url.startsWith("http://") || asset.url.startsWith("https://")) return asset.url;
  return `${API_ORIGIN}${asset.url}`;
}

function AssetPreview({ asset, label }: { asset?: EvaluationAsset; label: string }) {
  if (!asset) {
    return (
      <div className="rounded-2xl border border-dashed border-[var(--border)] bg-[var(--ivory)] p-6 text-sm text-[var(--mid-gray)] min-h-48 flex items-center justify-center">
        {label} 暂无
      </div>
    );
  }

  return (
    <a
      href={assetUrl(asset)}
      target="_blank"
      rel="noreferrer"
      className="group block rounded-2xl border border-[var(--border)] bg-[var(--ivory)] p-3 hover:border-[var(--terracotta-light)] hover:-translate-y-0.5 transition-all"
    >
      <div className="aspect-[4/3] overflow-hidden rounded-xl bg-[var(--warm-white)] border border-[var(--border)]">
        <img src={assetUrl(asset)} alt={asset.name} className="h-full w-full object-contain group-hover:scale-[1.015] transition-transform duration-300" />
      </div>
      <div className="mt-3 flex items-center justify-between gap-2">
        <p className="text-xs font-semibold text-[var(--charcoal)]">{label}</p>
        <p className="text-[10px] text-[var(--mid-gray)] font-mono truncate max-w-36" title={asset.name}>{asset.name}</p>
      </div>
    </a>
  );
}

function AssetGallery({ title, assets, empty }: { title: string; assets: EvaluationAsset[]; empty: string }) {
  return (
    <div>
      <h3 className="text-sm font-semibold text-[var(--charcoal)] mb-3">{title}</h3>
      {assets.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-[var(--border)] bg-[var(--ivory)] p-6 text-center text-sm text-[var(--mid-gray)]">
          {empty}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {assets.map((asset) => (
            <a
              key={asset.path}
              href={assetUrl(asset)}
              target="_blank"
              rel="noreferrer"
              className="group rounded-2xl border border-[var(--border)] bg-[var(--ivory)] p-3 hover:border-[var(--terracotta-light)] transition-all"
            >
              <div className="aspect-[4/3] overflow-hidden rounded-xl bg-[var(--warm-white)] border border-[var(--border)]">
                <img src={assetUrl(asset)} alt={asset.name} className="h-full w-full object-contain group-hover:scale-[1.01] transition-transform" />
              </div>
              <p className="mt-2 text-xs text-[var(--warm-gray)] font-mono truncate" title={asset.name}>{asset.name}</p>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

function EvaluationReportsPanel({
  reports,
  selectedReportId,
  selectedReport,
  loading,
  onSelect,
  onRefresh,
}: {
  reports: EvaluationReportSummary[];
  selectedReportId: string;
  selectedReport: EvaluationReportDetail | null;
  loading: boolean;
  onSelect: (reportId: string) => void;
  onRefresh: () => void;
}) {
  const [activeView, setActiveView] = useState<"overview" | "curves" | "samples">("overview");
  const metrics = selectedReport?.metrics;
  const metricItems = [
    { label: "mAP50", value: metrics?.map50 },
    { label: "mAP50-95", value: metrics?.map50_95 },
    { label: "Precision", value: metrics?.precision },
    { label: "Recall", value: metrics?.recall },
    { label: "Fitness", value: metrics?.fitness },
  ];
  const confusion = selectedReport?.assets.confusion ?? [];
  const curves = selectedReport?.assets.curves ?? [];
  const samples = selectedReport?.assets.samples ?? [];
  const primaryConfusion = confusion.find((asset) => asset.name.includes("normalized")) ?? confusion[0];
  const primaryCurve = curves.find((asset) => asset.name.toLowerCase().includes("maskpr")) ?? curves.find((asset) => asset.name.toLowerCase().includes("pr_curve")) ?? curves[0];
  const primarySample = samples.find((asset) => asset.name.includes("pred")) ?? samples[0];
  const tabs = [
    { key: "overview", label: "核心概览", icon: "fa-compass" },
    { key: "curves", label: `曲线 ${curves.length}`, icon: "fa-chart-line" },
    { key: "samples", label: `样例 ${samples.length}`, icon: "fa-images" },
  ] as const;

  return (
    <div className="card overflow-hidden">
      <div className="p-8 pb-6 bg-gradient-to-br from-[var(--warm-white)] to-[var(--ivory)] border-b border-[var(--border)]">
      <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>评估报告</h2>
          <p className="text-xs text-[var(--mid-gray)] mt-1">先看核心指标和代表图，需要深挖时再切换曲线与预测样例。</p>
        </div>
        <div className="flex gap-2">
          <select
            value={selectedReportId}
            onChange={(event) => onSelect(event.target.value)}
            className="min-w-52 px-3 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--warm-white)] text-sm text-[var(--charcoal)] focus:outline-none focus:border-[var(--terracotta)] transition-all"
          >
            {reports.length === 0 && <option value="">暂无报告</option>}
            {reports.map((report) => (
              <option key={report.report_id} value={report.report_id}>
                {report.name} · {report.updated_at.replace("T", " ").slice(0, 16)}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={onRefresh}
            className="px-4 py-2.5 rounded-xl border border-[var(--border)] text-sm text-[var(--warm-gray)] hover:text-[var(--charcoal)] hover:border-[var(--warm-gray)] transition-all"
          >
            <i className="fa-solid fa-rotate-right text-xs mr-2" />
            刷新
          </button>
        </div>
      </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-10">
          <div className="w-6 h-6 border-2 border-[var(--terracotta)] border-t-transparent rounded-full animate-spin" />
        </div>
      ) : !selectedReport ? (
        <div className="rounded-2xl border border-dashed border-[var(--border)] bg-[var(--ivory)] p-10 text-center">
          <i className="fa-solid fa-chart-simple text-3xl text-[var(--light-gray)] mb-3" />
          <p className="text-sm text-[var(--mid-gray)]">暂无评估报告。发起 odp-val 后，这里会展示结果图片和指标。</p>
        </div>
      ) : (
        <div className="p-8 space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {metricItems.map((item) => (
              <div key={item.label} className="rounded-2xl bg-[var(--ivory)] border border-[var(--border)] p-4">
                <p className="text-[10px] text-[var(--mid-gray)] uppercase tracking-wider">{item.label}</p>
                <p className="mt-2 text-xl font-semibold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>{formatMetric(item.value)}</p>
              </div>
            ))}
          </div>

          <div className="flex flex-wrap gap-2 rounded-2xl bg-[var(--ivory)] border border-[var(--border)] p-1.5">
            {tabs.map((tab) => (
              <button
                key={tab.key}
                type="button"
                onClick={() => setActiveView(tab.key)}
                className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                  activeView === tab.key
                    ? "bg-[var(--warm-white)] text-[var(--terracotta)] shadow-sm"
                    : "text-[var(--warm-gray)] hover:text-[var(--charcoal)]"
                }`}
              >
                <i className={`fa-solid ${tab.icon} text-xs mr-2`} />
                {tab.label}
              </button>
            ))}
          </div>

          {activeView === "overview" && (
            <div className="space-y-5">
              <div className="rounded-3xl border border-[var(--border)] bg-[var(--parchment)]/45 p-5">
                <div className="flex items-start gap-3">
                  <div className="w-9 h-9 rounded-2xl bg-[var(--sage-pale)] text-[var(--sage)] flex items-center justify-center">
                    <i className="fa-solid fa-lightbulb text-sm" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-[var(--charcoal)]">评估阅读顺序</h3>
                    <p className="mt-1 text-xs leading-6 text-[var(--warm-gray)]">
                      先用 mAP50 / Recall 判断整体可用性，再看混淆矩阵定位易混类别，最后用预测样例检查分割边界是否符合户型图语义。
                    </p>
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <AssetPreview asset={primaryConfusion} label="混淆矩阵" />
                <AssetPreview asset={primaryCurve} label="Mask PR 曲线" />
                <AssetPreview asset={primarySample} label="预测样例" />
              </div>
            </div>
          )}

          {activeView === "curves" && (
            <AssetGallery title="PR / P / R / F1 曲线" assets={curves} empty="未找到曲线图片。" />
          )}

          {activeView === "samples" && (
            <AssetGallery title="预测样例" assets={samples.slice(0, 8)} empty="未找到 val_batch 预测样例。" />
          )}
          </div>
      )}
    </div>
  );
}

export function Training() {
  const CHECKPOINTS_PER_PAGE = 6;
  const [jobs, setJobs] = useState<JobResponse[]>([]);
  const [datasets, setDatasets] = useState<DatasetProfile[]>([]);
  const [checkpoints, setCheckpoints] = useState<CheckpointItem[]>([]);
  const [trainingModels, setTrainingModels] = useState<ServingModel[]>([]);
  const [trainingModelsError, setTrainingModelsError] = useState<string | null>(null);
  const [curves, setCurves] = useState<Array<{
    run_id: string;
    dataset: string;
    task: string;
    model: string;
    epochs: number;
    status: string;
    metric: number | null;
    best_checkpoint: string | null;
    last_checkpoint: string | null;
    curve: Array<{ epoch: number; miou: number; loss: number }>;
  }>>([]);
  const [evaluationReports, setEvaluationReports] = useState<EvaluationReportSummary[]>([]);
  const [selectedReportId, setSelectedReportId] = useState("");
  const [selectedReport, setSelectedReport] = useState<EvaluationReportDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [curvesLoading, setCurvesLoading] = useState(true);
  const [reportsLoading, setReportsLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitNotice, setSubmitNotice] = useState<string | null>(null);
  const [recentTrainJobId, setRecentTrainJobId] = useState<string>("");
  const [evalNotice, setEvalNotice] = useState<string | null>(null);
  const [recentEvaluateJobId, setRecentEvaluateJobId] = useState<string>("");
  const [selectedCurveRunId, setSelectedCurveRunId] = useState<string>("");
  const [checkpointPage, setCheckpointPage] = useState(1);
  const [form, setForm] = useState({
    model: "",
    data: "room_separation_3",
    epochs: 10,
    batch: 8,
    device: "cpu",
    resume: false,
  });
  const [evalForm, setEvalForm] = useState({
    model: "",
    data: "room_separation_3",
    device: "cpu",
    batch: 8,
    split: "val",
  });

  const loadData = useCallback(async () => {
    const [jobsResult, curvesResult, checkpointsResult, datasetsResult, reportsResult, modelsResult] = await Promise.allSettled([
      fetchJobs(),
      fetchTrainingCurves(),
      fetchCheckpoints(),
      fetchDatasets(),
      fetchEvaluationReports(),
      fetchPretrainedModels(),
    ]);

    if (jobsResult.status === "fulfilled") {
      setJobs(jobsResult.value);
    }

    if (curvesResult.status === "fulfilled") {
      const curvesData = curvesResult.value;
      setCurves(curvesData);
      setSelectedCurveRunId((current) => {
        if (current && curvesData.some((item) => item.run_id === current)) return current;
        return curvesData[0]?.run_id ?? "";
      });
    }

    if (checkpointsResult.status === "fulfilled") {
      const segmentCheckpoints = checkpointsResult.value.filter((item) => item.task === "segment");
      setCheckpoints(segmentCheckpoints);
      if (!evalForm.model && segmentCheckpoints.length > 0) {
        const preferred = segmentCheckpoints.find((item) => item.kind === "best") ?? segmentCheckpoints[0];
        setEvalForm((current) => current.model ? current : { ...current, model: preferred.path });
      }
    }

    if (datasetsResult.status === "fulfilled") {
      setDatasets(datasetsResult.value);
    }

    if (modelsResult.status === "fulfilled") {
      const modelData = modelsResult.value;
      setTrainingModelsError(null);
      setTrainingModels(modelData);
      setForm((current) => {
        if (current.model && modelData.some((item) => item.name === current.model)) {
          return current;
        }
        const preferred = modelData.find((item) => item.is_default) ?? modelData[0];
        return preferred ? { ...current, model: preferred.name } : current;
      });
    } else {
      setTrainingModels([]);
      setTrainingModelsError(formatTrainingModelLoadError(modelsResult.reason));
    }

    if (reportsResult.status === "fulfilled") {
      const reportData = reportsResult.value;
      setEvaluationReports(reportData);
      setSelectedReportId((current) => {
        if (current && reportData.some((report) => report.report_id === current)) return current;
        return reportData[0]?.report_id ?? "";
      });
    }
  }, [evalForm.model]);

  useEffect(() => {
    loadData().finally(() => { setLoading(false); setCurvesLoading(false); setReportsLoading(false); });
    const id = setInterval(loadData, 3000);
    return () => clearInterval(id);
  }, [loadData]);

  useEffect(() => {
    if (!selectedReportId) {
      setSelectedReport(null);
      return;
    }
    setReportsLoading(true);
    fetchEvaluationReport(selectedReportId)
      .then(setSelectedReport)
      .catch(() => setSelectedReport(null))
      .finally(() => setReportsLoading(false));
  }, [selectedReportId]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!form.model.trim()) {
      setSubmitError("未发现可用的预训练权重，请先将 .pt 文件放入 models/pretrained。");
      return;
    }
    setSubmitting(true);
    setSubmitError(null);
    setSubmitNotice(null);
    try {
      const job = await submitTrainJob(form);
      setRecentTrainJobId(job.job_id);
      setSubmitNotice(`训练任务已提交：${job.job_id}`);
      await loadData();
    } catch {
      setSubmitError("提交失败，请检查后端服务是否正常运行");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleCancel(jobId: string) {
    await cancelJob(jobId);
    await loadData();
  }

  function handleCheckpointSelect(item: CheckpointItem) {
    setEvalForm((current) => ({ ...current, model: item.path }));
    const linkedRunId = findCurveRunIdByCheckpoint(item.path, curves);
    if (linkedRunId) {
      setSelectedCurveRunId(linkedRunId);
    }
  }

  async function handleEvaluate(e: React.FormEvent) {
    e.preventDefault();
    if (!evalForm.model.trim()) {
      setSubmitError("请先选择一个 checkpoint。");
      return;
    }
    setEvaluating(true);
    setSubmitError(null);
    setEvalNotice(null);
    try {
      const job = await submitEvaluateJob({
        model: evalForm.model,
        data: evalForm.data,
        device: evalForm.device,
        batch: evalForm.batch,
        split: evalForm.split,
        task_type: "segment",
      });
      setRecentEvaluateJobId(job.job_id);
      setEvalNotice(`评估任务已提交：${job.job_id}`);
      await loadData();
    } catch {
      setSubmitError("评估任务提交失败，请检查后端服务是否正常运行");
    } finally {
      setEvaluating(false);
    }
  }

  const runningJobs = jobs.filter((j) => j.status === "pending" || j.status === "running");
  const finishedJobs = jobs.filter((j) => !["pending", "running"].includes(j.status));
  const activeTrainJob =
    runningJobs.find((job) => job.job_id === recentTrainJobId) ??
    runningJobs.find((job) => job.task === "train");
  const activeEvaluateJob =
    runningJobs.find((job) => job.job_id === recentEvaluateJobId) ??
    runningJobs.find((job) => job.task === "evaluate");
  const selectedCurve = curves.find((item) => item.run_id === selectedCurveRunId) ?? curves[0];
  const totalCheckpointPages = Math.max(1, Math.ceil(checkpoints.length / CHECKPOINTS_PER_PAGE));
  const visibleCheckpoints = checkpoints.slice(
    (checkpointPage - 1) * CHECKPOINTS_PER_PAGE,
    checkpointPage * CHECKPOINTS_PER_PAGE
  );

  // Build chart option from real data
  const curveOption: EChartsOption | null = selectedCurve && selectedCurve.curve.length > 0 ? {
    grid: { top: 20, bottom: 30, left: 50, right: 20 },
    tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
    legend: { top: 0, right: 0, textStyle: { fontSize: 11 } },
    xAxis: { type: "category", data: selectedCurve.curve.map(p => p.epoch), axisLabel: { fontSize: 10, color: "#6B635A" }, axisLine: { lineStyle: { color: "#E5DDD3" } } },
    yAxis: [
      { type: "value", name: "mIoU", min: 0, max: 1, splitLine: { lineStyle: { type: "dashed", color: "#E5DDD3" } }, axisLabel: { fontSize: 10, color: "#6B635A" } },
      { type: "value", name: "Loss", splitLine: { show: false }, axisLabel: { fontSize: 10, color: "#6B635A" } },
    ],
    series: [
      { name: "mIoU", type: "line", data: selectedCurve.curve.map(p => p.miou), smooth: true, lineStyle: { width: 2 }, itemStyle: { color: "#C4714F" }, symbol: "none" },
      { name: "Loss", type: "line", yAxisIndex: 1, data: selectedCurve.curve.map(p => p.loss), smooth: true, lineStyle: { width: 2, type: "dashed" }, itemStyle: { color: "#7A8C7A" }, symbol: "none" },
    ],
  } : null;

  useEffect(() => {
    setCheckpointPage((current) => Math.min(current, totalCheckpointPages));
  }, [totalCheckpointPages]);

  return (
    <div className="min-h-screen px-6 py-10 max-w-5xl mx-auto space-y-10">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-[var(--charcoal)] mb-2" style={{ fontFamily: "var(--font-display)" }}>训练管理</h1>
        <p className="text-[var(--mid-gray)] text-sm">管理分割模型训练任务，追踪训练进度与性能指标</p>
      </div>

      {/* Training form */}
      <div className="card p-8">
        <h2 className="text-lg font-bold text-[var(--charcoal)] mb-6" style={{ fontFamily: "var(--font-display)" }}>发起新训练</h2>
        <form onSubmit={handleSubmit} className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-5">
          {[
            { label: "分割模型", key: "model", type: "select", options: trainingModels.map((item) => ({
              value: item.name,
              label: item.label,
            })) },
            { label: "数据集", key: "data", type: "text", placeholder: "room_separation_3" },
            { label: "训练轮数", key: "epochs", type: "number" },
            { label: "Batch Size", key: "batch", type: "number" },
            { label: "设备", key: "device", type: "text", placeholder: "cpu / 0" },
          ].map(({ label, key, type, options, placeholder }) => (
            <label key={key} className="flex flex-col gap-1.5">
              <span className="text-xs font-medium text-[var(--mid-gray)] uppercase tracking-wider">{label}</span>
              {type === "select" ? (
                <select
                  value={(form as Record<string, string | number | boolean>)[key] as string | number}
                  onChange={(e) => setForm({ ...form, [key]: e.target.value })}
                  className="px-3 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--warm-white)] text-sm text-[var(--charcoal)] focus:outline-none focus:border-[var(--terracotta)] transition-all"
                >
                  {options?.length === 0 && (
                    <option value="">
                      {trainingModelsError ?? "未发现 models/pretrained 下的 .pt 权重"}
                    </option>
                  )}
                  {options?.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                </select>
              ) : (
                <input
                  type={type}
                  value={(form as Record<string, string | number | boolean>)[key] as string | number}
                  onChange={(e) => setForm({ ...form, [key]: type === "number" ? parseInt(e.target.value) || 0 : e.target.value })}
                  placeholder={placeholder}
                  className="px-3 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--warm-white)] text-sm text-[var(--charcoal)] focus:outline-none focus:border-[var(--terracotta)] transition-all"
                />
              )}
            </label>
          ))}
          <div className="flex items-end">
            <button
              type="submit"
              disabled={submitting || !form.model.trim()}
              className="w-full py-2.5 px-4 bg-[var(--terracotta)] hover:bg-[var(--terracotta-light)] disabled:bg-[var(--light-gray)] text-white rounded-xl text-sm font-semibold transition-all flex items-center justify-center gap-2"
            >
              {submitting
                ? <><i className="fa-solid fa-circle-notch animate-spin text-xs" /> 提交中...</>
                : <><i className="fa-solid fa-play text-xs" /> 发起训练</>
              }
            </button>
          </div>
          <label className="flex items-center gap-2 text-sm text-[var(--warm-gray)]">
            <input
              type="checkbox"
              checked={form.resume}
              onChange={(event) => setForm({ ...form, resume: event.target.checked })}
              className="w-4 h-4 accent-[var(--terracotta)]"
            />
            从上次中断训练恢复
          </label>
        </form>
        {submitError && (
          <div className="mt-4 flex items-center gap-2 text-sm text-red-500">
            <i className="fa-solid fa-circle-exclamation" />
            {submitError}
          </div>
        )}
        {trainingModelsError && (
          <div className="mt-4 flex items-center gap-2 text-sm text-amber-600">
            <i className="fa-solid fa-triangle-exclamation" />
            {trainingModelsError}
          </div>
        )}
        {submitNotice && (
          <div className="mt-4 flex items-center gap-2 text-sm text-[var(--sage)]">
            <i className="fa-solid fa-circle-check" />
            {submitNotice}
          </div>
        )}
        {activeTrainJob && (
          <LiveTrainingPanel job={activeTrainJob} onCancel={(jobId) => void handleCancel(jobId)} />
        )}
      </div>

      {/* Evaluation form */}
      <div className="card p-8">
        <div className="flex items-start justify-between gap-4 mb-6">
          <div>
            <h2 className="text-lg font-bold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>模型评估</h2>
            <p className="text-xs text-[var(--mid-gray)] mt-1">选择 checkpoint 后调用 odp-val，对分割模型进行验证。</p>
          </div>
          <span className="px-3 py-1 rounded-full text-xs bg-[var(--ivory)] border border-[var(--border)] text-[var(--warm-gray)]">
            {checkpoints.length} 个 checkpoint
          </span>
        </div>
        <form onSubmit={handleEvaluate} className="grid grid-cols-1 md:grid-cols-5 gap-5">
          <label className="md:col-span-2 flex flex-col gap-1.5">
            <span className="text-xs font-medium text-[var(--mid-gray)] uppercase tracking-wider">Checkpoint</span>
            <select
              value={evalForm.model}
              onChange={(event) => setEvalForm({ ...evalForm, model: event.target.value })}
              className="px-3 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--warm-white)] text-sm text-[var(--charcoal)] focus:outline-none focus:border-[var(--terracotta)] transition-all"
            >
              <option value="">请选择模型权重</option>
              {checkpoints.map((item) => (
                <option key={item.path} value={item.path}>{item.name}</option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="text-xs font-medium text-[var(--mid-gray)] uppercase tracking-wider">数据集</span>
            <select
              value={evalForm.data}
              onChange={(event) => setEvalForm({ ...evalForm, data: event.target.value })}
              className="px-3 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--warm-white)] text-sm text-[var(--charcoal)] focus:outline-none focus:border-[var(--terracotta)] transition-all"
            >
              {datasets.length === 0 && <option value="room_separation_3">room_separation_3</option>}
              {datasets.map((item) => <option key={item.name} value={item.name}>{item.name}</option>)}
            </select>
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="text-xs font-medium text-[var(--mid-gray)] uppercase tracking-wider">设备</span>
            <input
              value={evalForm.device}
              onChange={(event) => setEvalForm({ ...evalForm, device: event.target.value })}
              className="px-3 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--warm-white)] text-sm text-[var(--charcoal)] focus:outline-none focus:border-[var(--terracotta)] transition-all"
              placeholder="cpu / 0"
            />
          </label>
          <div className="flex items-end">
            <button
              type="submit"
              disabled={evaluating || !evalForm.model}
              className="w-full py-2.5 px-4 bg-[var(--sage)] hover:opacity-90 disabled:bg-[var(--light-gray)] text-white rounded-xl text-sm font-semibold transition-all flex items-center justify-center gap-2"
            >
              {evaluating
                ? <><i className="fa-solid fa-circle-notch animate-spin text-xs" /> 提交中...</>
                : <><i className="fa-solid fa-square-poll-vertical text-xs" /> 发起评估</>
              }
            </button>
          </div>
        </form>
        {evalNotice && (
          <div className="mt-4 flex items-center gap-2 text-sm text-[var(--sage)]">
            <i className="fa-solid fa-circle-check" />
            {evalNotice}
          </div>
        )}
        {activeEvaluateJob && (
          <LiveEvaluatePanel job={activeEvaluateJob} onCancel={(jobId) => void handleCancel(jobId)} />
        )}
      </div>

      <EvaluationReportsPanel
        reports={evaluationReports}
        selectedReportId={selectedReportId}
        selectedReport={selectedReport}
        loading={reportsLoading}
        onSelect={setSelectedReportId}
        onRefresh={() => void loadData()}
      />

      {/* Checkpoints */}
      <div className="card p-8">
        <div className="flex items-start justify-between gap-4 mb-4">
          <div>
            <h2 className="text-lg font-bold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>Checkpoint 管理</h2>
            <p className="text-xs text-[var(--mid-gray)] mt-1">点击卡片会填充到评估模型，并联动切换下方训练曲线。</p>
          </div>
          {checkpoints.length > CHECKPOINTS_PER_PAGE && (
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setCheckpointPage((page) => Math.max(1, page - 1))}
                disabled={checkpointPage === 1}
                className="px-3 py-1.5 rounded-xl border border-[var(--border)] text-xs text-[var(--warm-gray)] hover:text-[var(--charcoal)] hover:border-[var(--warm-gray)] disabled:opacity-40 disabled:cursor-not-allowed transition-all"
              >
                上一页
              </button>
              <span className="text-xs text-[var(--mid-gray)] px-2">
                第 {checkpointPage} / {totalCheckpointPages} 页
              </span>
              <button
                type="button"
                onClick={() => setCheckpointPage((page) => Math.min(totalCheckpointPages, page + 1))}
                disabled={checkpointPage === totalCheckpointPages}
                className="px-3 py-1.5 rounded-xl border border-[var(--border)] text-xs text-[var(--warm-gray)] hover:text-[var(--charcoal)] hover:border-[var(--warm-gray)] disabled:opacity-40 disabled:cursor-not-allowed transition-all"
              >
                下一页
              </button>
            </div>
          )}
        </div>
        {checkpoints.length === 0 ? (
          <div className="text-center py-8">
            <i className="fa-solid fa-box-archive text-3xl text-[var(--light-gray)] mb-3" />
            <p className="text-sm text-[var(--mid-gray)]">暂无分割 checkpoint。训练完成后会显示在这里。</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {visibleCheckpoints.map((item) => (
              (() => {
                const linkedRunId = findCurveRunIdByCheckpoint(item.path, curves);
                const isActive = evalForm.model === item.path;
                const isLinkedCurve = selectedCurve?.run_id === linkedRunId;
                return (
                  <button
                    key={item.path}
                    onClick={() => handleCheckpointSelect(item)}
                    className={`text-left rounded-2xl border p-4 transition-all ${
                      isActive
                        ? "border-[var(--terracotta)] bg-[var(--terracotta-pale)]"
                        : "border-[var(--border)] bg-[var(--ivory)] hover:border-[var(--terracotta-light)]"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="text-sm font-semibold text-[var(--charcoal)] truncate">{item.name}</p>
                        <p className="text-[11px] text-[var(--mid-gray)] font-mono truncate mt-1">{item.path}</p>
                      </div>
                      <span className="px-2 py-0.5 rounded-lg text-[10px] border bg-[var(--warm-white)] text-[var(--warm-gray)]">{item.kind}</span>
                    </div>
                    <div className="mt-3 flex items-center justify-between gap-3">
                      <p className="text-[11px] text-[var(--mid-gray)]">{item.created_at.replace("T", " ").slice(0, 16)}</p>
                      <span
                        className={`px-2 py-0.5 rounded-full text-[10px] border ${
                          isLinkedCurve
                            ? "border-[var(--sage-light)] bg-[var(--sage-pale)] text-[var(--sage)]"
                            : "border-[var(--border)] bg-[var(--warm-white)] text-[var(--warm-gray)]"
                        }`}
                      >
                        {linkedRunId ? `曲线 ${linkedRunId}` : "未关联曲线"}
                      </span>
                    </div>
                  </button>
                );
              })()
            ))}
          </div>
        )}
      </div>

      {/* Training curve — real data */}
      {curvesLoading ? (
        <div className="card p-8 flex items-center justify-center">
          <div className="w-6 h-6 border-2 border-[var(--terracotta)] border-t-transparent rounded-full animate-spin" />
        </div>
      ) : curveOption ? (
        <div className="card p-8">
          <div className="flex items-start justify-between mb-6">
            <div>
              <h2 className="text-lg font-bold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>训练曲线</h2>
              <p className="text-xs text-[var(--mid-gray)] mt-0.5">
                {selectedCurve.model} · run {selectedCurve.run_id} · mIoU 最终 {(selectedCurve.metric ?? 0).toFixed(4)}
              </p>
            </div>
            <div className="flex flex-col items-end gap-2">
              <span className="px-3 py-1 rounded-full text-[11px] bg-[var(--ivory)] border border-[var(--border)] text-[var(--warm-gray)]">
                已加载 {curves.length} 条训练曲线
              </span>
              <select
                value={selectedCurve.run_id}
                onChange={(event) => setSelectedCurveRunId(event.target.value)}
                className="min-w-56 px-3 py-2 rounded-xl border border-[var(--border)] bg-[var(--warm-white)] text-sm text-[var(--charcoal)] focus:outline-none focus:border-[var(--terracotta)] transition-all"
              >
                {curves.map((item) => (
                  <option key={item.run_id} value={item.run_id}>
                    {item.run_id} · {item.model}
                  </option>
                ))}
              </select>
              <div className="flex items-center gap-4 text-xs">
                <span className="flex items-center gap-1.5">
                  <span className="w-6 h-0.5 bg-[var(--terracotta)] inline-block" />mIoU
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-6 h-0.5 border-t-2 border-dashed border-[var(--sage)] inline-block" />Loss
                </span>
              </div>
            </div>
          </div>
          <EChart key={selectedCurve.run_id} option={curveOption} className="w-full" style={{ height: "280px" }} />
        </div>
      ) : (
        <div className="card p-8 text-center">
          <i className="fa-solid fa-chart-line text-3xl text-[var(--light-gray)] mb-3" />
          <p className="text-[var(--mid-gray)] text-sm">暂无训练数据</p>
        </div>
      )}

      {/* Running jobs */}
      {runningJobs.length > 0 && (
        <div>
          <h2 className="text-lg font-bold text-[var(--charcoal)] mb-4" style={{ fontFamily: "var(--font-display)" }}>进行中的任务</h2>
          <div className="space-y-3">
            {runningJobs.map((job) => {
              const cfg = STATUS_CONFIG[job.status] ?? STATUS_CONFIG.pending;
              const elapsed = job.started_at
                ? Math.round((Date.now() - new Date(job.started_at).getTime()) / 1000)
                : null;
              return (
                <div key={job.job_id} className="card p-5 flex items-center gap-4">
                  <div className="w-10 h-10 rounded-xl bg-[var(--terracotta-pale)] flex items-center justify-center flex-shrink-0">
                    <i className={`fa-solid ${cfg.icon} text-[var(--terracotta)]`} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-[var(--charcoal)]">{job.task} · <span className="font-mono text-xs text-[var(--warm-gray)]">{job.job_id}</span></p>
                    {job.command.length > 0 && <p className="mt-1 text-[10px] text-[var(--mid-gray)] font-mono truncate">{job.command.join(" ")}</p>}
                    <div className="mt-1 flex items-center gap-3">
                      <div className="flex-1 h-1.5 bg-[var(--parchment)] rounded-full overflow-hidden max-w-48">
                        <div className="h-full bg-[var(--terracotta)] rounded-full animate-pulse" style={{ width: `${Math.max(job.progress_percent || 0, job.status === "running" ? 12 : 3)}%` }} />
                      </div>
                      <span className="text-xs text-[var(--mid-gray)]">
                        {job.progress_percent ? `${job.progress_percent}%` : elapsed !== null ? `${elapsed}s` : "等待中"}
                      </span>
                    </div>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-xs font-medium border ${cfg.badge}`}>{cfg.label}</span>
                  <button onClick={() => handleCancel(job.job_id)} className="text-xs text-[var(--mid-gray)] hover:text-red-400 transition-colors px-2 py-1 rounded-lg hover:bg-red-50">取消</button>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Job history */}
      <div>
        <h2 className="text-lg font-bold text-[var(--charcoal)] mb-4" style={{ fontFamily: "var(--font-display)" }}>任务历史</h2>
        {finishedJobs.length === 0 && !loading ? (
          <div className="card p-12 text-center">
            <i className="fa-solid fa-brain text-3xl text-[var(--light-gray)] mb-3" />
            <p className="text-[var(--mid-gray)]">暂无任务记录</p>
            <p className="text-xs text-[var(--light-gray)] mt-1">发起训练或评估后，任务会显示在这里</p>
          </div>
        ) : (
          <div className="card overflow-hidden">
            <table className="w-full text-sm table-fixed">
              <colgroup>
                <col className="w-[14%]" />
                <col className="w-[11%]" />
                <col className="w-[11%]" />
                <col className="w-[14%]" />
                <col className="w-[50%]" />
              </colgroup>
              <thead>
                <tr className="border-b border-[var(--border)] text-xs text-[var(--mid-gray)] uppercase tracking-wider">
                  <th className="text-left px-5 py-3.5 font-medium">任务 ID</th>
                  <th className="text-left px-5 py-3.5 font-medium">类型</th>
                  <th className="text-left px-5 py-3.5 font-medium">状态</th>
                  <th className="text-left px-5 py-3.5 font-medium">创建时间</th>
                  <th className="text-left px-5 py-3.5 font-medium">命令 / 结果</th>
                </tr>
              </thead>
              <tbody>
                {finishedJobs.map((job) => {
                  const cfg = STATUS_CONFIG[job.status] ?? STATUS_CONFIG.failed;
                  return (
                    <tr key={job.job_id} className="border-b border-[var(--border)] hover:bg-[var(--ivory)]/50 transition-colors">
                      <td className="px-5 py-3.5 font-mono text-xs text-[var(--warm-gray)] break-all">{job.job_id}</td>
                      <td className="px-5 py-3.5 text-[var(--charcoal)] whitespace-nowrap">{TASK_LABELS[job.task] ?? job.task}</td>
                      <td className="px-5 py-3.5 whitespace-nowrap">
                        <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium border whitespace-nowrap ${cfg.badge}`}>
                          <i className={`fa-solid ${cfg.icon} mr-1 text-[10px]`} />
                          {cfg.label}
                        </span>
                      </td>
                      <td className="px-5 py-3.5 text-[var(--mid-gray)] text-xs font-mono whitespace-nowrap">{job.created_at?.replace("T", " ").slice(0, 16)}</td>
                      <td className="px-5 py-3.5 align-top">
                        <JobDetail job={job} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
