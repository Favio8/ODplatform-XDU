import { useEffect, useState, useCallback } from "react";
import type { EChartsOption } from "echarts";
import { EChart } from "../components/EChart";
import {
  fetchCheckpoints,
  fetchDatasets,
  fetchEvaluationReport,
  fetchEvaluationReports,
  fetchJobs,
  fetchTrainingCurves,
  submitEvaluateJob,
  submitTrainJob,
  cancelJob,
} from "../lib/api";
import type { CheckpointItem, DatasetProfile, EvaluationAsset, EvaluationReportDetail, EvaluationReportSummary, JobResponse } from "../types";

const API_ORIGIN = (import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000/api").replace(/\/api\/?$/, "");

const STATUS_CONFIG: Record<string, { label: string; badge: string; icon: string }> = {
  pending:   { label: "等待中",   badge: "bg-[var(--parchment)] text-[var(--warm-gray)] border-[var(--border)]", icon: "fa-clock" },
  running:   { label: "进行中",   badge: "bg-[var(--terracotta-pale)] text-[var(--terracotta)] border-[var(--terracotta-light)]", icon: "fa-spinner fa-spin" },
  completed: { label: "已完成",  badge: "bg-[var(--sage-pale)] text-[var(--sage)] border-[var(--sage-light)]", icon: "fa-check" },
  failed:    { label: "失败",    badge: "bg-red-50 text-red-500 border border-red-200", icon: "fa-xmark" },
  cancelled:  { label: "已取消",  badge: "bg-amber-50 text-amber-600 border border-amber-200", icon: "fa-ban" },
};

function JobDetail({ job }: { job: JobResponse }) {
  const output = job.error || job.stderr_tail || job.stdout_tail;
  return (
    <div className="space-y-2">
      {job.command.length > 0 && (
        <p className="font-mono text-[11px] text-[var(--warm-gray)] break-all bg-[var(--ivory)] rounded-lg px-3 py-2">
          {job.command.join(" ")}
        </p>
      )}
      {output && (
        <p className="text-[11px] text-red-500 line-clamp-3 break-words">
          {output.slice(0, 260)}
        </p>
      )}
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
  const metrics = selectedReport?.metrics;
  const metricItems = [
    { label: "mAP50", value: metrics?.map50 },
    { label: "mAP50-95", value: metrics?.map50_95 },
    { label: "Precision", value: metrics?.precision },
    { label: "Recall", value: metrics?.recall },
    { label: "Fitness", value: metrics?.fitness },
  ];

  return (
    <div className="card p-8 space-y-6">
      <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>评估报告</h2>
          <p className="text-xs text-[var(--mid-gray)] mt-1">读取 Ultralytics 输出的指标、混淆矩阵、PR 曲线和预测样例。</p>
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
        <>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {metricItems.map((item) => (
              <div key={item.label} className="rounded-2xl bg-[var(--ivory)] border border-[var(--border)] p-4">
                <p className="text-[10px] text-[var(--mid-gray)] uppercase tracking-wider">{item.label}</p>
                <p className="mt-2 text-xl font-semibold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>{formatMetric(item.value)}</p>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-1 gap-8">
            <AssetGallery title="混淆矩阵" assets={selectedReport.assets.confusion} empty="未找到 confusion_matrix 图片。" />
            <AssetGallery title="PR / P / R / F1 曲线" assets={selectedReport.assets.curves} empty="未找到曲线图片。" />
            <AssetGallery title="预测样例" assets={selectedReport.assets.samples.slice(0, 8)} empty="未找到 val_batch 预测样例。" />
          </div>
        </>
      )}
    </div>
  );
}

export function Training() {
  const [jobs, setJobs] = useState<JobResponse[]>([]);
  const [datasets, setDatasets] = useState<DatasetProfile[]>([]);
  const [checkpoints, setCheckpoints] = useState<CheckpointItem[]>([]);
  const [curves, setCurves] = useState<Array<{ run_id: string; model: string; metric: number | null; curve: Array<{ epoch: number; miou: number; loss: number }>; status: string }>>([]);
  const [evaluationReports, setEvaluationReports] = useState<EvaluationReportSummary[]>([]);
  const [selectedReportId, setSelectedReportId] = useState("");
  const [selectedReport, setSelectedReport] = useState<EvaluationReportDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [curvesLoading, setCurvesLoading] = useState(true);
  const [reportsLoading, setReportsLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [form, setForm] = useState({
    model: "yolov8n-seg.pt",
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
    try {
      const [jobsData, curvesData, checkpointData, datasetData, reportData] = await Promise.all([
        fetchJobs(),
        fetchTrainingCurves(),
        fetchCheckpoints(),
        fetchDatasets(),
        fetchEvaluationReports(),
      ]);
      setJobs(jobsData);
      setCurves(curvesData);
      setCheckpoints(checkpointData.filter((item) => item.task === "segment"));
      setDatasets(datasetData);
      setEvaluationReports(reportData);
      setSelectedReportId((current) => {
        if (current && reportData.some((report) => report.report_id === current)) return current;
        return reportData[0]?.report_id ?? "";
      });
      if (!evalForm.model && checkpointData.length > 0) {
        const preferred = checkpointData.find((item) => item.task === "segment" && item.kind === "best") ?? checkpointData[0];
        setEvalForm((current) => current.model ? current : { ...current, model: preferred.path });
      }
    } catch {
      // silently fail - show whatever we have
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
    setSubmitting(true);
    setSubmitError(null);
    try {
      await submitTrainJob(form);
      setForm({ model: "yolov8n-seg.pt", data: "room_separation_3", epochs: 10, batch: 8, device: "cpu", resume: false });
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

  async function handleEvaluate(e: React.FormEvent) {
    e.preventDefault();
    if (!evalForm.model.trim()) {
      setSubmitError("请先选择一个 checkpoint。");
      return;
    }
    setEvaluating(true);
    setSubmitError(null);
    try {
      await submitEvaluateJob({
        model: evalForm.model,
        data: evalForm.data,
        device: evalForm.device,
        batch: evalForm.batch,
        split: evalForm.split,
        task_type: "segment",
      });
      await loadData();
    } catch {
      setSubmitError("评估任务提交失败，请检查后端服务是否正常运行");
    } finally {
      setEvaluating(false);
    }
  }

  const runningJobs = jobs.filter((j) => j.status === "pending" || j.status === "running");
  const finishedJobs = jobs.filter((j) => !["pending", "running"].includes(j.status));
  const latestCurve = curves[0];

  // Build chart option from real data
  const curveOption: EChartsOption | null = latestCurve && latestCurve.curve.length > 0 ? {
    grid: { top: 20, bottom: 30, left: 50, right: 20 },
    tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
    legend: { top: 0, right: 0, textStyle: { fontSize: 11 } },
    xAxis: { type: "category", data: latestCurve.curve.map(p => p.epoch), axisLabel: { fontSize: 10, color: "#6B635A" }, axisLine: { lineStyle: { color: "#E5DDD3" } } },
    yAxis: [
      { type: "value", name: "mIoU", min: 0, max: 1, splitLine: { lineStyle: { type: "dashed", color: "#E5DDD3" } }, axisLabel: { fontSize: 10, color: "#6B635A" } },
      { type: "value", name: "Loss", splitLine: { show: false }, axisLabel: { fontSize: 10, color: "#6B635A" } },
    ],
    series: [
      { name: "mIoU", type: "line", data: latestCurve.curve.map(p => p.miou), smooth: true, lineStyle: { width: 2 }, itemStyle: { color: "#C4714F" }, symbol: "none" },
      { name: "Loss", type: "line", yAxisIndex: 1, data: latestCurve.curve.map(p => p.loss), smooth: true, lineStyle: { width: 2, type: "dashed" }, itemStyle: { color: "#7A8C7A" }, symbol: "none" },
    ],
  } : null;

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
            { label: "分割模型", key: "model", type: "select", options: [
              { value: "yolov8n-seg.pt", label: "YOLOv8n-seg · 轻量版" },
              { value: "yolov8s-seg.pt", label: "YOLOv8s-seg · 标准版" },
              { value: "yolov8m-seg.pt", label: "YOLOv8m-seg · 高精度" },
            ]},
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
              disabled={submitting}
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
        <h2 className="text-lg font-bold text-[var(--charcoal)] mb-4" style={{ fontFamily: "var(--font-display)" }}>Checkpoint 管理</h2>
        {checkpoints.length === 0 ? (
          <div className="text-center py-8">
            <i className="fa-solid fa-box-archive text-3xl text-[var(--light-gray)] mb-3" />
            <p className="text-sm text-[var(--mid-gray)]">暂无分割 checkpoint。训练完成后会显示在这里。</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {checkpoints.slice(0, 6).map((item) => (
              <button
                key={item.path}
                onClick={() => setEvalForm({ ...evalForm, model: item.path })}
                className={`text-left rounded-2xl border p-4 transition-all ${
                  evalForm.model === item.path
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
                <p className="text-[11px] text-[var(--mid-gray)] mt-3">{item.created_at.replace("T", " ").slice(0, 16)}</p>
              </button>
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
              <p className="text-xs text-[var(--mid-gray)] mt-0.5">{latestCurve.model} · mIoU 最终 {(latestCurve.metric ?? 0).toFixed(4)}</p>
            </div>
            <div className="flex items-center gap-4 text-xs">
              <span className="flex items-center gap-1.5">
                <span className="w-6 h-0.5 bg-[var(--terracotta)] inline-block" />mIoU
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-6 h-0.5 border-t-2 border-dashed border-[var(--sage)] inline-block" />Loss
              </span>
            </div>
          </div>
          <EChart option={curveOption} className="w-full" style={{ height: "280px" }} />
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
            <table className="w-full text-sm">
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
                      <td className="px-5 py-3.5 font-mono text-xs text-[var(--warm-gray)]">{job.job_id}</td>
                      <td className="px-5 py-3.5 capitalize text-[var(--charcoal)]">{job.task}</td>
                      <td className="px-5 py-3.5">
                        <span className={`px-2.5 py-1 rounded-full text-xs font-medium border ${cfg.badge}`}>
                          <i className={`fa-solid ${cfg.icon} mr-1 text-[10px]`} />
                          {cfg.label}
                        </span>
                      </td>
                      <td className="px-5 py-3.5 text-[var(--mid-gray)] text-xs font-mono">{job.created_at?.replace("T", " ").slice(0, 16)}</td>
                      <td className="px-5 py-3.5">
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
