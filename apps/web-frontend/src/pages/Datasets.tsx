import { useEffect, useState, useCallback } from "react";
import {
  fetchDatasets,
  fetchJobs,
  fetchProjectStatus,
  createDataset,
  updateDataset,
  deleteDataset,
  uploadDatasetArchive,
  submitConfigJob,
  submitInitJob,
  submitTransformJob,
  submitValidateJob,
} from "../lib/api";
import type { DatasetProfile, DatasetUploadResult, JobResponse, ProjectStatus, TransformJobCreate } from "../types";

/* ─── Shared modal shell ─── */
function ModalShell({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: React.ReactNode;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "rgba(44,38,32,0.45)", backdropFilter: "blur(4px)" }}
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        className="card w-full max-w-lg p-8 animate-fade-up relative"
        style={{ animationDuration: "0.2s" }}
      >
        <button
          onClick={onClose}
          className="absolute top-5 right-5 w-8 h-8 rounded-lg flex items-center justify-center text-[var(--mid-gray)] hover:bg-[var(--parchment)] hover:text-[var(--charcoal)] transition-all"
        >
          <i className="fa-solid fa-xmark text-sm" />
        </button>
        <h2
          className="text-xl font-bold text-[var(--charcoal)] mb-6 pr-8"
          style={{ fontFamily: "var(--font-display)" }}
        >
          {title}
        </h2>
        {children}
      </div>
    </div>
  );
}

/* ─── Tag input for class names ─── */
function TagInput({
  tags,
  onChange,
  placeholder,
}: {
  tags: string[];
  onChange: (tags: string[]) => void;
  placeholder: string;
}) {
  const [input, setInput] = useState("");

  function addTag() {
    const trimmed = input.trim();
    if (trimmed && !tags.includes(trimmed)) {
      onChange([...tags, trimmed]);
    }
    setInput("");
  }

  return (
    <div className="min-h-[44px] p-2 rounded-xl border border-[var(--border)] bg-[var(--warm-white)] flex flex-wrap gap-1.5 items-start">
      {tags.map((tag) => (
        <span
          key={tag}
          className="flex items-center gap-1 px-2 py-1 rounded-lg bg-[var(--terracotta-pale)] text-[var(--terracotta)] text-xs font-medium border border-[var(--terracotta-light)]"
        >
          {tag}
          <button
            type="button"
            onClick={() => onChange(tags.filter((t) => t !== tag))}
            className="w-4 h-4 rounded flex items-center justify-center hover:bg-[var(--terracotta)] hover:text-white transition-all"
          >
            <i className="fa-solid fa-xmark text-[8px]" />
          </button>
        </span>
      ))}
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === ",") {
            e.preventDefault();
            addTag();
          }
          if (e.key === "Backspace" && !input && tags.length > 0) {
            onChange(tags.slice(0, -1));
          }
        }}
        onBlur={addTag}
        placeholder={tags.length === 0 ? placeholder : ""}
        className="flex-1 min-w-24 text-sm text-[var(--charcoal)] placeholder:text-[var(--light-gray)] bg-transparent outline-none"
      />
    </div>
  );
}

/* ─── Delete confirmation ─── */
function DeleteConfirm({
  name,
  onConfirm,
  onCancel,
}: {
  name: string;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <ModalShell title="确认删除" onClose={onCancel}>
      <div className="text-center mb-6">
        <div className="w-14 h-14 rounded-2xl bg-red-50 flex items-center justify-center mx-auto mb-4">
          <i className="fa-solid fa-trash text-red-400 text-xl" />
        </div>
        <p className="text-[var(--charcoal)] font-medium mb-1">确定删除数据集</p>
        <p className="text-sm text-[var(--warm-gray)]">
          <span className="font-mono bg-[var(--parchment)] px-1.5 py-0.5 rounded text-[var(--charcoal)]">{name}</span>
          &nbsp;删除后无法恢复，相关训练记录也将一并清除。
        </p>
      </div>
      <div className="flex gap-3">
        <button
          onClick={onCancel}
          className="flex-1 py-2.5 px-4 border border-[var(--border)] hover:border-[var(--warm-gray)] text-[var(--warm-gray)] rounded-xl text-sm font-medium transition-all"
        >
          取消
        </button>
        <button
          onClick={onConfirm}
          className="flex-1 py-2.5 px-4 bg-red-500 hover:bg-red-600 text-white rounded-xl text-sm font-medium transition-all"
        >
          确认删除
        </button>
      </div>
    </ModalShell>
  );
}

/* ─── Create / Edit form ─── */
type DatasetFormState = {
  name: string;
  class_names: string[];
  train: number;
  val: number;
  test: number;
};

const FORMAT_OPTIONS: Array<{ value: TransformJobCreate["format"]; label: string }> = [
  { value: "coco", label: "COCO" },
  { value: "pascal_voc", label: "VOC" },
  { value: "yolo", label: "YOLO" },
];

function toTransformFormat(format: string | undefined): TransformJobCreate["format"] {
  if (format === "coco" || format === "yolo" || format === "pascal_voc") return format;
  return "coco";
}

function JobBadge({ job }: { job: JobResponse }) {
  const label = job.status === "completed" ? "已完成" : job.status === "failed" ? "失败" : job.status === "running" ? "运行中" : "等待中";
  const tone = job.status === "completed"
    ? "bg-[var(--sage-pale)] text-[var(--sage)] border-[var(--sage-light)]"
    : job.status === "failed"
      ? "bg-red-50 text-red-500 border-red-200"
      : "bg-[var(--terracotta-pale)] text-[var(--terracotta)] border-[var(--terracotta-light)]";
  return (
    <div className={`rounded-xl border px-3 py-2 text-xs ${tone}`}>
      <div className="flex items-center justify-between gap-3">
        <span className="font-medium">{job.task} · {label}</span>
        <span className="font-mono">{job.return_code != null ? `code ${job.return_code}` : `${job.progress_percent || 0}%`}</span>
      </div>
      {job.status === "running" && (
        <div className="mt-2 h-1 rounded-full bg-white/60 overflow-hidden">
          <div className="h-full rounded-full bg-current" style={{ width: `${Math.max(job.progress_percent || 0, 8)}%` }} />
        </div>
      )}
      <p className="mt-1 truncate font-mono opacity-80" title={job.command.join(" ")}>{job.command.join(" ")}</p>
      {(job.error || job.stderr_tail) && (
        <p className="mt-1 line-clamp-2 text-red-500">{(job.error || job.stderr_tail).slice(0, 160)}</p>
      )}
    </div>
  );
}

function ProjectStatusCard({
  status,
  latestJobs,
  onInit,
  loading,
}: {
  status: ProjectStatus | null;
  latestJobs: JobResponse[];
  onInit: () => void;
  loading: boolean;
}) {
  const stats = [
    { label: "原始数据集", value: status?.raw_datasets.length ?? 0, icon: "fa-folder-open" },
    { label: "转换结果", value: status?.processed_datasets.length ?? 0, icon: "fa-layer-group" },
    { label: "Dataset YAML", value: status?.dataset_configs.length ?? 0, icon: "fa-file-code" },
    { label: "运行配置", value: status?.runtime_configs.length ?? 0, icon: "fa-sliders" },
  ];

  return (
    <div className="card p-6 mb-8">
      <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-5">
        <div>
          <h2 className="text-lg font-bold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>项目状态</h2>
          <p className="text-xs text-[var(--mid-gray)] mt-1 font-mono break-all">{status?.root ?? "正在读取工作区状态..."}</p>
        </div>
        <button
          onClick={onInit}
          disabled={loading}
          className="inline-flex items-center justify-center gap-2 px-4 py-2.5 bg-[var(--terracotta)] hover:bg-[var(--terracotta-light)] disabled:bg-[var(--light-gray)] text-white rounded-xl text-sm font-medium transition-all"
        >
          <i className={`fa-solid ${loading ? "fa-circle-notch fa-spin" : "fa-wand-magic-sparkles"} text-xs`} />
          安全初始化
        </button>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mt-5">
        {stats.map((item) => (
          <div key={item.label} className="rounded-2xl bg-[var(--ivory)] border border-[var(--border)] p-4">
            <div className="flex items-center justify-between">
              <span className="text-xs text-[var(--mid-gray)]">{item.label}</span>
              <i className={`fa-solid ${item.icon} text-[var(--terracotta)] text-xs`} />
            </div>
            <p className="text-2xl font-semibold text-[var(--charcoal)] mt-2" style={{ fontFamily: "var(--font-display)" }}>{item.value}</p>
          </div>
        ))}
      </div>

      {latestJobs.length > 0 && (
        <div className="mt-5 space-y-2">
          <p className="text-xs font-medium text-[var(--mid-gray)] uppercase tracking-wider">最近平台任务</p>
          {latestJobs.slice(0, 3).map((job) => <JobBadge key={job.job_id} job={job} />)}
        </div>
      )}
    </div>
  );
}

function DatasetUploadCard({
  onUploaded,
}: {
  onUploaded: (result: DatasetUploadResult) => void;
}) {
  const [datasetName, setDatasetName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault();
    if (!datasetName.trim() || !file) {
      setError("请填写数据集名称并选择 ZIP 文件。");
      return;
    }
    setUploading(true);
    setError(null);
    try {
      const result = await uploadDatasetArchive(file, datasetName.trim());
      onUploaded(result);
      setFile(null);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "上传失败。");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="card p-6 mb-8">
      <div className="flex items-start justify-between gap-4 mb-5">
        <div>
          <h2 className="text-lg font-bold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>上传训练数据集</h2>
          <p className="text-xs text-[var(--mid-gray)] mt-1">上传 ZIP 后只解压到 data/raw，不会自动转换或训练。</p>
        </div>
        <span className="px-3 py-1 rounded-full text-xs bg-[var(--ivory)] border border-[var(--border)] text-[var(--warm-gray)]">ZIP</span>
      </div>
      <form onSubmit={handleUpload} className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <label className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-[var(--mid-gray)] uppercase tracking-wider">数据集名称</span>
          <input
            value={datasetName}
            onChange={(event) => setDatasetName(event.target.value)}
            placeholder="room_v2"
            className="px-3 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--warm-white)] text-sm text-[var(--charcoal)] focus:outline-none focus:border-[var(--terracotta)] transition-all"
          />
        </label>
        <label className="md:col-span-2 flex flex-col gap-1.5">
          <span className="text-xs font-medium text-[var(--mid-gray)] uppercase tracking-wider">ZIP 文件</span>
          <input
            type="file"
            accept=".zip,application/zip"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            className="px-3 py-2 rounded-xl border border-[var(--border)] bg-[var(--warm-white)] text-sm text-[var(--warm-gray)] file:mr-3 file:border-0 file:bg-[var(--terracotta-pale)] file:text-[var(--terracotta)] file:rounded-lg file:px-3 file:py-1.5"
          />
        </label>
        <div className="flex items-end">
          <button
            type="submit"
            disabled={uploading}
            className="w-full py-2.5 px-4 bg-[var(--terracotta)] hover:bg-[var(--terracotta-light)] disabled:bg-[var(--light-gray)] text-white rounded-xl text-sm font-semibold transition-all flex items-center justify-center gap-2"
          >
            {uploading ? <><i className="fa-solid fa-circle-notch animate-spin text-xs" /> 上传中...</> : <><i className="fa-solid fa-cloud-arrow-up text-xs" /> 上传并识别</>}
          </button>
        </div>
      </form>
      {error && <p className="mt-3 text-sm text-red-500">{error}</p>}
    </div>
  );
}

function UploadResultCard({
  result,
  selectedFormat,
  onFormatChange,
  onTransform,
}: {
  result: DatasetUploadResult;
  selectedFormat: TransformJobCreate["format"];
  onFormatChange: (format: TransformJobCreate["format"]) => void;
  onTransform: () => void;
}) {
  const detected = result.detected_format;
  return (
    <div className="card p-5 mb-8 border-[var(--terracotta-light)]">
      <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-wider text-[var(--terracotta)] mb-1">上传完成</p>
          <h3 className="text-lg font-bold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>{result.dataset_name}</h3>
          <p className="text-xs text-[var(--mid-gray)] font-mono break-all mt-1">{result.raw_path}</p>
        </div>
        <div className="flex gap-2 text-xs">
          <span className="px-3 py-1 rounded-full bg-[var(--ivory)] border border-[var(--border)]">{result.file_count} 文件</span>
          <span className="px-3 py-1 rounded-full bg-[var(--ivory)] border border-[var(--border)]">{result.dir_count} 目录</span>
        </div>
      </div>
      <div className="mt-4 grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 rounded-2xl bg-[var(--ivory)] border border-[var(--border)] p-4">
          <p className="text-sm font-semibold text-[var(--charcoal)]">自动识别：{detected.format === "unknown" ? "未识别" : detected.format}</p>
          <p className="text-xs text-[var(--mid-gray)] mt-1">置信度：{detected.confidence}</p>
          <ul className="mt-2 space-y-1 text-xs text-[var(--warm-gray)]">
            {detected.reasons.slice(0, 3).map((reason) => <li key={reason}>• {reason}</li>)}
          </ul>
        </div>
        <label className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-[var(--mid-gray)] uppercase tracking-wider">用于转换的格式</span>
          <select
            value={selectedFormat}
            onChange={(event) => onFormatChange(event.target.value as TransformJobCreate["format"])}
            className="px-3 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--warm-white)] text-sm text-[var(--charcoal)] focus:outline-none focus:border-[var(--terracotta)] transition-all"
          >
            {FORMAT_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
          <p className="text-xs text-[var(--mid-gray)]">{result.next_step}</p>
          <button
            type="button"
            onClick={onTransform}
            className="mt-2 inline-flex items-center justify-center gap-2 px-4 py-2 bg-[var(--terracotta)] hover:bg-[var(--terracotta-light)] text-white rounded-xl text-sm font-semibold transition-all"
          >
            <i className="fa-solid fa-arrows-rotate text-xs" />
            用此格式转换
          </button>
        </label>
      </div>
    </div>
  );
}

function DatasetForm({
  initial,
  mode,
  onSubmit,
  loading,
}: {
  initial?: DatasetProfile;
  mode: "create" | "edit";
  onSubmit: (data: DatasetFormState) => void;
  loading: boolean;
}) {
  const [form, setForm] = useState<DatasetFormState>({
    name: initial?.name ?? "",
    class_names: initial?.class_names ?? [],
    train: initial?.splits.train ?? 0,
    val: initial?.splits.val ?? 0,
    test: initial?.splits.test ?? 0,
  });
  const [error, setError] = useState<string | null>(null);

  function set<K extends keyof DatasetFormState>(key: K, value: DatasetFormState[K]) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!form.name.trim()) {
      setError("数据集名称不能为空");
      return;
    }
    if (form.class_names.length === 0) {
      setError("请至少添加一个类别");
      return;
    }
    setError(null);
    onSubmit(form);
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      {/* Name */}
      <label className="flex flex-col gap-1.5">
        <span className="text-xs font-medium text-[var(--mid-gray)] uppercase tracking-wider">数据集名称</span>
        <input
          type="text"
          value={form.name}
          onChange={(e) => set("name", e.target.value)}
          disabled={mode === "edit"}
          placeholder="例如：my_floorplan_v1"
          className="px-3 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--warm-white)] text-sm text-[var(--charcoal)] placeholder:text-[var(--light-gray)] focus:outline-none focus:border-[var(--terracotta)] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        />
      </label>

      {/* Class names */}
      <label className="flex flex-col gap-1.5">
        <span className="text-xs font-medium text-[var(--mid-gray)] uppercase tracking-wider">
          房间类别 <span className="normal-case tracking-normal text-[var(--light-gray)]">(按 Enter 或逗号添加)</span>
        </span>
        <TagInput
          tags={form.class_names}
          onChange={(tags) => set("class_names", tags)}
          placeholder="例如：客厅、卧室、厨房"
        />
      </label>

      {/* Splits */}
      <div className="grid grid-cols-3 gap-3">
        {(["train", "val", "test"] as const).map((key) => (
          <label key={key} className="flex flex-col gap-1.5">
            <span className="text-xs font-medium text-[var(--mid-gray)] uppercase tracking-wider">
              {key === "train" ? "训练" : key === "val" ? "验证" : "测试"}
            </span>
            <input
              type="number"
              min="0"
              value={form[key]}
              onChange={(e) => set(key, parseInt(e.target.value) || 0)}
              className="px-3 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--warm-white)] text-sm text-[var(--charcoal)] focus:outline-none focus:border-[var(--terracotta)] transition-all"
            />
          </label>
        ))}
      </div>

      {error && (
        <div className="flex items-center gap-2 text-sm text-red-400">
          <i className="fa-solid fa-circle-exclamation" />
          {error}
        </div>
      )}

      <button
        type="submit"
        disabled={loading}
        className="w-full py-2.5 px-4 bg-[var(--terracotta)] hover:bg-[var(--terracotta-light)] disabled:bg-[var(--light-gray)] text-white rounded-xl text-sm font-semibold transition-all flex items-center justify-center gap-2"
      >
        {loading ? (
          <><i className="fa-solid fa-circle-notch animate-spin text-xs" /> 处理中...</>
        ) : mode === "create" ? (
          <><i className="fa-solid fa-plus text-xs" /> 创建数据集</>
        ) : (
          <><i className="fa-solid fa-check text-xs" /> 保存修改</>
        )}
      </button>
    </form>
  );
}

/* ─── Dataset card ─── */
function DatasetCard({
  ds,
  sourceFormat,
  onSourceFormatChange,
  onEdit,
  onDelete,
  onTransform,
  onValidate,
  onConfig,
}: {
  ds: DatasetProfile;
  sourceFormat: TransformJobCreate["format"];
  onSourceFormatChange: (format: TransformJobCreate["format"]) => void;
  onEdit: (ds: DatasetProfile) => void;
  onDelete: (name: string) => void;
  onTransform: (name: string, format: TransformJobCreate["format"]) => void;
  onValidate: (name: string) => void;
  onConfig: () => void;
}) {
  const total = ds.splits.train + (ds.splits.val ?? 0) + (ds.splits.test ?? 0);

  return (
    <div className="card p-6 flex flex-col gap-4 hover:shadow-[var(--shadow-md)] transition-all">
      <div className="flex items-start justify-between gap-3 min-w-0">
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-10 h-10 rounded-xl bg-[var(--terracotta-pale)] flex items-center justify-center flex-shrink-0">
            <i className="fa-solid fa-image text-[var(--terracotta)]" />
          </div>
          <div className="min-w-0">
            <p className="font-semibold text-[var(--charcoal)] truncate" title={ds.name}>{ds.name}</p>
            <p className="text-xs text-[var(--mid-gray)] mt-0.5 font-mono truncate" title={ds.yaml_path}>{ds.yaml_path.split(/[\\/]/).pop()}</p>
          </div>
        </div>
        <span className={`px-2.5 py-1 rounded-lg text-xs font-medium border flex-shrink-0 ${
          ds.status === "ready"
            ? "bg-[var(--sage-pale)] text-[var(--sage)] border-[var(--sage-light)]"
            : "bg-[var(--parchment)] text-[var(--warm-gray)] border-[var(--border)]"
        }`}>
          {ds.status === "ready" ? "已就绪" : ds.status}
        </span>
      </div>

      <div className="grid grid-cols-3 gap-2">
        {[
          { label: "训练", value: ds.splits.train, color: "var(--terracotta)" },
          { label: "验证", value: ds.splits.val ?? 0, color: "var(--sage)" },
          { label: "测试", value: ds.splits.test ?? 0, color: "var(--warm-gray)" },
        ].map(({ label, value, color }) => (
          <div key={label} className="text-center p-2 rounded-xl bg-[var(--ivory)]">
            <p className="text-lg font-semibold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>{value}</p>
            <p className="text-[10px] text-[var(--mid-gray)]">{label}</p>
          </div>
        ))}
      </div>

      <div className="space-y-1.5">
        <div className="flex flex-wrap gap-1.5">
          {ds.class_names.map((name) => (
            <span key={name} className="max-w-full px-2 py-0.5 rounded-md bg-[var(--parchment)] text-[var(--warm-gray)] text-xs border border-[var(--border)] truncate" title={name}>{name}</span>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <div className="flex-1 h-1 bg-[var(--parchment)] rounded-full overflow-hidden">
            <div className="h-full bg-[var(--sage)] rounded-full" style={{ width: `${Math.round(ds.coverage * 100)}%` }} />
          </div>
          <span className="text-[10px] text-[var(--mid-gray)] flex-shrink-0">覆盖率 {Math.round(ds.coverage * 100)}%</span>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-2 pt-1 border-t border-[var(--border)]">
        <div className="flex gap-2 pt-2">
          <select
            value={sourceFormat}
            onChange={(event) => onSourceFormatChange(event.target.value as TransformJobCreate["format"])}
            className="w-24 px-2 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--warm-white)] text-xs text-[var(--warm-gray)] focus:outline-none focus:border-[var(--terracotta)]"
          >
            {FORMAT_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
          <button
            onClick={() => onTransform(ds.name, sourceFormat)}
            className="flex-1 py-1.5 px-3 text-xs text-[var(--terracotta)] hover:text-white border border-[var(--terracotta-light)] hover:bg-[var(--terracotta)] rounded-lg transition-all flex items-center justify-center gap-1.5"
          >
            <i className="fa-solid fa-arrows-rotate text-[10px]" />
            转换数据
          </button>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => onValidate(ds.name)}
            className="flex-1 py-1.5 px-3 text-xs text-[var(--sage)] hover:text-white border border-[var(--sage-light)] hover:bg-[var(--sage)] rounded-lg transition-all flex items-center justify-center gap-1.5"
          >
            <i className="fa-solid fa-clipboard-check text-[10px]" />
            数据质检
          </button>
          <button
            onClick={onConfig}
            className="flex-1 py-1.5 px-3 text-xs text-[var(--warm-gray)] hover:text-[var(--charcoal)] border border-[var(--border)] hover:border-[var(--warm-gray)] rounded-lg transition-all flex items-center justify-center gap-1.5"
          >
            <i className="fa-solid fa-sliders text-[10px]" />
            生成配置
          </button>
        </div>
        <div className="flex gap-2">
        <button
          onClick={() => onEdit(ds)}
          className="flex-1 py-1.5 px-3 text-xs text-[var(--mid-gray)] hover:text-[var(--charcoal)] border border-[var(--border)] hover:border-[var(--warm-gray)] rounded-lg transition-all flex items-center justify-center gap-1.5"
        >
          <i className="fa-solid fa-pen text-[10px]" />
          编辑
        </button>
        <button
          onClick={() => onDelete(ds.name)}
          className="flex-1 py-1.5 px-3 text-xs text-red-400 hover:text-red-500 border border-red-100 hover:border-red-200 rounded-lg transition-all flex items-center justify-center gap-1.5"
        >
          <i className="fa-solid fa-trash text-[10px]" />
          删除
        </button>
        </div>
      </div>
    </div>
  );
}

/* ─── Main page ─── */
export function Datasets() {
  const [datasets, setDatasets] = useState<DatasetProfile[]>([]);
  const [projectStatus, setProjectStatus] = useState<ProjectStatus | null>(null);
  const [jobs, setJobs] = useState<JobResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [showCreate, setShowCreate] = useState(false);
  const [editTarget, setEditTarget] = useState<DatasetProfile | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [jobSubmitting, setJobSubmitting] = useState(false);
  const [jobError, setJobError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [uploadResult, setUploadResult] = useState<DatasetUploadResult | null>(null);
  const [formatByDataset, setFormatByDataset] = useState<Record<string, TransformJobCreate["format"]>>({});

  const load = useCallback(async () => {
    try {
      const items = await fetchDatasets();
      setDatasets(items);
      setLoadError(null);
    } catch {
      setLoadError("无法加载数据集");
      return;
    }

    try {
      const status = await fetchProjectStatus();
      setProjectStatus(status);
    } catch {
      setProjectStatus(null);
    }

    try {
      const jobItems = await fetchJobs();
      setJobs(jobItems);
    } catch {
      setJobs([]);
    }
  }, []);

  useEffect(() => {
    load().finally(() => setLoading(false));
  }, [load]);

  async function handleCreate(form: DatasetFormState) {
    setSubmitting(true);
    setSubmitError(null);
    try {
      await createDataset({
        name: form.name.trim(),
        class_names: form.class_names,
        train: form.train,
        val: form.val,
        test: form.test,
      });
      setShowCreate(false);
      await load();
    } catch {
      setSubmitError("创建失败，请检查名称是否已存在");
    } finally {
      setSubmitting(false);
    }
  }

  async function submitPlatformAction(action: () => Promise<JobResponse>) {
    setJobSubmitting(true);
    setJobError(null);
    try {
      await action();
      await load();
    } catch {
      setJobError("任务提交失败，请确认后端服务和 odp-platform 可用。");
    } finally {
      setJobSubmitting(false);
    }
  }

  function handleInit() {
    void submitPlatformAction(() => submitInitJob());
  }

  function handleTransform(name: string, format: TransformJobCreate["format"]) {
    void submitPlatformAction(() => submitTransformJob({ dataset: name, format, task_type: "segment" }));
  }

  async function handleUploaded(result: DatasetUploadResult) {
    const detectedFormat = toTransformFormat(result.detected_format.format);
    setUploadResult(result);
    setFormatByDataset((current) => ({ ...current, [result.dataset_name]: detectedFormat }));
    await load();
  }

  function handleFormatChange(name: string, format: TransformJobCreate["format"]) {
    setFormatByDataset((current) => ({ ...current, [name]: format }));
  }

  function handleValidate(name: string) {
    void submitPlatformAction(() => submitValidateJob({ dataset: name, task_type: "segment" }));
  }

  function handleConfig() {
    void submitPlatformAction(() => submitConfigJob({ task: "train", force: true }));
  }

  async function handleEdit(form: DatasetFormState) {
    if (!editTarget) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      await updateDataset(editTarget.name, {
        class_names: form.class_names,
        train: form.train,
        val: form.val,
        test: form.test,
      });
      setEditTarget(null);
      await load();
    } catch {
      setSubmitError("保存失败，请稍后重试");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleDeleteConfirm() {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      await deleteDataset(deleteTarget);
      setDeleteTarget(null);
      await load();
    } catch {
      setDeleteTarget(null);
    } finally {
      setDeleting(false);
    }
  }

  if (loading) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center py-20">
        <div className="w-8 h-8 border-2 border-[var(--terracotta)] border-t-transparent rounded-full animate-spin mb-3" />
        <p className="text-[var(--mid-gray)] text-sm">加载中...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen px-6 py-10 max-w-5xl mx-auto">
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-[var(--charcoal)] mb-2" style={{ fontFamily: "var(--font-display)" }}>数据集管理</h1>
          <p className="text-[var(--mid-gray)] text-sm">管理训练数据集、类别与划分比例</p>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="flex items-center gap-2 px-4 py-2.5 bg-[var(--terracotta)] hover:bg-[var(--terracotta-light)] text-white rounded-xl text-sm font-medium transition-all"
        >
          <i className="fa-solid fa-plus" />
          新增数据集
        </button>
      </div>

      <ProjectStatusCard
        status={projectStatus}
        latestJobs={jobs.filter((job) => ["init", "transform", "validate", "config"].includes(job.task))}
        onInit={handleInit}
        loading={jobSubmitting}
      />

      <DatasetUploadCard onUploaded={(result) => void handleUploaded(result)} />

      {uploadResult && (
        <UploadResultCard
          result={uploadResult}
          selectedFormat={formatByDataset[uploadResult.dataset_name] ?? toTransformFormat(uploadResult.detected_format.format)}
          onFormatChange={(format) => handleFormatChange(uploadResult.dataset_name, format)}
          onTransform={() => handleTransform(
            uploadResult.dataset_name,
            formatByDataset[uploadResult.dataset_name] ?? toTransformFormat(uploadResult.detected_format.format)
          )}
        />
      )}

      {jobError && (
        <div className="mb-6 rounded-xl border border-red-100 bg-red-50 px-4 py-3 text-sm text-red-500">
          <i className="fa-solid fa-circle-exclamation mr-2" />
          {jobError}
        </div>
      )}

      {loadError ? (
        <div className="card p-16 text-center">
          <i className="fa-solid fa-circle-exclamation text-3xl text-[var(--terracotta)] mb-3" />
          <p className="text-[var(--warm-gray)]">{loadError}</p>
          <button onClick={load} className="mt-2 text-sm text-[var(--terracotta)] underline">重试</button>
        </div>
      ) : datasets.length === 0 ? (
        <div className="card p-16 text-center">
          <div className="w-16 h-16 rounded-2xl bg-[var(--parchment)] flex items-center justify-center mx-auto mb-4">
            <i className="fa-solid fa-folder-open text-2xl text-[var(--light-gray)]" />
          </div>
          <h3 className="font-semibold text-[var(--charcoal)] text-lg mb-2" style={{ fontFamily: "var(--font-display)" }}>还没有训练数据集</h3>
          <p className="text-sm text-[var(--mid-gray)] mb-6 max-w-sm mx-auto">
            创建数据集后，可用于后续数据转换、质检与模型训练。
          </p>
          <button
            onClick={() => setShowCreate(true)}
            className="inline-flex items-center gap-2 px-5 py-2.5 bg-[var(--terracotta)] text-white rounded-xl text-sm font-medium"
          >
            <i className="fa-solid fa-upload" />
            立即上传
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {datasets.map((ds) => (
            <DatasetCard
              key={ds.name}
              ds={ds}
              sourceFormat={formatByDataset[ds.name] ?? "coco"}
              onSourceFormatChange={(format) => handleFormatChange(ds.name, format)}
              onEdit={setEditTarget}
              onDelete={setDeleteTarget}
              onTransform={handleTransform}
              onValidate={handleValidate}
              onConfig={handleConfig}
            />
          ))}
        </div>
      )}

      {/* Create modal */}
      {showCreate && (
        <ModalShell title="新增数据集" onClose={() => setShowCreate(false)}>
          <DatasetForm mode="create" onSubmit={handleCreate} loading={submitting} />
          {submitError && (
            <div className="mt-3 flex items-center gap-2 text-sm text-red-400">
              <i className="fa-solid fa-circle-exclamation" />
              {submitError}
            </div>
          )}
        </ModalShell>
      )}

      {/* Edit modal */}
      {editTarget && (
        <ModalShell title="编辑数据集" onClose={() => setEditTarget(null)}>
          <DatasetForm mode="edit" initial={editTarget} onSubmit={handleEdit} loading={submitting} />
          {submitError && (
            <div className="mt-3 flex items-center gap-2 text-sm text-red-400">
              <i className="fa-solid fa-circle-exclamation" />
              {submitError}
            </div>
          )}
        </ModalShell>
      )}

      {/* Delete confirm */}
      {deleteTarget && (
        <DeleteConfirm
          name={deleteTarget}
          onConfirm={handleDeleteConfirm}
          onCancel={() => setDeleteTarget(null)}
        />
      )}
    </div>
  );
}
