import { useEffect, useState, useCallback } from "react";
import { Link } from "react-router-dom";
import type { EChartsOption } from "echarts";
import { EChart } from "../components/EChart";
import { fetchOverview } from "../lib/api";
import type { OverviewPayload } from "../types";

function formatDate(value: string | undefined) {
  if (!value) return "-";
  return value.replace("T", " ");
}

export function Dashboard() {
  const [overview, setOverview] = useState<OverviewPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchOverview()
      .then((data) => { if (!cancelled) { setOverview(data); setError(null); } })
      .catch(() => { if (!cancelled) setError("无法连接到后端服务，请确保 odp-web-backend 正在运行。"); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-400">
        <i className="fa-solid fa-circle-notch animate-spin text-2xl mr-3" />
        正在载入工作台...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center space-y-4 max-w-md">
          <i className="fa-solid fa-circle-exclamation text-5xl text-red-400" />
          <h2 className="text-xl font-bold text-slate-700">后端未连接</h2>
          <p className="text-slate-500 text-sm">{error}</p>
          <p className="text-xs text-slate-400">
            提示：运行 <code className="bg-slate-100 px-1 rounded">PYTHONPATH=apps/web-backend/src python -m uvicorn odp_web_backend.main:app --port 8000</code>
          </p>
        </div>
      </div>
    );
  }

  if (!overview) return null;

  const dataset = overview.datasets[0];
  const latestRun = overview.runs[0];
  const validation = overview.metrics.validation_summary;
  const curve = overview.metrics.training_curve ?? [];
  const inference = overview.inference;
  const agent = overview.agent;

  const validationCounts = validation?.counts_by_severity ?? { PASS: 0, INFO: 0, WARNING: 0, ERROR: 0 };
  const validationTotal = validationCounts.PASS + validationCounts.INFO + validationCounts.WARNING + validationCounts.ERROR;

  const qcOption: EChartsOption = {
    tooltip: { trigger: "item" as const },
    legend: { bottom: "0", itemWidth: 10, itemHeight: 10, textStyle: { fontSize: 11, color: "#64748b" } },
    series: [
      {
        name: "数据状态",
        type: "pie" as const,
        radius: ["50%", "75%"],
        avoidLabelOverlap: false,
        itemStyle: { borderRadius: 6, borderColor: "#fff", borderWidth: 2 },
        label: { show: true, position: "center", formatter: `${validationTotal}\n总检查项`, fontSize: 13, fontWeight: "bold", color: "#1e293b" },
        data: [
          { value: validationCounts.PASS, name: "PASS", itemStyle: { color: "#10b981" } },
          { value: validationCounts.INFO, name: "INFO", itemStyle: { color: "#0ea5e9" } },
          { value: validationCounts.WARNING, name: "WARNING", itemStyle: { color: "#f59e0b" } },
          { value: validationCounts.ERROR, name: "ERROR", itemStyle: { color: "#ef4444" } },
        ],
      },
    ],
  };

  const trainingOption: EChartsOption = {
    grid: { top: "10%", bottom: "12%", left: "5%", right: "5%", containLabel: true },
    xAxis: { type: "category" as const, data: curve.map((p) => p.epoch), axisLine: { lineStyle: { color: "#cbd5e1" } }, axisLabel: { color: "#64748b" } },
    yAxis: [
      { type: "value" as const, name: "mIoU", min: 0, max: 1, splitLine: { lineStyle: { type: "dashed", color: "#f1f5f9" } }, axisLabel: { color: "#64748b" } },
      { type: "value" as const, name: "Loss", min: 0, splitLine: { show: false }, axisLabel: { color: "#64748b" } },
    ],
    series: [
      { name: "mIoU", type: "line" as const, data: curve.map((p) => p.miou), symbol: "none", itemStyle: { color: "#10b981" }, lineStyle: { width: 2 }, smooth: true },
      { name: "Loss", type: "line" as const, yAxisIndex: 1, data: curve.map((p) => p.loss), symbol: "none", itemStyle: { color: "#3b82f6" }, lineStyle: { width: 2 }, smooth: true },
    ],
  };

  const spaceOption: EChartsOption = {
    tooltip: { trigger: "item" as const },
    series: [{
      type: "pie" as const, radius: "80%", center: ["50%", "50%"],
      data: (inference?.regions ?? []).map((r) => ({ value: Math.round(r.area_ratio * 100), name: r.name, itemStyle: { color: r.color } })),
      label: { show: true, position: "inside", formatter: "{b}", fontSize: 10, color: "#fff" },
    }],
  };

  const pipelineStages = [
    { icon: "fa-folder-open", title: "原始数据", status: "done" },
    { icon: "fa-wand-magic-sparkles", title: "数据转换", status: overview.pipeline.find((s) => s.key === "convert")?.status ?? "ready" },
    { icon: "fa-clipboard-check", title: "质检", status: overview.pipeline.find((s) => s.key === "qc")?.status ?? "ready" },
    { icon: "fa-sliders", title: "配置", status: overview.pipeline.find((s) => s.key === "config")?.status ?? "ready" },
    { icon: "fa-scissors", title: "分割训练", status: overview.pipeline.find((s) => s.key === "train")?.status ?? "ready" },
    { icon: "fa-square-poll-vertical", title: "分割评估", status: overview.pipeline.find((s) => s.key === "eval")?.status ?? "ready" },
    { icon: "fa-robot", title: "Agent分析", status: overview.pipeline.find((s) => s.key === "agent")?.status ?? "ready" },
  ];

  return (
    <div className="flex-1 p-6 overflow-y-auto space-y-6 min-w-0">
      {/* Welcome */}
      <div className="bg-white rounded-2xl p-6 border border-slate-200/80 shadow-xs flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-800 flex items-center gap-2">您好，设计师 👋</h2>
          <p className="text-slate-400 text-xs mt-1">{overview.project_name}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Link to="/datasets" className="px-4 py-2 bg-blue-50 hover:bg-blue-100 text-blue-600 border border-blue-200 rounded-xl font-medium transition-all flex items-center gap-2 text-xs">
            <i className="fa-solid fa-cloud-arrow-up" />上传数据
          </Link>
          <Link to="/training" className="px-4 py-2 bg-slate-50 hover:bg-slate-100 text-slate-600 border border-slate-200 rounded-xl font-medium transition-all flex items-center gap-2 text-xs">
            <i className="fa-solid fa-play" />运行训练
          </Link>
        </div>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs">
          <p className="text-xs font-medium text-slate-400">系统状态</p>
          <h4 className="text-lg font-bold text-green-600 mt-1 flex items-center gap-1.5">
            <i className="fa-solid fa-circle-check text-base" />正常
          </h4>
          <p className="text-[11px] text-slate-400 mt-1">{overview.datasets.length} 个数据集</p>
        </div>
        <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs">
          <p className="text-xs font-medium text-slate-400">数据概览</p>
          <div className="flex items-baseline gap-3 mt-1">
            <span className="text-xl font-bold text-slate-800">{overview.datasets.length}</span>
            <span className="text-xs text-slate-400">数据集</span>
            {dataset && (
              <>
                <span className="text-xl font-bold text-slate-800">{dataset.splits.train + (dataset.splits.val ?? 0) + (dataset.splits.test ?? 0)}</span>
                <span className="text-xs text-slate-400">图像</span>
              </>
            )}
          </div>
        </div>
        <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs">
          <p className="text-xs font-medium text-slate-400">训练状态</p>
          <h4 className="text-lg font-bold text-blue-600 mt-1 flex items-center gap-1.5">
            <i className={`fa-solid ${latestRun?.status === "completed" ? "fa-circle-check" : "fa-circle-notch animate-spin"} text-base`} />
            {latestRun?.status === "completed" ? "已完成" : latestRun?.status === "running" ? "进行中" : "暂无训练"}
          </h4>
          <p className="text-[11px] text-slate-400 mt-1">最新: {formatDate(latestRun?.started_at)}</p>
        </div>
        <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs">
          <p className="text-xs font-medium text-slate-400">Agent分析</p>
          <h4 className="text-lg font-bold text-purple-600 mt-1 flex items-center gap-1.5">
            <i className="fa-solid fa-bolt text-base" />已就绪
          </h4>
          <p className="text-[11px] text-slate-400 mt-1">{agent.advice?.length ?? 0} 条改造建议</p>
        </div>
      </div>

      {/* Pipeline strip */}
      <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs">
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4">整体流程</h3>
        <div className="flex flex-wrap items-center gap-3">
          {pipelineStages.map(({ icon, title, status }, i) => (
            <div key={title} className="flex items-center gap-2">
              {i > 0 && <i className="fa-solid fa-chevron-right text-slate-300 text-xs" />}
              <div className={`flex items-center gap-2 px-3 py-2 rounded-xl border text-xs ${
                status === "done" ? "bg-green-50 border-green-200" :
                status === "running" ? "bg-blue-50 border-blue-200" :
                status === "warning" ? "bg-amber-50 border-amber-200" :
                "bg-slate-50 border-slate-200"
              }`}>
                <i className={`fa-solid ${icon} ${status === "done" ? "text-green-500" : status === "running" ? "text-blue-500" : "text-slate-400"}`} />
                <div>
                  <p className={`font-medium ${status === "done" ? "text-green-700" : status === "running" ? "text-blue-700" : "text-slate-600"}`}>{title}</p>
                  {status === "done" && <span className="text-[10px] text-green-600"><i className="fa-solid fa-check" /> 已完成</span>}
                  {status === "running" && <span className="text-[10px] text-blue-600"><i className="fa-solid fa-spinner fa-spin" /> 进行中</span>}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Three-column row */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Dataset overview */}
        <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-slate-700 mb-4">数据集概览</h3>
            <div className="space-y-3">
              {overview.datasets.map((ds) => (
                <div key={ds.name} className="flex items-center gap-3 p-2.5 rounded-xl bg-slate-50 border border-slate-200/60">
                  <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center text-blue-600">
                    <i className="fa-solid fa-images text-lg" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h4 className="font-semibold text-slate-800 text-xs truncate">{ds.name}</h4>
                    <p className="text-[11px] text-slate-400 mt-0.5">
                      {(ds.splits.train ?? 0) + (ds.splits.val ?? 0) + (ds.splits.test ?? 0)} 张图像 · {ds.class_names.join(", ")}
                    </p>
                  </div>
                  <span className={`px-2 py-0.5 rounded text-[10px] border ${
                    ds.status === "ready" ? "bg-green-50 text-green-600 border-green-200" : "bg-slate-50 text-slate-500 border-slate-200"
                  }`}>{ds.status === "ready" ? "已就绪" : ds.status}</span>
                </div>
              ))}
              {overview.datasets.length === 0 && (
                <div className="text-center py-8 text-slate-400 text-xs">
                  <i className="fa-solid fa-folder-open text-2xl mb-2" />
                  <p>暂无数据集，请先上传数据</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* QC chart */}
        <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs flex flex-col justify-between">
          <h3 className="text-sm font-bold text-slate-700 mb-2">质检摘要</h3>
          <EChart option={qcOption} className="flex-1 h-36" />
          <Link to="/datasets" className="w-full py-2 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-xl text-xs font-medium text-slate-600 transition-all text-center mt-3 block">
            查看质检报告
          </Link>
        </div>

        {/* Config trace */}
        <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-slate-700 mb-4">配置追踪</h3>
            <div className="space-y-2.5">
              {overview.metrics.config_trace.map((item) => (
                <div key={item.source} className="flex items-center justify-between p-2.5 rounded-xl bg-slate-50 border border-slate-200/60 text-xs">
                  <span className="text-slate-500">{item.source}</span>
                  <span className={`font-mono px-2 py-0.5 rounded border ${
                    item.source === "CLI" ? "text-red-600 bg-red-50 border-red-100" : "text-slate-700 bg-white border-slate-200"
                  }`}>{item.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Training chart + Activity log */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs lg:col-span-2 flex flex-col">
          <div className="flex justify-between items-center mb-4">
            <div>
              <h3 className="text-sm font-bold text-slate-700">
                训练状态
                {latestRun && (
                  <span className="ml-2 px-2 py-0.5 bg-green-50 text-green-600 rounded text-[11px] font-normal border border-green-200">
                    {latestRun.status === "completed" ? "已完成" : latestRun.status === "running" ? "进行中" : latestRun.status}
                  </span>
                )}
              </h3>
              {latestRun && <p className="text-[11px] text-slate-400 mt-0.5">模型: {latestRun.model} · 轮数: {latestRun.epochs}</p>}
            </div>
            <div className="flex items-center gap-4 text-xs">
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-green-500 inline-block" />mIoU ({(latestRun?.metric ?? 0).toFixed(4)})
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-blue-500 inline-block" />Loss ({curve.at(-1)?.loss?.toFixed(2) ?? "0.00"})
              </span>
            </div>
          </div>
          {curve.length > 0
            ? <EChart option={trainingOption} className="flex-1 min-h-[220px]" />
            : (
              <div className="flex-1 min-h-[220px] flex items-center justify-center text-slate-400 text-sm">
                <div className="text-center">
                  <i className="fa-solid fa-chart-line text-3xl mb-2" />
                  <p>暂无训练曲线数据</p>
                  <Link to="/training" className="text-blue-600 text-xs mt-2 inline-block">去发起训练 →</Link>
                </div>
              </div>
            )
          }
        </div>

        <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-sm font-bold text-slate-700">活动日志</h3>
            </div>
            <div className="space-y-4 max-h-[220px] overflow-y-auto pr-1">
              {latestRun && (
                <div className="flex gap-3 text-xs">
                  <span className="text-slate-400 font-mono flex-shrink-0">{formatDate(latestRun.started_at).slice(11, 19)}</span>
                  <p className="text-slate-600 flex-1"><strong className="text-slate-800">分割训练{latestRun.status === "completed" ? "完成" : "进行中"}</strong>，指标 {(latestRun.metric ?? 0).toFixed(4)}</p>
                </div>
              )}
              <div className="flex gap-3 text-xs">
                <span className="text-slate-400 font-mono flex-shrink-0">{formatDate(overview.generated_at).slice(11, 19)}</span>
                <p className="text-slate-600 flex-1"><strong className="text-slate-800">系统就绪</strong>，质检通过</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
