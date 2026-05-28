import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchOverview } from "../lib/api";
import type { OverviewPayload, RunSummary } from "../types";

function formatDate(value: string | undefined) {
  if (!value) return "";
  return value.replace("T", " ").slice(0, 16);
}

function StatCard({ label, value, sub, accent }: { label: string; value: string | number; sub?: string; accent?: boolean }) {
  return (
    <div className="card p-5 flex flex-col gap-1">
      <p className="text-xs text-[var(--mid-gray)] font-medium uppercase tracking-wider">{label}</p>
      <p className={`text-2xl font-semibold ${accent ? "text-[var(--terracotta)]" : "text-[var(--charcoal)]"}`} style={{ fontFamily: "var(--font-display)" }}>{value}</p>
      {sub && <p className="text-xs text-[var(--mid-gray)]">{sub}</p>}
    </div>
  );
}

function ProjectCard({ run }: { run: RunSummary }) {
  return (
    <Link to="/analysis" className="card p-4 flex flex-col gap-3 hover:shadow-[var(--shadow-md)] transition-all group cursor-pointer">
      <div className="flex items-start justify-between">
        <div>
          <p className="font-semibold text-[var(--charcoal)] text-sm">{run.dataset}</p>
          <p className="text-xs text-[var(--mid-gray)] mt-0.5">{run.model}</p>
        </div>
        <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium border ${
          run.status === "completed"
            ? "bg-[var(--sage-pale)] text-[var(--sage)] border-[var(--sage-light)]"
            : run.status === "running"
            ? "bg-[var(--terracotta-pale)] text-[var(--terracotta)] border-[var(--terracotta-light)]"
            : "bg-[var(--parchment)] text-[var(--warm-gray)] border-[var(--border)]"
        }`}>
          {run.status === "completed" ? "已完成" : run.status === "running" ? "分析中" : "待处理"}
        </span>
      </div>
      {run.metric != null && (
        <div className="flex items-baseline gap-2">
          <span className="text-xl font-semibold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>{(run.metric * 100).toFixed(1)}</span>
          <span className="text-xs text-[var(--mid-gray)]">% mIoU</span>
        </div>
      )}
      <div className="text-xs text-[var(--mid-gray)]">{formatDate(run.started_at)}</div>
    </Link>
  );
}

export function Home() {
  const [overview, setOverview] = useState<OverviewPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchOverview()
      .then(setOverview)
      .catch(() => setError("无法连接服务器"))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-[var(--terracotta)] border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-[var(--mid-gray)] text-sm">正在加载...</p>
        </div>
      </div>
    );
  }

  if (error || !overview) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center space-y-3">
          <i className="fa-solid fa-wifi text-3xl text-[var(--light-gray)]" />
          <p className="text-[var(--warm-gray)]">{error ?? "暂无数据"}</p>
          <button onClick={() => window.location.reload()} className="text-sm text-[var(--terracotta)] underline">重新加载</button>
        </div>
      </div>
    );
  }

  const datasets = overview.datasets;
  const latestRun = overview.runs[0];
  const pipeline = overview.pipeline;
  const totalImages = datasets.reduce((sum, ds) => sum + ds.splits.train + (ds.splits.val ?? 0) + (ds.splits.test ?? 0), 0);

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section
        className="relative overflow-hidden px-6 py-16 md:py-24"
        style={{ background: "linear-gradient(160deg, var(--ivory) 0%, var(--cream) 60%, var(--parchment) 100%)" }}
      >
        {/* Decorative grid pattern */}
        <div className="absolute inset-0 opacity-[0.035]" style={{
          backgroundImage: `linear-gradient(var(--terracotta) 1px, transparent 1px), linear-gradient(90deg, var(--terracotta) 1px, transparent 1px)`,
          backgroundSize: "48px 48px",
        }} />

        <div className="relative max-w-5xl mx-auto">
          <div className="max-w-2xl stagger-children">
            <p className="text-xs font-medium uppercase tracking-widest text-[var(--terracotta)] mb-3 animate-fade-up" style={{ letterSpacing: "0.12em" }}>
              户型图智能分析
            </p>
            <h1 className="text-4xl md:text-5xl font-bold text-[var(--charcoal)] leading-tight mb-4 animate-fade-up" style={{ fontFamily: "var(--font-display)" }}>
              让家的每一平米<br />
              <em className="not-italic text-[var(--terracotta)]">物尽其用</em>
            </h1>
            <p className="text-base text-[var(--warm-gray)] leading-relaxed mb-8 animate-fade-up">
              上传您的户型图，RoomWise 将自动识别房间类型、计算空间面积，并生成专业的装修建议与动线优化方案。
            </p>
            <div className="flex flex-wrap gap-3 animate-fade-up">
              <Link
                to="/analysis?mode=upload"
                className="inline-flex items-center gap-2 px-6 py-3 bg-[var(--terracotta)] hover:bg-[var(--terracotta-light)] text-white rounded-xl font-medium text-sm transition-all shadow-[var(--shadow-md)] hover:shadow-[var(--shadow-lg)]"
              >
                <i className="fa-solid fa-upload" />
                立即上传户型图
              </Link>
              <Link
                to="/datasets"
                className="inline-flex items-center gap-2 px-6 py-3 bg-[var(--warm-white)] hover:bg-[var(--ivory)] text-[var(--charcoal)] border border-[var(--border)] rounded-xl font-medium text-sm transition-all"
              >
                <i className="fa-solid fa-folder-open" />
                查看已有户型图
              </Link>
            </div>
          </div>

          {/* Floating floor plan preview */}
          <div className="hidden lg:block absolute right-8 top-8 bottom-8 w-72 animate-fade-in" style={{ animationDelay: "0.3s" }}>
            <div className="w-full h-full rounded-2xl border border-[var(--border)] shadow-[var(--shadow-lg)] p-4" style={{ background: "var(--warm-white)" }}>
              <p className="text-[10px] uppercase tracking-wider text-[var(--mid-gray)] mb-3">户型示意</p>
              <div className="flex flex-col gap-2 flex-1">
                <div className="flex gap-2 flex-1">
                  <div className="room-living rounded-xl flex-1 flex items-center justify-center text-xs font-medium">客厅 35%</div>
                  <div className="flex flex-col gap-2 flex-1">
                    <div className="room-bedroom rounded-xl flex-1 flex items-center justify-center text-xs font-medium">主卧 24%</div>
                    <div className="room-kitchen rounded-xl flex-1 flex items-center justify-center text-xs font-medium">厨房 16%</div>
                  </div>
                </div>
                <div className="flex gap-2" style={{ height: "33%" }}>
                  <div className="room-bathroom rounded-xl flex-1 flex items-center justify-center text-xs font-medium">卫生间 9%</div>
                  <div className="room-balcony rounded-xl flex-1 flex items-center justify-center text-xs font-medium">阳台 16%</div>
                </div>
              </div>
              <div className="mt-3 flex items-center justify-between">
                <span className="text-[10px] text-[var(--mid-gray)]">置信度 91.2%</span>
                <span className="text-[10px] text-[var(--sage)]">✓ 已完成分析</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Bar */}
      <section className="px-6 -mt-4 relative z-10 max-w-5xl mx-auto">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard label="已分析户型图" value={datasets.length} sub="张" accent />
          <StatCard label="图像总数" value={totalImages} sub="张训练图像" />
          <StatCard label="训练模型" value={overview.runs.length} sub="个已完成训练" />
          <StatCard label="装修建议" value={overview.agent.advice.length} sub="条专业建议" accent />
        </div>
      </section>

      {/* How it works */}
      <section className="px-6 py-14 max-w-5xl mx-auto">
        <h2 className="text-2xl font-bold text-[var(--charcoal)] mb-2" style={{ fontFamily: "var(--font-display)" }}>分析流程</h2>
        <p className="text-[var(--mid-gray)] text-sm mb-8">从上传到建议，三步完成户型分析</p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            { step: "01", icon: "fa-upload", title: "上传户型图", desc: "支持 JPG / PNG / PDF 格式，系统自动识别建筑轮廓线与房间边界", color: "var(--terracotta)" },
            { step: "02", icon: "fa-magic", title: "AI 智能分割", desc: "基于 YOLO 分割模型识别客厅、卧室、厨房等区域，计算各空间面积与比例", color: "var(--sage)" },
            { step: "03", icon: "fa-lightbulb", title: "生成装修建议", desc: "结合空间结构与面积数据，提供动线优化、收纳规划与采光改善建议", color: "var(--warm-gray)" },
          ].map(({ step, icon, title, desc, color }, i) => (
            <div key={title} className="relative animate-fade-up" style={{ animationDelay: `${i * 0.1}s` }}>
              <div className="card p-6 flex flex-col gap-4 h-full">
                <div className="flex items-start justify-between">
                  <span className="text-4xl font-bold text-[var(--parchment)] select-none" style={{ fontFamily: "var(--font-display)" }}>{step}</span>
                  <i className={`fa-solid ${icon} text-lg`} style={{ color }} />
                </div>
                <div>
                  <h3 className="font-semibold text-[var(--charcoal)] mb-1">{title}</h3>
                  <p className="text-sm text-[var(--warm-gray)] leading-relaxed">{desc}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Recent projects */}
      <section className="px-6 pb-16 max-w-5xl mx-auto">
        <div className="flex items-center justify-between mb-5">
          <div>
            <h2 className="text-2xl font-bold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>最近分析</h2>
            <p className="text-[var(--mid-gray)] text-sm mt-1">您的分析记录</p>
          </div>
          <Link to="/datasets" className="text-sm text-[var(--terracotta)] hover:text-[var(--terracotta-light)] font-medium flex items-center gap-1">
            查看全部 <i className="fa-solid fa-arrow-right text-xs" />
          </Link>
        </div>

        {overview.runs.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {overview.runs.slice(0, 3).map((run) => (
              <ProjectCard key={run.run_id} run={run} />
            ))}
          </div>
        ) : (
          <div className="card p-10 text-center">
            <i className="fa-solid fa-folder-open text-3xl text-[var(--light-gray)] mb-3" />
            <p className="text-[var(--mid-gray)] mb-3">还没有分析记录</p>
            <Link to="/analysis?mode=upload" className="text-sm text-[var(--terracotta)] font-medium">
              上传第一张户型图 →
            </Link>
          </div>
        )}
      </section>

      {/* Space optimization teaser */}
      {overview.inference?.regions && overview.inference.regions.length > 0 && (
        <section className="px-6 pb-16 max-w-5xl mx-auto">
          <div className="card p-8">
            <div className="flex flex-col md:flex-row gap-8 items-start">
              <div className="flex-1">
                <p className="text-xs uppercase tracking-widest text-[var(--terracotta)] mb-2" style={{ letterSpacing: "0.1em" }}>空间洞察</p>
                <h2 className="text-2xl font-bold text-[var(--charcoal)] mb-3" style={{ fontFamily: "var(--font-display)" }}>各空间面积分布</h2>
                <p className="text-sm text-[var(--warm-gray)] leading-relaxed mb-6">{overview.inference.summary || "基于最新分析结果的空间面积统计"}</p>
                <div className="space-y-3">
                  {overview.inference.regions.map((region) => (
                    <div key={region.name} className="flex items-center gap-3">
                      <div className="w-20 text-xs text-[var(--warm-gray)]">{region.name}</div>
                      <div className="flex-1 h-1.5 bg-[var(--parchment)] rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${Math.round(region.area_ratio * 100)}%`,
                            background: region.color || "var(--terracotta)"
                          }}
                        />
                      </div>
                      <div className="w-10 text-xs text-right font-medium text-[var(--charcoal)]">{Math.round(region.area_ratio * 100)}%</div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="w-full md:w-56">
                <div className="text-xs uppercase tracking-wider text-[var(--mid-gray)] mb-3">快速建议</div>
                <div className="space-y-2">
                  {overview.agent.advice.slice(0, 3).map((item) => (
                    <div key={item.title} className="flex items-start gap-2 p-3 rounded-xl bg-[var(--ivory)] text-xs">
                      <i className={`fa-solid mt-0.5 ${
                        item.priority === "high" ? "fa-triangle-exclamation text-[var(--terracotta)]" :
                        item.priority === "medium" ? "fa-circle text-[var(--terracotta-light)]" :
                        "fa-circle text-[var(--sage-light)]"
                      }`} />
                      <div>
                        <p className="font-medium text-[var(--charcoal)]">{item.title}</p>
                        <p className="text-[var(--warm-gray)] mt-0.5 leading-relaxed">{item.description}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
