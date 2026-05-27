import { useEffect, useState } from "react";
import { fetchOverview } from "../lib/api";
import type { AgentReport, InferenceResult } from "../types";

export function Inference() {
  const [inference, setInference] = useState<InferenceResult | null>(null);
  const [agent, setAgent] = useState<AgentReport | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchOverview()
      .then((data) => {
        setInference(data.inference);
        setAgent(data.agent);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-400">
        <i className="fa-solid fa-circle-notch animate-spin text-2xl mr-3" />
        加载推理结果...
      </div>
    );
  }

  return (
    <div className="flex-1 p-6 overflow-y-auto space-y-6 min-w-0">
      <div className="bg-white rounded-2xl p-6 border border-slate-200/80 shadow-xs">
        <h2 className="text-lg font-bold text-slate-800 mb-1">Agent 分析与推理</h2>
        <p className="text-slate-400 text-xs">基于最新分割模型的户型图分析与改造建议</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Floor plan visualization */}
        <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs lg:col-span-1">
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-bold text-slate-700 text-sm">分割结果可视化</h3>
            <span className="px-2 py-0.5 bg-green-50 text-green-600 rounded text-[11px] border border-green-200">
              <i className="fa-solid fa-check mr-1" />已就绪
            </span>
          </div>
          <p className="text-xs text-slate-400 mb-3">分割结果：{inference?.summary}</p>
          <div className="w-full aspect-square bg-slate-100 rounded-xl p-3 relative border border-slate-200 flex flex-col gap-1.5 overflow-hidden">
            <div className="flex gap-1.5 flex-1">
              <div className="bg-emerald-200/70 border border-emerald-400/40 rounded flex-1 flex items-center justify-center text-[10px] text-emerald-800 font-medium">
                {inference?.regions[0]?.name ?? "客厅"}
              </div>
              <div className="flex flex-col gap-1.5 flex-1">
                <div className="bg-amber-200/70 border border-amber-400/40 rounded flex-1 flex items-center justify-center text-[10px] text-amber-800 font-medium">
                  {inference?.regions[1]?.name ?? "主卧"}
                </div>
                <div className="bg-orange-200/70 border border-orange-400/40 rounded flex-1 flex items-center justify-center text-[10px] text-orange-800 font-medium">
                  {inference?.regions[2]?.name ?? "次卧"}
                </div>
              </div>
            </div>
            <div className="flex gap-1.5 h-1/3">
              <div className="bg-rose-200/70 border border-rose-400/40 rounded flex-1 flex items-center justify-center text-[10px] text-rose-800 font-medium">
                {inference?.regions[2]?.name ?? "厨房"}
              </div>
              <div className="bg-purple-200/70 border border-purple-400/40 rounded w-1/3 flex items-center justify-center text-[10px] text-purple-800 font-medium">
                {inference?.regions[3]?.name ?? "卫生间"}
              </div>
            </div>
          </div>
          {inference?.confidence != null && inference.confidence > 0 && (
            <p className="text-xs text-slate-400 text-center mt-3">置信度: <span className="text-blue-600 font-semibold">{(inference.confidence * 100).toFixed(1)}%</span></p>
          )}
        </div>

        {/* Agent advice */}
        <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs lg:col-span-2">
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-bold text-slate-700 text-sm">AI 智能改造建议</h3>
            <span className="px-2 py-0.5 bg-purple-50 text-purple-600 rounded text-[11px] border border-purple-200">
              {agent?.advice.length ?? 0} 条建议
            </span>
          </div>
          {agent?.advice && agent.advice.length > 0 ? (
            <div className="space-y-3">
              {agent.advice.map((item, index) => (
                <div key={item.title} className="p-4 bg-slate-50 rounded-xl border border-slate-200/60">
                  <div className="flex items-center gap-2 font-semibold text-slate-800 text-sm mb-1">
                    <i className={`fa-solid ${index === 0 ? "fa-wand-magic-sparkles text-blue-500" : index === 1 ? "fa-box-open text-purple-500" : "fa-sun text-amber-500"}`} />
                    {index + 1}. {item.title}
                    <span className={`ml-auto px-2 py-0.5 rounded text-[10px] border ${
                      item.priority === "high" ? "bg-red-50 text-red-600 border-red-100" :
                      item.priority === "medium" ? "bg-amber-50 text-amber-600 border-amber-100" :
                      "bg-slate-50 text-slate-500 border-slate-200"
                    }`}>
                      {item.priority === "high" ? "高优先级" : item.priority === "medium" ? "中优先级" : "低优先级"}
                    </span>
                  </div>
                  <p className="text-slate-500 text-xs leading-relaxed pl-6">{item.description}</p>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-slate-400 text-sm">
              <i className="fa-solid fa-robot text-2xl mb-2" />
              <p>暂无建议，请先完成训练</p>
            </div>
          )}

          {agent && (
            <div className="mt-5 pt-4 border-t border-slate-100">
              <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">空间流线分析</h4>
              <p className="text-sm text-slate-600">{agent.circulation || "暂无流线数据"}</p>
              <div className="mt-3 flex flex-wrap gap-2">
                {agent.spaces.map((space) => (
                  <span key={space} className="px-2 py-0.5 bg-blue-50 text-blue-600 rounded text-xs border border-blue-100">{space}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
