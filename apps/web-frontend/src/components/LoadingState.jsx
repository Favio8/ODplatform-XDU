import { motion, AnimatePresence } from "framer-motion";
import { Loader2, CheckCircle2, Wrench, ScanEye, Circle } from "lucide-react";

const TOOL_LABELS = {
  get_room_detail: "获取房间尺寸",
  analyze_adjacency: "分析空间邻接",
  estimate_natural_light: "评估采光条件",
  estimate_renovation_budget: "估算装修预算",
};

export default function LoadingState({ reasoningSteps = [], status }) {
  if (status === "waiting") {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex-1 flex flex-col items-center justify-center"
      >
        <ScanEye className="w-10 h-10 text-brand-400 animate-pulse mb-4" />
        <p className="text-sm text-zinc-500">YOLO 正在分割房间...</p>
      </motion.div>
    );
  }

  if (status === "segmenting") {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex-1 flex flex-col items-center justify-center"
      >
        <Loader2 className="w-10 h-10 text-brand-400 animate-spin mb-4" />
        <p className="text-sm text-zinc-500">正在上传分析...</p>
      </motion.div>
    );
  }

  // Show real reasoning steps
  const stepCount = reasoningSteps.length;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flex-1 flex flex-col"
    >
      <div className="flex items-center gap-3 mb-5">
        <div className="relative">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-100 to-brand-200 animate-pulse" />
          <Loader2 className="absolute inset-0 m-auto w-5 h-5 text-brand-600 animate-spin" />
        </div>
        <div>
          <p className="text-sm font-medium text-zinc-600">AI Agent 推理中</p>
          <p className="text-xs text-zinc-400">
            工具调用 · 数据收集 · 生成建议
          </p>
        </div>
      </div>

      <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
        <AnimatePresence initial={false}>
          {reasoningSteps.map((step, i) => {
            const isTool = step.type === "tool_call" || step.type === "phase1-tool_call";
            const isPhase1 = (step.step || "").toString().startsWith("phase1");
            const label = TOOL_LABELS[step.tool] || step.tool;

            return (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.25 }}
                className="flex items-start gap-3 py-1.5"
              >
                {isTool ? (
                  <>
                    <Wrench className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                    <div className="min-w-0">
                      <span className="text-xs font-medium text-zinc-600">
                        {label || step.tool}
                      </span>
                      {step.result_preview && (
                        <p className="text-[11px] text-zinc-400 truncate mt-0.5 max-w-[320px]">
                          {typeof step.result_preview === "string"
                            ? step.result_preview.slice(0, 120)
                            : ""}
                        </p>
                      )}
                    </div>
                  </>
                ) : step.type === "phase1_done" ? (
                  <>
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                    <span className="text-xs text-zinc-500">数据收集完成</span>
                  </>
                ) : step.type === "final" ? (
                  <>
                    <CheckCircle2 className="w-4 h-4 text-brand-400 shrink-0 mt-0.5" />
                    <span className="text-xs text-zinc-500">
                      {step.content
                        ? `分析生成中: ${step.content.slice(0, 80)}...`
                        : "生成分析报告..."}
                    </span>
                  </>
                ) : (
                  <div className="flex items-center gap-3">
                    <Circle className="w-3 h-3 text-zinc-300 shrink-0" />
                    <span className="text-xs text-zinc-400">
                      {(step.content || "").slice(0, 80)}
                    </span>
                  </div>
                )}
              </motion.div>
            );
          })}
        </AnimatePresence>

        {stepCount === 0 && (
          <>
            <div className="flex items-center gap-3 py-1.5">
              <Loader2 className="w-4 h-4 text-brand-400 animate-spin shrink-0" />
              <span className="text-xs text-zinc-400">Agent 正在调用工具...</span>
            </div>
          </>
        )}

        {stepCount > 0 && (
          <div className="flex items-center gap-3 py-1.5">
            <Loader2 className="w-4 h-4 text-brand-400 animate-spin shrink-0" />
            <span className="text-xs text-zinc-400">继续推理...</span>
          </div>
        )}
      </div>
    </motion.div>
  );
}
