import { motion, AnimatePresence } from "framer-motion";
import { Wrench, Brain, CheckCircle2, Loader2, ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";

const TOOL_LABELS = {
  get_room_detail: "获取房间数据",
  analyze_adjacency: "分析邻接关系",
  estimate_natural_light: "评估采光条件",
  estimate_renovation_budget: "估算装修预算",
};

export default function ReasoningSteps({ steps, isStreaming }) {
  const [expanded, setExpanded] = useState(true);

  if (!steps || steps.length === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="rounded-xl overflow-hidden"
      style={{
        background: "rgba(255,255,255,0.5)",
        backdropFilter: "blur(10px)",
        border: "1px solid rgba(0,0,0,0.05)",
      }}
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-white/30 transition-colors cursor-pointer"
      >
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-brand-500" />
          <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
            AI 推理过程
          </span>
          <span className="text-xs text-zinc-300">({steps.length} 步)</span>
        </div>
        <motion.div animate={{ rotate: expanded ? 90 : 0 }} transition={{ duration: 0.2 }}>
          <ChevronRight className="w-4 h-4 text-zinc-400" />
        </motion.div>
      </button>

      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 space-y-2">
              {steps.map((step, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                  className="flex items-start gap-3 py-1.5"
                >
                  {step.type === "tool_call" ? (
                    <>
                      <Wrench className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
                      <div className="min-w-0">
                        <span className="text-xs font-medium text-zinc-600">
                          {TOOL_LABELS[step.tool] || step.tool}
                        </span>
                        {step.result_preview && (
                          <p className="text-xs text-zinc-400 truncate mt-0.5 max-w-[300px]">
                            {step.result_preview.slice(0, 100)}
                          </p>
                        )}
                      </div>
                    </>
                  ) : (
                    <>
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                      <p className="text-xs text-zinc-500 leading-relaxed">
                        {step.content?.slice(0, 150) || "分析完成"}
                      </p>
                    </>
                  )}
                </motion.div>
              ))}

              {isStreaming && (
                <div className="flex items-center gap-3 py-1.5">
                  <Loader2 className="w-3.5 h-3.5 text-brand-400 animate-spin shrink-0" />
                  <span className="text-xs text-zinc-400">思考中...</span>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
