import { motion } from "framer-motion";
import {
  Sparkles, LayoutGrid, Lightbulb, Footprints, BoxSelect,
  AlertCircle, TrendingUp, ThumbsUp, ThumbsDown, Award, Zap
} from "lucide-react";
import ScoreRing from "./ScoreRing";
import RoomAccordion from "./RoomAccordion";
import ReasoningSteps from "./ReasoningSteps";
import ChatPanel from "./ChatPanel";

const ratingColors = {
  "S": "from-amber-400 to-orange-500 text-amber-700",
  "A+": "from-emerald-400 to-teal-500 text-emerald-700",
  "A": "from-emerald-400 to-teal-500 text-emerald-700",
  "A-": "from-green-400 to-emerald-500 text-green-700",
  "B+": "from-sky-400 to-blue-500 text-sky-700",
  "B": "from-sky-400 to-blue-500 text-sky-700",
  "C": "from-zinc-400 to-zinc-500 text-zinc-600",
};

export default function AnalysisPanel({ analysis, yoloRooms, reasoningSteps, sessionId }) {
  const scores = analysis.scores || {};
  const rooms = analysis.rooms || [];
  const issues = analysis.core_issues || [];
  const pros = analysis.pros || [];
  const cons = analysis.cons || [];
  const rating = analysis.rating || "N/A";
  const rc = ratingColors[rating] || ratingColors["B"];

  const avgScore = Object.values(scores).length
    ? Math.round(Object.values(scores).reduce((a, b) => a + b, 0) / Object.values(scores).length)
    : 0;

  return (
    <div className="space-y-6">
      {/* ---------- AI Summary Hero ---------- */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        className="rounded-2xl overflow-hidden"
        style={{
          background: "linear-gradient(135deg, rgba(124,58,237,0.10), rgba(99,102,241,0.04))",
          border: "1px solid rgba(124,58,237,0.12)",
        }}
      >
        <div className="p-6">
          {/* Rating + Type */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2.5">
              <Sparkles className="w-5 h-5 text-brand-500" />
              <span className="text-sm font-semibold tracking-tight text-zinc-800">
                AI 分析报告
              </span>
              {analysis.house_type && (
                <span className="text-xs px-2 py-0.5 rounded-full bg-brand-100/80 text-brand-600 border border-brand-200/60">
                  {analysis.house_type}
                </span>
              )}
            </div>
            <div className={`text-xs font-bold px-3 py-1 rounded-full bg-gradient-to-r ${rc} bg-opacity-10`}>
              <span className={rc.split(" ")[2]}>
                {rating} 级户型
              </span>
            </div>
          </div>

          <p className="text-sm text-zinc-500 leading-relaxed mb-4">
            {analysis.overall_assessment || "分析中..."}
          </p>

          {/* Pros / Cons */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex items-center gap-1.5 text-xs font-medium text-emerald-600">
                <ThumbsUp className="w-3.5 h-3.5" />
                优势
              </div>
              {pros.length > 0 ? pros.map((p, i) => (
                <div key={i} className="flex items-start gap-2 text-xs text-zinc-600">
                  <span className="text-emerald-400 mt-0.5">✓</span>
                  {p}
                </div>
              )) : (
                <p className="text-xs text-zinc-400">分析中...</p>
              )}
            </div>
            <div className="space-y-2">
              <div className="flex items-center gap-1.5 text-xs font-medium text-amber-600">
                <ThumbsDown className="w-3.5 h-3.5" />
                劣势
              </div>
              {cons.length > 0 ? cons.map((c, i) => (
                <div key={i} className="flex items-start gap-2 text-xs text-zinc-600">
                  <span className="text-amber-400 mt-0.5">✗</span>
                  {c}
                </div>
              )) : (
                <p className="text-xs text-zinc-400">分析中...</p>
              )}
            </div>
          </div>
        </div>
      </motion.div>

      {/* ---------- Dashboard Scores ---------- */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.06 }}
        className="rounded-2xl overflow-hidden"
        style={{
          background: "rgba(255,255,255,0.55)",
          backdropFilter: "blur(12px)",
          border: "1px solid rgba(0,0,0,0.05)",
        }}
      >
        <div className="p-5">
          <div className="flex items-center gap-2 mb-4">
            <LayoutGrid className="w-4 h-4 text-zinc-400" />
            <span className="text-xs font-semibold uppercase tracking-wide text-zinc-400">量化评分</span>
            <span className="text-[22px] font-bold text-zinc-700 ml-auto">{avgScore}</span>
            <span className="text-xs text-zinc-400">/ 100</span>
          </div>
          <div className="flex justify-around">
            <ScoreRing label="空间利用" value={scores.space_utilization || 0} icon={BoxSelect} />
            <ScoreRing label="采光" value={scores.lighting || 0} icon={Lightbulb} />
            <ScoreRing label="动线" value={scores.traffic_flow || 0} icon={Footprints} />
            <ScoreRing label="收纳潜力" value={scores.storage_potential || 0} icon={BoxSelect} />
          </div>
        </div>
      </motion.div>

      {/* ---------- Core Issues ---------- */}
      {issues.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="rounded-2xl overflow-hidden"
          style={{
            background: "rgba(251,243,219,0.5)",
            backdropFilter: "blur(12px)",
            border: "1px solid rgba(245,158,11,0.15)",
          }}
        >
          <div className="p-5">
            <div className="flex items-center gap-2 mb-3">
              <AlertCircle className="w-4 h-4 text-amber-500" />
              <span className="text-xs font-semibold uppercase tracking-wide text-amber-600">核心问题</span>
            </div>
            <div className="space-y-2.5">
              {issues.map((issue, i) => (
                <div key={i} className="flex items-start gap-3 text-sm text-zinc-600">
                  <span className="w-5 h-5 rounded-full bg-amber-100 text-amber-600 flex items-center justify-center text-xs font-semibold shrink-0 mt-0.5">
                    {i + 1}
                  </span>
                  {issue}
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      )}

      {/* ---------- Rooms ---------- */}
      <div>
        <div className="flex items-center gap-2 mb-3 px-1">
          <LayoutGrid className="w-4 h-4 text-zinc-400" />
          <span className="text-xs font-semibold uppercase tracking-wide text-zinc-400">
            房间详情
          </span>
          <span className="text-xs text-zinc-300">{rooms.length} 个区域</span>
        </div>
        <div className="space-y-2.5">
          {rooms.map((room, i) => (
            <RoomAccordion
              key={room.room_label || i}
              room={room}
              yoloRoom={yoloRooms?.find((y) => `Room ${y.id}` === room.room_label)}
              index={i}
            />
          ))}
          {rooms.length === 0 && (
            <p className="text-sm text-zinc-400 text-center py-8">AI 暂未返回房间分析</p>
          )}
        </div>
      </div>

      {/* ---------- Overall Suggestions ---------- */}
      {analysis.overall_suggestions && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.14 }}
          className="rounded-2xl overflow-hidden"
          style={{
            background: "rgba(255,255,255,0.55)",
            backdropFilter: "blur(12px)",
            border: "1px solid rgba(0,0,0,0.05)",
          }}
        >
          <div className="p-5">
            <div className="flex items-center gap-2 mb-3">
              <Zap className="w-4 h-4 text-brand-500" />
              <span className="text-xs font-semibold uppercase tracking-wide text-zinc-400">优化建议</span>
            </div>
            <p className="text-sm text-zinc-600 leading-relaxed">
              {analysis.overall_suggestions}
            </p>
          </div>
        </motion.div>
      )}

      {/* ---------- AI Reasoning ---------- */}
      {reasoningSteps && reasoningSteps.length > 0 && (
        <ReasoningSteps steps={reasoningSteps} isStreaming={false} />
      )}

      {/* ---------- Chat ---------- */}
      {sessionId && sendMessage && (
        <ChatPanel sessionId={sessionId} />
      )}
    </div>
  );
}
