import { ChevronDown } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";
import SuggestionGrid from "./SuggestionGrid";

const DOT_COLORS = [
  "bg-rose-400", "bg-sky-400", "bg-amber-400", "bg-emerald-400",
  "bg-violet-400", "bg-orange-400", "bg-teal-400", "bg-pink-400",
];

export default function RoomAccordion({ room, yoloRoom, index }) {
  const [open, setOpen] = useState(index === 0);
  const dot = DOT_COLORS[index % DOT_COLORS.length];

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06, duration: 0.35 }}
      className="rounded-xl overflow-hidden"
      style={{
        background: open ? "rgba(255,255,255,0.7)" : "rgba(255,255,255,0.45)",
        backdropFilter: "blur(10px)",
        border: "1px solid rgba(0,0,0,0.04)",
        transition: "background 0.25s",
      }}
    >
      <button
        onClick={() => setOpen(!open)}
        className="w-full px-4 py-3.5 flex items-center justify-between text-left hover:bg-white/30 transition-colors cursor-pointer"
      >
        <div className="flex items-center gap-3">
          <span className={`w-2.5 h-2.5 rounded-full ${dot} shrink-0`} />
          <div>
            <span className="font-medium text-sm text-zinc-700">
              {room.room_type || `Room ${room.room_label}`}
            </span>
            {yoloRoom && (
              <span className="ml-2 text-xs text-zinc-400">
                {yoloRoom.confidence >= 0.5 ? (
                  <span className="text-emerald-500">高置信</span>
                ) : (
                  <span className="text-amber-500">低置信</span>
                )}
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-3">
          {yoloRoom && (
            <span className="text-xs text-zinc-400">
              {(yoloRoom.area_ratio * 100).toFixed(1)}%
            </span>
          )}
          <motion.div
            animate={{ rotate: open ? 180 : 0 }}
            transition={{ duration: 0.2 }}
          >
            <ChevronDown className="w-4 h-4 text-zinc-300" />
          </motion.div>
        </div>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 pt-1">
              <p className="text-[13px] text-zinc-500 leading-relaxed mb-4">
                {room.analysis || "等待分析..."}
              </p>
              {room.suggestions && <SuggestionGrid suggestions={room.suggestions} />}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
