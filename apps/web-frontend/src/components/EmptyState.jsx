import { motion } from "framer-motion";
import { Home, AlertTriangle, ArrowLeft } from "lucide-react";

export default function EmptyState({ onReset, message }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flex-1 flex flex-col items-center justify-center text-center"
    >
      {message ? (
        <>
          <div className="w-16 h-16 rounded-2xl bg-red-50 flex items-center justify-center mb-4">
            <AlertTriangle className="w-8 h-8 text-red-400" />
          </div>
          <p className="text-sm text-zinc-600 mb-1">分析出错</p>
          <p className="text-xs text-zinc-400 max-w-sm mb-4">{message}</p>
          {onReset && (
            <button
              onClick={onReset}
              className="inline-flex items-center gap-2 text-sm text-brand-600 hover:text-brand-700 transition-colors cursor-pointer"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              重新上传
            </button>
          )}
        </>
      ) : (
        <>
          <div className="w-16 h-16 rounded-2xl bg-zinc-100 flex items-center justify-center mb-4">
            <Home className="w-8 h-8 text-zinc-300" />
          </div>
          <p className="text-sm text-zinc-400">上传户型图后</p>
          <p className="text-xs text-zinc-300 mt-1">AI 分析结果将在此处展示</p>
        </>
      )}
    </motion.div>
  );
}
