import { useRef, useState } from "react";
import { Upload, X, Sparkles } from "lucide-react";
import { motion } from "framer-motion";

export default function UploadZone({
  previewUrl,
  fileName,
  onSelect,
  onAnalyze,
  onReset,
}) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f?.type.startsWith("image/")) onSelect(f);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex-1 flex flex-col"
    >
      {!previewUrl ? (
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`
            flex-1 flex flex-col items-center justify-center rounded-2xl border-2 border-dashed
            cursor-pointer transition-all duration-300
            ${dragOver
              ? "border-brand-400 bg-brand-50/50 scale-[1.01]"
              : "border-zinc-200 hover:border-zinc-300 bg-white/50 hover:bg-white/80"
            }
          `}
          style={{
            backdropFilter: "blur(12px)",
          }}
        >
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-brand-100 to-brand-200 flex items-center justify-center mb-5">
            <Upload className="w-7 h-7 text-brand-600" />
          </div>
          <p className="text-base font-medium text-zinc-700">拖拽户型图到此处</p>
          <p className="text-sm text-zinc-400 mt-1.5">或点击选择图片 · 支持 JPG / PNG</p>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => {
              const f = e.target.files[0];
              if (f) onSelect(f);
            }}
          />
        </div>
      ) : (
        <div className="flex-1 flex flex-col">
          <div className="relative flex-1 rounded-2xl overflow-hidden bg-zinc-100 border border-zinc-200/60 group">
            <img
              src={previewUrl}
              alt="预览"
              className="w-full h-full object-contain transition-transform duration-500 group-hover:scale-105"
            />
            <button
              onClick={onReset}
              className="absolute top-3 right-3 w-8 h-8 rounded-full bg-black/40 text-white flex items-center justify-center hover:bg-black/60 transition-colors cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
            <div className="absolute bottom-3 left-3 text-xs text-zinc-500 bg-white/80 backdrop-blur px-2.5 py-1 rounded-full">
              {fileName}
            </div>
          </div>
          <motion.button
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            onClick={onAnalyze}
            className="mt-4 w-full py-3.5 rounded-xl bg-gradient-to-r from-brand-600 to-brand-500 text-white font-semibold
              hover:from-brand-700 hover:to-brand-600 shadow-lg shadow-brand-500/25
              transition-all duration-200 hover:shadow-xl hover:shadow-brand-500/30
              flex items-center justify-center gap-2 cursor-pointer"
          >
            <Sparkles className="w-4 h-4" />
            开始 AI 分析
          </motion.button>
        </div>
      )}
    </motion.div>
  );
}
