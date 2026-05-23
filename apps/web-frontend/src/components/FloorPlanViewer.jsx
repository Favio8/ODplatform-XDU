import { useState, useRef, useCallback } from "react";
import { createPortal } from "react-dom";
import { Maximize2, X, Plus, Minus, RotateCcw, Download } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function FloorPlanViewer({ visualization, imageSize, roomCount }) {
  const [modal, setModal] = useState(false);
  const [scale, setScale] = useState(1);
  const imgRef = useRef(null);

  const zoomIn = useCallback(() => setScale((s) => Math.min(s + 0.2, 3)), []);
  const zoomOut = useCallback(() => setScale((s) => Math.max(s - 0.2, 0.3)), []);
  const fitScreen = useCallback(() => setScale(1), []);

  const downloadImg = useCallback(() => {
    const a = document.createElement("a");
    a.href = `data:image/jpeg;base64,${visualization}`;
    a.download = "odplatform-segmentation.jpg";
    a.click();
  }, [visualization]);

  return (
    <>
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="flex-1 flex flex-col h-full"
      >
        {/* Toolbar */}
        <div className="flex items-center justify-between mb-2 px-1">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-zinc-400">分割结果</p>
            <p className="text-[11px] text-zinc-300">
              {imageSize?.width}×{imageSize?.height} · {roomCount} 个房间
            </p>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={zoomOut}
              className="w-7 h-7 rounded-lg border border-zinc-200 flex items-center justify-center text-zinc-400 hover:text-zinc-600 hover:bg-zinc-50 transition-colors cursor-pointer"
              title="缩小"
            >
              <Minus className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={fitScreen}
              className="text-[11px] px-2 h-7 rounded-lg border border-zinc-200 text-zinc-400 hover:text-zinc-600 hover:bg-zinc-50 transition-colors cursor-pointer"
              title="适应"
            >
              {Math.round(scale * 100)}%
            </button>
            <button
              onClick={zoomIn}
              className="w-7 h-7 rounded-lg border border-zinc-200 flex items-center justify-center text-zinc-400 hover:text-zinc-600 hover:bg-zinc-50 transition-colors cursor-pointer"
              title="放大"
            >
              <Plus className="w-3.5 h-3.5" />
            </button>
            <span className="w-px h-5 bg-zinc-200 mx-1" />
            <button
              onClick={downloadImg}
              className="w-7 h-7 rounded-lg border border-zinc-200 flex items-center justify-center text-zinc-400 hover:text-zinc-600 hover:bg-zinc-50 transition-colors cursor-pointer"
              title="下载"
            >
              <Download className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setModal(true)}
              className="w-7 h-7 rounded-lg border border-zinc-200 flex items-center justify-center text-zinc-400 hover:text-zinc-600 hover:bg-zinc-50 transition-colors cursor-pointer"
              title="全屏"
            >
              <Maximize2 className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Image */}
        <div
          className="relative flex-1 rounded-2xl overflow-hidden bg-zinc-100/80 border border-zinc-200/50"
        >
          <img
            ref={imgRef}
            src={`data:image/jpeg;base64,${visualization}`}
            alt="AI 分割结果"
            className="w-full h-full object-contain transition-transform duration-300"
            style={{ transform: `scale(${scale})` }}
          />
        </div>
      </motion.div>

      {/* Fullscreen Modal via Portal */}
      {modal &&
        createPortal(
          <AnimatePresence>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-[9999] bg-black/85 backdrop-blur-md flex items-center justify-center p-8"
              onClick={() => setModal(false)}
            >
              <div className="absolute top-6 right-6 flex items-center gap-2">
                <button
                  onClick={(e) => { e.stopPropagation(); zoomOut(); }}
                  className="w-9 h-9 rounded-lg bg-white/10 text-white flex items-center justify-center hover:bg-white/20 transition-colors cursor-pointer"
                >
                  <Minus className="w-4 h-4" />
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); fitScreen(); }}
                  className="text-sm px-3 h-9 rounded-lg bg-white/10 text-white hover:bg-white/20 transition-colors cursor-pointer"
                >
                  {Math.round(scale * 100)}%
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); zoomIn(); }}
                  className="w-9 h-9 rounded-lg bg-white/10 text-white flex items-center justify-center hover:bg-white/20 transition-colors cursor-pointer"
                >
                  <Plus className="w-4 h-4" />
                </button>
                <span className="w-px h-5 bg-white/20 mx-1" />
                <button
                  onClick={() => setModal(false)}
                  className="w-9 h-9 rounded-full bg-white/10 text-white flex items-center justify-center hover:bg-white/20 transition-colors cursor-pointer"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              <motion.img
                initial={{ scale: 0.9 }}
                animate={{ scale: 1 }}
                exit={{ scale: 0.9 }}
                src={`data:image/jpeg;base64,${visualization}`}
                alt="放大预览"
                className="max-w-full max-h-full object-contain rounded-2xl transition-transform duration-300"
                style={{ transform: `scale(${scale})` }}
                onClick={(e) => e.stopPropagation()}
              />
            </motion.div>
          </AnimatePresence>,
          document.body
        )}
    </>
  );
}
