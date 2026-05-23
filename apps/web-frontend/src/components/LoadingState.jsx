import { motion } from "framer-motion";
import { Loader2, CheckCircle2, Circle } from "lucide-react";
import { useState, useEffect } from "react";

const STEPS = [
  { label: "YOLO 房间分割", time: 800 },
  { label: "识别户型结构", time: 1800 },
  { label: "动线分析评估", time: 2800 },
  { label: "采光条件评估", time: 3600 },
  { label: "空间利用率计算", time: 4500 },
  { label: "生成装修建议", time: 6000 },
];

export default function LoadingState() {
  const [doneSteps, setDoneSteps] = useState([]);
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    const timers = STEPS.map((step, i) =>
      setTimeout(() => {
        setDoneSteps((prev) => [...prev, i]);
        setActiveStep(i + 1);
      }, step.time)
    );
    return () => timers.forEach(clearTimeout);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="flex-1 flex flex-col items-center justify-center"
    >
      <div className="relative mb-8">
        <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-brand-100 to-brand-200 animate-pulse" />
        <Loader2 className="absolute inset-0 m-auto w-8 h-8 text-brand-600 animate-spin" />
      </div>

      <p className="text-base font-medium text-zinc-600 mb-6">
        AI 正在分析户型图...
      </p>

      <div className="space-y-2.5 w-full max-w-[280px]">
        {STEPS.map((step, i) => {
          const done = doneSteps.includes(i);
          return (
            <motion.div
              key={step.label}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: done || i === activeStep ? 1 : 0.3, x: 0 }}
              transition={{ duration: 0.3 }}
              className="flex items-center gap-3"
            >
              {done ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
              ) : i === activeStep ? (
                <motion.div
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  <Circle className="w-4 h-4 text-brand-400 shrink-0 fill-brand-400/20" />
                </motion.div>
              ) : (
                <Circle className="w-4 h-4 text-zinc-200 shrink-0" />
              )}
              <span
                className={`text-sm transition-colors ${
                  done ? "text-zinc-500" : i === activeStep ? "text-zinc-700 font-medium" : "text-zinc-300"
                }`}
              >
                {step.label}
              </span>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
}
