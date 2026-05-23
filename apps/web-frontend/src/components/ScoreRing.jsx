import { motion } from "framer-motion";

export default function ScoreRing({ label, value, icon: Icon }) {
  const color =
    value >= 80
      ? "text-emerald-500 stroke-emerald-500"
      : value >= 60
        ? "text-amber-500 stroke-amber-500"
        : "text-red-400 stroke-red-400";

  const bgColor =
    value >= 80
      ? "text-emerald-200 stroke-emerald-200"
      : value >= 60
        ? "text-amber-200 stroke-amber-200"
        : "text-red-200 stroke-red-200";

  const r = 28;
  const circ = 2 * Math.PI * r;
  const offset = circ - (value / 100) * circ;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
      className="flex flex-col items-center gap-1.5"
    >
      <div className="relative w-[72px] h-[72px] flex items-center justify-center">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 72 72">
          <circle cx="36" cy="36" r={r} fill="none" strokeWidth="5" className={bgColor} />
          <motion.circle
            cx="36" cy="36" r={r} fill="none" strokeWidth="5" stroke="currentColor"
            strokeLinecap="round"
            className={color}
            strokeDasharray={circ}
            initial={{ strokeDashoffset: circ }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1, ease: "easeOut" }}
          />
        </svg>
        <span className={`absolute text-sm font-bold ${color}`}>{value}</span>
      </div>
      {Icon && <Icon className="w-4 h-4 text-zinc-400" />}
      <span className="text-xs text-zinc-500">{label}</span>
    </motion.div>
  );
}
