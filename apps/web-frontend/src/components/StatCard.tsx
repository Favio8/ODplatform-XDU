import type { ReactNode } from "react";

interface StatCardProps {
  label: string;
  value: string;
  hint: string;
  tone?: "green" | "blue" | "amber" | "teal";
  icon?: ReactNode;
}

const toneMap = {
  green: "text-emerald-300",
  blue: "text-sky-300",
  amber: "text-amber-300",
  teal: "text-teal-300",
};

export function StatCard({ label, value, hint, tone = "blue", icon }: StatCardProps) {
  return (
    <div className="panel stat-card">
      <div className="card-label">{label}</div>
      <div className={`card-value ${toneMap[tone]}`}>{value}</div>
      <div className="card-hint">{hint}</div>
      {icon ? <div className="card-icon">{icon}</div> : null}
    </div>
  );
}

