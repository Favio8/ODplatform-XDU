import { Sofa, Palette, Archive, Lightbulb } from "lucide-react";

const icons = {
  furniture: Sofa,
  color: Palette,
  storage: Archive,
  lighting: Lightbulb,
};

const labels = {
  furniture: "家具布局",
  color: "配色方案",
  storage: "收纳设计",
  lighting: "灯光设计",
};

export default function SuggestionGrid({ suggestions }) {
  return (
    <div className="grid grid-cols-2 gap-2.5">
      {Object.entries(suggestions).map(([key, value]) => {
        const Icon = icons[key] || Lightbulb;
        return (
          <div
            key={key}
            className="rounded-lg p-3"
            style={{ background: "rgba(249,250,251,0.8)" }}
          >
            <div className="flex items-center gap-1.5 mb-1">
              <Icon className="w-3.5 h-3.5 text-brand-500" />
              <span className="text-xs font-semibold text-zinc-600">{labels[key] || key}</span>
            </div>
            <p className="text-xs text-zinc-500 leading-relaxed">{value || "-"}</p>
          </div>
        );
      })}
    </div>
  );
}
