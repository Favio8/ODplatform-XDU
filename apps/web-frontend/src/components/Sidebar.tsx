import { NavLink } from "react-router-dom";

const navItems = [
  { to: "/", icon: "fa-chart-pie", label: "工作台" },
  { to: "/datasets", icon: "fa-database", label: "数据集" },
  { to: "/training", icon: "fa-scissors", label: "图像分割训练" },
  { to: "/inference", icon: "fa-robot", label: "Agent分析" },
];

export function Sidebar() {
  return (
    <aside className="w-64 bg-[#0f172a] text-slate-400 flex flex-col justify-between flex-shrink-0 border-r border-slate-800">
      <div>
        <div className="p-5 flex items-center gap-3 border-b border-slate-800">
          <div className="bg-blue-600 text-white p-2 rounded-lg flex items-center justify-center">
            <i className="fa-solid fa-layer-group text-lg" />
          </div>
          <div>
            <h1 className="text-white font-bold text-base tracking-wide">ODPlatform</h1>
            <p className="text-[10px] text-slate-500">AI Floorplan Segmentation</p>
          </div>
        </div>

        <nav className="p-4 space-y-1">
          {navItems.map(({ to, icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
                  isActive
                    ? "bg-blue-600 text-white font-medium"
                    : "hover:bg-slate-800 hover:text-slate-200"
                }`
              }
            >
              <i className={`fa-solid ${icon} w-5`} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
      </div>

      <div className="p-4 m-4 bg-slate-900/50 rounded-xl border border-slate-800/80">
        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">项目概览</h3>
        <div className="space-y-2 text-xs">
          <div className="flex justify-between">
            <span className="text-slate-500">项目名称</span>
            <span className="text-slate-300 font-medium">Room Intelligence</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">任务类型</span>
            <span className="text-slate-300">房间分割 + 改造建议</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">后端状态</span>
            <span className="px-2 py-0.5 bg-green-500/10 text-green-400 rounded text-[10px]">已连接</span>
          </div>
        </div>
        <button className="w-full mt-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium transition-all flex items-center justify-center gap-2">
          <i className="fa-solid fa-gear" />
          项目设置
        </button>
      </div>
    </aside>
  );
}
