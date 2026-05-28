import { BrowserRouter, Routes, Route, NavLink, Link } from "react-router-dom";
import { Sidebar } from "./components/Sidebar";
import { Home } from "./pages/Home";
import { Analysis } from "./pages/Analysis";
import { Datasets } from "./pages/Datasets";
import { Floorplans } from "./pages/Floorplans";
import { Training } from "./pages/Training";

function TopNav() {
  return (
    <header className="sticky top-0 z-50 bg-[var(--warm-white)] border-b border-[var(--border)] backdrop-blur-sm">
      <div className="flex items-center justify-between px-6 h-16">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-[var(--terracotta)] flex items-center justify-center flex-shrink-0">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="1" y="1" width="5.5" height="5.5" rx="1" fill="white" opacity="0.9"/>
              <rect x="9.5" y="1" width="5.5" height="5.5" rx="1" fill="white" opacity="0.7"/>
              <rect x="1" y="9.5" width="5.5" height="5.5" rx="1" fill="white" opacity="0.7"/>
              <rect x="9.5" y="9.5" width="5.5" height="3.5" rx="1" fill="white" opacity="0.5"/>
            </svg>
          </div>
          <span style={{ fontFamily: "var(--font-display)" }} className="text-lg font-semibold text-[var(--charcoal)] tracking-tight">RoomWise</span>
        </div>

        <nav className="hidden md:flex items-center gap-1">
          {[
            { to: "/", label: "首页", icon: "fa-home" },
            { to: "/analysis", label: "智能分析", icon: "fa-magic" },
            { to: "/floorplans", label: "我的户型图", icon: "fa-images" },
            { to: "/datasets", label: "数据集管理", icon: "fa-database" },
            { to: "/training", label: "训练管理", icon: "fa-brain" },
          ].map(({ to, label, icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                  isActive
                    ? "bg-[var(--terracotta-pale)] text-[var(--terracotta)]"
                    : "text-[var(--warm-gray)] hover:bg-[var(--ivory)] hover:text-[var(--charcoal)]"
                }`
              }
            >
              <i className={`fa-solid ${icon} text-xs`} />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="flex items-center gap-3">
          <Link
            to="/analysis?mode=upload"
            className="flex items-center gap-2 px-4 py-2 bg-[var(--terracotta)] hover:bg-[var(--terracotta-light)] text-white rounded-xl text-sm font-medium transition-all"
          >
            <i className="fa-solid fa-plus text-xs" />
            <span className="hidden sm:inline">上传户型图</span>
          </Link>
        </div>
      </div>
    </header>
  );
}

function MobileNav() {
  return (
    <nav className="md:hidden fixed bottom-0 left-0 right-0 z-50 bg-[var(--warm-white)] border-t border-[var(--border)] px-2 pb-2 pt-1 flex items-center justify-around">
      {[
        { to: "/", icon: "fa-home", label: "首页" },
        { to: "/analysis", icon: "fa-magic", label: "分析" },
        { to: "/floorplans", icon: "fa-images", label: "户型图" },
        { to: "/datasets", icon: "fa-database", label: "数据集" },
        { to: "/training", icon: "fa-brain", label: "训练" },
      ].map(({ to, icon, label }) => (
        <NavLink
          key={to}
          to={to}
          end={to === "/"}
          className={({ isActive }) =>
            `flex flex-col items-center gap-0.5 px-3 py-1.5 rounded-xl text-[10px] font-medium transition-all ${
              isActive ? "text-[var(--terracotta)]" : "text-[var(--mid-gray)]"
            }`
          }
        >
          <i className={`fa-solid ${icon}`} />
          {label}
        </NavLink>
      ))}
    </nav>
  );
}

function AppLayout() {
  return (
    <div className="min-h-screen flex flex-col" style={{ background: "var(--cream)" }}>
      <TopNav />
      <div className="flex-1 flex pb-16 md:pb-0">
        <main className="flex-1 min-w-0">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/analysis" element={<Analysis />} />
            <Route path="/floorplans" element={<Floorplans />} />
            <Route path="/datasets" element={<Datasets />} />
            <Route path="/training" element={<Training />} />
          </Routes>
        </main>
      </div>
      <MobileNav />
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppLayout />
    </BrowserRouter>
  );
}
