export default function Header({ onReset, hasResult }) {
  return (
    <header className="h-16 border-b border-zinc-200/60 bg-white/70 backdrop-blur-xl sticky top-0 z-50">
      <div className="h-full max-w-[1600px] mx-auto px-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-500 to-brand-700 flex items-center justify-center text-white text-sm font-bold">
            O
          </div>
          <span className="font-semibold text-lg tracking-tight">
            ODPlatform
          </span>
          <span className="hidden sm:inline text-xs font-medium text-brand-500 bg-brand-50 px-2 py-0.5 rounded-full">
            AI Agent
          </span>
        </div>
        <div className="flex items-center gap-3">
          {hasResult && (
            <button
              onClick={onReset}
              className="text-sm text-zinc-500 hover:text-zinc-700 transition-colors cursor-pointer"
            >
              新分析
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
