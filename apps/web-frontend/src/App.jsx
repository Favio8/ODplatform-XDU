import { useAnalysis } from "./hooks/useAnalysis";
import Header from "./components/Header";
import UploadZone from "./components/UploadZone";
import FloorPlanViewer from "./components/FloorPlanViewer";
import AnalysisPanel from "./components/AnalysisPanel";
import LoadingState from "./components/LoadingState";
import EmptyState from "./components/EmptyState";

export default function App() {
  const a = useAnalysis();

  const showFloorPlan = a.visualization && (a.phase === "loading" || a.phase === "done");
  const showAnalysis = a.phase === "done" && a.analysis;

  return (
    <div className="min-h-screen flex flex-col bg-zinc-50">
      <Header onReset={a.reset} hasResult={showAnalysis} />

      <main className="flex-1 mx-auto w-full max-w-[1600px] flex flex-col lg:flex-row">
        {/* Left: Floor Plan */}
        <div className="lg:w-[45%] flex-shrink-0">
          <div className="lg:sticky lg:top-0 lg:h-[calc(100vh-64px)] flex flex-col p-4 lg:p-6">
            {a.phase === "idle" || a.phase === "uploading" ? (
              <UploadZone
                previewUrl={a.previewUrl}
                fileName={a.file?.name}
                onSelect={a.selectFile}
                onAnalyze={a.startAnalysis}
                onReset={a.reset}
              />
            ) : showFloorPlan ? (
              <FloorPlanViewer
                visualization={a.visualization}
                imageSize={a.imageSize}
                roomCount={a.yoloRooms?.length || 0}
              />
            ) : a.phase === "loading" && !a.yoloReady ? (
              <LoadingState reasoningSteps={[]} status="segmenting" />
            ) : (
              <EmptyState onReset={a.reset} message={a.error} />
            )}
          </div>
        </div>

        {/* Right: Analysis */}
        <div className="lg:w-[55%] flex flex-col min-h-0">
          <div className="flex-1 flex flex-col overflow-y-auto p-4 lg:p-6 lg:h-[calc(100vh-64px)]">
            {showAnalysis ? (
              <AnalysisPanel
                analysis={a.analysis}
                yoloRooms={a.yoloRooms}
                reasoningSteps={a.reasoningSteps}
                sessionId={a.sessionId}
              />
            ) : a.phase === "loading" ? (
              <LoadingState
                reasoningSteps={a.reasoningSteps}
                status={a.yoloReady ? "analyzing" : "waiting"}
              />
            ) : (
              <div className="flex-1 flex items-center justify-center">
                <EmptyState
                  onReset={a.phase === "error" ? a.reset : null}
                  message={a.phase === "error" ? a.error : null}
                />
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
