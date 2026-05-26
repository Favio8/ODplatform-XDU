import { useAnalysis } from "./hooks/useAnalysis";
import Header from "./components/Header";
import UploadZone from "./components/UploadZone";
import FloorPlanViewer from "./components/FloorPlanViewer";
import AnalysisPanel from "./components/AnalysisPanel";
import LoadingState from "./components/LoadingState";
import EmptyState from "./components/EmptyState";

export default function App() {
  const analysis = useAnalysis();

  return (
    <div className="min-h-screen flex flex-col bg-zinc-50">
      <Header onReset={analysis.reset} hasResult={analysis.phase === "done"} />

      <main className="flex-1 mx-auto w-full max-w-[1600px] flex flex-col lg:flex-row">
        {/* Left: Floor Plan */}
        <div className="lg:w-[45%] flex-shrink-0">
          <div className="lg:sticky lg:top-0 lg:h-[calc(100vh-64px)] flex flex-col p-4 lg:p-6">
            {analysis.phase === "idle" || analysis.phase === "uploading" ? (
              <UploadZone
                previewUrl={analysis.previewUrl}
                fileName={analysis.file?.name}
                onSelect={analysis.selectFile}
                onAnalyze={analysis.startAnalysis}
                onReset={analysis.reset}
              />
            ) : analysis.phase === "loading" ? (
              <LoadingState />
            ) : analysis.phase === "done" && analysis.result ? (
              <FloorPlanViewer
                visualization={analysis.result.visualization}
                imageSize={analysis.result.image_size}
                roomCount={analysis.result.yolo_rooms?.length || 0}
              />
            ) : (
              <EmptyState onReset={analysis.reset} message={analysis.error} />
            )}
          </div>
        </div>

        {/* Right: Analysis */}
        <div className="lg:w-[55%] flex flex-col min-h-0">
          <div className="flex-1 flex flex-col overflow-y-auto p-4 lg:p-6 lg:h-[calc(100vh-64px)]">
            {analysis.phase === "done" && analysis.result ? (
              <AnalysisPanel
                analysis={analysis.result.analysis}
                yoloRooms={analysis.result.yolo_rooms}
                reasoningSteps={analysis.reasoningSteps}
                sessionId={analysis.sessionId}
                sendMessage={analysis.sendMessage}
              />
            ) : (
              <div className="flex-1 flex items-center justify-center">
                <EmptyState
                  onReset={analysis.phase === "error" ? analysis.reset : null}
                  message={analysis.phase === "error" ? analysis.error : null}
                />
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
