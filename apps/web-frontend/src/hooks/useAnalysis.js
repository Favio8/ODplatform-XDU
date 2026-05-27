import { useState, useRef, useCallback } from "react";

export function useAnalysis() {
  const [state, setState] = useState({
    phase: "idle",
    file: null,
    previewUrl: null,
    sessionId: null,
    // YOLO 结果（立即返回）
    yoloReady: false,
    visualization: null,
    yoloRooms: [],
    imageSize: null,
    // Agent 推理（逐步返回）
    reasoningSteps: [],
    analysis: null,
    polling: false,
    error: null,
  });

  const pollRef = useRef(null);

  const selectFile = (file) => {
    const url = URL.createObjectURL(file);
    setState({
      phase: "uploading", file, previewUrl: url,
      sessionId: null, yoloReady: false,
      visualization: null, yoloRooms: [], imageSize: null,
      reasoningSteps: [], analysis: null, polling: false, error: null,
    });
  };

  const startAnalysis = async () => {
    if (!state.file) return;
    setState((s) => ({ ...s, phase: "loading", error: null, reasoningSteps: [], analysis: null }));

    const form = new FormData();
    form.append("file", state.file);

    try {
      // 阶段 1: YOLO（立即返回）
      const res = await fetch("/api/analyze", { method: "POST", body: form });
      if (!res.ok) throw new Error((await res.text()) || `HTTP ${res.status}`);
      const yoloData = await res.json();

      setState((s) => ({
        ...s,
        sessionId: yoloData.session_id,
        yoloReady: true,
        visualization: yoloData.visualization,
        yoloRooms: yoloData.yolo_rooms,
        imageSize: yoloData.image_size,
        phase: "loading",
      }));

      // 阶段 2: 轮询 Agent 推理步骤
      const sid = yoloData.session_id;
      await new Promise((resolve, reject) => {
        pollRef.current = setInterval(async () => {
          try {
            const r = await fetch(`/api/session/${sid}`);
            const sess = await r.json();

            if (sess.error) {
              clearInterval(pollRef.current);
              reject(new Error(sess.error));
              return;
            }

            setState((s) => ({
              ...s,
              reasoningSteps: sess.reasoning_steps || [],
            }));

            if (sess.status === "done" && sess.analysis) {
              clearInterval(pollRef.current);
              setState((s) => ({
                ...s,
                phase: "done",
                analysis: sess.analysis,
                reasoningSteps: sess.reasoning_steps || [],
                polling: false,
              }));
              resolve();
            } else if (sess.status === "error") {
              clearInterval(pollRef.current);
              reject(new Error(sess.error || "Agent 分析失败"));
            }
          } catch (e) {
            // 网络错误忽略，继续轮询
          }
        }, 800);
      });
    } catch (err) {
      if (pollRef.current) clearInterval(pollRef.current);
      setState((s) => ({ ...s, phase: "error", error: err.message, polling: false }));
    }
  };

  const sendMessage = async (message) => {
    if (!state.sessionId) return null;
    const form = new FormData();
    form.append("message", message);
    try {
      const res = await fetch(`/api/chat/${state.sessionId}`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    } catch (err) {
      return { reply: `发送失败: ${err.message}`, reasoning_steps: [] };
    }
  };

  const reset = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    setState({
      phase: "idle", file: null, previewUrl: null,
      sessionId: null, yoloReady: false,
      visualization: null, yoloRooms: [], imageSize: null,
      reasoningSteps: [], analysis: null, polling: false, error: null,
    });
  }, [state.previewUrl]);

  return { ...state, selectFile, startAnalysis, sendMessage, reset };
}
