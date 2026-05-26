import { useState } from "react";

export function useAnalysis() {
  const [state, setState] = useState({
    phase: "idle",
    file: null,
    previewUrl: null,
    result: null,
    sessionId: null,
    reasoningSteps: [],
    error: null,
  });

  const selectFile = (file) => {
    const url = URL.createObjectURL(file);
    setState({
      phase: "uploading", file, previewUrl: url,
      result: null, sessionId: null, reasoningSteps: [], error: null,
    });
  };

  const startAnalysis = async () => {
    if (!state.file) return;
    setState((s) => ({ ...s, phase: "loading", error: null, reasoningSteps: [] }));

    const form = new FormData();
    form.append("file", state.file);

    try {
      const res = await fetch("/api/analyze", { method: "POST", body: form });
      if (!res.ok) throw new Error((await res.text()) || `HTTP ${res.status}`);
      const data = await res.json();
      setState((s) => ({
        ...s,
        phase: "done",
        result: data,
        sessionId: data.session_id,
        reasoningSteps: data.reasoning_steps || [],
      }));
    } catch (err) {
      setState((s) => ({ ...s, phase: "error", error: err.message }));
    }
  };

  const sendMessage = async (message) => {
    if (!state.sessionId) return null;
    const form = new FormData();
    form.append("message", message);

    try {
      const res = await fetch(`/api/chat/${state.sessionId}`, {
        method: "POST", body: form,
      });
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    } catch (err) {
      return { reply: `发送失败: ${err.message}`, reasoning_steps: [] };
    }
  };

  const reset = () => {
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    setState({
      phase: "idle", file: null, previewUrl: null,
      result: null, sessionId: null, reasoningSteps: [], error: null,
    });
  };

  return { ...state, selectFile, startAnalysis, sendMessage, reset };
}
