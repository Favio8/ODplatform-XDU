import { useState } from "react";

export function useAnalysis() {
  const [state, setState] = useState({
    phase: "idle", // idle | uploading | loading | done | error
    file: null,
    previewUrl: null,
    result: null,
    error: null,
  });

  const selectFile = (file) => {
    const url = URL.createObjectURL(file);
    setState({ phase: "uploading", file, previewUrl: url, result: null, error: null });
  };

  const startAnalysis = async () => {
    if (!state.file) return;
    setState((s) => ({ ...s, phase: "loading", error: null }));

    const form = new FormData();
    form.append("file", state.file);

    try {
      const res = await fetch("/api/analyze", { method: "POST", body: form });
      if (!res.ok) throw new Error((await res.text()) || `HTTP ${res.status}`);
      const data = await res.json();
      setState((s) => ({ ...s, phase: "done", result: data }));
    } catch (err) {
      setState((s) => ({ ...s, phase: "error", error: err.message }));
    }
  };

  const reset = () => {
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    setState({ phase: "idle", file: null, previewUrl: null, result: null, error: null });
  };

  return { ...state, selectFile, startAnalysis, reset };
}
