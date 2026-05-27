import { Component } from "react";
import { AlertTriangle } from "lucide-react";

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error("ErrorBoundary caught:", error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="flex-1 flex flex-col items-center justify-center p-8 text-center">
          <AlertTriangle className="w-10 h-10 text-red-400 mb-4" />
          <p className="text-sm font-medium text-zinc-600 mb-2">渲染出错</p>
          <p className="text-xs text-zinc-400 max-w-md mb-4">
            {this.state.error.message || String(this.state.error)}
          </p>
          <button
            onClick={() => {
              this.setState({ error: null });
              this.props.onRetry?.();
            }}
            className="text-sm text-brand-600 hover:text-brand-700 cursor-pointer"
          >
            重试
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
