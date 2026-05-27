import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Bot, User, Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";

export default function ChatPanel({ sessionId }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || loading) return;

    setInput("");
    setMessages((m) => [...m, { role: "user", content: text }]);
    setLoading(true);

    try {
      const form = new FormData();
      form.append("message", text);

      const res = await fetch(`/api/chat/${sessionId}/stream`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) throw new Error(await res.text());

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let replyContent = "";

      setMessages((m) => [...m, { role: "assistant", content: "" }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            if (data === "[DONE]") continue;
            try {
              const parsed = JSON.parse(data);
              if (parsed.token) {
                replyContent += parsed.token;
                setMessages((m) => {
                  const updated = [...m];
                  updated[updated.length - 1] = {
                    role: "assistant",
                    content: replyContent,
                  };
                  return updated;
                });
              }
            } catch {}
          }
        }
      }
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `发送失败: ${err.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full min-h-[300px]">
      <div className="flex items-center gap-2 mb-3 px-1">
        <Sparkles className="w-4 h-4 text-brand-500" />
        <span className="text-xs font-semibold uppercase tracking-wide text-zinc-400">
          追问 AI
        </span>
      </div>

      {/* Messages — no auto-scroll */}
      <div className="flex-1 overflow-y-auto space-y-3 mb-3 min-h-[120px] max-h-[280px]">
        <AnimatePresence initial={false}>
          {messages.map((msg, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              className={`flex gap-2.5 ${msg.role === "user" ? "flex-row-reverse" : ""}`}
            >
              <div
                className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 ${
                  msg.role === "user"
                    ? "bg-brand-100"
                    : "bg-gradient-to-br from-brand-400 to-brand-600"
                }`}
              >
                {msg.role === "user" ? (
                  <User className="w-3 h-3 text-brand-600" />
                ) : (
                  <Bot className="w-3 h-3 text-white" />
                )}
              </div>
              <div
                className={`text-[13px] leading-relaxed px-3 py-2 rounded-2xl max-w-[85%] ${
                  msg.role === "user"
                    ? "bg-brand-500 text-white rounded-tr-md"
                    : "bg-zinc-100 text-zinc-700 rounded-tl-md"
                }`}
              >
                {msg.role === "user" ? (
                  msg.content
                ) : (
                  <div className="prose prose-sm max-w-none prose-p:my-1 prose-ul:my-1 prose-li:my-0.5 prose-headings:my-2 prose-code:text-xs prose-code:bg-zinc-200 prose-code:px-1 prose-code:rounded">
                    <ReactMarkdown>{msg.content || "..."}</ReactMarkdown>
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Input */}
      <div className="flex items-center gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="追问任何问题…"
          className="flex-1 h-9 px-3.5 text-sm rounded-xl border border-zinc-200 bg-white/70
            focus:outline-none focus:border-brand-300 focus:ring-2 focus:ring-brand-100
            placeholder:text-zinc-300 transition-all"
        />
        <button
          onClick={handleSend}
          disabled={!input.trim() || loading}
          className="w-9 h-9 rounded-xl bg-brand-500 text-white flex items-center justify-center
            hover:bg-brand-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors cursor-pointer"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
