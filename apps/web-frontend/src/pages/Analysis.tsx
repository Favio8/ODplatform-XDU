import { useEffect, useState, useRef, useCallback, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { analyzeFloorplan, fetchAgentSession, fetchFloorplan, fetchFloorplans, fetchServingModels, streamAgentChat } from "../lib/api";
import type { AgentAdvice, AgentReport, AgentRequirements, AgentSessionResponse, AgentYoloRoom, ChatMessage, FloorplanRecord, InferenceResult, RoomDisplayInfo, ServingModel } from "../types";

type ViewMode = "upload" | "analyzing" | "result";

const ROOM_COLORS: Record<string, string> = {
  "客厅": "#E8D5C4",
  "主卧": "#D4E4D4",
  "次卧": "#C8DCC8",
  "厨房": "#F0D4D0",
  "卫生间": "#D8E4F0",
  "阳台": "#F0EDD8",
  "餐厅": "#EDE4D8",
  "书房": "#E4DDE8",
  "玄关": "#EDE8DF",
  "default": "#E5DDD3",
};

const ROOM_COLOR_VALUES = Object.entries(ROOM_COLORS)
  .filter(([key]) => key !== "default")
  .map(([, value]) => value);

const DEFAULT_REQUIREMENTS: AgentRequirements = {
  family_size: "三口之家",
  has_pet: false,
  pet_type: "",
  style: "现代简约",
  budget: "20-35万",
  priorities: ["收纳", "采光", "动线"],
  notes: "",
};

type AgentRoomRaw = Record<string, unknown>;

function escapeHtml(value: unknown): string {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderInlineMarkdown(text: string, keyPrefix: string) {
  const tokens = text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g).filter(Boolean);
  return tokens.map((token, index) => {
    const key = `${keyPrefix}-${index}`;
    if (token.startsWith("**") && token.endsWith("**")) {
      return <strong key={key} className="font-semibold text-[var(--charcoal)]">{token.slice(2, -2)}</strong>;
    }
    if (token.startsWith("`") && token.endsWith("`")) {
      return (
        <code
          key={key}
          className="px-1.5 py-0.5 rounded-md bg-[var(--parchment)] text-[var(--charcoal)] font-mono text-[0.9em]"
        >
          {token.slice(1, -1)}
        </code>
      );
    }
    return <span key={key}>{token}</span>;
  });
}

function renderParagraphLines(text: string, keyPrefix: string) {
  const lines = text.split("\n");
  return lines.map((line, index) => (
    <span key={`${keyPrefix}-line-${index}`}>
      {renderInlineMarkdown(line, `${keyPrefix}-inline-${index}`)}
      {index < lines.length - 1 && <br />}
    </span>
  ));
}

function renderMarkdownContent(content: string) {
  const normalized = content.replace(/\r\n/g, "\n").trim();
  if (!normalized) {
    return <span className="text-[var(--mid-gray)]">正在组织回答...</span>;
  }

  const blocks = normalized.split(/\n\s*\n/);
  return (
    <div className="space-y-3">
      {blocks.map((block, blockIndex) => {
        const trimmed = block.trim();
        if (!trimmed) return null;

        const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
        if (heading) {
          const level = heading[1].length;
          const className =
            level === 1
              ? "text-lg font-bold text-[var(--charcoal)]"
              : level === 2
                ? "text-base font-bold text-[var(--charcoal)]"
                : "text-sm font-semibold text-[var(--charcoal)]";
          return (
            <div key={`block-${blockIndex}`} className={className}>
              {renderInlineMarkdown(heading[2], `heading-${blockIndex}`)}
            </div>
          );
        }

        const orderedLines = trimmed.split("\n").filter((line) => /^\d+\.\s+/.test(line.trim()));
        if (orderedLines.length > 0 && orderedLines.length === trimmed.split("\n").length) {
          return (
            <ol key={`block-${blockIndex}`} className="list-decimal pl-5 space-y-1.5 text-[var(--warm-gray)]">
              {orderedLines.map((line, index) => (
                <li key={`ordered-${blockIndex}-${index}`}>
                  {renderInlineMarkdown(line.trim().replace(/^\d+\.\s+/, ""), `ordered-${blockIndex}-${index}`)}
                </li>
              ))}
            </ol>
          );
        }

        const bulletLines = trimmed.split("\n").filter((line) => /^[-*]\s+/.test(line.trim()));
        if (bulletLines.length > 0 && bulletLines.length === trimmed.split("\n").length) {
          return (
            <ul key={`block-${blockIndex}`} className="list-disc pl-5 space-y-1.5 text-[var(--warm-gray)]">
              {bulletLines.map((line, index) => (
                <li key={`bullet-${blockIndex}-${index}`}>
                  {renderInlineMarkdown(line.trim().replace(/^[-*]\s+/, ""), `bullet-${blockIndex}-${index}`)}
                </li>
              ))}
            </ul>
          );
        }

        return (
          <p key={`block-${blockIndex}`} className="text-[var(--warm-gray)] leading-8">
            {renderParagraphLines(trimmed, `paragraph-${blockIndex}`)}
          </p>
        );
      })}
    </div>
  );
}

function extractAgentRooms(analysis: Record<string, unknown> | null): AgentRoomRaw[] {
  return Array.isArray(analysis?.rooms) ? analysis.rooms as AgentRoomRaw[] : [];
}

function getRoomNumericId(room: AgentRoomRaw): number | null {
  const rawId = room.id ?? room.room_id ?? room.room_index;
  if (typeof rawId === "number" && Number.isFinite(rawId)) return rawId;
  if (typeof rawId === "string") {
    const parsed = Number(rawId.replace(/[^\d.-]/g, ""));
    if (Number.isFinite(parsed)) return parsed;
  }
  const label = String(room.room_label ?? "");
  const match = label.match(/\d+/);
  return match ? Number(match[0]) : null;
}

function getRoomSemanticName(room: AgentRoomRaw | undefined, fallback: string): string {
  if (!room) return fallback;
  return String(room.room_type || room.room_label || room.name || fallback).trim() || fallback;
}

function flattenSuggestions(room: AgentRoomRaw | undefined): string {
  if (!room) return "";
  const suggestions = room.suggestions;
  if (typeof suggestions === "string") return suggestions;
  if (Array.isArray(suggestions)) return suggestions.map(String).join(" ");
  if (typeof suggestions === "object" && suggestions) {
    return Object.values(suggestions as Record<string, unknown>).map(String).join(" ");
  }
  return "";
}

function getRoomAdvice(room: AgentRoomRaw | undefined, semanticName: string): AgentAdvice | undefined {
  if (!room) return undefined;
  const analysis = String(room.analysis || "").trim();
  const suggestions = flattenSuggestions(room).trim();
  const description = [analysis, suggestions].filter(Boolean).join(" ");
  if (!description) return undefined;
  return {
    title: semanticName,
    description,
    priority: "medium",
  };
}

function findAgentRoomForYolo(agentRooms: AgentRoomRaw[], yoloRoom: AgentYoloRoom, index: number): AgentRoomRaw | undefined {
  return agentRooms.find((room) => getRoomNumericId(room) === yoloRoom.id) ?? agentRooms[index];
}

function buildRoomDisplayInfos(rooms: AgentYoloRoom[], analysis: Record<string, unknown> | null): RoomDisplayInfo[] {
  const agentRooms = extractAgentRooms(analysis);
  return rooms.map((room, index) => {
    const agentRoom = findAgentRoomForYolo(agentRooms, room, index);
    const fallback = `Room ${room.id}`;
    const semanticName = getRoomSemanticName(agentRoom, fallback);
    const displayName = semanticName === fallback ? fallback : `${semanticName} · ${fallback}`;
    const advice = getRoomAdvice(agentRoom, semanticName);
    const confidenceText = `置信度 ${Math.round(room.confidence * 100)}%`;
    const note = advice?.description
      ? advice.description.slice(0, 80)
      : `${confidenceText}，已进入空间分析。`;

    return {
      id: room.id,
      label: fallback,
      semanticName,
      displayName,
      confidence: room.confidence,
      areaRatio: room.area_ratio,
      color: ROOM_COLOR_VALUES[index % ROOM_COLOR_VALUES.length],
      note,
      advice,
    };
  });
}

function formatRequirementItems(requirements: AgentRequirements): Array<{ label: string; value: string }> {
  return [
    { label: "家庭人数", value: requirements.family_size || "未填写" },
    { label: "宠物", value: requirements.has_pet ? (requirements.pet_type ? `有${requirements.pet_type}` : "有宠物") : "无宠物" },
    { label: "预算", value: requirements.budget || "未填写" },
    { label: "风格", value: requirements.style || "未填写" },
    { label: "关注点", value: requirements.priorities.length ? requirements.priorities.join("、") : "未选择" },
    { label: "备注", value: requirements.notes || "无" },
  ];
}

function buildRequirementBasis(requirements: AgentRequirements): string {
  const basis = [
    requirements.budget ? `预算 ${requirements.budget}` : "",
    requirements.style ? `${requirements.style} 风格` : "",
    requirements.has_pet ? `${requirements.pet_type || "宠物"}友好` : "",
    ...requirements.priorities,
  ].filter(Boolean);
  return basis.length ? basis.join("、") : "当前填写的居住需求";
}

function createDefaultRequirements(): AgentRequirements {
  return {
    ...DEFAULT_REQUIREMENTS,
    priorities: [...DEFAULT_REQUIREMENTS.priorities],
  };
}

function normalizeRequirements(requirements?: Partial<AgentRequirements> | null): AgentRequirements {
  const defaults = createDefaultRequirements();
  return {
    family_size: typeof requirements?.family_size === "string" && requirements.family_size.trim()
      ? requirements.family_size
      : defaults.family_size,
    has_pet: Boolean(requirements?.has_pet),
    pet_type: typeof requirements?.pet_type === "string" ? requirements.pet_type : "",
    style: typeof requirements?.style === "string" && requirements.style.trim()
      ? requirements.style
      : defaults.style,
    budget: typeof requirements?.budget === "string" && requirements.budget.trim()
      ? requirements.budget
      : defaults.budget,
    priorities: Array.isArray(requirements?.priorities)
      ? requirements.priorities.map(String).filter(Boolean)
      : defaults.priorities,
    notes: typeof requirements?.notes === "string" ? requirements.notes : "",
  };
}

function toChatMessages(messages: Record<string, unknown>[] | undefined): ChatMessage[] {
  if (!Array.isArray(messages)) return [];
  const chatMessages: ChatMessage[] = [];
  for (const message of messages) {
    const role = message.role === "user" || message.role === "assistant" ? message.role : null;
    const content = typeof message.content === "string" ? message.content.trim() : "";
    if (!role || !content) continue;
    chatMessages.push({ role, content, status: "done" });
  }
  return chatMessages;
}

function downloadHtmlReport(params: {
  imageSrc: string | null;
  imageName: string;
  modelName: string;
  requirements: AgentRequirements;
  rooms: RoomDisplayInfo[];
  agent: AgentReport | null;
  chatMessages: ChatMessage[];
}) {
  const requirementRows = formatRequirementItems(params.requirements)
    .map((item) => `<div class="pill"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></div>`)
    .join("");
  const roomRows = params.rooms.map((room) => `
    <div class="room">
      <div><strong>${escapeHtml(room.displayName)}</strong><p>${escapeHtml(room.note)}</p></div>
      <span>${Math.round(room.areaRatio * 100)}%</span>
    </div>
  `).join("");
  const adviceRows = (params.agent?.advice ?? []).map((item) => `
    <section class="section">
      <h3>${escapeHtml(item.title)}</h3>
      <p>${escapeHtml(item.description)}</p>
    </section>
  `).join("");
  const chatRows = params.chatMessages
    .filter((message) => message.content.trim())
    .map((message) => `<p><strong>${message.role === "user" ? "我" : "RoomWise Agent"}：</strong>${escapeHtml(message.content)}</p>`)
    .join("");

  const html = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>RoomWise 户型分析报告</title>
  <style>
    :root { --paper:#F8F3EB; --card:#FFFDF8; --ink:#2F2924; --muted:#7B7068; --line:#E7D9C7; --terra:#B86F55; --sage:#70866C; }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 32px; color: var(--ink); background: var(--paper); font-family: "Microsoft YaHei", "PingFang SC", sans-serif; }
    .wrap { max-width: 960px; margin: 0 auto; }
    header { margin-bottom: 24px; }
    h1 { margin: 0 0 8px; font-size: 30px; }
    .meta { color: var(--muted); font-size: 13px; }
    .card, .section { background: var(--card); border: 1px solid var(--line); border-radius: 22px; padding: 22px; box-shadow: 0 14px 40px rgba(88, 65, 44, .08); margin-bottom: 18px; }
    img { width: 100%; max-height: 520px; object-fit: contain; border-radius: 18px; background: #EFE4D5; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .pill { border: 1px solid var(--line); background: #FBF6EE; border-radius: 16px; padding: 12px; display: flex; justify-content: space-between; gap: 16px; }
    .pill span, p { color: var(--muted); line-height: 1.7; }
    .room { border: 1px solid var(--line); border-radius: 16px; padding: 14px; display: flex; justify-content: space-between; gap: 16px; margin-bottom: 10px; background: #FBF6EE; }
    .room span { color: var(--terra); font-weight: 700; }
    h2, h3 { margin: 0 0 10px; }
    @media print { body { background: #fff; padding: 0; } .card, .section { box-shadow: none; break-inside: avoid; } }
  </style>
</head>
<body>
  <main class="wrap">
    <header>
      <h1>RoomWise 户型分析报告</h1>
      <div class="meta">${escapeHtml(params.imageName)} · ${escapeHtml(params.modelName || "未记录模型")} · ${new Date().toLocaleString()}</div>
    </header>
    ${params.imageSrc ? `<div class="card"><img src="${params.imageSrc}" alt="分割结果" /></div>` : ""}
    <section class="card"><h2>本次需求</h2><div class="grid">${requirementRows}</div></section>
    <section class="card"><h2>空间识别</h2>${roomRows || "<p>暂无空间识别结果。</p>"}</section>
    <section class="card"><h2>Agent 总结</h2><p>${escapeHtml(params.agent?.summary || "暂无总结。")}</p></section>
    ${adviceRows || `<section class="section"><h3>装修建议</h3><p>暂无建议。</p></section>`}
    <section class="card"><h2>空间优化</h2><p>${escapeHtml(params.agent?.circulation || "暂无空间优化建议。")}</p></section>
    <section class="card"><h2>继续沟通摘要</h2>${chatRows || "<p>暂无继续沟通记录。</p>"}</section>
  </main>
</body>
</html>`;

  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `roomwise-report-${Date.now()}.html`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function RoomRegion({ name, color, area_ratio, note }: { name: string; color?: string; area_ratio: number; note?: string }) {
  const bg = color || ROOM_COLORS[name] || ROOM_COLORS["default"];
  return (
    <div
      className="rounded-xl p-3 flex flex-col gap-1 text-xs min-w-0"
      style={{ background: bg }}
    >
      <div className="flex items-center justify-between">
        <span className="font-semibold text-[var(--charcoal)]">{name}</span>
        <span className="text-[var(--warm-gray)] font-medium">{Math.round(area_ratio * 100)}%</span>
      </div>
      {note && <p className="text-[var(--warm-gray)] leading-relaxed line-clamp-2">{note}</p>}
    </div>
  );
}

function AdviceCard({
  title,
  description,
  priority,
  index,
  active = false,
  onClick,
}: {
  title: string;
  description: string;
  priority: string;
  index: number;
  active?: boolean;
  onClick?: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`w-full text-left flex gap-4 p-5 rounded-xl border bg-[var(--warm-white)] hover:shadow-[var(--shadow-sm)] transition-all ${
        active ? "border-[var(--terracotta-light)] ring-1 ring-[var(--terracotta-light)] bg-[var(--terracotta-pale)]" : "border-[var(--border)]"
      }`}
    >
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-[var(--parchment)] flex items-center justify-center text-sm font-semibold text-[var(--warm-gray)]">
        {index + 1}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <h4 className="font-semibold text-[var(--charcoal)] text-sm">{title}</h4>
          <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium border ${
            priority === "high" ? "badge-high" :
            priority === "medium" ? "badge-medium" : "badge-low"
          }`}>
            {priority === "high" ? "高优先级" : priority === "medium" ? "建议关注" : "可参考"}
          </span>
        </div>
        <p className="text-xs text-[var(--warm-gray)] leading-relaxed">{description}</p>
      </div>
    </button>
  );
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function toInferenceResult(file: File, rooms: AgentYoloRoom[], visualization: string): InferenceResult {
  const confidence = rooms.length
    ? rooms.reduce((sum, room) => sum + room.confidence, 0) / rooms.length
    : 0;
  return {
    dataset: "room_separation_3",
    image_name: file.name,
    image_path: "",
    mask_path: `data:image/jpeg;base64,${visualization}`,
    confidence,
    regions: rooms.map((room, index) => ({
      name: `Room ${room.id}`,
      color: ROOM_COLOR_VALUES[index % ROOM_COLOR_VALUES.length],
      area_ratio: room.area_ratio,
      note: `置信度 ${Math.round(room.confidence * 100)}%，已进入 Agent 空间分析。`,
    })),
    summary: `YOLO 已识别 ${rooms.length} 个空间区域，分割结果已进入 Agent 分析流程。`,
  };
}

function toInferenceResultFromRecord(record: {
  filename: string;
  rooms: AgentYoloRoom[];
  visualization: string;
  summary?: string | null;
}): InferenceResult {
  const confidence = record.rooms.length
    ? record.rooms.reduce((sum, room) => sum + room.confidence, 0) / record.rooms.length
    : 0;
  return {
    dataset: "room_separation_3",
    image_name: record.filename,
    image_path: "",
    mask_path: `data:image/jpeg;base64,${record.visualization}`,
    confidence,
    regions: record.rooms.map((room, index) => ({
      name: `Room ${room.id}`,
      color: ROOM_COLOR_VALUES[index % ROOM_COLOR_VALUES.length],
      area_ratio: room.area_ratio,
      note: `置信度 ${Math.round(room.confidence * 100)}%，已进入 Agent 空间分析。`,
    })),
    summary: record.summary || `已恢复最近一次分析结果，共识别 ${record.rooms.length} 个空间区域。`,
  };
}

function RequirementForm({
  requirements,
  onChange,
  models,
  selectedModel,
  onModelChange,
}: {
  requirements: AgentRequirements;
  onChange: (requirements: AgentRequirements) => void;
  models: ServingModel[];
  selectedModel: string;
  onModelChange: (modelName: string) => void;
}) {
  function set<K extends keyof AgentRequirements>(key: K, value: AgentRequirements[K]) {
    onChange({ ...requirements, [key]: value });
  }

  function togglePriority(value: string) {
    const exists = requirements.priorities.includes(value);
    set("priorities", exists
      ? requirements.priorities.filter((item) => item !== value)
      : [...requirements.priorities, value]);
  }

  return (
    <div className="card p-6 mb-6 animate-fade-up">
      <div className="flex items-start justify-between gap-4 mb-5">
        <div>
          <h2 className="text-lg font-bold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>居住需求</h2>
          <p className="text-xs text-[var(--mid-gray)] mt-1">这些信息会交给 Agent，用于生成更贴近真实家庭的装修建议。</p>
        </div>
        <span className="px-3 py-1 rounded-full bg-[var(--terracotta-pale)] text-[var(--terracotta)] text-xs border border-[var(--terracotta-light)]">
          个性化分析
        </span>
      </div>
      <div className="mb-5 rounded-2xl border border-[var(--border)] bg-[var(--ivory)] p-4">
        <div className="flex flex-col md:flex-row md:items-center gap-3 justify-between">
          <div>
            <p className="text-sm font-semibold text-[var(--charcoal)]">分割模型</p>
            <p className="text-xs text-[var(--mid-gray)] mt-0.5">默认使用当前最佳模型 yolo26m_seg_best.pt，也可切换模型对比效果。</p>
          </div>
          <select
            value={selectedModel}
            onChange={(event) => onModelChange(event.target.value)}
            className="md:min-w-72 px-3 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--warm-white)] text-sm text-[var(--charcoal)] focus:outline-none focus:border-[var(--terracotta)] transition-all"
          >
            {models.length === 0 && <option value="">未发现 serving 模型</option>}
            {models.map((model) => (
              <option key={model.name} value={model.name}>
                {model.label}{model.is_default ? " · 默认" : ""}
              </option>
            ))}
          </select>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <label className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-[var(--mid-gray)] uppercase tracking-wider">家庭人数</span>
          <input
            value={requirements.family_size}
            onChange={(event) => set("family_size", event.target.value)}
            placeholder="例如：三口之家"
            className="px-3 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--warm-white)] text-sm text-[var(--charcoal)] focus:outline-none focus:border-[var(--terracotta)] transition-all"
          />
        </label>
        <label className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-[var(--mid-gray)] uppercase tracking-wider">装修风格</span>
          <select
            value={requirements.style}
            onChange={(event) => set("style", event.target.value)}
            className="px-3 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--warm-white)] text-sm text-[var(--charcoal)] focus:outline-none focus:border-[var(--terracotta)] transition-all"
          >
            {["现代简约", "奶油风", "原木风", "中古风", "轻奢", "北欧", "无偏好"].map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-[var(--mid-gray)] uppercase tracking-wider">预算</span>
          <select
            value={requirements.budget}
            onChange={(event) => set("budget", event.target.value)}
            className="px-3 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--warm-white)] text-sm text-[var(--charcoal)] focus:outline-none focus:border-[var(--terracotta)] transition-all"
          >
            {["10万以内", "10-20万", "20-35万", "35万以上", "暂不确定"].map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-[var(--mid-gray)] uppercase tracking-wider">宠物</span>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => set("has_pet", !requirements.has_pet)}
              className={`px-3 py-2.5 rounded-xl border text-sm transition-all ${requirements.has_pet ? "bg-[var(--sage-pale)] border-[var(--sage-light)] text-[var(--sage)]" : "border-[var(--border)] text-[var(--mid-gray)]"}`}
            >
              {requirements.has_pet ? "有宠物" : "无宠物"}
            </button>
            <input
              value={requirements.pet_type}
              onChange={(event) => set("pet_type", event.target.value)}
              disabled={!requirements.has_pet}
              placeholder="猫/狗"
              className="min-w-0 flex-1 px-3 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--warm-white)] text-sm text-[var(--charcoal)] disabled:opacity-50 focus:outline-none focus:border-[var(--terracotta)] transition-all"
            />
          </div>
        </label>
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        {["收纳", "儿童友好", "宠物友好", "采光", "动线", "居家办公", "低预算"].map((item) => (
          <button
            key={item}
            type="button"
            onClick={() => togglePriority(item)}
            className={`px-3 py-1.5 rounded-full text-xs border transition-all ${requirements.priorities.includes(item) ? "bg-[var(--terracotta)] text-white border-[var(--terracotta)]" : "bg-[var(--ivory)] text-[var(--warm-gray)] border-[var(--border)]"}`}
          >
            {item}
          </button>
        ))}
      </div>
      <textarea
        value={requirements.notes}
        onChange={(event) => set("notes", event.target.value)}
        rows={2}
        placeholder="补充说明：例如希望保留一间书房、厨房要开放式、老人同住需要防滑和夜灯..."
        className="mt-4 w-full px-3 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--warm-white)] text-sm text-[var(--charcoal)] focus:outline-none focus:border-[var(--terracotta)] transition-all resize-none"
      />
    </div>
  );
}

function RequirementSummaryCard({ requirements }: { requirements: AgentRequirements }) {
  return (
    <div className="card p-6">
      <div className="flex items-start justify-between gap-3 mb-4">
        <div>
          <h2 className="text-lg font-bold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>本次需求摘要</h2>
          <p className="text-xs text-[var(--mid-gray)] mt-0.5">Agent 建议会围绕这些居住条件展开。</p>
        </div>
        <span className="px-3 py-1 rounded-full bg-[var(--sage-pale)] text-[var(--sage)] text-xs border border-[var(--sage-light)]">
          {buildRequirementBasis(requirements)}
        </span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {formatRequirementItems(requirements).map((item) => (
          <div key={item.label} className="rounded-xl border border-[var(--border)] bg-[var(--ivory)] px-4 py-3">
            <p className="text-[10px] uppercase tracking-wider text-[var(--light-gray)] mb-1">{item.label}</p>
            <p className="text-sm font-semibold text-[var(--charcoal)] line-clamp-2">{item.value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function ChatPanel({
  disabled,
  messages,
  input,
  streaming,
  onInput,
  onSend,
}: {
  disabled: boolean;
  messages: ChatMessage[];
  input: string;
  streaming: boolean;
  onInput: (value: string) => void;
  onSend: () => void;
}) {
  return (
    <div className="card p-6">
      <div className="flex items-start justify-between gap-4 mb-5">
        <div>
          <h2 className="text-lg font-bold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>继续沟通改造想法</h2>
          <p className="text-xs text-[var(--mid-gray)] mt-0.5">可以继续追问预算、房间改造、宠物收纳或动线细节。</p>
        </div>
        <span className="px-3 py-1 rounded-full bg-[var(--ivory)] text-[var(--warm-gray)] text-xs border border-[var(--border)]">
          流式对话
        </span>
      </div>

      <div className="space-y-3 mb-4 max-h-72 overflow-y-auto pr-1">
        {messages.length === 0 && (
          <div className="rounded-2xl border border-dashed border-[var(--border)] bg-[var(--ivory)] p-5 text-sm text-[var(--mid-gray)]">
            试试追问：“如果预算降到10万以内怎么办？”、“主卧能不能改大？”、“我家有猫怎么做收纳？”
          </div>
        )}
        {messages.map((message, index) => (
          <div
            key={`${message.role}-${index}`}
            className={`rounded-2xl px-4 py-3 text-sm leading-relaxed border ${
              message.role === "user"
                ? "ml-8 bg-[var(--terracotta-pale)] border-[var(--terracotta-light)] text-[var(--charcoal)]"
                : "mr-8 bg-[var(--ivory)] border-[var(--border)] text-[var(--warm-gray)]"
            }`}
          >
            <div className="text-[10px] uppercase tracking-wider mb-1 text-[var(--light-gray)]">
              {message.role === "user" ? "我的追问" : "RoomWise Agent"}
              {message.status === "streaming" ? " · 正在生成" : ""}
              {message.status === "error" ? " · 失败" : ""}
            </div>
            {message.role === "assistant"
              ? renderMarkdownContent(message.content)
              : <p className="text-[var(--charcoal)] leading-8">{message.content || "正在组织回答..."}</p>}
          </div>
        ))}
      </div>

      <div className="flex flex-col md:flex-row gap-3">
        <input
          value={input}
          onChange={(event) => onInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              onSend();
            }
          }}
          disabled={disabled || streaming}
          placeholder={disabled ? "请重新上传户型图后继续沟通" : "输入你的新想法，例如：预算降到10万以内怎么调整？"}
          className="flex-1 px-4 py-3 rounded-xl border border-[var(--border)] bg-[var(--warm-white)] text-sm text-[var(--charcoal)] disabled:opacity-50 focus:outline-none focus:border-[var(--terracotta)] transition-all"
        />
        <button
          type="button"
          onClick={onSend}
          disabled={disabled || streaming || !input.trim()}
          className="px-5 py-3 rounded-xl bg-[var(--terracotta)] text-white text-sm font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-[var(--shadow-sm)] transition-all"
        >
          {streaming ? "生成中" : "发送"}
        </button>
      </div>
    </div>
  );
}

function toAgentReport(analysis: Record<string, unknown> | null | undefined): AgentReport {
  const rooms = Array.isArray(analysis?.rooms) ? analysis.rooms as Array<Record<string, unknown>> : [];
  const pros = Array.isArray(analysis?.pros) ? analysis.pros.map(String) : [];
  const cons = Array.isArray(analysis?.cons) ? analysis.cons.map(String) : [];
  const coreIssues = Array.isArray(analysis?.core_issues) ? analysis.core_issues.map(String) : [];
  const overall = String(analysis?.overall_suggestions || analysis?.overall_assessment || "Agent 已完成户型分析。");

  const advice = [
    ...coreIssues.map((item) => ({
      title: "核心问题",
      description: item,
      priority: "high" as const,
    })),
    ...rooms.slice(0, 4).map((room) => ({
      title: String(room.room_type || room.room_label || "空间建议"),
      description: [
        room.analysis,
        typeof room.suggestions === "object" && room.suggestions
          ? Object.values(room.suggestions as Record<string, unknown>).join(" ")
          : "",
      ].filter(Boolean).join(" "),
      priority: "medium" as const,
    })),
  ].filter((item) => item.description);

  return {
    report_id: `agent-${Date.now()}`,
    dataset: "room_separation_3",
    scene_type: String(analysis?.house_type || "户型图空间理解"),
    spaces: rooms.map((room) => String(room.room_type || room.room_label || "未命名空间")),
    advice: advice.length ? advice : [{
      title: "整体建议",
      description: overall,
      priority: "medium",
    }],
    circulation: String(analysis?.overall_assessment || overall),
    summary: [
      analysis?.rating ? `评级：${String(analysis.rating)}` : "",
      pros.length ? `优势：${pros.join("、")}` : "",
      cons.length ? `待优化：${cons.join("、")}` : "",
      overall,
    ].filter(Boolean).join("；"),
    export_path: "",
  };
}

export function Analysis() {
  const [searchParams, setSearchParams] = useSearchParams();
  const recordId = searchParams.get("recordId")?.trim() || "";
  const uploadMode = searchParams.get("mode") === "upload";
  const [view, setView] = useState<ViewMode>("result");
  const [inference, setInference] = useState<InferenceResult | null>(null);
  const [agent, setAgent] = useState<AgentReport | null>(null);
  const [agentAnalysis, setAgentAnalysis] = useState<Record<string, unknown> | null>(null);
  const [agentStatus, setAgentStatus] = useState<"idle" | "analyzing" | "done" | "error">("idle");
  const [agentError, setAgentError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState("");
  const [servingModels, setServingModels] = useState<ServingModel[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [usedModelName, setUsedModelName] = useState("");
  const [loading, setLoading] = useState(true);
  const [dragOver, setDragOver] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadPreview, setUploadPreview] = useState<string | null>(null);
  const [yoloRooms, setYoloRooms] = useState<AgentYoloRoom[]>([]);
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null);
  const [selectedRoomId, setSelectedRoomId] = useState<number | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatStreaming, setChatStreaming] = useState(false);
  const [requirements, setRequirements] = useState<AgentRequirements>(() => createDefaultRequirements());
  const fileInputRef = useRef<HTMLInputElement>(null);

  const resetToUploadState = useCallback(() => {
    setView("upload");
    setUploadedFile(null);
    setUploadPreview(null);
    setInference(null);
    setAgent(null);
    setAgentAnalysis(null);
    setAgentStatus("idle");
    setAgentError(null);
    setSessionId("");
    setUsedModelName("");
    setYoloRooms([]);
    setImageSize(null);
    setSelectedRoomId(null);
    setChatMessages([]);
    setChatInput("");
    setChatStreaming(false);
    setRequirements(createDefaultRequirements());
  }, []);

  const hydrateFromRecord = useCallback((record: FloorplanRecord, session: AgentSessionResponse | null = null) => {
    const analysis = session?.analysis ?? record.analysis ?? null;
    const rooms = session?.yolo_rooms?.length ? session.yolo_rooms : record.rooms;
    const visualization = session?.visualization || record.visualization;
    const imageSizeValue = session?.image_size ?? record.image_size;
    const status = session?.status ?? record.agent_status;

    setView("result");
    setUploadedFile(null);
    setUploadPreview(visualization ? `data:image/jpeg;base64,${visualization}` : null);
    setInference(toInferenceResultFromRecord({
      filename: record.filename,
      rooms,
      visualization,
      summary: record.summary,
    }));
    setAgentAnalysis(analysis);
    setAgent(analysis ? toAgentReport(analysis) : null);
    setAgentStatus(
      status === "done" && analysis
        ? "done"
        : status === "error"
          ? "error"
          : status === "analyzing"
            ? "analyzing"
            : analysis
              ? "done"
              : "idle"
    );
    setAgentError(session?.error ?? record.agent_error ?? null);
    setSessionId(record.session_id || "");
    setUsedModelName("");
    setYoloRooms(rooms);
    setImageSize(imageSizeValue);
    setSelectedRoomId(rooms[0]?.id ?? null);
    setChatMessages(toChatMessages(session?.messages));
    setChatInput("");
    setChatStreaming(false);
    setRequirements(normalizeRequirements(session?.requirements ?? record.requirements));
  }, []);

  const loadPersistedAnalysis = useCallback(async (): Promise<boolean> => {
    if (uploadMode) return false;

    let record: FloorplanRecord | null = null;
    if (recordId) {
      try {
        record = await fetchFloorplan(recordId);
      } catch {
        record = null;
      }
    }

    if (!record) {
      const records = await fetchFloorplans();
      record = records[0] ?? null;
    }
    if (!record) return false;

    let session: AgentSessionResponse | null = null;
    if (record.session_id) {
      try {
        session = await fetchAgentSession(record.session_id);
      } catch {
        session = null;
      }
    }

    hydrateFromRecord(record, session);
    return true;
  }, [hydrateFromRecord, recordId, uploadMode]);

  useEffect(() => {
    let cancelled = false;

    async function loadPage() {
      setLoading(true);
      const [modelsResult, historyResult] = await Promise.allSettled([
        fetchServingModels(),
        loadPersistedAnalysis(),
      ]);
      if (cancelled) return;

      if (modelsResult.status === "fulfilled") {
        setServingModels(modelsResult.value);
        const defaultModel = modelsResult.value.find((model) => model.is_default) ?? modelsResult.value[0];
        setSelectedModel(defaultModel?.name ?? "");
      }

      const hasPersistedHistory = historyResult.status === "fulfilled" && historyResult.value;
      if (!hasPersistedHistory) {
        resetToUploadState();
      }

      setLoading(false);
    }

    loadPage().catch(() => {
      if (!cancelled) {
        resetToUploadState();
        setLoading(false);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [loadPersistedAnalysis, resetToUploadState]);

  useEffect(() => {
    if (selectedModel || servingModels.length === 0) return;
    const defaultModel = servingModels.find((model) => model.is_default) ?? servingModels[0];
    setSelectedModel(defaultModel?.name ?? "");
  }, [selectedModel, servingModels]);

  const [uploadingProgress, setUploadingProgress] = useState(0);

  const handleFile = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) return;
    setUploadedFile(file);
    const url = URL.createObjectURL(file);
    setUploadPreview(url);
    setInference(null);
    setAgent(null);
    setAgentAnalysis(null);
    setAgentError(null);
    setAgentStatus("idle");
    setSessionId("");
    setUsedModelName("");
    setYoloRooms([]);
    setImageSize(null);
    setSelectedRoomId(null);
    setChatMessages([]);
    setChatInput("");
    setChatStreaming(false);
    setView("analyzing");
    setUploadingProgress(0);

    const tick = setInterval(() => {
      setUploadingProgress(p => Math.min(p + 8, 92));
    }, 300);

    try {
      const yolo = await analyzeFloorplan(file, requirements, selectedModel);
      setSessionId(yolo.session_id);
      setUsedModelName(yolo.model?.name ?? selectedModel);
      setInference(toInferenceResult(file, yolo.yolo_rooms, yolo.visualization));
      setYoloRooms(yolo.yolo_rooms);
      setImageSize(yolo.image_size);
      setSelectedRoomId(yolo.yolo_rooms[0]?.id ?? null);
      setUploadPreview(`data:image/jpeg;base64,${yolo.visualization}`);
      setAgentStatus("analyzing");
      clearInterval(tick);
      setUploadingProgress(100);
      setView("result");

      for (let i = 0; i < 180; i += 1) {
        const session = await fetchAgentSession(yolo.session_id);
        if (session.status === "done" && session.analysis) {
          setAgent(toAgentReport(session.analysis));
          setAgentAnalysis(session.analysis);
          setInference(toInferenceResult(file, session.yolo_rooms, session.visualization));
          setYoloRooms(session.yolo_rooms);
          setImageSize(session.image_size);
          setUploadPreview(`data:image/jpeg;base64,${session.visualization}`);
          setAgentStatus("done");
          setView("result");
          return;
        }
        if (session.status === "error") {
          setAgentStatus("error");
          setAgentError(session.error || "Agent 分析失败，但图像分割结果已生成。");
          setView("result");
          return;
        }
        await sleep(1000);
      }
      setAgentStatus("error");
      setAgentError("Agent 分析超时，但图像分割结果已生成。");
      setView("result");
    } catch (error) {
      clearInterval(tick);
      console.error(error);
      setAgentStatus("error");
      setAgentError(error instanceof Error ? error.message : "分析失败，请稍后重试。");
      setView("upload");
    }
  }, [requirements, selectedModel]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const roomDisplayInfos = useMemo(
    () => buildRoomDisplayInfos(yoloRooms, agentAnalysis),
    [agentAnalysis, yoloRooms]
  );

  const selectedRoom = useMemo(
    () => roomDisplayInfos.find((room) => room.id === selectedRoomId) ?? roomDisplayInfos[0],
    [roomDisplayInfos, selectedRoomId]
  );

  const selectedAdviceTitle = selectedRoom?.advice?.title ?? selectedRoom?.semanticName;

  const handleAdviceClick = useCallback((title: string) => {
    const matchedRoom = roomDisplayInfos.find((room) => title.includes(room.semanticName) || title.includes(room.label));
    if (matchedRoom) setSelectedRoomId(matchedRoom.id);
  }, [roomDisplayInfos]);

  const handleSendChat = useCallback(async () => {
    const message = chatInput.trim();
    if (!message || !sessionId || chatStreaming) return;

    setChatInput("");
    setChatStreaming(true);
    setChatMessages((prev) => [
      ...prev,
      { role: "user", content: message, status: "done" },
      { role: "assistant", content: "", status: "streaming" },
    ]);

    try {
      await streamAgentChat(sessionId, message, (token) => {
        setChatMessages((prev) => {
          const next = [...prev];
          const index = next.length - 1;
          if (index >= 0 && next[index].role === "assistant") {
            next[index] = { ...next[index], content: next[index].content + token };
          }
          return next;
        });
      });
      setChatMessages((prev) => {
        const next = [...prev];
        const index = next.length - 1;
        if (index >= 0 && next[index].role === "assistant") {
          next[index] = { ...next[index], status: "done" };
        }
        return next;
      });
    } catch (error) {
      setChatMessages((prev) => {
        const next = [...prev];
        const index = next.length - 1;
        if (index >= 0 && next[index].role === "assistant") {
          next[index] = {
            ...next[index],
            status: "error",
            content: next[index].content || (error instanceof Error ? error.message : "对话失败，请重新上传户型图后再试。"),
          };
        }
        return next;
      });
    } finally {
      setChatStreaming(false);
    }
  }, [chatInput, chatStreaming, sessionId]);

  const handleExportReport = useCallback(() => {
    downloadHtmlReport({
      imageSrc: uploadPreview,
      imageName: inference?.image_name || uploadedFile?.name || "户型图分析",
      modelName: usedModelName || selectedModel,
      requirements,
      rooms: roomDisplayInfos,
      agent,
      chatMessages,
    });
  }, [agent, chatMessages, inference?.image_name, requirements, roomDisplayInfos, selectedModel, uploadPreview, uploadedFile?.name, usedModelName]);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-[var(--terracotta)] border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-[var(--mid-gray)] text-sm">正在加载分析结果...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen px-6 py-10 max-w-5xl mx-auto">
      {/* Page header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-[var(--charcoal)] mb-2" style={{ fontFamily: "var(--font-display)" }}>智能分析</h1>
        <p className="text-[var(--mid-gray)] text-sm">上传户型图，获取 AI 房间分割与专业装修建议</p>
      </div>

      {view === "upload" && (
        <RequirementForm
          requirements={requirements}
          onChange={setRequirements}
          models={servingModels}
          selectedModel={selectedModel}
          onModelChange={setSelectedModel}
        />
      )}

      {/* Upload zone */}
      {view === "upload" && (
        <div
          className={`upload-zone p-16 flex flex-col items-center justify-center gap-4 text-center cursor-pointer mb-10 animate-fade-up`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileInput}
          />
          <div className="w-16 h-16 rounded-2xl bg-[var(--parchment)] flex items-center justify-center">
            <i className="fa-solid fa-upload text-2xl text-[var(--terracotta)]" />
          </div>
          <div>
            <p className="font-semibold text-[var(--charcoal)] text-lg mb-1" style={{ fontFamily: "var(--font-display)" }}>
              将户型图拖拽到此处
            </p>
            <p className="text-sm text-[var(--mid-gray)]">
              或点击选择文件 · 支持 JPG / PNG / PDF
            </p>
          </div>
          <p className="text-xs text-[var(--light-gray)]">建议上传清晰标注房间名称的户型图</p>
        </div>
      )}

      {/* Analyzing state */}
      {view === "analyzing" && (
        <div className="card p-16 flex flex-col items-center justify-center gap-4 text-center mb-10 animate-fade-in">
          <div className="w-16 h-16 relative">
            <div className="absolute inset-0 rounded-full border-3 border-[var(--terracotta)] border-t-transparent animate-spin" />
            <div className="absolute inset-3 rounded-full bg-[var(--terracotta-pale)] flex items-center justify-center">
              <i className="fa-solid fa-magic text-[var(--terracotta)]" />
            </div>
          </div>
          <div>
            <p className="font-semibold text-[var(--charcoal)] text-lg mb-1" style={{ fontFamily: "var(--font-display)" }}>AI 正在分析中</p>
            <p className="text-sm text-[var(--mid-gray)]">识别房间区域 · 计算空间面积 · 生成装修建议</p>
          </div>
          <div className="w-64 h-1.5 bg-[var(--parchment)] rounded-full overflow-hidden">
            <div className="h-full bg-[var(--terracotta)] rounded-full transition-all duration-300" style={{ width: `${uploadingProgress}%` }} />
          </div>
          <p className="text-xs text-[var(--light-gray)]">{uploadedFile?.name}</p>
        </div>
      )}

      {/* Results */}
      {view === "result" && (
        <div className="space-y-10 animate-fade-up">
          <div className="flex flex-col md:flex-row md:items-stretch gap-3 justify-between">
            <div className="flex-1">
              <RequirementSummaryCard requirements={requirements} />
            </div>
            <button
              type="button"
              onClick={handleExportReport}
              className="md:self-stretch inline-flex items-center justify-center gap-2 px-5 py-3 rounded-2xl border border-[var(--border)] bg-[var(--warm-white)] text-[var(--warm-gray)] hover:text-[var(--terracotta)] hover:border-[var(--terracotta-light)] hover:shadow-[var(--shadow-sm)] transition-all text-sm font-semibold"
            >
              <i className="fa-solid fa-file-export" />
              导出 HTML 报告
            </button>
          </div>

          {/* Floor plan visualization + room list */}
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
            {/* Floor plan viz */}
            <div className="lg:col-span-3">
              <div className="card p-6 h-full">
                <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between mb-5">
                  <div className="min-w-0 flex-1">
                    <h2 className="text-lg font-bold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>分割结果</h2>
                    <p
                      className="text-xs text-[var(--mid-gray)] mt-0.5 truncate"
                      title={inference?.image_name || (uploadedFile ? uploadedFile.name : "户型图分析")}
                    >
                      {inference?.image_name || (uploadedFile ? uploadedFile.name : "户型图分析")}
                    </p>
                    {usedModelName && (
                      <p className="text-[10px] text-[var(--light-gray)] mt-1 font-mono truncate" title={`Model: ${usedModelName}`}>
                        Model: {usedModelName}
                      </p>
                    )}
                  </div>
                  {inference?.confidence != null && inference.confidence > 0 && (
                    <div className="self-start flex-shrink-0 whitespace-nowrap flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[var(--sage-pale)] text-[var(--sage)] text-xs border border-[var(--sage-light)]">
                      <i className="fa-solid fa-check" />
                      {Math.round(inference.confidence * 100)}% 置信度
                    </div>
                  )}
                </div>

                {/* Uploaded image preview or room grid fallback */}
                {uploadPreview ? (
                  <div className="relative rounded-xl overflow-hidden" style={{ minHeight: "280px", background: "var(--parchment)" }}>
                    <img
                      src={uploadPreview}
                      alt="上传的户型图"
                      className="w-full h-full object-contain"
                      style={{ maxHeight: "320px" }}
                    />
                    {imageSize && yoloRooms.length > 0 && (
                      <svg
                        className="absolute inset-0 w-full h-full"
                        viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
                        preserveAspectRatio="xMidYMid meet"
                      >
                        {yoloRooms.map((room, index) => {
                          const active = selectedRoomId === room.id;
                          const display = roomDisplayInfos.find((item) => item.id === room.id);
                          const color = display?.color ?? ROOM_COLOR_VALUES[index % ROOM_COLOR_VALUES.length];
                          const points = room.polygon?.map((point) => point.join(",")).join(" ");
                          const bbox = room.bbox;
                          return (
                            <g key={room.id} onClick={() => setSelectedRoomId(room.id)} style={{ cursor: "pointer" }}>
                              {points ? (
                                <polygon
                                  points={points}
                                  fill={active ? `${color}88` : `${color}22`}
                                  stroke={active ? "var(--terracotta)" : color}
                                  strokeWidth={active ? 5 : 2}
                                />
                              ) : (
                                <rect
                                  x={bbox.x1}
                                  y={bbox.y1}
                                  width={bbox.x2 - bbox.x1}
                                  height={bbox.y2 - bbox.y1}
                                  fill={active ? `${color}88` : `${color}22`}
                                  stroke={active ? "var(--terracotta)" : color}
                                  strokeWidth={active ? 5 : 2}
                                  rx={8}
                                />
                              )}
                              <text
                                x={bbox.x1 + 8}
                                y={Math.max(18, bbox.y1 + 22)}
                                fill={active ? "var(--charcoal)" : "#4E463F"}
                                fontSize={active ? 22 : 18}
                                fontWeight="700"
                              >
                                {display?.semanticName ?? `Room ${room.id}`}
                              </text>
                            </g>
                          );
                        })}
                      </svg>
                    )}
                  </div>
                ) : (
                  /* Room grid visualization */
                  <div className="flex flex-col gap-2" style={{ minHeight: "280px" }}>
                    <div className="flex gap-2 flex-1">
                      <div
                        className="flex-1 rounded-xl flex items-center justify-center text-xs font-semibold"
                        style={{ background: inference?.regions?.[0] ? "#E8D5C4" : "var(--parchment)", border: "1.5px solid #C4A882" }}
                      >
                        {inference?.regions?.[0]?.name ?? "客厅"} {inference?.regions?.[0] ? `${Math.round(inference.regions[0].area_ratio * 100)}%` : ""}
                      </div>
                      <div className="flex flex-col gap-2 flex-1">
                        <div
                          className="flex-1 rounded-xl flex items-center justify-center text-xs font-semibold"
                          style={{ background: inference?.regions?.[1] ? "#D4E4D4" : "var(--parchment)", border: "1.5px solid #9DB89D" }}
                        >
                          {inference?.regions?.[1]?.name ?? "主卧"} {inference?.regions?.[1] ? `${Math.round(inference.regions[1].area_ratio * 100)}%` : ""}
                        </div>
                        <div
                          className="flex-1 rounded-xl flex items-center justify-center text-xs font-semibold"
                          style={{ background: inference?.regions?.[2] ? "#F0D4D0" : "var(--parchment)", border: "1.5px solid #D4A09A" }}
                        >
                          {inference?.regions?.[2]?.name ?? "厨房"} {inference?.regions?.[2] ? `${Math.round(inference.regions[2].area_ratio * 100)}%` : ""}
                        </div>
                      </div>
                    </div>
                    <div className="flex gap-2" style={{ minHeight: "80px" }}>
                      <div
                        className="flex-1 rounded-xl flex items-center justify-center text-xs font-semibold"
                        style={{ background: inference?.regions?.[3] ? "#D8E4F0" : "var(--parchment)", border: "1.5px solid #9AB5CC" }}
                      >
                        {inference?.regions?.[3]?.name ?? "卫生间"} {inference?.regions?.[3] ? `${Math.round(inference.regions[3].area_ratio * 100)}%` : ""}
                      </div>
                      <div
                        className="flex-1 rounded-xl flex items-center justify-center text-xs font-semibold"
                        style={{ background: "var(--parchment)", border: "1.5px solid var(--border)" }}
                      >
                        {inference?.regions?.[4]?.name ?? "其他区域"}
                      </div>
                    </div>
                  </div>
                )}

                {/* Summary */}
                {inference?.summary && (
                  <p className="mt-4 text-xs text-[var(--warm-gray)] leading-relaxed italic">
                    "{inference.summary}"
                  </p>
                )}
              </div>
            </div>

            {/* Room list */}
            <div className="lg:col-span-2">
              <div className="card p-6 h-full flex flex-col">
                <h2 className="text-lg font-bold text-[var(--charcoal)] mb-5" style={{ fontFamily: "var(--font-display)" }}>空间明细</h2>
                <div className="flex-1 space-y-2 overflow-y-auto">
                  {roomDisplayInfos.length > 0 ? roomDisplayInfos.map((region) => (
                    <button
                      key={region.id}
                      onClick={() => setSelectedRoomId(region.id)}
                      className={`w-full text-left flex items-center gap-3 p-3 rounded-xl transition-all ${
                        selectedRoomId === region.id
                          ? "bg-[var(--terracotta-pale)] ring-1 ring-[var(--terracotta-light)]"
                          : "bg-[var(--ivory)] hover:bg-[var(--parchment)]"
                      }`}
                    >
                      <div
                        className="w-3 h-3 rounded-full flex-shrink-0"
                        style={{ background: region.color || "var(--mid-gray)" }}
                      />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-[var(--charcoal)]">{region.displayName}</p>
                        <p className="text-[10px] text-[var(--mid-gray)] mt-0.5 truncate">
                          {region.note}
                        </p>
                        <p className="text-[10px] text-[var(--light-gray)] mt-0.5">
                          置信度 {Math.round(region.confidence * 100)}%
                        </p>
                      </div>
                      <div className="text-sm font-semibold text-[var(--charcoal)] flex-shrink-0">{Math.round(region.areaRatio * 100)}%</div>
                    </button>
                  )) : (inference?.regions ?? []).map((region) => (
                    <RoomRegion key={region.name} {...region} />
                  ))}
                  {roomDisplayInfos.length === 0 && (!inference?.regions || inference.regions.length === 0) && (
                    <div className="text-center py-8 text-[var(--mid-gray)] text-xs">
                      <i className="fa-solid fa-map text-2xl mb-2 block text-[var(--light-gray)]" />
                      暂无空间数据
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Renovation advice */}
          <div>
            <div className="flex items-center gap-3 mb-5">
              <div>
                <h2 className="text-lg font-bold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>装修建议</h2>
                <p className="text-xs text-[var(--mid-gray)] mt-0.5">基于 {buildRequirementBasis(requirements)} 生成的专业改造方案</p>
              </div>
              <div className="ml-auto flex items-center gap-2 text-xs text-[var(--mid-gray)]">
                <i className="fa-solid fa-robot text-[var(--sage)]" />
                AI 分析 · {agent?.advice.length ?? 0} 条建议
              </div>
            </div>

            {selectedRoom?.advice && (
              <div className="mb-4 rounded-2xl border border-[var(--terracotta-light)] bg-[var(--terracotta-pale)] p-4">
                <div className="flex items-center gap-2 mb-1">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ background: selectedRoom.color }} />
                  <h3 className="text-sm font-semibold text-[var(--charcoal)]">{selectedRoom.displayName}</h3>
                  <span className="text-xs text-[var(--terracotta)]">当前高亮空间</span>
                </div>
                <p className="text-xs text-[var(--warm-gray)] leading-relaxed">{selectedRoom.advice.description}</p>
              </div>
            )}

            {(agent?.advice ?? []).length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {agent!.advice.map((item, i) => (
                  <AdviceCard
                    key={`${item.title}-${i}`}
                    {...item}
                    index={i}
                    active={Boolean(selectedAdviceTitle && item.title.includes(selectedAdviceTitle))}
                    onClick={() => handleAdviceClick(item.title)}
                  />
                ))}
              </div>
            ) : agentStatus === "analyzing" ? (
              <div className="card p-10 text-center">
                <div className="w-8 h-8 border-2 border-[var(--terracotta)] border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                <p className="text-[var(--warm-gray)]">分割结果已生成，Agent 正在生成装修建议...</p>
              </div>
            ) : agentStatus === "error" ? (
              <div className="card p-10 text-center">
                <i className="fa-solid fa-triangle-exclamation text-3xl text-[var(--terracotta)] mb-3" />
                <p className="text-[var(--warm-gray)]">图像分割已完成，Agent 建议暂时不可用。</p>
                {agentError && <p className="text-xs text-[var(--mid-gray)] mt-2 break-words">{agentError}</p>}
              </div>
            ) : (
              <div className="card p-10 text-center">
                <i className="fa-solid fa-lightbulb text-3xl text-[var(--light-gray)] mb-3" />
                <p className="text-[var(--mid-gray)]">完成训练后可获得装修建议</p>
              </div>
            )}
          </div>

          {/* Space optimization */}
          {agent && (
            <div className="card p-8">
              <h2 className="text-lg font-bold text-[var(--charcoal)] mb-2" style={{ fontFamily: "var(--font-display)" }}>空间优化参考</h2>
              <p className="text-sm text-[var(--mid-gray)] mb-6">基于户型结构的空间规划建议</p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <div className="w-10 h-10 rounded-xl bg-[var(--terracotta-pale)] flex items-center justify-center mb-3">
                    <i className="fa-solid fa-route text-[var(--terracotta)]" />
                  </div>
                  <h4 className="font-semibold text-[var(--charcoal)] mb-1">动线规划</h4>
                  <p className="text-xs text-[var(--warm-gray)] leading-relaxed">{agent.circulation || "建议形成从玄关到客餐厅再到厨房的连续动线，减少交叉干扰"}</p>
                </div>
                <div>
                  <div className="w-10 h-10 rounded-xl bg-[var(--sage-pale)] flex items-center justify-center mb-3">
                    <i className="fa-solid fa-boxes-stacked text-[var(--sage)]" />
                  </div>
                  <h4 className="font-semibold text-[var(--charcoal)] mb-1">收纳空间</h4>
                  <p className="text-xs text-[var(--warm-gray)] leading-relaxed">建议在玄关、主卧和过道增加通顶柜体，补足日常收纳需求</p>
                </div>
                <div>
                  <div className="w-10 h-10 rounded-xl bg-[var(--parchment)] flex items-center justify-center mb-3">
                    <i className="fa-solid fa-sun text-[var(--terracotta-light)]" />
                  </div>
                  <h4 className="font-semibold text-[var(--charcoal)] mb-1">采光优化</h4>
                  <p className="text-xs text-[var(--warm-gray)] leading-relaxed">保持客厅开敞，减小视觉遮挡，最大化自然采光</p>
                </div>
              </div>

              {agent.spaces && agent.spaces.length > 0 && (
                <div className="mt-6 pt-6 border-t border-[var(--border)]">
                  <p className="text-xs text-[var(--mid-gray)] mb-3">识别出的空间类型</p>
                  <div className="flex flex-wrap gap-2">
                    {agent.spaces.map((space) => (
                      <span key={space} className="px-3 py-1.5 rounded-full text-xs bg-[var(--ivory)] border border-[var(--border)] text-[var(--warm-gray)]">{space}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          <ChatPanel
            disabled={!sessionId || agentStatus === "analyzing"}
            messages={chatMessages}
            input={chatInput}
            streaming={chatStreaming}
            onInput={setChatInput}
            onSend={handleSendChat}
          />

          {/* CTA to upload new */}
          <div className="text-center">
            <button
              onClick={() => {
                setSearchParams({ mode: "upload" });
                resetToUploadState();
              }}
              className="inline-flex items-center gap-2 px-6 py-3 border border-[var(--border)] hover:border-[var(--terracotta)] text-[var(--warm-gray)] hover:text-[var(--terracotta)] rounded-xl text-sm font-medium transition-all"
            >
              <i className="fa-solid fa-repeat" />
              分析另一张户型图
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
