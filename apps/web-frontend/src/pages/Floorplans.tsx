import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { deleteFloorplan, fetchFloorplans } from "../lib/api";
import type { FloorplanRecord } from "../types";

function statusLabel(status: string) {
  if (status === "done") return "已完成";
  if (status === "error") return "Agent异常";
  return "分析中";
}

function statusClass(status: string) {
  if (status === "done") return "bg-[var(--sage-pale)] text-[var(--sage)] border-[var(--sage-light)]";
  if (status === "error") return "bg-red-50 text-red-500 border-red-100";
  return "bg-[var(--parchment)] text-[var(--warm-gray)] border-[var(--border)]";
}

function ratingOf(record: FloorplanRecord) {
  const rating = record.analysis?.rating;
  return typeof rating === "string" && rating ? rating : "未评级";
}

function FloorplanCard({
  record,
  onDelete,
}: {
  record: FloorplanRecord;
  onDelete: (id: string) => void;
}) {
  return (
    <div className="card overflow-hidden flex flex-col hover:shadow-[var(--shadow-md)] transition-all animate-fade-up">
      <div className="relative bg-[var(--parchment)] h-48 overflow-hidden">
        <img
          src={`data:image/jpeg;base64,${record.visualization}`}
          alt={record.filename}
          className="w-full h-full object-contain"
        />
        <span className={`absolute top-3 right-3 px-2.5 py-1 rounded-lg text-xs font-medium border ${statusClass(record.agent_status)}`}>
          {statusLabel(record.agent_status)}
        </span>
      </div>

      <div className="p-5 flex flex-col gap-4 flex-1">
        <div className="min-w-0">
          <p className="font-semibold text-[var(--charcoal)] truncate" title={record.filename}>
            {record.filename}
          </p>
          <p className="text-xs text-[var(--mid-gray)] mt-1">
            {new Date(record.created_at).toLocaleString()} · {record.room_count} 个空间
          </p>
        </div>

        <div className="grid grid-cols-3 gap-2">
          <div className="text-center p-2 rounded-xl bg-[var(--ivory)]">
            <p className="text-lg font-semibold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>{record.room_count}</p>
            <p className="text-[10px] text-[var(--mid-gray)]">空间</p>
          </div>
          <div className="text-center p-2 rounded-xl bg-[var(--ivory)]">
            <p className="text-lg font-semibold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>{ratingOf(record)}</p>
            <p className="text-[10px] text-[var(--mid-gray)]">评级</p>
          </div>
          <div className="text-center p-2 rounded-xl bg-[var(--ivory)]">
            <p className="text-lg font-semibold text-[var(--charcoal)]" style={{ fontFamily: "var(--font-display)" }}>{record.image_size.width}</p>
            <p className="text-[10px] text-[var(--mid-gray)]">宽度</p>
          </div>
        </div>

        <p className="text-xs text-[var(--warm-gray)] leading-relaxed line-clamp-3 min-h-[3rem]">
          {record.summary || "图像分割结果已保存，等待 Agent 生成装修建议。"}
        </p>

        <div className="flex gap-2 pt-1 border-t border-[var(--border)] mt-auto">
          <Link
            to={`/analysis?recordId=${record.record_id}`}
            className="flex-1 py-1.5 px-3 text-xs text-[var(--mid-gray)] hover:text-[var(--charcoal)] border border-[var(--border)] hover:border-[var(--warm-gray)] rounded-lg transition-all flex items-center justify-center gap-1.5"
          >
            <i className="fa-solid fa-eye text-[10px]" />
            重新分析
          </Link>
          <button
            onClick={() => onDelete(record.record_id)}
            className="flex-1 py-1.5 px-3 text-xs text-red-400 hover:text-red-500 border border-red-100 hover:border-red-200 rounded-lg transition-all flex items-center justify-center gap-1.5"
          >
            <i className="fa-solid fa-trash text-[10px]" />
            删除
          </button>
        </div>
      </div>
    </div>
  );
}

export function Floorplans() {
  const [records, setRecords] = useState<FloorplanRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const items = await fetchFloorplans();
      setRecords(items);
      setLoadError(null);
    } catch {
      setLoadError("无法加载户型图记录");
    }
  }, []);

  useEffect(() => {
    load().finally(() => setLoading(false));
  }, [load]);

  async function handleDelete(recordId: string) {
    await deleteFloorplan(recordId);
    await load();
  }

  if (loading) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center py-20">
        <div className="w-8 h-8 border-2 border-[var(--terracotta)] border-t-transparent rounded-full animate-spin mb-3" />
        <p className="text-[var(--mid-gray)] text-sm">加载中...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen px-6 py-10 max-w-5xl mx-auto">
      <div className="flex items-start justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold text-[var(--charcoal)] mb-2" style={{ fontFamily: "var(--font-display)" }}>我的户型图</h1>
          <p className="text-[var(--mid-gray)] text-sm">查看已上传、分割并由 Agent 分析过的户型图</p>
        </div>
        <Link
          to="/analysis?mode=upload"
          className="flex items-center gap-2 px-4 py-2.5 bg-[var(--terracotta)] hover:bg-[var(--terracotta-light)] text-white rounded-xl text-sm font-medium transition-all flex-shrink-0"
        >
          <i className="fa-solid fa-plus" />
          上传户型图
        </Link>
      </div>

      {loadError ? (
        <div className="card p-16 text-center">
          <i className="fa-solid fa-circle-exclamation text-3xl text-[var(--terracotta)] mb-3" />
          <p className="text-[var(--warm-gray)]">{loadError}</p>
          <button onClick={load} className="mt-2 text-sm text-[var(--terracotta)] underline">重试</button>
        </div>
      ) : records.length === 0 ? (
        <div className="card p-16 text-center animate-fade-up">
          <div className="w-16 h-16 rounded-2xl bg-[var(--parchment)] flex items-center justify-center mx-auto mb-4">
            <i className="fa-solid fa-images text-2xl text-[var(--light-gray)]" />
          </div>
          <h3 className="font-semibold text-[var(--charcoal)] text-lg mb-2" style={{ fontFamily: "var(--font-display)" }}>还没有已分析户型图</h3>
          <p className="text-sm text-[var(--mid-gray)] mb-6 max-w-sm mx-auto">
            上传一张户型图后，系统会保存分割结果和 Agent 装修建议，方便后续回看。
          </p>
          <Link
            to="/analysis?mode=upload"
            className="inline-flex items-center gap-2 px-5 py-2.5 bg-[var(--terracotta)] text-white rounded-xl text-sm font-medium"
          >
            <i className="fa-solid fa-upload" />
            上传第一张户型图
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {records.map((record) => (
            <FloorplanCard key={record.record_id} record={record} onDelete={handleDelete} />
          ))}
        </div>
      )}
    </div>
  );
}
