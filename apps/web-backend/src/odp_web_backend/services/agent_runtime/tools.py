"""Agent 工具集 —— 可被 LLM 调用的结构化函数。"""

import math
import json


def get_room_detail(room: dict) -> dict:
    """根据 YOLO 数据计算单个房间的精确指标。"""
    bbox = room.get("bbox", {})
    w = bbox.get("x2", 0) - bbox.get("x1", 0)
    h = bbox.get("y2", 0) - bbox.get("y1", 0)
    area_px = w * h
    area_ratio = room.get("area_ratio", 0)
    polygon = room.get("polygon", [])

    # 用 polygon 估算实际形状复杂度
    if polygon and len(polygon) > 0:
        pts = polygon[0] if isinstance(polygon[0], list) else polygon
        perimeter = sum(
            math.hypot(pts[i][0] - pts[(i + 1) % len(pts)][0],
                       pts[i][1] - pts[(i + 1) % len(pts)][1])
            for i in range(len(pts))
        ) if len(pts) >= 3 else 0
        circularity = (4 * math.pi * area_px) / (perimeter * perimeter) if perimeter > 0 else 0
    else:
        perimeter = 0
        circularity = 0

    # 形状评估
    if circularity > 0.85:
        shape = "接近正方形"
    elif circularity > 0.7:
        shape = "矩形"
    elif circularity > 0.5:
        shape = "L 形或不规则"
    else:
        shape = "复杂多边形"

    # 面积等级（相对于整图）
    if area_ratio > 0.25:
        size_label = "超大"
    elif area_ratio > 0.12:
        size_label = "大"
    elif area_ratio > 0.05:
        size_label = "中等"
    elif area_ratio > 0.02:
        size_label = "小"
    else:
        size_label = "极小"

    return {
        "room_id": room["id"],
        "bbox_width_px": round(w, 1),
        "bbox_height_px": round(h, 1),
        "area_pixels": round(area_px, 0),
        "area_ratio": round(area_ratio, 4),
        "size_label": size_label,
        "shape": shape,
        "perimeter_px": round(perimeter, 1),
        "confidence": room.get("confidence", 0),
    }


def analyze_adjacency(rooms: list) -> list:
    """分析房间邻接关系（基于 bbox 重叠）。"""
    pairs = []
    for i, r1 in enumerate(rooms):
        for j, r2 in enumerate(rooms):
            if i >= j:
                continue
            b1 = r1["bbox"]
            b2 = r2["bbox"]

            # 计算重叠
            ox = max(0, min(b1["x2"], b2["x2"]) - max(b1["x1"], b2["x1"]))
            oy = max(0, min(b1["y2"], b2["y2"]) - max(b1["y1"], b2["y1"]))

            # 计算间距
            gap_x = max(0, max(b1["x1"], b2["x1"]) - min(b1["x2"], b2["x2"]))
            gap_y = max(0, max(b1["y1"], b2["y1"]) - min(b1["y2"], b2["y2"]))
            gap = max(gap_x, gap_y)

            if ox > 0 and oy > 0:
                relation = "相邻（共享墙面）"
            elif gap < 30:
                relation = "紧邻（间距 < 30px）"
            elif gap < 80:
                relation = "靠近"
            else:
                relation = "远离"

            pairs.append({
                "room_a": r1["id"],
                "room_b": r2["id"],
                "relation": relation,
                "gap_px": round(gap, 1),
            })
    return pairs


def estimate_natural_light(room: dict, image_w: int, image_h: int) -> dict:
    """估算房间采光条件（基于在图中的位置）。"""
    bbox = room["bbox"]
    cx = (bbox["x1"] + bbox["x2"]) / 2
    cy = (bbox["y1"] + bbox["y2"]) / 2

    # 距离边缘越近 → 越可能有窗户 → 采光越好
    edge_dist = min(cx / image_w, 1 - cx / image_w, cy / image_h, 1 - cy / image_h)
    edge_dist = max(edge_dist, 0.01)

    if edge_dist < 0.1:
        score = 85 + (0.1 - edge_dist) * 150
    elif edge_dist < 0.25:
        score = 65 + (0.25 - edge_dist) * 130
    elif edge_dist < 0.4:
        score = 40 + (0.4 - edge_dist) * 100
    else:
        score = 25

    score = min(100, max(10, round(score)))

    if score >= 80:
        level = "优秀"
    elif score >= 60:
        level = "良好"
    elif score >= 40:
        level = "一般"
    else:
        level = "较差"

    return {
        "room_id": room["id"],
        "lighting_score": score,
        "lighting_level": level,
        "position_x_ratio": round(cx / image_w, 3),
        "position_y_ratio": round(cy / image_h, 3),
        "near_exterior": edge_dist < 0.1,
    }


def estimate_renovation_budget(rooms: list, house_type: str = "") -> dict:
    """估算装修预算范围（基于房间数量和面积）。"""
    n_rooms = len(rooms)
    total_area_ratio = sum(r.get("area_ratio", 0) for r in rooms)

    # 假设户型图对应实际面积 ~100m²
    est_total_m2 = 100
    room_m2 = total_area_ratio * est_total_m2

    # 参考价：简装 800/m², 精装 1500/m², 豪装 3000+/m²
    budget_simple = round(room_m2 * 800)
    budget_standard = round(room_m2 * 1500)
    budget_premium = round(room_m2 * 3500)

    return {
        "estimated_area_m2": round(room_m2, 0),
        "room_count": n_rooms,
        "budget_simple_yuan": budget_simple,
        "budget_standard_yuan": budget_standard,
        "budget_premium_yuan": budget_premium,
        "unit": "元（人民币）",
        "disclaimer": "基于户型图面积估算，实际费用以装修公司报价为准",
    }


# ── OpenAI Function Calling 工具定义 ──

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_room_detail",
            "description": "获取某个房间的精确尺寸、形状、面积等物理指标。用于需要精确数据时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "room_index": {
                        "type": "integer",
                        "description": "房间在列表中的索引（0-based）",
                    }
                },
                "required": ["room_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_adjacency",
            "description": "分析所有房间之间的邻接关系，返回每对房间是相邻、紧邻还是远离。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_natural_light",
            "description": "估算某个房间的采光评分（0-100），基于房间在户型图中的位置。",
            "parameters": {
                "type": "object",
                "properties": {
                    "room_index": {
                        "type": "integer",
                        "description": "房间在列表中的索引（0-based）",
                    }
                },
                "required": ["room_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_renovation_budget",
            "description": "根据房间数量和面积估算装修预算范围（简装/精装/豪装三档）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "house_type": {
                        "type": "string",
                        "description": "LLM 已判断的户型类型",
                    }
                },
                "required": [],
            },
        },
    },
]


def execute_tool(name: str, args: dict, context: dict) -> str:
    """执行工具并返回 JSON 字符串结果。"""
    rooms = context.get("rooms", [])
    image_size = context.get("image_size", {})
    house_type = context.get("house_type", "")

    if name == "get_room_detail":
        idx = args.get("room_index", 0)
        if 0 <= idx < len(rooms):
            result = get_room_detail(rooms[idx])
        else:
            result = {"error": f"无效的 room_index: {idx}，共 {len(rooms)} 个房间"}
        return json.dumps(result, ensure_ascii=False)

    elif name == "analyze_adjacency":
        result = analyze_adjacency(rooms)
        return json.dumps(result, ensure_ascii=False)

    elif name == "estimate_natural_light":
        idx = args.get("room_index", 0)
        if 0 <= idx < len(rooms):
            w = image_size.get("width", 640)
            h = image_size.get("height", 640)
            result = estimate_natural_light(rooms[idx], w, h)
        else:
            result = {"error": f"无效的 room_index: {idx}"}
        return json.dumps(result, ensure_ascii=False)

    elif name == "estimate_renovation_budget":
        ht = args.get("house_type", house_type)
        result = estimate_renovation_budget(rooms, ht)
        return json.dumps(result, ensure_ascii=False)

    else:
        return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
