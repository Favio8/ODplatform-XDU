"""AI Agent —— 工具调用 + 多步推理 + 交互对话。"""

import json
from openai import OpenAI
from tools import TOOL_DEFINITIONS, execute_tool

SYSTEM_PROMPT = """你是一个资深室内设计师和户型分析专家，配备了一套专业的户型分析工具。

## 你的能力

1. **工具使用**：你可以调用工具获取房间精确尺寸、邻接关系、采光评分、装修预算等数据。在分析前应先调用相关工具获取数据，而不是凭空猜测。

2. **多步推理**：遵循[观察->工具调用->分析->结论]的流程。每次分析一个维度，层层递进：
   - 第一步：调用 get_room_detail 获取各房间精确数据
   - 第二步：调用 analyze_adjacency 分析空间关系
   - 第三步：调用 estimate_natural_light 评估采光
   - 第四步：综合以上数据，给出最终分析和建议
   - 最后：调用 estimate_renovation_budget 估算预算

3. **交互对话**：如果用户追问某个细节（如"主卧能改大吗"），你应该基于已有的户型数据给出针对性回答。

4. **记忆上下文**：如果你已经分析过一次户型，后续对话应该引用之前的分析结果，而不是重新分析。

## 输出规范

- 最终分析结果必须包含完整的结构化 JSON，格式与之前分析一致
- 工具调用获取的数据要转化为用户能理解的自然语言
- 每个房间的建议要基于工具返回的真实数据，不要编造
- 评分要基于工具返回的客观指标

## 户型分析时的 JSON 格式

{
  "rating": "S/A+/A/A-/B+/B/C",
  "house_type": "户型类型",
  "overall_assessment": "整体评价",
  "pros": ["优势1", "优势2", "优势3"],
  "cons": ["劣势1", "劣势2"],
  "scores": {
    "space_utilization": 85,
    "lighting": 78,
    "traffic_flow": 72,
    "storage_potential": 65
  },
  "core_issues": ["问题1", "问题2"],
  "rooms": [
    {
      "room_label": "Room 1",
      "room_type": "主卧",
      "analysis": "基于数据的具体分析...",
      "suggestions": {
        "furniture": "...",
        "color": "...",
        "storage": "...",
        "lighting": "..."
      }
    }
  ],
  "overall_suggestions": "整体优化建议",
  "renovation_budget": {
    "simple": 80000,
    "standard": 150000,
    "premium": 350000,
    "unit": "元"
  }
}"""


class Agent:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def analyze(self, image_base64: str, rooms: list, image_size: dict,
                session_id: str = None, memory=None) -> dict:
        """两阶段 Agent 分析：
        阶段1: LLM 调用工具收集数据（短上下文）
        阶段2: 后端汇总数据，独立调用 LLM 生成 JSON（干净上下文）
        """

        room_context = "\n".join(
            f"Room {r['id']}: 面积占比 {r.get('area_ratio', '?')}, "
            f"置信度 {r.get('confidence', '?')}"
            for r in rooms
        )

        tool_context = {
            "rooms": rooms,
            "image_size": image_size,
            "house_type": "",
        }

        # ========== 阶段1: 工具调用收集数据 ==========
        messages = [
            {"role": "system", "content": "你是一个室内设计分析助手。请调用工具收集户型数据，每次尽量多调用。收集完成后回复 DONE。"},
            {"role": "user", "content": f"请收集以下房间的数据：\n{room_context}\n\n对每个房间调用 get_room_detail 和 estimate_natural_light，然后调用 analyze_adjacency 和 estimate_renovation_budget。"},
        ]

        reasoning_steps = []
        all_tool_results = {}

        for iteration in range(1, 8):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                max_completion_tokens=2048,
                temperature=0.3,
            )

            msg = response.choices[0].message

            if not msg.tool_calls:
                reasoning_steps.append({
                    "step": f"phase1-{iteration}",
                    "type": "phase1_done",
                    "content": (msg.content or "工具收集完成")[:100],
                })
                break

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                tool_result = execute_tool(fn_name, fn_args, tool_context)
                all_tool_results[fn_name] = all_tool_results.get(fn_name, [])
                all_tool_results[fn_name].append(tool_result)

                reasoning_steps.append({
                    "step": f"phase1-{iteration}",
                    "type": "tool_call",
                    "tool": fn_name,
                    "arguments": fn_args,
                    "result_preview": tool_result[:200],
                })

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tc.id, "type": "function",
                        "function": {"name": fn_name, "arguments": tc.function.arguments},
                    }],
                })
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": tool_result,
                })

        # ========== 阶段2: 独立生成 JSON 分析 ==========
        # 精简工具结果，避免 prompt 过长
        compact = {}
        for k, v in all_tool_results.items():
            # 每条结果只保留关键字段
            trimmed = []
            for item in v:
                try:
                    obj = json.loads(item) if isinstance(item, str) else item
                    if k == "get_room_detail":
                        trimmed.append({rk: obj[rk] for rk in ["room_id", "size_label", "area_ratio", "shape", "area_pixels"] if rk in obj})
                    elif k == "estimate_natural_light":
                        trimmed.append({rk: obj[rk] for rk in ["room_id", "lighting_score", "lighting_level", "near_exterior"] if rk in obj})
                    elif k == "analyze_adjacency":
                        trimmed = obj[:20]  # 最多 20 对关系
                    else:
                        trimmed.append(obj)
                except json.JSONDecodeError:
                    trimmed.append(str(item)[:200])
            compact[k] = trimmed

        data_summary = json.dumps(compact, ensure_ascii=False, indent=2)
        if len(data_summary) > 3000:
            data_summary = data_summary[:3000] + "\n...[truncated]"

        json_template = """{
  "rating": "S/A+/A/A-/B+/B/C",
  "house_type": "户型类型",
  "overall_assessment": "整体评价，100字内",
  "pros": ["优势1", "优势2", "优势3"],
  "cons": ["劣势1", "劣势2"],
  "scores": {
    "space_utilization": 85,
    "lighting": 78,
    "traffic_flow": 72,
    "storage_potential": 65
  },
  "core_issues": ["问题1", "问题2"],
  "rooms": [
    {
      "room_label": "Room 1",
      "room_type": "主卧",
      "analysis": "...",
      "suggestions": {
        "furniture": "...",
        "color": "...",
        "storage": "...",
        "lighting": "..."
      }
    }
  ],
  "overall_suggestions": "整体优化建议",
  "renovation_budget": {"simple": 80000, "standard": 150000, "premium": 350000, "unit": "元"}
}"""

        phase2_prompt = (
            "以下是通过工具获取的户型精确数据：\n\n"
            f"{data_summary}\n\n"
            f"YOLO 检测到 {len(rooms)} 个房间，面积占比和置信度如下：\n"
            f"{room_context}\n\n"
            "请基于以上真实数据，结合你看到的户型标注图，输出完整的 JSON 分析结果。\n\n"
            f"{json_template}\n\n"
            "评分范围 0-100。只输出 JSON，不要任何其他内容。"
        )

        phase2_msg = [
            {"role": "system", "content": "你是一个资深室内设计师。你已获取了精确的户型数据。请基于数据输出 JSON 分析结果。只输出 JSON。"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": phase2_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            },
        ]

        final_response = self.client.chat.completions.create(
            model=self.model,
            messages=phase2_msg,
            max_completion_tokens=4096,
            temperature=0.5,
        )

        final_content = final_response.choices[0].message.content or ""
        parsed = self._parse_response(final_content)

        reasoning_steps.append({
            "step": "phase2",
            "type": "final",
            "content": final_content[:300],
        })

        if memory and session_id:
            for step in reasoning_steps:
                memory.add_reasoning(session_id, step)

        return {
            "analysis": parsed,
            "reasoning_steps": reasoning_steps,
        }

    def chat(self, session_id: str, user_message: str, memory) -> dict:
        """交互式对话，基于已有的分析上下文。"""
        sess = memory.get(session_id)
        if not sess:
            return {"reply": "会话已过期，请重新上传户型图开始分析。", "reasoning_steps": []}

        # 构建对话上下文
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 注入之前的分析摘要
        if sess["analyses"]:
            last_analysis = sess["analyses"][-1]
            context_msg = (
                "以下是之前对用户户型图的分析摘要：\n"
                f"户型类型：{last_analysis.get('house_type', '未知')}\n"
                f"评级：{last_analysis.get('rating', 'N/A')}\n"
                f"检测到 {len(sess['yolo_rooms'])} 个房间\n"
                "用户现在想进一步讨论这个户型。请基于之前的数据给出针对性回答。"
            )
            messages.append({"role": "user", "content": context_msg})
            messages.append({
                "role": "assistant",
                "content": "我已了解这个户型。请问您想进一步了解什么？",
            })

        # 加入历史对话
        messages.extend(sess["messages"][-10:])  # 最多保留 10 条历史

        # 加入用户新消息
        messages.append({"role": "user", "content": user_message})

        reasoning_steps = []
        max_iterations = 8
        iteration = 0

        tool_context = {
            "rooms": sess["yolo_rooms"],
            "image_size": sess["image_size"],
            "house_type": (
                sess["analyses"][-1].get("house_type", "")
                if sess["analyses"] else ""
            ),
        }

        while iteration < max_iterations:
            iteration += 1

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto" if iteration < 5 else "none",
                max_completion_tokens=2048,
                temperature=0.7,
            )

            msg = response.choices[0].message

            if not msg.tool_calls:
                reasoning_steps.append({
                    "step": iteration,
                    "type": "final",
                    "content": msg.content[:200] if msg.content else "",
                })
                memory.add_message(session_id, "user", user_message)
                memory.add_message(session_id, "assistant", msg.content or "")
                for step in reasoning_steps:
                    memory.add_reasoning(session_id, step)
                return {
                    "reply": msg.content or "",
                    "reasoning_steps": reasoning_steps,
                }

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                tool_result = execute_tool(fn_name, fn_args, tool_context)
                reasoning_steps.append({
                    "step": iteration,
                    "type": "tool_call",
                    "tool": fn_name,
                    "arguments": fn_args,
                    "result_preview": tool_result[:200],
                })

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": tc.function.arguments,
                        },
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

        return {"reply": "抱歉，推理超时。请重新描述您的问题。", "reasoning_steps": reasoning_steps}

    def _parse_response(self, content: str) -> dict:
        text = content.strip()

        # 剥离 markdown 代码块
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        # 找第一个 { 到最后一个 } 之间的 JSON
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {
                "rating": "N/A",
                "house_type": "未知",
                "overall_assessment": content,
                "pros": [], "cons": [],
                "scores": {}, "core_issues": [],
                "rooms": [], "overall_suggestions": "",
            }
