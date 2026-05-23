import json

from openai import OpenAI

SYSTEM_PROMPT = """你是一个资深室内设计师和户型分析专家。

用户会给你一张经过AI分割标注的户型平面图。图中用不同颜色标记了各个房间区域，并用 "Room 1", "Room 2" 等文字标签标注在对应区域上。

请仔细观察图片，为每个被标记为 Room N 的房间提供专业分析，并给出整体评价。

必须严格按照以下JSON格式返回，不要包含其他文字：

{
  "rating": "A-",
  "house_type": "两室一厅",
  "overall_assessment": "整体户型评价，100字以内",
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
      "room_type": "客厅",
      "analysis": "该区域位于户型中心，面积约XX平米...",
      "suggestions": {
        "furniture": "家具布局建议",
        "color": "配色建议",
        "storage": "收纳建议",
        "lighting": "灯光建议"
      }
    }
  ],
  "overall_suggestions": "整体优化建议，150字以内"
}

注意：
- rating 用 S/A+/A/A-/B+/B/C 评级体系，S为顶级户型
- pros 列出 3 个户型优势，cons 列出 2 个劣势
- scores 中每个分数为 0-100 的整数
- core_issues 列出 2-3 个户型核心问题
- room_type 用中文：客厅/主卧/次卧/厨房/卫生间/阳台/走廊/储物间/书房/餐厅
- 每个建议 60 字以内，精炼专业"""


class Agent:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def analyze(self, image_base64: str, rooms: list) -> dict:
        room_context = "\n".join(
            f"Room {r['id']}: 面积占比 {r.get('area_ratio', '?')}, "
            f"置信度 {r.get('confidence', '?')}"
            for r in rooms
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "请分析这张户型图。AI已检测到以下房间区域：\n"
                                f"{room_context}\n\n"
                                "请为每个标记区域识别房间类型、给出装修建议，并提供整体评分。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                },
            ],
            max_tokens=4096,
            temperature=0.7,
        )

        content = response.choices[0].message.content
        return self._parse_response(content)

    def _parse_response(self, content: str) -> dict:
        text = content.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return {
                "rating": "N/A",
                "house_type": "未知",
                "overall_assessment": content,
                "pros": [],
                "cons": [],
                "scores": {},
                "core_issues": [],
                "rooms": [],
                "overall_suggestions": "",
            }
