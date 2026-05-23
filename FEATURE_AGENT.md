# feature/agent 分支变更摘要

## 概述

基于 `main` 分支，新增 **AI 户型分析 Agent** 模块。用户上传户型图 → YOLO 分割房间 → 多模态 LLM（Gemini）看图分析 → 给出装修建议。

## 文件变更清单

### 新增文件

```
apps/web-backend/
├── .env                      # API 配置（已 gitignore，不提交）
├── requirements.txt           # Python 依赖
├── main.py                    # FastAPI 入口（端口 7860）
├── model_handler.py           # YOLO 模型加载 + 推理 + 可视化
└── agent.py                   # LLM Agent（Gemini 多模态分析）

apps/web-frontend/             # React + Vite + Tailwind v4 项目
├── index.html
├── package.json
├── vite.config.js             # API proxy → :7860
└── src/
    ├── main.jsx
    ├── App.jsx                # 左右双栏布局
    ├── index.css              # Tailwind + 背景渐变
    ├── hooks/
    │   └── useAnalysis.js     # 状态管理 hook
    └── components/
        ├── Header.jsx          # 顶栏（毛玻璃）
        ├── UploadZone.jsx      # 拖拽上传（玻璃态）
        ├── FloorPlanViewer.jsx # 左侧户型图 + 工具栏 + 全屏预览
        ├── AnalysisPanel.jsx   # 右侧分析容器（Hero + 评分 + 房间）
        ├── ScoreRing.jsx       # SVG 环形评分动画
        ├── RoomAccordion.jsx   # 折叠房间卡片
        ├── SuggestionGrid.jsx  # 2×2 建议网格
        ├── LoadingState.jsx    # 6 步分析过程动画
        └── EmptyState.jsx      # 空状态 / 错误状态
```

### 修改文件

| 文件 | 变更说明 |
|------|----------|
| `.gitignore` | 新增 `.env` 忽略规则 |
| `apps/web-frontend/README.md` | 更新为 React + Vite 说明 |
| `apps/platform/pyproject.toml` | `requires-python` 从 `>=3.11,<3.12` 改为 `>=3.11` |

### 未修改的文件

`apps/platform/src/odp_platform/` 下所有核心模块、`apps/desktop/`、`apps/web-backend/README.md`（原占位文件保留）、`pyproject.toml`（根目录）均未改动，确保与组长 `main` 分支无冲突。

## 技术栈

| 层 | 选型 | 说明 |
|------|------|------|
| 后端框架 | FastAPI | 轻量、异步、自动 OpenAPI |
| 模型推理 | YOLO (ultralytics) | 加载 `best.pt` 做房间实例分割 |
| AI 分析 | Gemini 3 Flash (多模态) | 看图识别房间类型 + 生成装修建议 |
| LLM SDK | openai >= 1.50 | OpenAI 兼容协议，通过中转 API 调用 |
| 前端框架 | React 19 + Vite 8 | HMR 开发体验 |
| 样式 | Tailwind CSS v4 | 原子化 CSS |
| 动画 | Framer Motion | stagger、spring、渐变入场 |
| 图标 | Lucide React | 轻量开源图标库 |

## API 接口

### POST /api/analyze

- Content-Type: multipart/form-data
- 参数: `file` (image/jpeg, image/png)

**返回结构：**

```json
{
  "image_size": { "width": 640, "height": 640 },
  "visualization": "base64 JPEG",
  "yolo_rooms": [
    { "id": 1, "bbox": {...}, "polygon": [...], "area_ratio": 0.15, "confidence": 0.98 }
  ],
  "analysis": {
    "rating": "B+",
    "house_type": "两室一厅",
    "overall_assessment": "...",
    "pros": ["优势1", "优势2", "优势3"],
    "cons": ["劣势1", "劣势2"],
    "scores": {
      "space_utilization": 82,
      "lighting": 85,
      "traffic_flow": 75,
      "storage_potential": 70
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
    "overall_suggestions": "..."
  }
}
```

## 启动方式

### 后端

```bash
cd apps/web-backend
cp .env.example .env   # 填入 API Key
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 7860
```

### 前端（开发）

```bash
cd apps/web-frontend
npm install
npm run dev             # Vite 开发服务器 :5173，API 自动代理到 :7860
```

### 环境变量（.env）

```
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.ikuncode.cc/v1
LLM_MODEL=gemini-3-flash-preview
MODEL_PATH=path/to/best.pt
```

## 合并注意事项

1. `.env` 已加入 `.gitignore`，API key 不会进入仓库
2. `apps/web-frontend/` 是 Vite 项目，需要 `npm install` 后开发
3. `best.pt`（模型权重 23MB）不在本仓库中，需单独提供
4. 本分支仅新增文件，未修改 `main` 分支已有模块，合并无冲突
