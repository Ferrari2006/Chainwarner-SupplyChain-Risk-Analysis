# ChainWarner — 开源供应链风险智能分析平台

## 项目简介
ChainWarner 是一个面向开源供应链的风险智能分析平台，聚焦依赖拓扑、结构洞约束、生态声誉（OpenRank）、活动度（Activity）、CVE 漏洞等维度进行综合评估，可实时构建依赖图并输出关键风险指标与传播路径。

- 参赛作品：OpenRank（ECNU 团队）
- 许可证：MIT License（详见仓库许可证文件）

## 功能亮点
- 实时依赖图构建：基于仓库配置文件（pyproject.toml、requirements.txt、package.json 等）解析主要依赖
- 图算法分析：结构洞约束、有效规模、介数中心性、PageRank、K-Core 等（EasyGraph / NetworkX）
- 生态声誉与活动度：对接 OpenDigger 的 OpenRank 与 Activity 实时数据
- 风险传播仿真：从依赖子图向主项目的风险传播（迭代衰减）
- 并发与缓存：HTTP 流式抓取 + 本地缓存，加速重复分析与页面交互

## 技术栈
- 后端：FastAPI、Uvicorn、Pydantic
- 图算法：EasyGraph / NetworkX
- HTTP：httpx（流式获取 + 连接复用）
- 前端模板：Jinja2 + 原生 HTML/CSS/JS（AntV G6、ECharts）

## 项目结构
```
ChainWarner/
├─ backend/
│  ├─ app/
│  │  ├─ api/
│  │  │  └─ endpoints.py          # 主路由与风险分析接口
│  │  ├─ core/
│  │  │  └─ stream_processor.py   # OpenDigger 流式抓取与缓存
│  │  ├─ db/
│  │  │  └─ database.py           # 内存/SQLite 存储抽象
│  │  ├─ engines/
│  │  │  ├─ graph_engine.py       # 图构建与结构洞/中心性等算法
│  │  │  ├─ ml_engine.py          # 机器学习占位（可拓展）
│  │  │  ├─ nlp_engine.py         # NLP 占位（可拓展）
│  │  │  └─ agent_engine.py       # 智能问答占位（可对接 LLM）
│  │  ├─ models/
│  │  │  └─ graph.py              # Pydantic 数据模型
│  │  ├─ templates/
│  │  │  └─ index.html            # 前端页面（图谱与榜单）
│  │  ├─ static/                  # 静态资源
│  │  └─ main.py                  # FastAPI 入口（模板挂载、路由注册）
│  ├─ requirements.txt            # 后端依赖
│  ├─ runtime.txt / Procfile / vercel.json  # 部署配置（可选）
│  └─ test_api.py                 # 基本接口测试脚本
└─ render.yaml                    # Render 部署配置（可选）
```

## 快速开始（本地运行）
1. 进入后端目录（Windows PowerShell）
   ```powershell
   cd D:\PROJECT\DASEPROJECT\ChainWarner\backend
   ```
2. 创建并激活虚拟环境（首次）
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. 安装依赖（首次或更新后）
   ```powershell
   python -m pip install -U pip
   pip install -r requirements.txt
   ```
4. 启动服务
   ```powershell
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
5. 访问页面
   - 前端页面：`http://localhost:8000/`
   - 健康检查：`http://localhost:8000/health`

可选：更快的计算（牺牲部分图算法）
```powershell
$env:LITE_MODE="true"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 主要接口
- GET `/` 前端仪表盘页面
- GET `/health` 健康检查
- GET `/api/v1/graph/{owner}/{repo}` 构建依赖图并计算风险指标
  - 查询参数：
    - `fresh`（bool）：是否强制重新计算（默认 false，命中缓存更快）
    - `max_deps`（int）：限制依赖数量（默认 15，越大越慢）
    - `expand`（bool）：递归扩展下游依赖（PyPI），更丰富但更慢
- GET `/api/v1/leaderboard` 榜单（风险/安全）
- GET `/api/v1/compare/{owner1}/{repo1}/{owner2}/{repo2}` 双项目对比（简化版）

## 运行机制简述
- 依赖解析：按语言生态识别配置文件，过滤测试/类型等“琐碎”依赖
- 数据拉取：通过 OpenDigger 获取 OpenRank/Activity，PyPI 获取元信息，OSV 获取 CVE
- 图计算：EasyGraph/NetworkX 计算结构洞约束、有效规模、介数、PageRank 等
- 分数融合：声誉优先（OpenRank 70% + 活动度 30%），并对 CVE 风险做声誉抑制
- 风险传播：从子依赖向主项目做迭代传播（带衰减），避免“一片红”的极端
- 并发与缓存：httpx 连接复用 + 本地缓存，降低重复分析的等待时间

## 部署与环境
- 本地：Uvicorn 开发模式（`--reload`）
- Render / Vercel：支持基础部署配置（`render.yaml`、`vercel.json`）
- 存储：默认内存（可设 `CHAINWARNER_STORAGE=sqlite` 切至 SQLite）
