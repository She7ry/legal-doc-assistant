# Legal Document Assistant

基于 RAG（Retrieval-Augmented Generation）的法律文档分析与法律信息辅助系统。支持合同、政策、租约、隐私协议、校规及合规文档的智能问答、风险审查与冲突检测。

系统以**引用优先（citation-first）**为核心设计原则——所有回答必须基于文档原文，标注来源引用。它帮助用户理解文档内容、识别风险、组织问题清单，为与律师沟通做好准备。本系统不替代律师，不提供最终法律意见、诉讼策略或个案判定。

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| LLM 编排 | LangChain + LangGraph |
| 向量存储 | ChromaDB |
| 检索策略 | Hybrid（Dense + BM25 + RRF 融合 + MMR 去重） |
| 默认 LLM | DeepSeek（Chat）/ DashScope（Embedding） / 任意 OpenAI-compatible |
| Embedding | DashScope text-embedding-v3 |
| 中文分词 | jieba |
| 前端 | Vue 3 + TypeScript + Vite + Element Plus + Pinia |
| 数据持久化 | SQLite（任务/记忆/Matter） |
| 测试 | pytest + pytest-asyncio + coverage |
| 代码规范 | Ruff |

---

## 核心功能

### 文档管理与检索

- 支持上传 PDF、DOCX、TXT、Markdown 格式文档
- 法律章节感知的智能分块（legal-section-aware chunking）
- Hybrid 检索：向量语义搜索 + BM25 关键词匹配 + RRF 融合排序
- 轻量级 lexical rerank + MMR 多样性选择，减少近似重复
- 后台异步文档摄入，支持进度查询与阶段性警告
- 同名重复上传自动创建新版本，旧版本保留但不参与检索

### 智能问答

- 基于检索结果的引用式回答，每条回答标注 `[S1]`、`[S2]` 等来源编号
- 支持多轮对话历史
- Tool Calling 模式：模型自主决定调用 `search_documents` 或 `web_search`
- Answer Guard 机制：二次校验引用有效性、过强法律结论和证据缺失

### Agent 工作流

基于 LangGraph 的六阶段法律审查 Agent：

```
Plan → Execute → Collect Findings → Build Deliverables → Synthesize Report → Finalize
```

- **LLM Planner**：根据用户目标与文档内容自动规划审查步骤
- **并行执行**：支持多步骤并行执行，含指数退避重试
- **ReAct 循环**：执行阶段可启用受控 ReAct 循环，自动补充弱证据
- **Confirmation Gates**：关键事实需用户确认后才写入正式报告
- **Matter 管理**：审查结果持久化为 Matter 记录，含 findings、artifacts、risk matrix
- **SSE 实时推送**：任务进度通过 Server-Sent Events 流式推送

生成的交付物包括：
- 风险矩阵（Risk Matrix）
- 律师问题清单（Lawyer Questions）
- 谈判 Checklist
- 义务日历（Obligation Calendar）
- 正式审查报告（Formal Report）

### 条款审查与冲突检测

- 条款审查：对特定条款类型进行风险等级评估
- 冲突检测：比对合同条款与政策条款，识别冲突与矛盾

### 用户记忆系统

独立于文档 RAG 的记忆系统：

- 存储用户偏好、对话上下文、任务状态和反馈
- 支持语义去重、置信度衰减、过期清理
- LLM 驱动的自动记忆提取
- 对话摘要压缩为 session memory
- 记忆注入 prompt 时作为数据处理，不作为文档证据引用

### 安全与多租户

- API Key 认证（`X-API-Key` 或 `Authorization: Bearer`）
- 多租户隔离（`X-Tenant-Id` 路由到独立存储）
- 可配置 CORS 策略
- 上传大小限制
- Sliding-window API 限流

---

## 项目结构

```text
legal_doc_assistant/
├── api/                          # FastAPI 后端
│   ├── main.py                   # 应用入口
│   ├── dependencies.py           # 单例依赖注入
│   ├── jobs.py                   # 文档摄入任务存储
│   ├── agent_tasks.py            # Agent 任务存储
│   ├── task_queue.py             # 后台任务队列
│   ├── sse.py                    # SSE 事件流
│   ├── routers/
│   │   ├── documents.py          # 文档上传与查询
│   │   ├── chat.py               # 问答与 Tool Calling
│   │   ├── agent.py              # Agent 任务生命周期
│   │   ├── matters.py            # Matter CRUD 与导出
│   │   ├── memories.py           # 用户记忆管理
│   │   ├── review.py             # 条款审查与冲突检测
│   │   └── feedback.py           # 反馈收集
│   ├── middleware/
│   │   └── rate_limit.py         # 限流中间件
│   └── schemas/                  # Pydantic 请求/响应模型
│
├── src/doc_assistant/            # 核心业务逻辑
│   ├── config/settings.py        # 全局配置（环境变量驱动）
│   ├── models/                   # LLM 与 Embedding 适配器
│   ├── ingestion/                # 文档加载、哈希、持久化
│   ├── retrieval/                # Chroma + BM25 混合检索
│   ├── services/
│   │   ├── qa_service.py         # 问答核心逻辑
│   │   ├── tool_calling_service.py  # Tool Calling 编排
│   │   ├── agent_service.py      # Agent 业务逻辑入口
│   │   ├── clause_review.py      # 条款风险评估
│   │   ├── conflict_check.py     # 冲突检测
│   │   ├── answer_guard.py       # 回答质量守卫
│   │   ├── evidence.py           # 证据分析
│   │   ├── review_taxonomy.py    # 审查分类体系
│   │   └── agent/                # Agent 工作流子模块
│   │       ├── planner.py        # LLM 规划器
│   │       ├── executor.py       # 步骤执行器
│   │       ├── workflow.py       # LangGraph 编排入口
│   │       └── schemas.py        # Agent 领域模型
│   ├── graphs/                   # LangGraph 状态图定义
│   ├── memory/                   # 记忆策略、存储、检索
│   ├── matter/                   # Matter 持久化与导出
│   ├── tools/                    # 外部工具（Web Search 等）
│   ├── prompts/                  # 分层 Prompt 模板
│   ├── evaluation/               # RAG 评估指标与 CLI
│   ├── schemas/                  # 共享领域模型
│   └── utils/                    # 工具函数
│
├── frontend/                     # Vue 3 前端
│   ├── src/
│   │   ├── pages/                # 页面组件
│   │   ├── components/           # 通用组件
│   │   ├── api/                  # HTTP 客户端
│   │   ├── stores/               # Pinia 状态管理
│   │   └── layouts/              # 布局组件
│   └── package.json
│
├── data/                         # 运行时数据（不入版本控制）
│   ├── uploads/                  # 上传文件
│   ├── vector_store/             # ChromaDB 持久化
│   └── eval/                     # 评估数据集与报告
│
├── tests/                        # 单元/集成测试
├── pyproject.toml                # Python 项目配置
└── .env.example                  # 环境变量模板
```

---

## 快速开始

### 环境要求

- Python 3.10 ~ 3.12
- Node.js 18+（前端开发）

### 后端安装

```powershell
cd E:\project\legal_doc_assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .env.example .env
```

编辑 `.env`，配置 LLM 和 Embedding 的 API Key。

### 前端安装

```powershell
cd frontend
npm.cmd install
```

> 如果 PowerShell 执行策略阻止了 `npm`，使用 `npm.cmd` 代替。

### 启动服务

```powershell
# 启动后端 API（默认 http://localhost:8000）
uvicorn api.main:app --reload

# 启动前端开发服务器（默认 http://127.0.0.1:5173）
cd frontend
npm.cmd run dev
```

- 后端 API 文档：http://localhost:8000/docs
- 前端界面：http://127.0.0.1:5173

---

## 配置说明

所有配置通过环境变量管理，写在 `.env` 文件中。

### LLM 配置

Chat 默认使用 DeepSeek，Embedding 默认使用阿里云 DashScope：

```env
# Chat: DeepSeek
DOC_ASSISTANT_CHAT_PROVIDER=deepseek
DOC_ASSISTANT_CHAT_API_KEY=<your-deepseek-key>
DOC_ASSISTANT_CHAT_MODEL=deepseek-v4-pro
DOC_ASSISTANT_CHAT_API=compatible
DOC_ASSISTANT_CHAT_BASE_URL=

# Embedding: DashScope
DOC_ASSISTANT_EMBEDDING_PROVIDER=dashscope
DOC_ASSISTANT_EMBEDDING_API_KEY=<your-dashscope-key>
DOC_ASSISTANT_EMBEDDING_MODEL=text-embedding-v3
```

Chat 切换到其他 OpenAI-compatible 服务：

```env
DOC_ASSISTANT_CHAT_PROVIDER=openai-compatible
DOC_ASSISTANT_CHAT_API_KEY=<provider-key>
DOC_ASSISTANT_CHAT_MODEL=<provider-model>
DOC_ASSISTANT_CHAT_BASE_URL=https://provider.example/v1
```

可通过 `DOC_ASSISTANT_CHAT_EXTRA_BODY` 传递 provider 特定参数：

```env
DOC_ASSISTANT_CHAT_EXTRA_BODY={"reasoning_effort":"high"}
```

### 检索配置

```env
DOC_ASSISTANT_TOP_K=5
DOC_ASSISTANT_RETRIEVAL_MODE=hybrid       # hybrid | dense | bm25
DOC_ASSISTANT_RETRIEVAL_FETCH_K=40
DOC_ASSISTANT_RETRIEVAL_MIN_RELEVANCE=0
DOC_ASSISTANT_RETRIEVAL_RRF_K=60
DOC_ASSISTANT_RETRIEVAL_DENSE_WEIGHT=1
DOC_ASSISTANT_RETRIEVAL_BM25_WEIGHT=1
DOC_ASSISTANT_RETRIEVAL_RERANK_MODE=lexical
DOC_ASSISTANT_RETRIEVAL_RERANK_WEIGHT=0.25
DOC_ASSISTANT_RETRIEVAL_MMR_LAMBDA=0.85
DOC_ASSISTANT_CHUNK_SIZE=900
DOC_ASSISTANT_CHUNK_OVERLAP=120
```

`hybrid` 模式将 Chroma 向量检索与进程内 BM25 通过 Reciprocal Rank Fusion 融合，经 lexical rerank 后使用 MMR 选择减少近似重复片段。

### Tool Calling 配置

```env
DOC_ASSISTANT_TOOL_CALL_MAX_ITERATIONS=6

# 默认禁用，避免敏感文档文本发送到公共搜索引擎
DOC_ASSISTANT_WEB_SEARCH_ENABLED=false
DOC_ASSISTANT_WEB_SEARCH_PROVIDER=duckduckgo   # duckduckgo | brave | bing
DOC_ASSISTANT_WEB_SEARCH_API_KEY=
DOC_ASSISTANT_WEB_SEARCH_MAX_RESULTS=5
DOC_ASSISTANT_WEB_SEARCH_TIMEOUT_SECONDS=10
```

### Agent 配置

```env
DOC_ASSISTANT_AGENT_MAX_PARALLEL_STEPS=3
DOC_ASSISTANT_AGENT_STEP_MAX_RETRIES=2
DOC_ASSISTANT_AGENT_STEP_RETRY_BACKOFF_SECONDS=2,5
DOC_ASSISTANT_AGENT_LLM_PLANNER_ENABLED=true
DOC_ASSISTANT_AGENT_REACT_ENABLED=true
DOC_ASSISTANT_AGENT_REACT_MAX_ITERATIONS=2
```

当 `REACT_ENABLED=true` 时，Agent 执行阶段会在计划 tool call 后运行受控 ReAct 循环：观测缺失引用、guard 警告和弱证据，然后使用白名单内的文档操作（如 `document_qa`、`build_evidence_profile`）修补证据。

### 记忆配置

```env
DOC_ASSISTANT_MEMORY_DB_PATH=
DOC_ASSISTANT_MEMORY_COLLECTION=user_memories
DOC_ASSISTANT_MEMORY_TOP_K=5
DOC_ASSISTANT_MEMORY_MIN_CONFIDENCE=0.55
DOC_ASSISTANT_MEMORY_SEMANTIC_DEDUP_MIN_SCORE=0.88
DOC_ASSISTANT_CHAT_HISTORY_WINDOW=12
DOC_ASSISTANT_MEMORY_SESSION_TTL_HOURS=24
DOC_ASSISTANT_MEMORY_TASK_TTL_HOURS=168
DOC_ASSISTANT_MEMORY_MAX_ACTIVE_PER_USER=500
DOC_ASSISTANT_MEMORY_DECAY_HALF_LIFE_DAYS=90
DOC_ASSISTANT_MEMORY_LLM_EXTRACTION_ENABLED=true
DOC_ASSISTANT_MEMORY_AUTO_SUMMARY_THRESHOLD=12
DOC_ASSISTANT_MEMORY_PROMPT_MAX_TOKENS=800
```

### 安全与隔离配置

```env
DOC_ASSISTANT_API_KEYS=                         # 为空则禁用认证
DOC_ASSISTANT_CORS_ORIGINS=http://localhost:3000,http://localhost:5173
DOC_ASSISTANT_DEFAULT_TENANT_ID=default
DOC_ASSISTANT_MAX_UPLOAD_BYTES=20971520         # 20MB
```

### 文档摄入配置

```env
DOC_ASSISTANT_INGEST_JOBS_DB_PATH=              # 默认 data/ingest_jobs.sqlite3
DOC_ASSISTANT_AGENT_TASKS_DB_PATH=              # 默认 data/agent_tasks.sqlite3
DOC_ASSISTANT_MATTER_DB_PATH=                   # 默认 data/matters.sqlite3
DOC_ASSISTANT_PDF_OCR_ENABLED=false
DOC_ASSISTANT_PDF_OCR_LANG=eng
```

PDF OCR 默认禁用。启用时需安装 `pdf2image`、`pytesseract` 及本地 OCR 运行时。

---

## API 接口

### 健康检查

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/health` | 服务健康检查 |

### 文档管理

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/v1/documents/ingest` | 上传文档并排队摄入 |
| GET | `/api/v1/documents/jobs/{job_id}` | 查询摄入任务状态 |
| GET | `/api/v1/documents` | 列出已索引文档 |

### 智能问答

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/v1/chat/ask` | 基础问答（支持对话历史） |
| POST | `/api/v1/chat/ask/stream` | SSE 流式问答 |
| POST | `/api/v1/chat/tools` | Tool Calling 模式问答 |
| GET | `/api/v1/chat/conversations` | 列出对话 |
| POST | `/api/v1/chat/conversations` | 创建对话 |
| PATCH | `/api/v1/chat/conversations/{id}` | 更新对话 |
| GET | `/api/v1/chat/conversations/{id}/messages` | 获取对话历史消息 |

### Agent 任务

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/v1/agent/tasks` | 创建 Agent 审查任务 |
| GET | `/api/v1/agent/tasks/{task_id}` | 查询任务状态与结果 |
| POST | `/api/v1/agent/tasks/{task_id}/resume` | 补充上下文后恢复任务 |
| GET | `/api/v1/agent/tasks/{task_id}/events` | SSE 实时进度流 |

Agent 任务状态流转：`queued` → `running` → `succeeded` / `failed` / `needs_input`

### Matter 管理

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/api/v1/matters` | 列出 Matter 记录 |
| GET | `/api/v1/matters/{matter_id}` | 获取 Matter 详情 |
| GET | `/api/v1/matters/{matter_id}/artifacts` | 列出生成的交付物 |
| GET | `/api/v1/matters/{matter_id}/findings` | 列出审查发现 |
| PATCH | `/api/v1/matters/{matter_id}/findings/{finding_id}` | 更新 finding 人工审核状态 |
| PATCH | `/api/v1/matters/{matter_id}/confirmation-gates/{gate_id}` | 确认/放弃/请求补充 |
| POST | `/api/v1/matters/{matter_id}/formal-report` | 生成正式报告 |
| PATCH | `/api/v1/matters/{matter_id}/artifacts/{artifact_id}` | 更新交付物（版本化） |
| GET | `/api/v1/matters/{matter_id}/events` | 获取审计事件 |
| GET | `/api/v1/matters/{matter_id}/artifacts/export` | 批量导出 ZIP |
| GET | `/api/v1/matters/{matter_id}/artifacts/{id}/export` | 单 artifact 导出 |

### 用户记忆

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/api/v1/memories` | 列出活跃记忆 |
| GET | `/api/v1/memories/stats` | 记忆健康统计 |
| POST | `/api/v1/memories` | 创建记忆条目 |
| POST | `/api/v1/memories/batch` | 批量创建记忆 |
| POST | `/api/v1/memories/maintenance` | 执行维护（过期清理等） |
| POST | `/api/v1/memories/summarize-conversation` | 对话摘要压缩 |
| PATCH | `/api/v1/memories/{memory_id}` | 更新记忆 |
| DELETE | `/api/v1/memories/{memory_id}` | 软删除记忆 |
| POST | `/api/v1/memories/batch-delete` | 批量删除记忆 |

### 审查与反馈

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/v1/review/clause` | 条款风险评估 |
| POST | `/api/v1/review/conflict` | 合同与政策冲突检测 |
| POST | `/api/v1/feedback` | 提交回答反馈 |

---

## 使用示例

### 上传文档

```powershell
$headers = @{ "X-Tenant-Id" = "acme" }
$job = Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/api/v1/documents/ingest" `
  -Headers $headers `
  -Form @{ file = Get-Item ".\contract.pdf" }

# 查询摄入进度
Invoke-RestMethod -Method Get `
  -Uri "http://localhost:8000/api/v1/documents/jobs/$($job.job_id)" `
  -Headers $headers
```

### Tool Calling 问答

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/api/v1/chat/tools" `
  -ContentType "application/json" `
  -Body '{
    "question": "结合最近公开新闻和已上传合同，分析供应商履约风险。",
    "enable_web_search": true,
    "max_tool_iterations": 6
  }'
```

### 认证请求

设置 `DOC_ASSISTANT_API_KEYS` 后，请求需附带认证头：

```powershell
$headers = @{
  "X-API-Key" = "your-api-key"
  "X-Tenant-Id" = "acme"
  "X-User-Id" = "user-001"
}
```

---

## Prompt 体系

系统采用分层 Prompt 架构，模板位于 `src/doc_assistant/prompts/`：

| 模板文件 | 用途 |
|----------|------|
| `base_legal_assistant.txt` | 全局身份、安全边界、证据规则、司法管辖意识 |
| `document_qa.txt` | 结构化文档问答输出 |
| `clause_review.txt` | 条款审查与风险评级 |
| `conflict_check.txt` | 合同/政策冲突检测 |
| `tool_calling_system.txt` | Tool 使用策略 |
| `agent_planner.txt` | Agent 规划器指令 |
| `answer_repair.txt` | 引用不合格时的二次修复 |
| `legal_issue_spotting.txt` | 法律争点识别 |
| `lawyer_work_product.txt` | 律师工作底稿生成 |
| `plain_language_explain.txt` | 通俗语言解释 |
| `general_chat.txt` | 通用对话 |

---

## 评估系统

### 生成测试数据

```powershell
generate-eval-fixtures
```

### 运行 RAG 评估

```powershell
run-rag-eval --clean --concurrency 4
```

评估报告输出到 `data/eval/latest_report.json`。

### CI 集成用法

```powershell
run-rag-eval --skip-ingest `
  --baseline data/eval/baseline_report.json `
  --fail-on-regression `
  --min-score retrieval.at_5.recall=0.8 `
  --min-score generation.citation_accuracy=0.9
```

### 检索指标

| 指标 | 含义 |
|------|------|
| `recall` | 在 top-k 中找到的 gold source 比例 |
| `hit` | top-k 中至少命中一个 gold source 则为 1 |
| `precision` | top-k 中属于 gold source 的比例 |
| `mrr` | 第一个匹配 source 的倒数排名 |
| `ndcg` | 排序质量，靠前匹配权重更高 |

### 生成指标

| 指标 | 含义 |
|------|------|
| `answer_correctness` | 必需词汇出现且禁止词汇缺席 |
| `faithfulness` | 回答中的数字和关键词有引用上下文支撑 |
| `citation_accuracy` | 引用的 source ID 对应 gold source |
| `refusal_accuracy` | 不可回答问题包含预期的拒答表述 |

评估数据集 `data/eval/eval_dataset.json` 包含可回答、不可回答、中文查询和跨文档场景，并记录 chunking config hash 以检测分块变更后的陈旧预期。

---

## 开发

### 运行测试

```powershell
pytest
```

### 代码检查

```powershell
ruff check src/ api/ tests/
ruff format --check src/ api/ tests/
```

### 安装开发依赖

```powershell
python -m pip install -e ".[dev,eval]"
```

---

## 架构设计要点

### 记忆系统与文档 RAG 的隔离

- **文档 RAG**：存储上传文档的分块，返回带引用标注的原文摘录
- **用户记忆**：存储偏好、上下文、对话历史、任务状态和反馈元数据
- 记忆仅作为 `<user_memory>` 数据注入 prompt，不参与文档证据引用

### Agent 工作流设计

Agent 基于 LangGraph 状态图编排，六个线性节点：

1. **Plan**：LLM Planner 分析目标与文档，生成审查计划
2. **Execute Steps**：并行执行计划步骤，每步调用 tool registry 中的工具
3. **Collect Findings**：汇总发现，审计证据完整性
4. **Build Deliverables**：生成 Matter Profile、artifacts 和 confirmation gates
5. **Synthesize Report**：综合报告并通过 Answer Guard 校验
6. **Finalize Result**：确定最终状态（`completed` / `needs_human_review`）

ReAct 循环在 Execute Steps 内部运行，不改变图的拓扑结构。

### 引用与证据链

- 全局 `_CitationRegistry` 管理 `[S1]`、`[S2]` 编号
- 每条 finding 关联 evidence coverage、support level 和原文位置
- Answer Guard 校验引用有效性，不合格时触发 `answer_repair` 二次修复

---

## 路线图

### 近期

1. 文档原文并排审阅与可编辑 artifact 生命周期
2. CI 发布 RAG baseline 报告并设置回归门控
3. 记忆评估面板：precision、staleness、conflicts、leakage 指标

### 中期

1. 深化工作流策略：合同审查、版本比对、争议事实整理、合规检查、谈判准备
2. 接入外部 reranker（cross-encoder 或 provider rerank API）实现两阶段检索

### 远期

1. JWT 认证与租户管理后台
2. 按文档类型、语言和工作流类别扩展评估标签体系

---

## CI/CD

项目使用 GitHub Actions（`.github/workflows/ci.yml`）：

- **后端**：Python 3.11 + pytest 单元测试
- **前端**：Node 20 + TypeScript 编译 + Vite 构建
- **RAG 评估门控**：可选，需配置 LLM API secrets 后启用

---

## License

Private project — 未开源。
