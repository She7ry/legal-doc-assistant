# Legal Document Assistant — CLAUDE.md

引用优先（citation-first）的法律文档 RAG 分析系统。支持合同、政策、租约、隐私协议、校规及合规文档的智能问答、风险审查与冲突检测。

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | FastAPI + Uvicorn + LangChain + LangGraph |
| 向量存储 | ChromaDB |
| 检索 | Hybrid（Dense + BM25 + RRF 融合 + MMR 去重） |
| 默认 LLM | DeepSeek（Chat）/ DashScope（Embedding） |
| 中文分词 | jieba |
| 前端 | Vue 3 + TypeScript + Vite + Element Plus + Pinia |
| 数据持久化 | SQLite（任务/记忆/Matter） |
| 测试 | pytest + pytest-asyncio |

## 项目结构

```
api/                    — FastAPI 路由、中间件、依赖注入、SQLite store
src/doc_assistant/      — 核心业务逻辑
  config/               — 环境配置（Settings dataclass）
  models/               — LLM/Embedding 模型工厂
  ingestion/            — 文档加载（PDF/DOCX/TXT/MD）
  retrieval/            — Hybrid 向量检索引擎
  services/             — QA、Tool Calling、Agent、Review 服务
  memory/               — 用户记忆系统（SQLite + ChromaDB）
  matter/               — 法律事务管理（Matter store + 导出）
  prompts/              — Prompt 模板文件（11 个 .txt）
  graphs/               — LangGraph 状态图定义
  tools/                — Web 搜索客户端
  evaluation/           — RAG 评估指标与 CLI
frontend/               — Vue 3 SPA
tests/                  — 18 个测试文件
data/                   — 运行时数据（uploads、vector_store、SQLite）
scripts/                — RAG eval 脚本
```

## 命令

### 后端

```bash
# 先复制并编辑 .env
cp .env.example .env

# 启动开发服务器
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 生产模式（不自动重载）
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 前端

```bash
cd frontend
npm install
npm run dev        # 开发服务器 → http://127.0.0.1:5173
npm run build      # 生产构建
```

### 测试

```bash
# 运行全部测试
python -m pytest tests/ -v

# 运行单个模块
python -m pytest tests/test_qa_service.py -v

# 带覆盖率
python -m pytest tests/ -v --cov=src/doc_assistant --cov-report=term-missing
```

### RAG 评估

```bash
python -m doc_assistant.evaluation.cli run_eval
# 或
run-rag-eval
```

## 可用 Skills

项目 `.claude/skills/` 下有 14 个 skill，通过 `/<name>` 调用：

### 项目运行
| Skill | 用途 |
|-------|------|
| `backend-run` | 启动 FastAPI 后端 |
| `frontend-run` | 启动 Vue 前端 |
| `test` | 运行 pytest |
| `rag-eval` | RAG 评估 |
| `db-inspect` | 数据库检查 |

### 代码质量
| Skill | 用途 |
|-------|------|
| `adversarial-verify` | 对抗性验证——完成前质疑自己的结论 |
| `owasp-review` | API/后端 OWASP Top 10 安全审查 |
| `secret-scan` | 扫描硬编码密钥和凭证 |
| `systematic-debugging` | 系统性调试流程 |
| `clean-commits` | 规范化 Git 提交 |
| `write-failing-test-first` | TDD 红-绿-重构 |

### 法律领域
| Skill | 用途 |
|-------|------|
| `legal-review` | 多轮次法律文档审查工作流 |
| `citation-check` | 引用验证——RAG 回答质量核心机制 |
| `contract-risk` | 合同风险矩阵分析与条款审查 |

### 推荐安装的官方 Skills

在 Claude Code 中运行以下命令安装 Anthropic 官方 skills（用于文档生成）：

```
/plugin marketplace add anthropics/skills
/plugin install document-skills@anthropic-agent-skills
```

包含 `docx`（Word 生成）、`pdf`（PDF 处理）、`xlsx`（表格）、`pptx`（演示文稿）。

也可以添加社区 marketplace：

```
/plugin marketplace add jeremylongshore/claude-code-plugins-plus-skills
/plugin marketplace add Archive228/loopkit
```

## 关键设计原则

- **引用优先**：所有回答必须标注来源引用 `[S1]`、`[S2]`
- **多租户隔离**：通过 `X-Tenant-Id` header 隔离 Chroma 集合、BM25 索引、SQLite store
- **版本化摄入**：文档按 SHA-256 去重，内容变更时创建新版本
- **Hybrid 检索**：Dense + BM25 → RRF 融合 → 可选 lexical rerank → MMR 去重
- **Answer Guard**：回答后校验引用有效性、过强法律结论、证据缺失
- **Memory 隔离**：用户记忆独立于文档 RAG，注入为上下文但不作为文档证据引用
