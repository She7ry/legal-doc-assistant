# Legal Document Assistant

面向合同、政策与合规文档的法律信息辅助系统。用户可以建立个人文档库，通过带原文引用的问答、条款审查和 Agent 任务整理风险与事项报告。

> 本项目用于辅助理解和整理法律文档，不替代律师，也不提供最终法律意见。

## 主要功能

- 注册、登录与个人数据隔离
- 上传 PDF、DOCX、TXT、Markdown，后台完成解析与索引
- 基于 Qdrant 混合检索的引用式问答和多轮对话
- 条款风险审查、合同与政策冲突检测
- 分层 Agent：简单任务使用工具调用，复杂任务使用 LangGraph 分步执行
- 可选的 Web 搜索与 Docusign 只读协议查询

后端使用 FastAPI、LangChain/LangGraph、Qdrant 和 SQLite；前端使用 Vue 3、TypeScript、Element Plus 和 Pinia。

## 快速开始

环境要求：Python 3.10～3.12、Node.js 18+。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .env.example .env

npm.cmd --prefix frontend install
```

在 `.env` 中填写 Chat 与 Embedding 服务凭据，然后分别启动后端和前端：

```powershell
uvicorn backend.main:app --reload
```

```powershell
npm.cmd --prefix frontend run dev
```

- 前端：http://127.0.0.1:5173
- API 文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/health

具体可选项见 `.env.example`。

## 使用流程

1. 注册并登录。
2. 在“文档”页面上传资料，等待索引完成。
3. 在工作区进行带引用问答，或在“审查”页面检查条款和冲突。
4. 对复杂任务创建 Agent 任务，查看并自行保存需要保留的报告。

## 项目结构

```text
src/backend/
                        FastAPI 接口、认证与后台任务
src/ai/
                        Agent、RAG、记忆、审查与 MCP 能力
frontend/               Vue 前端
tests/                  后端测试
```

## 开发检查

```powershell
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
npm.cmd --prefix frontend run build
```
