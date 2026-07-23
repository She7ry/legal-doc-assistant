"""法律文档助手的 AI 核心包。

本包实现「引用优先」的 Agent 与 RAG 能力，`backend` 通过这里的
服务类对外暴露能力。整体数据流如下：

    上传文档 → rag.ingestion → rag.retrieval（向量 + BM25 混合检索）
                                    ↓
    用户提问 → rag.qa_service / agent.tool_calling → LLM 生成答案
                                    ↓
              rag.grounding → 校验引用、评估证据支持度
                                    ↓
    复杂任务 → react_task（ReAct tool calling）→ 工具轨迹 → 带引用报告

目录结构速查
------------
agent/        ReAct 运行时、Tool Calling 与 Agent 工具适配器
config/       环境变量与 Settings（模型、检索、Agent、记忆等开关）
rag/          文档入库、混合检索、引用校验与问答服务
review/       条款审阅与冲突分析
memory/       用户长期记忆、对话历史、语义检索
skills/       后端专用 Agent Skill
llm.py        LLM / Embedding 客户端（DeepSeek、OpenAI-compatible 等）
qdrant.py     RAG 与 Memory 共用的 Qdrant 基础能力
prompts/      LLM 系统提示词模板（.txt）
rag/evaluation/ 离线评测 CLI 与指标
utils/        小工具函数（如加载 prompt 文件）
"""

__version__ = "0.1.0"
