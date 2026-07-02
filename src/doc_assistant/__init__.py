"""legal_doc_assistant 核心 Python 包。

本包实现「引用优先」的法律文档助手后端逻辑，API 层（`api/`）通过这里的
服务类对外暴露能力。整体数据流如下：

    上传文档 → ingestion → retrieval（向量 + BM25 混合检索）
                                    ↓
    用户提问 → qa_service / tool_calling_service → LLM 生成答案
                                    ↓
              answer_guard + evidence → 校验引用、评估证据支持度
                                    ↓
    复杂任务 → agent_service（LangGraph 工作流）→ 计划 → 逐步执行 → 报告

目录结构速查
------------
config/       环境变量与 Settings（模型、检索、Agent、记忆等开关）
ingestion/    PDF/DOCX/TXT 解析与上传落盘
retrieval/    Chroma 向量库 + BM25 索引，混合检索与分块入库
models/       LLM / Embedding 客户端（DeepSeek、OpenAI-compatible 等）
schemas/      跨模块共享的数据结构（Citation、QAAnswer 等）
services/     业务核心：问答、工具调用聊天、法律 Agent
  agent/      Agent 工作流编排（LangGraph）与数据结构
  answer_guard  答案合规校验（引用、强结论、无依据事实）
  evidence      将答案拆成可审计主张并评估引用支持度
graphs/       LangGraph 状态图定义（Agent 流水线、工具调用循环）
memory/       用户长期记忆、对话历史、语义检索
matter/       案件（matter）持久化：finding、artifact、审计事件
tools/        Agent 可调用的外部工具（如网页搜索）
prompts/      LLM 系统提示词模板（.txt）
evaluation/   离线评测 CLI 与指标
utils/        小工具函数（如加载 prompt 文件）
"""

__version__ = "0.1.0"
