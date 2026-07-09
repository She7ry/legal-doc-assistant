# 法律文档智能分析 Agent（Legal Document Assistant）

> 项目周期：`【请填写实际起止时间】`  
> 项目角色：`【请按实际情况填写，如：Agent / 后端核心开发】`

## 推荐简历版本

**项目简介：** 面向合同、政策、隐私协议等法律文档构建引用优先的 RAG + ReAct Agent 系统，支持文档问答、条款风险审查、合同与政策冲突检测及带引用审查报告生成；系统以可追溯证据链和 Answer Guard 约束模型输出，不替代律师作最终法律判断。

**技术栈：** Python、FastAPI、LangChain、LangGraph、DeepSeek / OpenAI-compatible LLM、DashScope Embedding、Qdrant、SQLite、Vue 3、TypeScript、SSE、pytest

**核心工作：**

- 构建 ReAct-only Agent 入口：任务型审查直接复用 Tool Calling 循环，由模型按需调用 `search_documents`、`review_clause`、`check_conflict` 等白名单工具，并以迭代上限约束调用边界。
- 设计 citation-first 安全链路：工具返回的引用统一重编号为 `D1/D2/...`，最终报告通过 Answer Guard 校验无效引用、无来源事实/法条及过强法律结论，必要时触发自动修复。
- 打通前后端任务生命周期：FastAPI 创建/查询/恢复 Agent 任务，SQLite 持久化任务与 Matter 记录，SSE 推送 `queued/running/react_started/succeeded` 等进度事件，前端展示报告、引用、证据画像和 tool trace。
- 优化法律文档 RAG 证据底座：实现法律章节感知分块和文档版本管理，基于 Qdrant 统一承载 Dense 与 BM25 Sparse 检索，并在数据库侧完成 RRF 融合和 MMR 去重。
- 建立离线评估与回归测试：覆盖检索 Recall、Precision、MRR、nDCG，以及生成 Faithfulness、Citation Accuracy、Refusal Accuracy 等指标。

## 一页简历精简版

**法律文档智能分析 Agent｜Agent / 后端开发｜`【项目时间】`**  
Python、FastAPI、LangGraph、LangChain、DeepSeek / OpenAI-compatible LLM、Qdrant、SQLite、Vue 3

- 搭建 ReAct-only 法律审查 Agent，注册文档检索、条款审查、冲突检测等工具，通过 JSON Schema、工具白名单、迭代上限和超时机制约束模型调用边界。
- 建立 citation-first 安全链路，统一重编号工具引用，利用 Answer Guard 校验无效引用、无来源事实/法条及过强法律结论，并触发自动修复。
- 实现 SSE 进度流、SQLite 任务持久化、Matter 同步及前端 tool trace 展示，支持任务回放和问题定位。
- 构建 Dense + BM25 + RRF + MMR 的混合检索链路，配合法律章节感知分块、文档版本管理和证据画像，为条款审查、冲突检测及报告生成提供可追溯依据。

## 工作流面试表述

用户目标进入系统后，控制层先判断是否缺少关键输入；信息完整时创建 Agent 任务并进入 ReAct 工具调用循环。模型根据问题决定是否调用文档检索、条款审查或冲突检测工具；每个工具返回的引用会统一重编号并写入 trace。最终答案经过证据画像与 Answer Guard 校验，生成带引用报告，并将任务结果、引用和工具轨迹持久化到 SQLite / Matter。

## 使用说明

- 项目周期和个人角色无法从代码可靠推断，投递前请替换占位符。
- 自动化测试数量会随代码变化，投递前请以最新 `pytest` 结果为准。
- 项目目前具备调用轨迹、耗时日志和离线评估指标，但未实现完整 A/B 测试平台或工具成功率看板，因此简历中未写未经验证的性能提升百分比。
