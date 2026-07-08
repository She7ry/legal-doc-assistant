# 法律文档智能分析 Agent（Legal Document Assistant）

> 项目周期：`【请填写实际起止时间】`  
> 项目角色：`【请按实际情况填写，如：Agent / 后端核心开发】`

## 推荐简历版本

**项目简介：** 面向合同、政策、隐私协议等法律文档构建引用优先的 RAG + Agent 系统，支持文档问答、条款风险审查、合同与政策冲突检测、版本对比及审查报告生成；系统以可追溯证据链和人工确认机制约束模型输出，不替代律师作最终法律判断。

**技术栈：** Python、FastAPI、LangChain、LangGraph、DeepSeek / OpenAI-compatible LLM、DashScope Embedding、Qdrant、SQLite、Vue 3、TypeScript、SSE、pytest

**核心工作：**

- 设计双层工具体系：在任务型 Agent 中注册文档问答、条款审查、冲突检测、事实抽取、版本对比、义务日历、条款修订、证据审计、谈判清单、报告汇总等 10 个工具；在开放式对话中提供 `search_documents`、`web_search` 两类 Function Calling 工具，通过 JSON Schema 约束参数类型、长度、数量及必填项，并以统一注册表和执行适配层完成白名单路由、异常返回和后续扩展。
- 基于 LangGraph 实现两套控制器：对话链采用 `Model → Tools → Observe → Model` 循环，由 LLM 自主决定是否继续检索；审查链采用 `Plan → Execute → Collect Findings → Build Deliverables → Synthesize → Finalize` 六阶段工作流，将单步执行和 ReAct 补证拆成可路由、可检查点保存的图节点，并按任务类型动态跳转审查与报告分支。
- 落地“规则规划 + LLM Planner”混合决策：通过目标关键词、审查范围和任务类型识别版本对比、义务日历、条款修订、谈判准备、证据审计及冲突检查场景；在合规、隐私等复杂目标下启用 LLM 生成结构化计划，并对工具名、最大步骤数和最终汇总步骤进行校验，LLM 规划失败时自动回退启发式计划。
- 实现受控 ReAct 多轮补证机制：每个步骤完成后检查引用数量、Answer Guard 告警、弱支持主张及缺失证据，仅允许调用 `document_qa`、`build_evidence_profile` 等白名单动作；在有限迭代预算内合并新引用并重算证据画像，无法补齐时转入 `ask_user` 或人工复核，防止无边界循环和模型任意调用高风险工具。
- 构建法律输出安全控制链：统一维护跨步骤引用编号与原文证据，使用 Answer Guard 检测无效引用、无来源事实/法条和过强法律结论，并触发自动修复；对关键事实、条款改写和正式报告设置 Confirmation Gate，结合 LangGraph checkpoint / interrupt 支持人工确认后恢复执行，同时限制 Web Search 默认关闭且禁止向公共搜索服务发送合同正文或个人敏感信息。
- 完成生产化执行与可观测设计：对连续条款审查步骤进行受限并行执行，加入工具超时、步骤重试与退避策略；通过 SQLite 持久化任务、Matter、审查发现、交付物和记忆，通过 SSE 推送规划、执行、报告阶段进度，并记录工具名称、入参、结果、ReAct trace、检索耗时及错误上下文，支持问题追踪和任务回放。
- 优化法律文档 RAG 证据底座：实现法律章节感知分块和文档版本管理，基于 Qdrant 统一承载 Dense 与 BM25 Sparse 检索，并在数据库侧完成 RRF 融合和 MMR 去重；建立 Recall、Precision、MRR、nDCG、Faithfulness、Citation Accuracy、Refusal Accuracy 等评估指标。

## 一页简历精简版

**法律文档智能分析 Agent｜Agent / 后端开发｜`【项目时间】`**  
Python、FastAPI、LangGraph、LangChain、DeepSeek / OpenAI-compatible LLM、Qdrant、SQLite、Vue 3

- 基于 LangGraph 搭建法律审查 Plan–Execute 工作流与开放式 Tool Calling 循环，注册 10 个任务工具及 2 个对话工具，通过 JSON Schema、工具白名单、迭代上限和超时机制约束模型调用边界。
- 设计“规则规划 + LLM Planner + 受控 ReAct”决策链：按任务意图生成多步计划，在发现缺失引用、弱证据或 Guard 告警时自动调用文档检索/证据审计补证，失败时回退启发式计划或转人工确认。
- 建立 citation-first 安全链路，统一管理跨步骤证据编号，利用 Answer Guard 校验无效引用、无来源事实/法条及过强法律结论，并通过 Confirmation Gate、checkpoint / interrupt 支持高风险结果人工审核后恢复。
- 构建 Dense + BM25 + RRF + 词法重排 + MMR 的混合检索链路，配合法律章节感知分块、文档版本管理和证据画像，为条款审查、冲突检测、版本对比及报告生成提供可追溯依据。
- 实现并行步骤执行、失败重试、SSE 进度流、SQLite 任务持久化及工具/ReAct 轨迹审计；建立 RAG 与生成质量评估指标，当前 148 项自动化测试全部通过。

## 工作流面试表述

用户目标进入系统后，控制层先判断是否缺少法域、截止时间、当事方立场等关键输入；信息完整时由规则或 LLM Planner 从工具注册表中生成受约束计划。Executor 执行工具并统一登记引用，随后根据引用、证据画像和 Answer Guard 结果决定是否进入 ReAct 补证。补证只能使用白名单文档工具，并受迭代预算控制；证据仍不足时请求用户输入或进入人工确认闸门。最后系统汇总 findings、risk matrix、义务日历、谈判清单等交付物，生成带引用的报告并持久化任务与审计事件。

## 使用说明

- 项目周期和个人角色无法从代码可靠推断，投递前请替换占位符。
- “148 项测试全部通过”是本次生成说明时的实际回归结果；后续测试数量变化时应同步更新。
- 项目目前具备调用轨迹、耗时日志和离线评估指标，但未实现完整 A/B 测试平台或工具成功率看板，因此简历中未写未经验证的性能提升百分比。
