# Legal Document Assistant 优化执行任务书

更新时间：2026-07-11  
工作目录：`E:\project\legal_doc_assistant`  
当前分支：`codex/deepseek-chat-protocols`

## 1. 给执行者的任务

这是实施任务，不是再次审计。按本文顺序完成高置信度修复，补最小回归测试，并执行完整验证。除非遇到真实阻塞，不要只给建议后停止，也不要把任务改造成大规模重构。

实现原则：

- 先修数据正确性，再修运行时可靠性，最后处理性能、产品链路和 CI。
- 修根因所在的共享函数，不在每个调用者重复加补丁。
- 优先标准库、SQLite/Qdrant/LangChain 已有能力；不要新增队列框架、ORM、日志框架、前端测试框架或抽象层。
- 每个非平凡修复至少有一个能复现旧问题的测试。
- 一个阶段完成后立即运行该阶段的定向测试；全部完成后再跑全量检查。

## 2. 必须保护的当前工作树

当前工作树在本任务开始前已经有 23 个未提交文件，约 `160 insertions / 659 deletions`。这些改动属于既有工作，禁止使用 `git reset --hard`、`git checkout --`、`git restore` 或其他覆盖方式。

开始前执行：

```powershell
git -c safe.directory=E:/project/legal_doc_assistant status -sb
git -c safe.directory=E:/project/legal_doc_assistant diff --stat
```

与本任务存在重叠、修改前必须先查看现有 diff 的文件：

- `src/doc_assistant/services/qa_service.py`
- `src/doc_assistant/retrieval/vector_store.py`
- `api/routers/chat.py`
- `pyproject.toml`
- `README.md`

使用 `apply_patch` 做局部编辑，保留目标文件中与本任务无关的现有修改。不要自行 commit、push 或创建 PR，除非用户在新对话中明确要求。

## 3. 阶段 0：建立基线

先运行，不通过就先判断是否为现有工作树问题，不能通过 reset 解决：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src api tests scripts
Push-Location frontend
npm.cmd run build
Pop-Location
```

审计时基线为 136 项测试通过、Ruff 通过、前端可构建。测试数量可能因工作树继续变化而不同，重点是开始实施前记录真实结果。

## 4. 阶段 1：修复记忆数据正确性

### 4.1 SQLite 删除后仍被 Qdrant 召回

根因链路：

- `src/doc_assistant/memory/service.py::delete_memory` 先把 SQLite 状态改成 `deleted`。
- `src/doc_assistant/memory/vector_store.py::delete_memory` 吞掉 Qdrant 删除异常，旧向量可能继续存在。
- `src/doc_assistant/memory/service.py::_hydrate_vector_candidates` 只回查“不完整”的向量候选；元数据完整的旧向量直接被信任。
- 结果是 SQLite 已删除的记忆仍可能进入 `DocumentQAService` 的 prompt。

最小修复：

1. SQLite 是唯一权威。`_hydrate_vector_candidates` 对所有向量候选 ID 一次性调用 `MemoryStore.get_memories_by_ids`。
2. 只使用 SQLite 返回的最新记录构造候选；数据库不存在、非 `active` 或已过期的记录全部丢弃。
3. Qdrant 删除仍可 best-effort，但不能静默假装成功：返回布尔结果并以 warning 记录失败；不要因为向量删除失败回滚 SQLite 删除。
4. 不增加双写事务或消息队列；读取时校验权威状态已经解决一致性窗口。

测试放在 `tests/test_memory_service.py`：fake vector store 故意保留带完整元数据的旧向量，删除 SQLite 记录后，`retrieve_relevant_memories()` 必须返回空列表。

### 4.2 记忆创建和 PATCH 的并发丢失

根因链路：

- `MemoryService.create_memory` 的“查旧记录、插入新记录、标旧记录 stale”跨多个事务。
- `MemoryStore.update_memory` 在写锁和事务外读取旧值，随后覆盖全部字段。
- 两个连接并发时可能产生两个 active 同键记录，或让分别修改 `content`、`confidence` 的 PATCH 丢掉其中一个更新。

最小修复：

1. 在 `MemoryStore` 增加一个具体的原子用例方法，用 `BEGIN IMMEDIATE` 在一个事务内完成“查旧、写新、旧记录 stale”。不要创建通用 UnitOfWork/Repository 抽象。
2. `MemoryService.create_memory` 调用该方法，向量同步仍在 SQLite 提交成功后 best-effort 执行。
3. `update_memory` 在同一个 `BEGIN IMMEDIATE` 事务内重新读取并写入，或者只更新 `MemoryUpdate` 实际提供的列；不得继续锁外读、全字段覆盖。
4. 增加 active 同键的 SQLite 部分唯一索引：`(tenant_id, user_id, scope, type, key) WHERE status = 'active'`。创建索引前若存在历史重复，保留 `updated_at` 最新的一条，其他标为 `stale`；不要直接让应用启动失败。

测试仍放在 `tests/test_memory_service.py` 或最接近 Store 的现有测试文件：两个独立 Store/连接用 `threading.Barrier` 并发创建和更新，断言只有一个 active 同键记录，并且两个不同字段的修改都保留。

定向验证：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_memory_service.py
```

## 5. 阶段 2：修复文档入库一致性并顺手减少 Qdrant 写入

### 5.1 分批 upsert 失败留下 active 残片

根因链路：

- `QdrantDocumentRepository.embed_and_add` 分批 `upsert`，前一批成功、后一批失败时会留下部分点。
- `DocumentIngester._rollback_new_chunks` 的删除失败只记录日志。
- 下一次上传在 `DocumentIngester._ingest_file` 中只要发现任意相同 `file_id` 的 active 点就返回 `skipped=True`。

最小且完整的修复是两阶段激活：

1. `_prepare_chunks` 给每个点写入 `expected_chunk_count=len(chunks)`，新点先写 `active=False`。
2. 所有批次成功后，使用 Qdrant 原生一次 `set_payload(points=[...])` 把本版本全部 ID 改成 `active=True`。
3. 激活新版本成功后，再一次批量把旧版本写成 `active=False` 并设置 `superseded_by_file_id`。
4. 写入失败时残片保持 inactive，即使回滚再次失败也不会参与搜索、目录或去重。
5. `skipped=True` 前验证 `expected_chunk_count`、实际记录数及 `chunk_id` 集合 `0..N-1` 完整。缺少 `expected_chunk_count` 的同内容记录不能被当成已完整写入，应清理确定性 ID 后重建。

同时删除 `update_metadatas` 的逐点远程写循环。如果它只有旧版本失活这一个调用者，替换为接受 `ids + common_payload` 的具体批量方法，不保留两个重叠 API。

测试放在 `tests/test_vector_store_ingestion.py`：模拟第一批成功、第二批失败、rollback 失败；断言没有 active 新 chunk，并且重试会重新入库而不是 skipped。另加一个 call-count 断言，100 个旧 chunk 失活只能调用一次 `set_payload`。

### 5.2 `active` 缺失语义不一致

当前 `metadata_is_active()` 把缺少 `active` 当作 true，而 Qdrant `active_filter()` 只匹配显式 `true`。同一点可能出现在目录/去重逻辑中，却永远搜不到。

优先选择不迁移数据的最小一致方案：保留 Python 的兼容语义，让 Qdrant filter 同时匹配 `active=True` 和字段缺失（使用当前 qdrant-client 的 `IsEmptyCondition`/等价原生条件）。先用本地 Qdrant 测试确认 missing/true/false 三种 payload 的 Python 判定和实际查询完全一致。只有当前客户端无法表达缺失条件时，才改为严格 `active is True` 并写一次性回填脚本；不要建设通用迁移框架。

### 5.3 每次检索重复检查 collection

文档搜索和记忆搜索目前每次 cache miss 会重复调用 `collection_exists`，`ensure_dense_collection` 内又检查并读取 collection。

最小修复：每个 repository/vector store 实例只在首次拿到 embedding 维度时校验一次并记住已验证维度；后续直接 query。不要增加全局缓存或 collection manager。用假客户端断言首次查询执行校验，第二次不再出现额外检查。

定向验证：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_vector_store_ingestion.py tests\test_memory_service.py
```

## 6. 阶段 3：修复异步阻塞和后台任务生命周期

### 6.1 `/chat/ask` 的 async 路径仍执行同步 LLM

根因：`DocumentQAService.aprepare_answer` 直接调用 `_rewrite_query()`，`aask()` 又同步调用 `finalize_prepared_answer()`；两者在特定路径会进入 `chat_model.invoke`，冻结 FastAPI event loop。

最小修复：

- 在异步路径用 `await asyncio.to_thread(...)` 执行 `_rewrite_query` 和 `finalize_prepared_answer`。
- 保留同步 `ask()` 行为不变。
- 不为整个 QA service 建第二套 async 类；只有当 tracing/context 测试证明 `to_thread` 不可用时，才改成复用现有 `_ainvoke_chat_messages` 的 async rewrite/repair。

测试放在 `tests/test_qa_service.py`：慢 fake LLM 记录执行线程，同时运行一个短 heartbeat coroutine；断言同步模型不在 event-loop 线程执行，heartbeat 在模型返回前已经推进。

### 6.2 后台任务进程内去重和 executor 关闭

现状：`api/task_queue.py` 的 `_submitted_keys` 只在当前进程；多个 worker 都会扫描 queued/running 任务。`mark_running` 没有数据库条件，可能重复执行；executor 也没有随 lifespan 关闭。

最小安全方案：

1. `IngestJobStore` 与 `AgentTaskStore` 各增加具体的 `claim(id) -> bool`，使用单条 SQLite `UPDATE ... WHERE status='queued'`，依靠 `rowcount`/`RETURNING` 判断是否认领成功。
2. worker 执行业务副作用前先 claim；失败者直接返回。
3. `list_restartable()` 只返回 queued。没有 lease/heartbeat 时不要自动重跑 running，避免两个进程同时启动时把另一个进程刚认领的任务当作崩溃遗留任务。
4. README 明确当前后台执行器仍是单机方案；不要引入 Celery、Redis 或通用队列抽象。
5. 为 executor 增加显式 shutdown。考虑测试会多次进入 FastAPI lifespan，使用可懒重建的 executor，而不是关闭一次后永久不可用。

测试：两个独立 Store/连接并发 claim 同一个 queued job/task，必须恰好一个成功；重复 lifespan 后仍能提交任务，退出后没有遗留工作线程。

定向验证：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_qa_service.py tests\test_ingest_jobs.py tests\test_agent_tasks.py
```

## 7. 阶段 4：收紧 HTTP 错误边界并让现有日志真正可见

### 7.1 错误分类和异常泄漏

`api/main.py` 把任意 `ValueError` 映射成 422、任意 `RuntimeError` 映射成 502，并把 `str(exc)` 返回客户端。SSE、文档路由和后台任务也存在原始异常返回或持久化。

最小修复：

1. 删除全局 `ValueError`/`RuntimeError` handler；未知异常统一走已有安全 500 响应。
2. 只在明确的用户输入或上游服务边界捕获并转换成 `HTTPException`。
3. SSE 与后台任务对外只给稳定、安全的信息和 task/request ID；完整异常用 `logger.exception` 保留在服务端。
4. 不创建自定义异常继承树。现有 `HTTPException` 已足够。

至少增加 API 测试：内部 `ValueError` 不得伪装成 422，响应不得包含原始异常文本；已知输入错误仍保持原来的 4xx。

### 7.2 INFO 埋点在 Uvicorn 默认配置下丢失

`api/main.py` 和 `doc_assistant/observability.py` 已经记录 request/retrieval duration，但仓库没有应用日志配置。

使用标准库 `logging.config.dictConfig` 增加一个很小的配置，只配置 `api` 和 `doc_assistant` namespace 的 INFO handler；`propagate=False` 防止重复。Formatter 用 defaults 处理缺少 `request_id`、`operation`、`duration_ms` 的普通日志。不要新增 structlog 等依赖。

测试 formatter 不会因缺少 extra 字段报错，并能显示 request ID、operation 和 duration。

## 8. 阶段 5：让 Agent → Matter 链路真实可用

### 8.1 当前问题

真实 `run_react_agent_task` 固定返回 `findings=[]`、`matter_profile=None`、`artifacts=[]`，但 `api/routers/agent.py` 仍无条件写 Matter，`MatterStore.upsert_from_agent_result` 再制造“未生成 profile”的占位记录。现有 rich Matter API 测试使用的是 fake Agent，并未覆盖真实服务。

### 8.2 先保证不制造空壳

无论结构化生成是否成功，都先落实以下规则：

- 只有 `matter_profile`、`findings`、`artifacts` 至少一项具有实际内容时才调用 `upsert_from_agent_result`。
- 未持久化 Matter 时，任务响应不要伪造可打开的 matter ID；前端 `AgentPage.vue` 不显示 Matter 链接。
- `MatterStore` 不再自动制造占位 profile；调用者传空结果应拒绝或不调用。

### 8.3 使用已有 LangChain 结构化输出能力

Matter 页面和导出功能已经存在，因此这里按核心功能实现，不只停留在隐藏链接：

1. 复用仓库现有 `chat_model.with_structured_output(...)` 模式（见 `src/doc_assistant/tools/legal_review.py`），定义最小 Pydantic 输出模型，字段只覆盖 MatterStore 当前实际消费的 profile、finding 和 artifact 字段。
2. 在 ReAct 报告与引用已经产生后做一次结构化提取；输入只包含任务目标、最终报告、引用 ID/位置和必要 evidence，避免重新跑检索。
3. 模型不支持 structured output、解析失败或输出为空时，记录 warning 并返回“无 Matter”，不得回退到手写 JSON 提取或占位记录。
4. finding 的 citation 必须来自本次已有引用 ID；过滤不存在的引用，不允许模型发明来源。
5. 不把整个 Agent 替换成 LangChain `create_agent`，也不缓存当前 graph；这些改动不等价且没有明显收益。

测试：用真实 `LegalAgentService` 调用链加 fake chat model，断言结构化结果被写入 Matter；解析失败时任务仍可成功，但不会创建空 Matter。保留现有 rich fake 测试作为 MatterStore/API 行为测试。

### 8.4 正式报告准入规则只保留后端一份

`frontend/src/pages/MattersPage.vue` 重复实现了后端 `_formal_report_blockers` 规则。

- 在 `MatterRecordOut` 返回 `formal_report_blockers` 和 `can_generate_formal_report`。
- 前端直接消费后端结果，删除 `unresolvedRequiredGates`、`unresolvedFindings`、`isFindingFormalReady` 等重复判断。
- 暂不因为文件有 1339 行就机械拆组件；只有本轮修改后仍存在独立、反复变化的 UI 区块时再拆。不要新增前端测试框架，本轮以后端规则测试、TypeScript 构建和浏览器 smoke test 验收。

定向验证：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_agent_service.py tests\test_agent_api.py tests\test_matter_store.py
Push-Location frontend
npm.cmd run build
Pop-Location
```

## 9. 阶段 6：文档预览真正分页

现状：`_document_catalog.get_document_text` 扫描并返回一个版本的全部 chunk，前端一次渲染全部。大文件会同时压垮 Qdrant 响应、API JSON 和 DOM。

最小方案：

1. `/api/v1/documents/text` 增加 `offset>=0`、`limit`（默认 100，上限 200）。响应保留 `total_chunks`，增加足以判断是否还有下一页的 `offset/limit` 或 `next_offset`。
2. repository 使用 `document_key/file_id/version/active` 加 `chunk_id` 数值范围过滤，只读取本页；用 Qdrant count 获取总数。不要先全量 scroll 再在 Python 切片。
3. 前端预览首次只取一页，提供“加载更多”；保持 citation 定位到 chunk 的能力。如果目标 chunk 不在已加载页，根据 chunk ID 请求对应范围，而不是加载全文。
4. 暂不增加单独的文档 manifest 数据库。只有文档列表全量扫描经过测量仍慢时再加 summary point。

测试：构造超过一页的 chunks，断言页之间不重复、不遗漏，`total_chunks` 正确，API 上限生效。前端构建后用浏览器 smoke test 检查首屏和加载更多。

## 10. 阶段 7：CI、文档和依赖可复现性

1. `.github/workflows/ci.yml` 的 backend job 在 pytest 前增加：

   ```powershell
   python -m ruff check src api tests scripts
   ```

2. 不立即启用 `ruff format --check`；当前会要求机械修改约 57 个文件，应留给单独格式化变更。
3. README 的覆盖率命令改成当前已安装 `coverage` 能执行的形式：

   ```powershell
   python -m coverage run -m pytest
   python -m coverage report -m
   ```

4. 为 CI 的 Python 3.11 生成 constraints 文件并在安装时使用 `-c`。优先使用当前 pip 已有的锁定/导出能力；若必须临时使用 pip-tools，只用于生成文件，不加入运行时依赖。不要从混有无关包的开发环境盲目提交整份 `pip freeze`。在一个干净临时 venv 中验证 constraints 能安装 `.[dev,eval]` 并跑测试。
5. 不新增 Vitest/Jest。等出现需要独立测试的前端业务逻辑时再加；本轮应把正式报告规则移回后端。

## 11. 最终验证与完成定义

全部阶段完成后运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src api tests scripts
git -c safe.directory=E:/project/legal_doc_assistant diff --check
Push-Location frontend
npm.cmd run build
Pop-Location
git -c safe.directory=E:/project/legal_doc_assistant status -sb
git -c safe.directory=E:/project/legal_doc_assistant diff --stat
```

完成标准：

- 删除/过期记忆无法从旧向量重新进入 prompt。
- 文档写入失败不会留下可搜索残片，重试不会误报 skipped。
- 记忆并发创建/更新不重复、不丢字段。
- async QA 路径不在 event loop 执行同步 LLM。
- 同一 queued 后台任务只能被一个执行者 claim，executor 可正常关闭和重建。
- 未知内部异常不再以 422/502 和原始文本暴露。
- INFO 请求/检索耗时日志实际可见。
- 真实 Agent 能生成受 schema 约束的 Matter；失败时不制造空壳。
- 正式报告准入规则只由后端决定。
- 文档预览按页读取和渲染。
- Ruff 已进入 CI，README 覆盖率命令有效，CI Python 依赖可复现。
- 没有覆盖任务开始前的用户改动，没有新增不必要依赖或抽象。

如真实 RAG 凭据可用，再运行既有 eval；凭据不可用时不要阻塞完成，只在最终报告中明确说明未运行。最终回复按阶段列出修改文件、关键行为变化、测试结果和明确跳过项。

## 12. 明确不要做的事情

- 不把 Agent 全面替换为 `create_agent`。
- 不引入 Celery、Redis、SQLAlchemy、structlog 或新的前端测试框架。
- 不为单一实现创建 interface/factory/repository 层。
- 不机械拆分 `MattersPage.vue`。
- 不把 Qdrant 全量迁移到另一种向量数据库。
- 不做与上述验收标准无关的格式化、命名或目录重排。
- 不 reset、restore、覆盖或提交用户已有改动。
