<template>
  <div class="page-stack agent-page">
    <section class="summary-strip">
      <div class="metric">
        <span>任务状态</span>
        <strong>{{ task ? statusLabel(task.status) : "未运行" }}</strong>
      </div>
      <div class="metric">
        <span>执行进度</span>
        <strong>{{ task?.progress ?? 0 }}%</strong>
      </div>
      <div class="metric">
        <span>引用数量</span>
        <strong>{{ result?.citations.length ?? 0 }}</strong>
      </div>
      <div class="metric">
        <span>工具调用</span>
        <strong>{{ toolCalls.length }}</strong>
      </div>
    </section>

    <div class="agent-grid">
      <section class="tool-panel agent-run-panel">
        <div class="panel-heading">
          <div>
            <h2>Agent 任务</h2>
            <p>使用 ReAct 工具调用生成带引用的法律文档报告。</p>
          </div>
          <el-button :icon="Refresh" :disabled="loading" @click="resetForm">重置</el-button>
        </div>

        <el-form class="agent-form" label-position="top" @submit.prevent>
          <el-form-item label="任务目标">
            <el-input
              v-model="form.objective"
              type="textarea"
              :rows="4"
              maxlength="2000"
              show-word-limit
              resize="none"
              :disabled="loading"
              placeholder="例如：审查这份 SaaS MSA 的终止、付款和责任限制风险，并给出律师问题清单。"
            />
          </el-form-item>

          <el-form-item label="关注领域">
            <el-select
              v-model="form.focusAreas"
              multiple
              filterable
              allow-create
              default-first-option
              collapse-tags
              collapse-tags-tooltip
              :disabled="loading"
              placeholder="选择或输入条款类型"
            >
              <el-option
                v-for="area in focusAreaOptions"
                :key="area.value"
                :label="area.label"
                :value="area.value"
              />
            </el-select>
          </el-form-item>

          <div class="agent-form-row">
            <el-form-item label="用户模式">
              <el-radio-group v-model="form.userRole" :disabled="loading">
                <el-radio-button
                  v-for="option in userRoleOptions"
                  :key="option.value"
                  :value="option.value"
                >
                  {{ option.label }}
                </el-radio-button>
              </el-radio-group>
            </el-form-item>
            <el-form-item label="最大工具轮次">
              <el-input-number
                v-model="form.maxSteps"
                :min="3"
                :max="10"
                :disabled="loading"
                controls-position="right"
              />
            </el-form-item>
          </div>

          <div class="panel-actions">
            <el-button
              type="primary"
              :icon="MagicStick"
              :loading="loading"
              :disabled="!form.objective.trim()"
              @click="runTask"
            >
              运行 Agent
            </el-button>
          </div>
        </el-form>
      </section>

      <section v-if="task" class="tool-panel agent-status-panel">
        <div class="panel-heading">
          <div>
            <h2>任务进度</h2>
            <p>{{ task.task_id }}</p>
          </div>
          <el-tag :type="taskStatusType(task.status)" effect="dark">
            {{ statusLabel(task.status) }}
          </el-tag>
        </div>

        <el-progress
          :percentage="Math.max(0, Math.min(task.progress, 100))"
          :status="progressStatus"
          :stroke-width="10"
        />
        <p class="job-stage">{{ stageLabel(task.stage) }}</p>

        <el-alert
          v-if="task.error"
          type="error"
          :title="task.error"
          :closable="false"
          show-icon
        />

        <el-alert
          v-if="clarificationQuestions.length"
          type="warning"
          title="需要补充信息后再运行"
          :closable="false"
          show-icon
        >
          <ul class="structured-list structured-list--compact agent-clarification-list">
            <li v-for="question in clarificationQuestions" :key="question">{{ question }}</li>
          </ul>
        </el-alert>

        <el-form
          v-if="task.status === 'needs_input'"
          class="agent-resume-form"
          label-position="top"
          @submit.prevent
        >
          <el-form-item label="补充信息">
            <el-input
              v-model="resumeForm.clarificationAnswers"
              type="textarea"
              :rows="3"
              maxlength="1200"
              show-word-limit
              resize="none"
              :disabled="loading"
              placeholder="例如：请审查付款和终止风险。我代表客户方，合同适用纽约州法。"
            />
          </el-form-item>
          <div class="panel-actions">
            <el-button
              type="primary"
              :icon="MagicStick"
              :loading="loading"
              :disabled="!resumeForm.clarificationAnswers.trim()"
              @click="resumeTask"
            >
              补充并继续
            </el-button>
          </div>
        </el-form>

        <div v-if="events.length" class="agent-event-list">
          <article v-for="event in events" :key="event.event_id" class="agent-event">
            <el-tag size="small" :type="eventType(event.event_type)" effect="plain">
              {{ eventLabel(event.event_type) }}
            </el-tag>
            <div>
              <strong>{{ event.message }}</strong>
              <span>{{ event.progress }}% · {{ formatDate(event.created_at) }}</span>
            </div>
          </article>
        </div>
      </section>

      <section v-if="result" class="tool-panel agent-report-panel">
        <div class="panel-heading">
          <div>
            <h2>最终报告</h2>
            <p>任务 {{ result.task_id }}</p>
          </div>
          <el-tag :type="result.human_review_required ? 'warning' : 'success'" effect="dark">
            {{ result.human_review_required ? "需要人工复核" : "证据闭环完成" }}
          </el-tag>
        </div>

        <div class="agent-report">{{ result.report }}</div>

        <div v-if="result.guard_warnings.length" class="answer-trust">
          <el-alert type="warning" title="报告守卫提示" :closable="false" show-icon>
            <ul class="structured-list structured-list--compact">
              <li v-for="warning in result.guard_warnings" :key="warning">
                {{ guardWarningLabel(warning) }}
              </li>
            </ul>
          </el-alert>
        </div>

        <EvidencePanel v-if="result.evidence" :evidence="result.evidence" />
        <CitationList v-if="result.citations.length" :citations="result.citations" />
      </section>

      <section v-if="result" class="tool-panel agent-trace-panel">
        <div class="panel-heading">
          <div>
            <h2>工具轨迹</h2>
            <p>ReAct 循环中实际调用的工具和返回摘要。</p>
          </div>
        </div>

        <el-empty
          v-if="!toolCalls.length"
          :image-size="96"
          description="本次回答没有调用工具"
        />
        <div v-else class="agent-tool-call-list">
          <article
            v-for="(call, index) in toolCalls"
            :key="call.tool_call_id || `${call.name}-${index}`"
            class="agent-tool-call"
          >
            <div class="agent-tool-call__header">
              <strong>{{ toolLabel(call.name) }}</strong>
              <el-tag size="small" effect="plain">{{ call.tool_call_id || `#${index + 1}` }}</el-tag>
            </div>
            <dl class="agent-tool-call__details">
              <div>
                <dt>参数</dt>
                <dd><pre>{{ formatJson(call.arguments) }}</pre></dd>
              </div>
              <div>
                <dt>结果</dt>
                <dd>{{ toolResultSummary(call.result) }}</dd>
              </div>
            </dl>
          </article>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, reactive, ref } from "vue";
import { ElMessage } from "element-plus";
import { MagicStick, Refresh } from "@element-plus/icons-vue";

import { getAgentTask, resumeAgentTask, runAgentTask, streamAgentTaskEvents } from "../api/agent";
import { formatApiError } from "../api/http";
import type {
  AgentStepResult,
  AgentTaskEvent,
  AgentTaskRecordResponse,
} from "../api/types";
import { guardWarningLabel } from "../utils/legalDisplay";
import CitationList from "../components/CitationList.vue";
import EvidencePanel from "../components/EvidencePanel.vue";

const AGENT_CONVERSATION_STORAGE_KEY = "legal-doc-assistant.agentConversationId";

const focusAreaOptions = [
  { label: "付款", value: "payment" },
  { label: "终止", value: "termination" },
  { label: "责任限制", value: "liability limitation" },
  { label: "保密", value: "confidentiality" },
  { label: "数据隐私", value: "data privacy" },
  { label: "适用法律", value: "governing law" },
  { label: "补偿", value: "indemnification" },
  { label: "转让", value: "assignment" },
  { label: "审计权", value: "audit rights" },
];

const userRoleOptions = [
  { label: "普通用户", value: "ordinary" },
  { label: "法律专业", value: "lawyer" },
];

const form = reactive({
  objective: "",
  focusAreas: ["payment", "termination", "liability limitation"],
  userRole: "ordinary" as "ordinary" | "lawyer",
  maxSteps: 6,
});
const resumeForm = reactive({
  clarificationAnswers: "",
});
const loading = ref(false);
const task = ref<AgentTaskRecordResponse | null>(null);
const conversationId = ref(readConversationId());

interface AgentToolCallTrace {
  tool_call_id: string;
  name: string;
  arguments: Record<string, unknown>;
  result: Record<string, unknown>;
}

const result = computed(() => task.value?.result ?? null);
const toolCalls = computed(() => result.value?.steps.flatMap(stepToolCalls) ?? []);
const events = computed(() => task.value?.events ?? []);
const clarificationQuestions = computed(() => {
  if (task.value?.status !== "needs_input") {
    return [];
  }
  const needsInputEvent = [...events.value]
    .reverse()
    .find((event) => event.event_type === "needs_input");
  const questions = needsInputEvent?.payload?.questions;
  if (!Array.isArray(questions)) {
    return [];
  }
  return questions.map((question) => String(question)).filter(Boolean).slice(0, 3);
});
const progressStatus = computed(() => {
  if (!task.value) {
    return undefined;
  }
  if (task.value.status === "failed") {
    return "exception" as const;
  }
  if (task.value.status === "needs_input") {
    return "warning" as const;
  }
  if (task.value.status === "succeeded") {
    return "success" as const;
  }
  return undefined;
});

async function runTask() {
  if (!form.objective.trim() || loading.value) {
    return;
  }

  loading.value = true;
  try {
    task.value = await runAgentTask({
      objective: form.objective.trim(),
      focus_areas: form.focusAreas,
      user_role: form.userRole,
      max_steps: form.maxSteps,
      conversation_id: conversationId.value,
    });
    await streamCurrentTaskEvents();
  } catch (error) {
    await refreshCurrentTask();
    ElMessage.error(formatApiError(error));
  } finally {
    loading.value = false;
  }
}

async function resumeTask() {
  if (!task.value || task.value.status !== "needs_input" || loading.value) {
    return;
  }
  const answers = splitClarificationAnswers(resumeForm.clarificationAnswers);
  if (!answers.length) {
    return;
  }

  loading.value = true;
  const taskId = task.value.task_id;
  try {
    task.value = await resumeAgentTask(taskId, {
      clarification_answers: answers,
      focus_areas: form.focusAreas,
      user_role: form.userRole,
      max_steps: form.maxSteps,
      conversation_id: conversationId.value,
    });
    resumeForm.clarificationAnswers = "";
    if (task.value.status !== "needs_input") {
      await streamCurrentTaskEvents(latestEventId(task.value.events));
    }
    task.value = await getAgentTask(taskId);
  } catch (error) {
    await refreshCurrentTask();
    ElMessage.error(formatApiError(error));
  } finally {
    loading.value = false;
  }
}

async function streamCurrentTaskEvents(afterEventId = 0) {
  if (!task.value) {
    return;
  }
  const taskId = task.value.task_id;
  await streamAgentTaskEvents(
    taskId,
    {
      onEvent(event) {
        mergeEvent(event);
        applyEventProgress(event);
      },
      onError(message) {
        ElMessage.warning(message);
      },
    },
    afterEventId,
  );
  task.value = await getAgentTask(taskId);
}

async function refreshCurrentTask() {
  if (!task.value?.task_id) {
    return;
  }
  try {
    task.value = await getAgentTask(task.value.task_id);
  } catch {
    // Keep the existing local task state if status refresh also fails.
  }
}

function mergeEvent(event: AgentTaskEvent) {
  if (!task.value) {
    return;
  }
  const existing = task.value.events.findIndex((item) => item.event_id === event.event_id);
  if (existing >= 0) {
    task.value.events[existing] = event;
  } else {
    task.value.events.push(event);
  }
}

function applyEventProgress(event: AgentTaskEvent) {
  if (!task.value) {
    return;
  }
  task.value.stage = event.stage;
  task.value.progress = event.progress;
  if (event.event_type === "running") {
    task.value.status = "running";
  } else if (event.event_type === "queued" || event.event_type === "input_received") {
    task.value.status = "queued";
  } else if (event.event_type === "needs_input") {
    task.value.status = "needs_input";
  } else if (event.event_type === "failed") {
    task.value.status = "failed";
    task.value.error = event.message;
  } else if (event.event_type === "succeeded") {
    task.value.status = "succeeded";
    task.value.progress = 100;
  }
}

function resetForm() {
  task.value = null;
  form.objective = "";
  form.focusAreas = ["payment", "termination", "liability limitation"];
  form.userRole = "ordinary";
  form.maxSteps = 6;
  resumeForm.clarificationAnswers = "";
  conversationId.value = createConversationId();
}

function statusLabel(status: string) {
  const labels: Record<string, string> = {
    queued: "排队中",
    running: "运行中",
    needs_input: "需补充信息",
    succeeded: "已完成",
    failed: "失败",
    completed: "已完成",
    needs_review: "需复核",
    needs_human_review: "需人工复核",
  };
  return labels[status] ?? status;
}

function stageLabel(stage: string) {
  const labels: Record<string, string> = {
    queued: "任务已排队",
    answering: "正在运行 ReAct 工具调用",
    reporting: "正在生成最终报告",
    needs_input: "等待补充任务上下文",
    completed: "任务已完成",
    failed: "任务失败",
  };
  return labels[stage] ?? stage;
}

function taskStatusType(status: string) {
  if (status === "succeeded") {
    return "success";
  }
  if (status === "queued" || status === "running" || status === "needs_input") {
    return "warning";
  }
  return "danger";
}

function eventType(eventTypeValue: string) {
  if (eventTypeValue === "failed") {
    return "danger";
  }
  if (eventTypeValue === "succeeded") {
    return "success";
  }
  if (eventTypeValue === "input_received") {
    return "primary";
  }
  if (eventTypeValue === "needs_input") {
    return "warning";
  }
  if (eventTypeValue === "step_completed") {
    return "primary";
  }
  return "info";
}

function eventLabel(eventTypeValue: string) {
  const labels: Record<string, string> = {
    queued: "排队",
    running: "开始",
    input_received: "已补充",
    needs_input: "需补充",
    react_started: "ReAct",
    react_completed: "完成",
    succeeded: "完成",
    failed: "失败",
  };
  return labels[eventTypeValue] ?? eventTypeValue;
}

function toolLabel(tool: string) {
  const labels: Record<string, string> = {
    tool_calling_react: "ReAct 回答",
    search_documents: "文档检索",
    document_qa: "文档问答",
    review_clause: "条款审查",
    check_conflict: "冲突检查",
    web_search: "网页搜索",
  };
  return labels[tool] ?? tool;
}

function stepToolCalls(step: AgentStepResult): AgentToolCallTrace[] {
  const calls = step.output.tool_calls;
  if (!Array.isArray(calls)) {
    return [];
  }
  return calls.filter(isRecord).map((call) => ({
    tool_call_id: stringValue(call, "tool_call_id"),
    name: stringValue(call, "name") || "tool",
    arguments: recordValue(call.arguments),
    result: recordValue(call.result),
  }));
}

function formatJson(value: unknown) {
  try {
    return JSON.stringify(value ?? {}, null, 2);
  } catch {
    return String(value ?? "");
  }
}

function toolResultSummary(result: Record<string, unknown>) {
  const error = stringValue(result, "error");
  if (error) {
    return `Error: ${error}`;
  }
  const content = stringValue(result, "content");
  if (content) {
    return shortText(content);
  }
  const metadata = recordValue(result.metadata);
  const citationCount = metadata.citation_count;
  if (typeof citationCount === "number") {
    return `Citations: ${citationCount}`;
  }
  return shortText(formatJson(result));
}

function shortText(value: string) {
  return value.length > 280 ? `${value.slice(0, 277)}...` : value;
}

function stringValue(item: Record<string, unknown>, key: string) {
  const value = item[key];
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return "";
}

function recordValue(value: unknown): Record<string, unknown> {
  return isRecord(value) ? value : {};
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));
}

function splitClarificationAnswers(value: string) {
  return value
    .split(/\r?\n/)
    .map((line) => line.replace(/^[-*•\d.、\s]+/, "").trim())
    .filter(Boolean)
    .slice(0, 6);
}

function latestEventId(items: AgentTaskEvent[]) {
  return items.reduce((latest, event) => Math.max(latest, event.event_id), 0);
}

function readConversationId(): string {
  const stored = localStorage.getItem(AGENT_CONVERSATION_STORAGE_KEY);
  if (stored) {
    return stored;
  }
  return createConversationId();
}

function createConversationId(): string {
  const id = crypto.randomUUID();
  localStorage.setItem(AGENT_CONVERSATION_STORAGE_KEY, id);
  return id;
}
</script>
