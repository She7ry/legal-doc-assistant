<template>
  <section class="tool-panel job-panel">
    <div class="panel-heading">
      <div>
        <h2>索引任务</h2>
        <p>{{ current ? current.file_name : "暂无运行中的任务" }}</p>
      </div>
      <StatusBadge v-if="current" :status="current.status" />
    </div>

    <el-empty v-if="!current" :image-size="88" description="上传文档后显示任务状态" />

    <div v-else class="job-state">
      <el-progress :percentage="progress" :status="progressStatus" :stroke-width="10" />
      <p class="job-stage">{{ stageLabel }}</p>

      <dl class="job-meta">
        <div>
          <dt>Job ID</dt>
          <dd>{{ current.job_id }}</dd>
        </div>
        <div>
          <dt>提交时间</dt>
          <dd>{{ formatDate(current.submitted_at) }}</dd>
        </div>
        <div v-if="current.completed_at">
          <dt>完成时间</dt>
          <dd>{{ formatDate(current.completed_at) }}</dd>
        </div>
        <div>
          <dt>进度</dt>
          <dd>{{ current.progress }}%</dd>
        </div>
      </dl>

      <el-alert
        v-if="current.error"
        type="error"
        :title="current.error"
        :closable="false"
        show-icon
      />

      <el-alert
        v-if="warningText"
        type="warning"
        :title="warningText"
        :closable="false"
        show-icon
      />

      <div v-if="current.result" class="job-result">
        <el-statistic title="版本" :value="current.result.document_version" />
        <el-statistic title="文档数" :value="current.result.document_count" />
        <el-statistic title="分块数" :value="current.result.chunk_count" />
        <el-statistic v-if="current.result.page_count !== null" title="页数" :value="current.result.page_count" />
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { ElMessage } from "element-plus";

import { getIngestJob } from "../api/documents";
import { formatApiError } from "../api/http";
import type { IngestJobResponse } from "../api/types";
import StatusBadge from "./StatusBadge.vue";

const props = defineProps<{
  job: IngestJobResponse | null;
}>();

const emit = defineEmits<{
  updated: [job: IngestJobResponse];
  completed: [job: IngestJobResponse];
}>();

const current = ref<IngestJobResponse | null>(null);
const polling = ref(false);
let timer: number | undefined;

const progress = computed(() => {
  if (!current.value) {
    return 0;
  }
  return Math.max(0, Math.min(current.value.progress ?? 0, 100));
});

const stageLabel = computed(() => {
  if (!current.value) {
    return "";
  }
  const labels: Record<string, string> = {
    uploaded: "文件已接收",
    parsing: "正在解析文档",
    chunking: "正在按条款分块",
    embedding: "正在生成向量",
    indexing: "正在写入索引",
    completed: "索引完成",
    failed: "索引失败",
  };
  return labels[current.value.stage] ?? current.value.stage;
});

const warningText = computed(() => {
  if (!current.value?.warnings?.length) {
    return "";
  }
  return current.value.warnings.join("；");
});

const progressStatus = computed(() => {
  if (!current.value) {
    return undefined;
  }
  if (current.value.status === "failed") {
    return "exception" as const;
  }
  if (current.value.status === "succeeded") {
    return "success" as const;
  }
  return undefined;
});

watch(
  () => props.job,
  (job) => {
    current.value = job;
    resetPolling();
  },
  { immediate: true },
);

onBeforeUnmount(() => {
  clearPolling();
});

function resetPolling() {
  clearPolling();
  if (!current.value || isTerminal(current.value)) {
    return;
  }

  timer = window.setInterval(poll, 1600);
  void poll();
}

function clearPolling() {
  if (timer !== undefined) {
    window.clearInterval(timer);
    timer = undefined;
  }
}

async function poll() {
  if (!current.value || polling.value) {
    return;
  }

  polling.value = true;
  try {
    const next = await getIngestJob(current.value.job_id);
    current.value = next;
    emit("updated", next);
    if (isTerminal(next)) {
      clearPolling();
      emit("completed", next);
    }
  } catch (error) {
    clearPolling();
    ElMessage.error(formatApiError(error));
  } finally {
    polling.value = false;
  }
}

function isTerminal(job: IngestJobResponse) {
  return job.status === "succeeded" || job.status === "failed";
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));
}
</script>
