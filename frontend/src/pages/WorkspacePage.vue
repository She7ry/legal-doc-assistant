<template>
  <div class="page-stack">
    <section class="summary-strip">
      <div class="metric">
        <span>已索引文档</span>
        <strong>{{ documentTotal }}</strong>
      </div>
      <div class="metric">
        <span>当前租户</span>
        <strong>{{ settings.tenantId }}</strong>
      </div>
      <div class="metric">
        <span>API 地址</span>
        <strong>{{ settings.displayBaseUrl }}</strong>
      </div>
    </section>

    <div class="workspace-grid">
      <ChatPanel />

      <aside class="side-stack">
        <MemoryPanel />
        <DocumentUploader @job-created="handleJobCreated" />
        <IngestJobStatus
          :job="activeJob"
          @updated="activeJob = $event"
          @completed="handleJobCompleted"
        />
      </aside>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from "vue";
import { ElMessage } from "element-plus";

import { listDocuments } from "../api/documents";
import { formatApiError } from "../api/http";
import type { IngestJobResponse } from "../api/types";
import ChatPanel from "../components/ChatPanel.vue";
import DocumentUploader from "../components/DocumentUploader.vue";
import IngestJobStatus from "../components/IngestJobStatus.vue";
import MemoryPanel from "../components/MemoryPanel.vue";
import { useSettingsStore } from "../stores/settings";

const settings = useSettingsStore();
const documentTotal = ref(0);
const activeJob = ref<IngestJobResponse | null>(null);

onMounted(() => {
  void refreshDocuments();
});

function handleJobCreated(job: IngestJobResponse) {
  activeJob.value = job;
}

function handleJobCompleted(job: IngestJobResponse) {
  activeJob.value = job;
  if (job.status === "succeeded") {
    void refreshDocuments();
  }
}

async function refreshDocuments() {
  try {
    const response = await listDocuments();
    documentTotal.value = response.total;
  } catch (error) {
    ElMessage.warning(formatApiError(error));
  }
}
</script>
