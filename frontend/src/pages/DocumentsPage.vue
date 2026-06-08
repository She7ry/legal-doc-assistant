<template>
  <div class="page-stack">
    <div class="documents-grid">
      <section class="tool-panel document-list-panel">
        <div class="panel-heading">
          <div>
            <h2>已索引文档</h2>
            <p>{{ documents.length }} 个文件</p>
          </div>
          <el-button :icon="Refresh" :loading="loading" @click="loadDocuments">刷新</el-button>
        </div>

        <el-table
          v-loading="loading"
          :data="documents"
          row-key="file_id"
          empty-text="暂无已索引文档"
          class="document-table"
        >
          <el-table-column prop="file_name" label="文件名" min-width="260" />
          <el-table-column label="版本" width="90">
            <template #default="{ row }">v{{ row.document_version }}</template>
          </el-table-column>
          <el-table-column label="类型" width="90">
            <template #default="{ row }">{{ formatExtension(row.file_extension) }}</template>
          </el-table-column>
          <el-table-column prop="chunk_count" label="分块" width="90" />
          <el-table-column label="页数" width="90">
            <template #default="{ row }">{{ row.page_count ?? "-" }}</template>
          </el-table-column>
          <el-table-column label="警告" width="90">
            <template #default="{ row }">{{ row.warning_count || "-" }}</template>
          </el-table-column>
          <el-table-column label="索引时间" min-width="170">
            <template #default="{ row }">{{ formatIndexedAt(row.indexed_at) }}</template>
          </el-table-column>
          <el-table-column prop="file_id" label="File ID" min-width="320" />
        </el-table>
      </section>

      <aside class="side-stack">
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
import { Refresh } from "@element-plus/icons-vue";

import { listDocuments } from "../api/documents";
import { formatApiError } from "../api/http";
import type { DocumentInfo, IngestJobResponse } from "../api/types";
import DocumentUploader from "../components/DocumentUploader.vue";
import IngestJobStatus from "../components/IngestJobStatus.vue";

const documents = ref<DocumentInfo[]>([]);
const loading = ref(false);
const activeJob = ref<IngestJobResponse | null>(null);

onMounted(() => {
  void loadDocuments();
});

async function loadDocuments() {
  loading.value = true;
  try {
    const response = await listDocuments();
    documents.value = response.documents;
  } catch (error) {
    ElMessage.error(formatApiError(error));
  } finally {
    loading.value = false;
  }
}

function handleJobCreated(job: IngestJobResponse) {
  activeJob.value = job;
}

function handleJobCompleted(job: IngestJobResponse) {
  activeJob.value = job;
  if (job.status === "succeeded") {
    void loadDocuments();
  }
}

function formatExtension(value: string) {
  return value ? value.replace(".", "").toUpperCase() : "-";
}

function formatIndexedAt(value: string | null) {
  if (!value) {
    return "-";
  }
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}
</script>
