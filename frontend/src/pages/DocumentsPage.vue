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
          <el-table-column label="操作" width="110" fixed="right">
            <template #default="{ row }">
              <el-button :icon="View" size="small" plain @click="openDocumentPreview(row)">
                预览
              </el-button>
            </template>
          </el-table-column>
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

    <el-drawer
      v-model="previewVisible"
      title="文档预览"
      size="min(760px, 92vw)"
      class="document-preview-drawer"
      @opened="scrollToHighlightedChunk"
    >
      <div v-loading="previewLoading" class="document-preview">
        <el-empty
          v-if="!preview && !previewLoading"
          :image-size="96"
          description="暂无可预览内容"
        />

        <template v-else-if="preview">
          <header class="document-preview__header">
            <div>
              <h3>{{ preview.document.file_name }}</h3>
              <p>{{ preview.document.document_key }}</p>
            </div>
            <div class="document-preview__tags">
              <el-tag effect="plain">v{{ preview.document.document_version }}</el-tag>
              <el-tag effect="plain">{{ preview.total_chunks }} chunks</el-tag>
              <el-tag v-if="preview.document.page_count !== null" effect="plain">
                {{ preview.document.page_count }} 页
              </el-tag>
            </div>
          </header>

          <el-alert
            v-if="highlightedChunkId !== null"
            type="info"
            :title="`已定位到 chunk ${highlightedChunkId}`"
            :closable="false"
            show-icon
          />

          <div class="document-preview__chunks">
            <article
              v-for="(chunk, index) in preview.chunks"
              :id="chunkDomId(chunk)"
              :key="chunkKey(chunk, index)"
              class="document-preview-chunk"
              :class="{ 'document-preview-chunk--highlighted': isHighlightedChunk(chunk) }"
            >
              <div class="document-preview-chunk__meta">
                <el-tag size="small" type="primary" effect="dark">
                  {{ chunk.chunk_id !== null ? `Chunk ${chunk.chunk_id}` : `Part ${index + 1}` }}
                </el-tag>
                <span v-if="chunk.page_label">{{ chunk.page_label }}</span>
                <span v-if="chunk.section_heading">{{ chunk.section_heading }}</span>
              </div>
              <p>{{ chunk.text }}</p>
            </article>
          </div>
        </template>
      </div>
    </el-drawer>
  </div>
</template>

<script setup lang="ts">
import { nextTick, onMounted, ref, watch } from "vue";
import { useRoute } from "vue-router";
import { ElMessage } from "element-plus";
import { Refresh, View } from "@element-plus/icons-vue";

import { getDocumentText, listDocuments } from "../api/documents";
import { formatApiError } from "../api/http";
import type {
  DocumentInfo,
  DocumentTextChunk,
  DocumentTextResponse,
  IngestJobResponse,
} from "../api/types";
import DocumentUploader from "../components/DocumentUploader.vue";
import IngestJobStatus from "../components/IngestJobStatus.vue";

const route = useRoute();
const documents = ref<DocumentInfo[]>([]);
const loading = ref(false);
const activeJob = ref<IngestJobResponse | null>(null);
const preview = ref<DocumentTextResponse | null>(null);
const previewVisible = ref(false);
const previewLoading = ref(false);
const highlightedChunkId = ref<number | null>(null);

onMounted(async () => {
  await loadDocuments();
  await openPreviewFromRoute();
});

watch(
  () => route.query,
  () => {
    void openPreviewFromRoute();
  },
);

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

async function openDocumentPreview(
  target: {
    document_key?: string | null;
    file_id?: string | null;
    document_version?: number | null;
  },
  chunkId: number | null = null,
) {
  const documentKey = target.document_key?.trim() || null;
  const fileId = target.file_id?.trim() || null;
  if (!documentKey && !fileId) {
    ElMessage.warning("该文档缺少可预览的定位信息。");
    return;
  }

  highlightedChunkId.value = chunkId;
  previewVisible.value = true;
  previewLoading.value = true;
  try {
    preview.value = await getDocumentText({
      document_key: documentKey,
      file_id: fileId,
      document_version: target.document_version ?? null,
    });
    await scrollToHighlightedChunk();
  } catch (error) {
    preview.value = null;
    ElMessage.error(formatApiError(error));
  } finally {
    previewLoading.value = false;
  }
}

async function openPreviewFromRoute() {
  const documentKey = queryString(route.query.document_key);
  const fileId = queryString(route.query.file_id);
  if (!documentKey && !fileId) {
    return;
  }
  const documentVersion = parseOptionalInt(queryString(route.query.document_version));
  const chunkId = parseOptionalInt(queryString(route.query.chunk_id));
  await openDocumentPreview(
    {
      document_key: documentKey,
      file_id: fileId,
      document_version: documentVersion,
    },
    chunkId,
  );
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

function queryString(value: unknown): string | null {
  if (Array.isArray(value)) {
    return typeof value[0] === "string" ? value[0] : null;
  }
  return typeof value === "string" ? value : null;
}

function parseOptionalInt(value: string | null): number | null {
  if (!value) {
    return null;
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : null;
}

function chunkKey(chunk: DocumentTextChunk, index: number) {
  return chunk.chunk_id !== null ? `chunk-${chunk.chunk_id}` : `part-${index}`;
}

function chunkDomId(chunk: DocumentTextChunk) {
  return chunk.chunk_id !== null ? chunkDomIdForId(chunk.chunk_id) : undefined;
}

function chunkDomIdForId(chunkId: number) {
  return `document-preview-chunk-${chunkId}`;
}

function isHighlightedChunk(chunk: DocumentTextChunk) {
  return chunk.chunk_id !== null && chunk.chunk_id === highlightedChunkId.value;
}

async function scrollToHighlightedChunk() {
  await nextTick();
  if (highlightedChunkId.value === null) {
    return;
  }
  const element = document.getElementById(chunkDomIdForId(highlightedChunkId.value));
  element?.scrollIntoView({ block: "center" });
}
</script>
