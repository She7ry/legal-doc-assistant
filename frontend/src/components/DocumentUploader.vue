<template>
  <section class="tool-panel">
    <div class="panel-heading">
      <div>
        <h2>上传文档</h2>
        <p>PDF、DOCX、TXT、Markdown</p>
      </div>
    </div>

    <el-upload
      ref="uploadRef"
      class="upload-drop"
      drag
      :auto-upload="false"
      :limit="1"
      accept=".pdf,.docx,.txt,.md,.markdown"
      :on-change="handleFileChange"
      :on-remove="handleFileRemove"
    >
      <el-icon class="upload-drop__icon"><UploadFilled /></el-icon>
      <div class="el-upload__text">拖拽文件到此处，或点击选择</div>
      <template #tip>
        <div class="el-upload__tip">支持 PDF、DOCX、TXT、MD、Markdown；单文件上传。</div>
      </template>
    </el-upload>

    <div class="panel-actions">
      <el-button :disabled="!selectedFile || uploading" @click="clearFile">清空</el-button>
      <el-button
        type="primary"
        :icon="UploadFilled"
        :loading="uploading"
        :disabled="!selectedFile"
        @click="submit"
      >
        提交索引
      </el-button>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref } from "vue";
import { ElMessage, type UploadFile, type UploadInstance } from "element-plus";
import { UploadFilled } from "@element-plus/icons-vue";

import { ingestDocument } from "../api/documents";
import { formatApiError } from "../api/http";
import type { IngestJobResponse } from "../api/types";

const emit = defineEmits<{
  "job-created": [job: IngestJobResponse];
}>();

const uploadRef = ref<UploadInstance>();
const selectedFile = ref<File | null>(null);
const uploading = ref(false);
const supportedExtensions = new Set(["pdf", "docx", "txt", "md", "markdown"]);

function handleFileChange(uploadFile: UploadFile) {
  const file = uploadFile.raw ?? null;
  if (file && !isSupportedFile(file)) {
    ElMessage.error("暂不支持该文件类型，请选择 PDF、DOCX、TXT 或 Markdown 文件。");
    clearFile();
    return;
  }
  selectedFile.value = file;
}

function handleFileRemove() {
  selectedFile.value = null;
}

function clearFile() {
  uploadRef.value?.clearFiles();
  selectedFile.value = null;
}

function isSupportedFile(file: File) {
  const extension = file.name.split(".").pop()?.toLowerCase();
  return extension ? supportedExtensions.has(extension) : false;
}

async function submit() {
  if (!selectedFile.value) {
    return;
  }

  uploading.value = true;
  try {
    const job = await ingestDocument(selectedFile.value);
    emit("job-created", job);
    ElMessage.success("已提交索引任务");
    clearFile();
  } catch (error) {
    ElMessage.error(formatApiError(error));
  } finally {
    uploading.value = false;
  }
}
</script>
