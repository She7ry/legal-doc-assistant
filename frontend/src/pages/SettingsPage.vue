<template>
  <div class="page-stack settings-page">
    <section class="tool-panel settings-panel">
      <div class="panel-heading">
        <div>
          <h2>连接配置</h2>
          <p>当前配置保存在本机浏览器</p>
        </div>
      </div>

      <el-form label-position="top" class="settings-form" @submit.prevent>
        <el-form-item label="API Base URL">
          <el-input v-model="form.apiBaseUrl" placeholder="http://localhost:8000" />
        </el-form-item>

        <el-form-item label="X-API-Key">
          <el-input
            v-model="form.apiKey"
            type="password"
            show-password
            placeholder="后端未启用 DOC_ASSISTANT_API_KEYS 时可留空"
          />
        </el-form-item>

        <el-form-item label="X-Tenant-Id">
          <el-input v-model="form.tenantId" placeholder="default" />
        </el-form-item>

        <el-form-item label="X-User-Id">
          <el-input v-model="form.userId" placeholder="local-user" />
        </el-form-item>

        <div class="panel-actions panel-actions--left">
          <el-button :icon="Refresh" @click="reset">恢复默认</el-button>
          <el-button :icon="Connection" :loading="checking" @click="testConnection">
            测试连接
          </el-button>
          <el-button type="primary" @click="save">保存配置</el-button>
        </div>

        <el-alert
          v-if="connectionResult"
          class="connection-result"
          :type="connectionAlertType"
          :title="connectionTitle"
          :closable="false"
          show-icon
        >
          <div class="connection-summary">
            <span>{{ chatProviderLabel }}</span>
            <span>{{ embeddingProviderLabel }}</span>
            <span>上传上限 {{ uploadLimitLabel }}</span>
          </div>
          <dl class="health-check-list">
            <div v-for="check in connectionResult.checks" :key="check.name">
              <dt>{{ check.name }}</dt>
              <dd>
                <el-tag size="small" :type="check.status === 'ok' ? 'success' : 'warning'">
                  {{ check.status === "ok" ? "正常" : "告警" }}
                </el-tag>
                <span>{{ check.detail }}</span>
              </dd>
            </div>
          </dl>
        </el-alert>
      </el-form>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, reactive, ref } from "vue";
import { ElMessage } from "element-plus";
import { Connection, Refresh } from "@element-plus/icons-vue";

import { checkHealth } from "../api/health";
import { formatApiError } from "../api/http";
import type { HealthResponse } from "../api/types";
import { useSettingsStore } from "../stores/settings";

const settings = useSettingsStore();
const checking = ref(false);
const connectionResult = ref<HealthResponse | null>(null);
const form = reactive({
  apiBaseUrl: settings.apiBaseUrl,
  apiKey: settings.apiKey,
  tenantId: settings.tenantId,
  userId: settings.userId,
});

const connectionAlertType = computed(() =>
  connectionResult.value?.status === "ok" ? "success" : "warning",
);

const connectionTitle = computed(() => {
  if (!connectionResult.value) {
    return "";
  }
  return connectionResult.value.status === "ok"
    ? `连接正常：API v${connectionResult.value.version}`
    : "连接可用，但运行配置存在告警";
});

const chatProviderLabel = computed(() => {
  const chat = connectionResult.value?.providers.chat;
  if (!chat) {
    return "Chat provider 未知";
  }
  return `Chat ${chat.provider ?? "-"} / ${chat.model ?? "-"}`;
});

const embeddingProviderLabel = computed(() => {
  const embedding = connectionResult.value?.providers.embedding;
  if (!embedding) {
    return "Embedding provider 未知";
  }
  return `Embedding ${embedding.provider ?? "-"} / ${embedding.model ?? "-"}`;
});

const uploadLimitLabel = computed(() => formatBytes(connectionResult.value?.limits.max_upload_bytes));

function save() {
  settings.save(form);
  ElMessage.success("配置已保存");
}

function reset() {
  settings.reset();
  form.apiBaseUrl = settings.apiBaseUrl;
  form.apiKey = settings.apiKey;
  form.tenantId = settings.tenantId;
  form.userId = settings.userId;
  ElMessage.success("已恢复默认配置");
}

async function testConnection() {
  settings.save(form);
  checking.value = true;
  connectionResult.value = null;
  try {
    const response = await checkHealth();
    connectionResult.value = response;
    if (response.status === "ok") {
      ElMessage.success(`连接正常：${response.status}`);
    } else {
      ElMessage.warning("连接可用，但后端配置存在告警");
    }
  } catch (error) {
    ElMessage.error(formatApiError(error));
  } finally {
    checking.value = false;
  }
}

function formatBytes(value: number | undefined) {
  if (!value || value <= 0) {
    return "-";
  }
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(size >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}
</script>
