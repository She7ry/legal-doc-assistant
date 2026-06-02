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
      </el-form>
    </section>
  </div>
</template>

<script setup lang="ts">
import { reactive, ref } from "vue";
import { ElMessage } from "element-plus";
import { Connection, Refresh } from "@element-plus/icons-vue";

import { checkHealth } from "../api/health";
import { formatApiError } from "../api/http";
import { useSettingsStore } from "../stores/settings";

const settings = useSettingsStore();
const checking = ref(false);
const form = reactive({
  apiBaseUrl: settings.apiBaseUrl,
  apiKey: settings.apiKey,
  tenantId: settings.tenantId,
  userId: settings.userId,
});

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
  try {
    const response = await checkHealth();
    ElMessage.success(`连接正常：${response.status}`);
  } catch (error) {
    ElMessage.error(formatApiError(error));
  } finally {
    checking.value = false;
  }
}
</script>
