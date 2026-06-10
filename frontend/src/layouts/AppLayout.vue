<template>
  <el-container class="app-shell">
    <el-aside class="app-sidebar" width="264px">
      <div class="brand">
        <div class="brand-mark">LD</div>
        <div class="brand-copy">
          <strong>Legal Assistant</strong>
          <span>企业文档审查工作台</span>
        </div>
      </div>

      <el-menu class="nav-menu" :default-active="route.path" router>
        <el-menu-item index="/">
          <el-icon><Monitor /></el-icon>
          <span>工作台</span>
        </el-menu-item>
        <el-menu-item index="/agent">
          <el-icon><MagicStick /></el-icon>
          <span>Agent 审查</span>
        </el-menu-item>
        <el-menu-item index="/matters">
          <el-icon><Collection /></el-icon>
          <span>事项库</span>
        </el-menu-item>
        <el-menu-item index="/documents">
          <el-icon><Files /></el-icon>
          <span>文档库</span>
        </el-menu-item>
        <el-menu-item index="/review">
          <el-icon><Document /></el-icon>
          <span>审查工具</span>
        </el-menu-item>
        <el-menu-item index="/settings">
          <el-icon><Setting /></el-icon>
          <span>系统设置</span>
        </el-menu-item>
      </el-menu>
    </el-aside>

    <el-container class="app-content">
      <el-header class="app-header" height="64px">
        <div>
          <h1>{{ pageTitle }}</h1>
          <p>{{ pageSubtitle }}</p>
        </div>
        <div class="runtime-state">
          <el-tag effect="plain" :type="healthTagType">
            <el-icon v-if="healthState === 'ok'"><CircleCheck /></el-icon>
            <el-icon v-else-if="healthState === 'degraded'"><Warning /></el-icon>
            <el-icon v-else-if="healthState === 'error'"><CircleClose /></el-icon>
            {{ healthLabel }}
          </el-tag>
          <el-button
            :icon="Refresh"
            circle
            size="small"
            :loading="healthLoading"
            aria-label="刷新服务状态"
            @click="refreshHealth"
          />
          <el-tag effect="plain" type="info">Tenant: {{ settings.tenantId }}</el-tag>
          <el-tag effect="plain" type="info">User: {{ settings.userId }}</el-tag>
          <el-tag effect="plain" :type="settings.hasApiKey ? 'success' : 'warning'">
            {{ settings.hasApiKey ? "API Key 已配置" : "本地免密或未配置" }}
          </el-tag>
        </div>
      </el-header>

      <el-main class="app-main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { useRoute } from "vue-router";
import {
  CircleCheck,
  CircleClose,
  Collection,
  Document,
  Files,
  MagicStick,
  Monitor,
  Refresh,
  Setting,
  Warning,
} from "@element-plus/icons-vue";

import { checkHealth } from "../api/health";
import { formatApiError } from "../api/http";
import type { HealthResponse } from "../api/types";
import { useSettingsStore } from "../stores/settings";

const route = useRoute();
const settings = useSettingsStore();
const health = ref<HealthResponse | null>(null);
const healthError = ref("");
const healthLoading = ref(false);

const routeMeta = computed(() => {
  switch (route.path) {
    case "/agent":
      return {
        title: "Agent 审查",
        subtitle: "拆解法律文档任务，跟踪工具执行、证据和人工复核点",
      };
    case "/matters":
      return {
        title: "事项库",
        subtitle: "查看 Matter Profile、结构化交付物和待确认信息",
      };
    case "/documents":
      return {
        title: "文档库",
        subtitle: "管理已索引文件和上传任务",
      };
    case "/review":
      return {
        title: "审查工具",
        subtitle: "执行条款风险审查和冲突检测",
      };
    case "/settings":
      return {
        title: "系统设置",
        subtitle: "配置 API 地址、租户和访问密钥",
      };
    default:
      return {
        title: "工作台",
        subtitle: "上传文档、跟踪索引状态并进行引用式问答",
      };
  }
});

const pageTitle = computed(() => routeMeta.value.title);
const pageSubtitle = computed(() => routeMeta.value.subtitle);

const healthState = computed(() => {
  if (healthError.value) {
    return "error";
  }
  if (!health.value) {
    return "checking";
  }
  return health.value?.status === "ok" ? "ok" : "degraded";
});

const healthLabel = computed(() => {
  if (healthError.value) {
    return "服务离线";
  }
  if (!health.value) {
    return "检查中";
  }
  if (health.value.status === "ok") {
    return `服务正常 v${health.value.version}`;
  }
  const warningCount = health.value.checks.filter((check) => check.status !== "ok").length;
  return `配置告警 ${warningCount}`;
});

const healthTagType = computed(() => {
  if (healthState.value === "ok") {
    return "success";
  }
  if (healthState.value === "degraded") {
    return "warning";
  }
  if (healthState.value === "checking") {
    return "info";
  }
  return "danger";
});

onMounted(() => {
  void refreshHealth();
});

async function refreshHealth() {
  healthLoading.value = true;
  healthError.value = "";
  try {
    health.value = await checkHealth();
  } catch (error) {
    health.value = null;
    healthError.value = formatApiError(error);
  } finally {
    healthLoading.value = false;
  }
}
</script>
