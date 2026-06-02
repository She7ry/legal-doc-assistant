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
import { computed } from "vue";
import { useRoute } from "vue-router";
import { Document, Files, Monitor, Setting } from "@element-plus/icons-vue";

import { useSettingsStore } from "../stores/settings";

const route = useRoute();
const settings = useSettingsStore();

const routeMeta = computed(() => {
  switch (route.path) {
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
</script>
