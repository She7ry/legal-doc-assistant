<template>
  <main class="auth-page">
    <section class="tool-panel auth-panel">
      <div class="auth-heading">
        <div class="brand-mark">LD</div>
        <div>
          <h1>Legal Assistant</h1>
          <p>登录后仅检索你自己上传的文档</p>
        </div>
      </div>

      <el-tabs v-model="mode" stretch>
        <el-tab-pane label="登录" name="login" />
        <el-tab-pane label="注册" name="register" />
      </el-tabs>

      <el-form label-position="top" @submit.prevent="submit">
        <el-form-item label="用户名">
          <el-input v-model="username" maxlength="64" autocomplete="username" />
        </el-form-item>
        <el-form-item label="密码">
          <el-input
            v-model="password"
            type="password"
            show-password
            maxlength="128"
            :autocomplete="mode === 'login' ? 'current-password' : 'new-password'"
            @keyup.enter="submit"
          />
        </el-form-item>
        <el-form-item v-if="mode === 'register'" label="确认密码">
          <el-input
            v-model="passwordConfirmation"
            type="password"
            show-password
            maxlength="128"
            autocomplete="new-password"
            @keyup.enter="submit"
          />
        </el-form-item>
        <el-button type="primary" class="auth-submit" :loading="submitting" @click="submit">
          {{ mode === "login" ? "登录" : "创建账号" }}
        </el-button>
      </el-form>
    </section>
  </main>
</template>

<script setup lang="ts">
import { ref } from "vue";
import { ElMessage } from "element-plus";
import { useRoute, useRouter } from "vue-router";

import { formatApiError } from "../api/http";
import { useAuthStore } from "../stores/auth";

const auth = useAuthStore();
const route = useRoute();
const router = useRouter();
const mode = ref<"login" | "register">("login");
const username = ref("");
const password = ref("");
const passwordConfirmation = ref("");
const submitting = ref(false);

async function submit() {
  const normalizedUsername = username.value.trim();
  if (!normalizedUsername || password.value.length < 8) {
    ElMessage.warning("请输入用户名和至少 8 位密码");
    return;
  }
  if (mode.value === "register" && password.value !== passwordConfirmation.value) {
    ElMessage.warning("两次输入的密码不一致");
    return;
  }

  submitting.value = true;
  try {
    const credentials = { username: normalizedUsername, password: password.value };
    if (mode.value === "login") {
      await auth.login(credentials);
    } else {
      await auth.register(credentials);
    }
    const redirect = typeof route.query.redirect === "string" ? route.query.redirect : "/";
    await router.replace(redirect.startsWith("/") ? redirect : "/");
  } catch (error) {
    ElMessage.error(formatApiError(error));
  } finally {
    submitting.value = false;
  }
}
</script>

<style scoped>
.auth-page {
  min-height: 100vh;
  display: grid;
  place-items: center;
  padding: 24px;
  background: var(--app-background, #f5f7fa);
}

.auth-panel {
  width: min(420px, 100%);
}

.auth-heading {
  display: flex;
  gap: 14px;
  align-items: center;
  margin-bottom: 20px;
}

.auth-heading h1,
.auth-heading p {
  margin: 0;
}

.auth-heading p {
  margin-top: 4px;
  color: var(--el-text-color-secondary);
}

.auth-submit {
  width: 100%;
}
</style>
