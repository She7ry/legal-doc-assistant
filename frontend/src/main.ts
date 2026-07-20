import { createApp } from "vue";
import { createPinia } from "pinia";
import {
  ElAlert,
  ElAside,
  ElButton,
  ElContainer,
  ElDialog,
  ElDrawer,
  ElEmpty,
  ElForm,
  ElFormItem,
  ElHeader,
  ElIcon,
  ElInput,
  ElInputNumber,
  ElLoading,
  ElMain,
  ElMenu,
  ElMenuItem,
  ElOption,
  ElProgress,
  ElRadioButton,
  ElRadioGroup,
  ElSelect,
  ElStatistic,
  ElTabPane,
  ElTable,
  ElTableColumn,
  ElTabs,
  ElTag,
  ElUpload,
} from "element-plus";
import "element-plus/dist/index.css";

import App from "./App.vue";
import { router } from "./app/router";
import { useAuthStore } from "./stores/auth";
import "./styles/main.css";

const app = createApp(App);
const pinia = createPinia();

app.use(pinia);
app.use(router);
router.beforeEach(async (to) => {
  const auth = useAuthStore(pinia);
  if (!auth.initialized) {
    try {
      await auth.loadCurrentUser();
    } catch {
      auth.user = null;
    }
  }
  if (!auth.user && to.name !== "login") {
    return { name: "login", query: { redirect: to.fullPath } };
  }
  if (auth.user && to.name === "login") {
    return { name: "workspace" };
  }
  return true;
});
[
  ElAlert,
  ElAside,
  ElButton,
  ElContainer,
  ElDialog,
  ElDrawer,
  ElEmpty,
  ElForm,
  ElFormItem,
  ElHeader,
  ElIcon,
  ElInput,
  ElInputNumber,
  ElMain,
  ElMenu,
  ElMenuItem,
  ElOption,
  ElProgress,
  ElRadioButton,
  ElRadioGroup,
  ElSelect,
  ElStatistic,
  ElTabPane,
  ElTable,
  ElTableColumn,
  ElTabs,
  ElTag,
  ElUpload,
].forEach((component) => {
  app.component(component.name!, component);
});
app.use(ElLoading);

app.mount("#app");
