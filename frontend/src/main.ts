import { createApp } from "vue";
import { createPinia } from "pinia";
import {
  ElAlert,
  ElAside,
  ElButton,
  ElContainer,
  ElDialog,
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
import "./styles/main.css";

const app = createApp(App);

app.use(createPinia());
app.use(router);
[
  ElAlert,
  ElAside,
  ElButton,
  ElContainer,
  ElDialog,
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
