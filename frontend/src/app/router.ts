import { createRouter, createWebHistory } from "vue-router";

import DocumentsPage from "../pages/DocumentsPage.vue";
import ReviewPage from "../pages/ReviewPage.vue";
import SettingsPage from "../pages/SettingsPage.vue";
import WorkspacePage from "../pages/WorkspacePage.vue";

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", name: "workspace", component: WorkspacePage },
    { path: "/documents", name: "documents", component: DocumentsPage },
    { path: "/review", name: "review", component: ReviewPage },
    { path: "/settings", name: "settings", component: SettingsPage },
  ],
});
