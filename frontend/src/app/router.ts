import { createRouter, createWebHistory } from "vue-router";

import WorkspacePage from "../pages/WorkspacePage.vue";

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/login", name: "login", component: () => import("../pages/LoginPage.vue") },
    { path: "/", name: "workspace", component: WorkspacePage },
    { path: "/agent", name: "agent", component: () => import("../pages/AgentPage.vue") },
    { path: "/documents", name: "documents", component: () => import("../pages/DocumentsPage.vue") },
    { path: "/review", name: "review", component: () => import("../pages/ReviewPage.vue") },
    { path: "/settings", name: "settings", component: () => import("../pages/SettingsPage.vue") },
  ],
});
