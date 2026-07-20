import { defineStore } from "pinia";

import * as authApi from "../api/auth";
import { ApiError } from "../api/http";
import type { AuthRequest, AuthUser } from "../api/types";

export const useAuthStore = defineStore("auth", {
  state: () => ({
    user: null as AuthUser | null,
    initialized: false,
  }),
  actions: {
    async loadCurrentUser() {
      try {
        this.user = await authApi.getCurrentUser();
      } catch (error) {
        if (!(error instanceof ApiError) || error.status !== 401) {
          throw error;
        }
        this.user = null;
      } finally {
        this.initialized = true;
      }
    },
    async login(credentials: AuthRequest) {
      this.user = await authApi.login(credentials);
      this.initialized = true;
    },
    async register(credentials: AuthRequest) {
      this.user = await authApi.register(credentials);
      this.initialized = true;
    },
    async logout() {
      try {
        await authApi.logout();
      } finally {
        this.user = null;
        this.initialized = true;
      }
    },
  },
});
