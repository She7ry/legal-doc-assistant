import { defineStore } from "pinia";

import {
  clearRuntimeSettings,
  DEFAULT_RUNTIME_SETTINGS,
  readRuntimeSettings,
  type RuntimeSettings,
  writeRuntimeSettings,
} from "../config/runtime";

export const useSettingsStore = defineStore("settings", {
  state: (): RuntimeSettings => readRuntimeSettings(),
  getters: {
    displayBaseUrl: (state) => state.apiBaseUrl.replace(/^https?:\/\//, ""),
  },
  actions: {
    save(settings: RuntimeSettings) {
      this.apiBaseUrl = settings.apiBaseUrl.trim();
      writeRuntimeSettings({ apiBaseUrl: this.apiBaseUrl });
    },
    reset() {
      clearRuntimeSettings();
      const defaults = readRuntimeSettings();
      this.apiBaseUrl = defaults.apiBaseUrl;
    },
  },
});
