import { defineStore } from 'pinia'
import type { BreadcrumbStates } from '@/stores/interface'

export const useBreadcrumbStore = defineStore('app-breadcrumb', {
  state: (): BreadcrumbStates => ({
    pathTitle: {}
  }),
  actions: {
    setTitle(path: string, title: string) {
      this.pathTitle[path] = title
    },
    resetTitle(path: string) {
      this.pathTitle[path] = ''
    },
    getTitle(path: string) {
      return this.pathTitle[path]
    }
  },
  persist: false
})
