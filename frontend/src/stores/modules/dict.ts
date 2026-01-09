import { defineStore } from 'pinia'
import { cloneDeep } from 'lodash'
import { type DictStates } from '@/stores/interface/index'
/**
 * 字典列表
 */
export const useDictStore = defineStore('app-dict-list', {
  state: (): DictStates => ({}),
  actions: {},
  persist: true
})
