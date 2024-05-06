import router from '@/router'
import { defineStore } from 'pinia'
import { getUrlWithParams } from '@/utils'
import { useKeepAliveStore } from '@/stores/modules/keepAlive'
import type { TabMenuStates, TabListProps } from '@/stores/interface'
import pinia from '@/stores'

export const useTabMenuStore = defineStore('tabMenu', {
  state: (): TabMenuStates => ({
    tabList: [],
    curTab: 0,
    show: false,
    tabTitle: ''
  }),
  actions: {
    initTab(tabList: TabListProps[], title = '') {
      this.tabList = tabList
      this.curTab = tabList[0].name
      this.show = true
      this.tabTitle = title
    },
    changeTab(name: string) {
      this.curTab = name
      this.tabList = JSON.parse(JSON.stringify(this.tabList))
    },
    resetTab() {
      this.show = false
      this.curTab = null
      this.tabList = []
    },
    getCurTabInfo() {
      return this.tabList.find((item) => item.name === this.curTab)
    }
  },
  persist: false
})
