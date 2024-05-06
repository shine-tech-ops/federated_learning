import router from '@/router'
import { defineStore } from 'pinia'
import { getUrlWithParams } from '@/utils'
import { useKeepAliveStore } from '@/stores/modules/keepAlive'
import type { TagsViewState, tagsListProps } from '@/stores/interface'
import pinia from '@/stores'

const keepAliveStore = useKeepAliveStore(pinia)

export const useTagsViewsStore = defineStore('tagsView', {
  state: (): TagsViewState => ({
    tagsList: []
  }),
  actions: {
    async addTags(tabItem: tagsListProps) {
      if (this.tagsList.every((item) => item.path !== tabItem.path)) {
        this.tagsList.push(tabItem)
      }
      // keepalive
      if (!keepAliveStore.keepAliveName.includes(tabItem.name) && tabItem.isKeepAlive) {
        keepAliveStore.addKeepAliveName(tabItem.name)
      }
    },
    async removeTags(tabPath: string, isCurrent: boolean = true) {
      if (isCurrent) {
        this.tagsList.forEach((item, index) => {
          if (item.path !== tabPath) return
          const nextTab = this.tagsList[index + 1] || this.tagsList[index - 1]
          if (!nextTab) return
          router.push(nextTab.path)
        })
      }
      // remove keepalive
      const tabItem = this.tagsList.find((item) => item.path === tabPath)
      tabItem?.isKeepAlive && keepAliveStore.removeKeepAliveName(tabItem.name)
      // set tabs
      this.tagsList = this.tagsList.filter((item) => item.path !== tabPath)
    },
    // Close Tabs On Side
    async closeTagsOnSide(path: string, type: 'left' | 'right') {
      const currentIndex = this.tagsList.findIndex((item) => item.path === path)
      if (currentIndex !== -1) {
        const range = type === 'left' ? [0, currentIndex] : [currentIndex + 1, this.tagsList.length]
        this.tagsList = this.tagsList.filter((item, index) => {
          return index < range[0] || index >= range[1] || !item.close
        })
      }
      // set keepalive
      const KeepAliveList = this.tagsList.filter((item) => item.isKeepAlive)
      keepAliveStore.setKeepAliveName(KeepAliveList.map((item) => item.name))
    },
    // Close MultipleTab
    async closeMultipleTag(tagsValue?: string) {
      this.tagsList = this.tagsList.filter((item) => {
        return item.path === tagsValue || !item.close
      })
      // set keepalive
      const KeepAliveList = this.tagsList.filter((item) => item.isKeepAlive)
      keepAliveStore.setKeepAliveName(KeepAliveList.map((item) => item.name))
    },
    async setTags(tagsList: tagsListProps[]) {
      this.tagsList = tagsList
    },
    // Set Tags Title
    async setTagsTitle(title: string) {
      this.tagsList.forEach((item) => {
        if (item.path == getUrlWithParams()) item.title = title
      })
    }
  },
  persist: true
})
