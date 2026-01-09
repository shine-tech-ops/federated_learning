import { defineStore } from 'pinia'
import { type RouteRecordRaw } from 'vue-router'
/**
 * 路由列表
 */
export const useRoutesList = defineStore('routesList', {
  state: (): RoutesListState => ({
    routesList: [],
    homeUrl: '/'
  }),
  actions: {
    setRoutesList(data: Array<RouteRecordRaw>) {
      this.routesList = data
      this.setHomeUrl()
    },
    async addRoutesList(data: Array<RouteRecordRaw>) {
      this.routesList.push(data)
      this.setHomeUrl()
    },
    setHomeUrl() {
      this.routesList.find((route) => {
        if (route.meta && !route.meta.isFull) {
          this.homeUrl = route.path
          return true
        }
      })
      console.log('🚀 首页路由更新为:', this.homeUrl)
    }
  }
})
