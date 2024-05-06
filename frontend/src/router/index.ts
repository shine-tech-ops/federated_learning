import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'
import { notFoundAndNoPower, staticRoutes, dynamicRoutes } from './route'
import NProgress from 'nprogress'
import { storeToRefs } from 'pinia'
import pinia from '@/stores'
import 'nprogress/nprogress.css'
import { Session } from '@/utils/storage'
import { useRoutesList } from '@/stores/modules/routesList'
import { useUserStore } from '@/stores/modules/user'
import { useTagsViewsStore } from '@/stores/modules/tagsView'
import { cloneDeep } from 'lodash'
import { useGlobalStore } from '@/stores/modules/global'

const storesRoutesList = useRoutesList(pinia)
const userStore = useUserStore(pinia)
const tagsViewsStore = useTagsViewsStore(pinia)
// const globalStore = useGlobalStore(pinia)

// 根据权限过滤路由
function filterRoutesByRoles_bak(routes: RouteRecordRaw[], roles: any[]) {
  return routes.filter((route) => {
    if (roles.includes(route.name)) {
      if (route.children) {
        route.children = filterRoutesByRoles(route.children, roles)
      }
      return true
    } else {
      return false
    }
  })
}

function filterRoutesByRoles(routes: RouteRecordRaw[], permissions: any[]) {
  return routes.filter((route) => {
    if (route.children) {
      route.children = filterRoutesByRoles(route.children, permissions)
    }
    if (route.meta) {
      const auths = route.meta.auth as string[]
      // 如果配置了权限
      if (auths) {
        let flag = false
        auths.map((val: string) => {
          if (permissions.includes(val)) {
            flag = true
          }
        })
        return flag
      } else {
        // 没有配置就根据子项判断
        if (route.children?.length === 0) {
          return false
        }
      }
    }

    return true
  })
}

// 更新路由重定向
function updateRedirect(routes: RouteRecordRaw[]) {
  routes.forEach((route) => {
    if (route.children && route.children.length > 0 && !route.component) {
      route.redirect = route.children[0].path
      updateRedirect(route.children)
    }
  })
}

function getPermission(): string | string[] {
  // const userStore = useUserStore(pinia)
  // const { users, permissions } = storeToRefs(userStore)
  const isSuperAdmin = userStore.isSuperAdmin
  if (!isSuperAdmin) {
    // const roles = users.value.roles
    console.log('🚀 ~ 当前用户权限:', userStore.permissions)
    // return roles
    return userStore.permissions
  } else {
    console.log('当前是超管')
    return 'all'
  }
}

// 获取对应权限的路由
function routePermission(permission: string | string[]): RouteRecordRaw[] {
  let routes = cloneDeep(dynamicRoutes)
  if (Array.isArray(permission)) {
    routes = filterRoutesByRoles(routes, permission)
  }
  updateRedirect(routes)
  return routes
}

function initSyncRoute() {
  const permission = getPermission()
  const routes = routePermission(permission)
  // console.log('初始化异步路由', '\n全部路由', dynamicRoutes, '\n过滤后', routes)
  routes.forEach((route) => {
    if (route.meta?.isFull) {
      router.addRoute(route as unknown as RouteRecordRaw)
    } else {
      router.addRoute('layout', route as unknown as RouteRecordRaw)
    }
  })

  // 存储路由
  // const storesRoutesList = useRoutesList(pinia)
  storesRoutesList.setRoutesList(routes)

  // 更新tagView
  if (permission !== 'all') {
    // const tagsViewsStore = useTagsViewsStore(pinia)
    const { tagsList } = storeToRefs(tagsViewsStore)
    const filterTags = tagsList.value.filter((item) => permission.includes(item.name))
    tagsList.value = filterTags
    console.log('🚀 ~ 过滤后的 tagsList:', filterTags)
  }
}

export async function initRouter() {
  if (!Session.get('token')) return false
  // 初始化用户信息
  await userStore.setUsers()
  // await useUserStore(pinia).setUsers()
  // 初始化系统信息
  // await globalStore.initSysInfo()
  await useGlobalStore(pinia).initSysInfo()
  // 添加动态路由
  initSyncRoute()
  // 设置首页
  // const storesRoutesList = useRoutesList(pinia)
  const root = router.resolve({ name: 'root' })
  if (root) {
    const rootRoute = root.matched[0]
    // 当首页路径不一致时更新
    if (rootRoute.redirect !== storesRoutesList.homeUrl) {
      router.addRoute({ path: '/', redirect: storesRoutesList.homeUrl, name: 'root' })
    }
  }
}

/** 重置路由 */
export function resetRouter() {
  router.getRoutes().forEach((route) => {
    const { name, meta } = route
    if (name && router.hasRoute(name)) {
      router.removeRoute(name)
    }
  })
  // const storesRoutesList = useRoutesList(pinia)
  storesRoutesList.setRoutesList([])
}

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [...notFoundAndNoPower, ...staticRoutes],
  strict: false,
  scrollBehavior: () => ({ left: 0, top: 0 })
})

// 路由加载前
router.beforeEach(async (to, from, next) => {
  // const globalStore = useGlobalStore(pinia)
  // 初始化系统信息
  // if (!globalStore.sysLoaded) {
  //   globalStore.initSysInfo()
  // }
  // console.log('allRoutes', JSON.parse(JSON.stringify(router.getRoutes())));
  NProgress.configure({ showSpinner: false })
  if (to.meta.title) NProgress.start()
  const token = Session.get('token')
  if (to.path === '/login' && !token) {
    next()
    NProgress.done()
  } else {
    if (!token) {
      next(`/login?redirect=${to.path}&params=${JSON.stringify(to.query ? to.query : to.params)}`)
      window.localStorage.clear()
      Session.clear()
      NProgress.done()
    } else if (token && to.path === '/login') {
      next('/')
      NProgress.done()
    } else {
      if (['/403', '/404', '/500'].includes(to.path)) {
        next()
        NProgress.done()
        return
      }

      const { routesList } = storeToRefs(storesRoutesList)
      if (routesList.value.length === 0) {
        await initRouter()
        // 有可能权限判断后路由为空
        if (routesList.value.length === 0) {
          next()
        } else {
          next({ path: to.path, query: to.query })
        }
        NProgress.done()
      } else {
        next()
        NProgress.done()
      }
    }
  }
})

router.onError((error) => {
  NProgress.done()
  console.warn('路由错误', error.message)
})

// 路由跳转结束
router.afterEach(() => {
  NProgress.done()
})

export default router
