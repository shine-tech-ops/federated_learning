<template>
  <el-breadcrumb class="h-[50px] flex items-center">
    <transition-group name="breadcrumb">
      <el-breadcrumb-item v-for="(item, index) in breadcrumbs" :key="item.path">
        <span v-if="item.redirect === 'noredirect' || index === breadcrumbs.length - 1">{{
          getRouteTitle(item)
        }}</span>
        <a v-else @click.prevent="handleLink(item)">
          {{ i18n(item.meta.title) }}
        </a>
      </el-breadcrumb-item>
    </transition-group>
  </el-breadcrumb>
</template>

<script setup lang="ts">
import { onBeforeMount, ref, watch } from 'vue'
import { useRoute, type RouteLocationMatched } from 'vue-router'
// import { compile } from "path-to-regexp";
import router from '@/router'
import { useBreadcrumbStore } from '@/stores/modules/breadcrumb'
import pinia from '@/stores'
import { useI18n } from 'vue-i18n'

const { t: $t } = useI18n()

const currentRoute = useRoute()
const breadcrumbStore = useBreadcrumbStore(pinia)

const pathCompile = (path: string) => {
  // const { params } = currentRoute;
  // const toPath = compile(path);
  // return toPath(params);
  return path
}

const i18n = (name: any) => {
  return $t(name)
}

const breadcrumbs = ref([] as Array<RouteLocationMatched>)

const allRouter = router.getRoutes()

const checkRoute = (item: RouteLocationMatched) => {
  return item.meta && item.meta.title && item.meta.breadcrumb !== false
}

function getBreadcrumb() {
  let matched = currentRoute.matched.filter((item) => item.meta && item.meta.title)
  const list: any = []
  matched.forEach((item) => {
    const filterRes = checkRoute(item)
    if (filterRes) {
      // 如果有链接的父级菜单
      if (item.meta.activeMenu) {
        const parentRoute = allRouter.find((route) => route.path === item.meta.activeMenu)
        if (parentRoute && checkRoute(parentRoute)) {
          list.push(parentRoute)
        }
      }
      list.push(item)
    }
  })

  breadcrumbs.value = list
}

// 获取路由标题
const getRouteTitle = (item: RouteLocationMatched) => {
  const path = item.path
  // 首先获取缓存的标题
  const cacheTitle = breadcrumbStore.getTitle(path)
  const title = (cacheTitle || item.meta.title) as string
  return $t(title)
}

function handleLink(item: any) {
  const { redirect, path } = item
  if (redirect) {
    router.push(redirect).catch((err) => {
      console.warn(err)
    })
    return
  }
  router.push(pathCompile(path)).catch((err) => {
    console.warn(err)
  })
}

watch(
  () => currentRoute.path,
  (path) => {
    if (path.startsWith('/redirect/')) {
      return
    }
    getBreadcrumb()
  }
)

onBeforeMount(() => {
  getBreadcrumb()
})
</script>

<style lang="scss" scoped>
.app-breadcrumb.el-breadcrumb {
  display: inline-block;
  margin-left: 8px;
  font-size: 14px;
  line-height: 50px;
}

// 覆盖 element-plus 的样式
.el-breadcrumb__inner,
.el-breadcrumb__inner a {
  font-weight: 400 !important;
}
</style>
