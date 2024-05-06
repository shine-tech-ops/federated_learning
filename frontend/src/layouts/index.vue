<template>
  <el-container class="layout is-vertical">
    <el-header>
      <Header></Header>
    </el-header>
    <el-container class="classic-content">
      <el-aside>
        <div class="aside-box" :style="{ width: isCollapse ? '65px' : '210px' }">
          <el-scrollbar>
            <el-menu
              :router="false"
              :default-active="activeMenu"
              :collapse="isCollapse"
              :unique-opened="accordion"
              :collapse-transition="false"
              class="py-5 !px-2.5"
            >
              <SubMenu :menu-list="menuList"></SubMenu>
            </el-menu>
          </el-scrollbar>
        </div>
      </el-aside>
      <el-container class="!flex-col">
        <!-- <TagsView></TagsView> -->
        <tab-menu></tab-menu>
        <el-main>
          <el-scrollbar
            ref="layoutMainScrollbarRef"
            class="layout-main-scroll layout-backtop-header-fixed"
            wrap-class="layout-main-scroll"
            view-class="layout-main-scroll"
          >
            <transition appear name="fade-transform" mode="out-in">
              <div :class="tabMenuStore.show ? 'min-h-main-sub' : 'min-h-main'">
                <!-- <el-card> -->
                <router-view v-slot="{ Component, route }">
                  <!-- <keep-alive :include="keepAliveName"> -->
                  <component :is="Component" :key="route.fullPath" />
                  <!-- </keep-alive> -->
                </router-view>
                <!-- </el-card> -->
              </div>
            </transition>
          </el-scrollbar>
        </el-main>
        <Footer></Footer>
      </el-container>
    </el-container>
  </el-container>
</template>

<script setup lang="ts" name="layoutVertical">
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import pinia from '@/stores'
import { useGlobalStore } from '@/stores/modules/global'
import Header from '@/layouts/components/Header.vue'
import SubMenu from '@/layouts/components/SubMenu.vue'
import Footer from '@/layouts/components/Footer.vue'
import TagsView from '@/layouts/components/TagsView.vue'
import { useRoutesList } from '@/stores/modules/routesList'
import { useKeepAliveStore } from '@/stores/modules/keepAlive'
import { storeToRefs } from 'pinia'
import TabMenu from '@/layouts/components/TabMenu.vue'
import { useTabMenuStore } from '@/stores/modules/tabMenu'
import { ElScrollbar } from 'element-plus'
import mittBus from '@/utils/mittBus'
import useIdleTimer from '@/hooks/useIdleTimer'
import router from '@/router'

// 滚动条
const layoutMainScrollbarRef = ref<InstanceType<typeof ElScrollbar>>()

const tabMenuStore = useTabMenuStore(pinia)
// const title = import.meta.env.VITE_APP_TITLE
const storesRoutesList = useRoutesList(pinia)
const keepAliveStore = useKeepAliveStore()
const { keepAliveName } = storeToRefs(keepAliveStore)
const route = useRoute()
const globalStore = useGlobalStore(pinia)
const accordion = true
const isCollapse = computed(() => globalStore.isCollapse)
const activeMenu = computed(
  () => (route.meta.activeMenu ? route.meta.activeMenu : route.path) as string
)

// 路由过滤递归函数
const filterRoutesFun = <T extends Menu.MenuOptions>(arr: T[]): T[] => {
  return arr
    .filter((item: T) => !item.meta?.isHide)
    .map((item: T) => {
      item = Object.assign({}, item)
      if (item.children) item.children = filterRoutesFun(item.children)
      return item
    })
}

const genMenuList = () => {
  return filterRoutesFun(storesRoutesList.routesList)
}

const menuList = genMenuList()

// 主区域滚动事件
// @ts-ignore
mittBus.on('main-scroll-to', (to: number) => {
  layoutMainScrollbarRef.value!.setScrollTop(to)
})
</script>

<style scoped lang="scss">
@import './index.scss';
</style>
