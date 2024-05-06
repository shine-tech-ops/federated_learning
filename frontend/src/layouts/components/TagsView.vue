<template>
  <div class="tags-box">
    <div class="tags-menu">
      <!-- <el-scrollbar> -->
      <el-tabs v-model="tabsMenuValue" @tab-click="tabClick" @tab-remove="tabRemove">
        <el-tab-pane
          v-for="item in tagsList"
          :key="item.path"
          :label="$t(item.title)"
          :name="item.path"
          :closable="item.close"
        >
          <template #label>
            <el-icon v-if="item.icon && globalStore.tagsViewIcon" class="tabs-icon">
              <component :is="item.icon"></component>
            </el-icon>
            {{ $t(item.title) }}
          </template>
        </el-tab-pane>
      </el-tabs>
      <!-- </el-scrollbar> -->
    </div>
    <el-dropdown trigger="click" class="tabs-action ml-5">
      <el-icon size="16"><ArrowDown /></el-icon>
      <template #dropdown>
        <el-dropdown-menu>
          <el-dropdown-item @click="closeCurrentTab">
            <el-icon :size="14"><FolderRemove /></el-icon>
            关闭当前
          </el-dropdown-item>
          <el-dropdown-item @click="closeOtherTab">
            <el-icon :size="14"><Close /></el-icon>
            关闭其他
          </el-dropdown-item>
          <el-dropdown-item @click="closeAllTab">
            <el-icon :size="14"><FolderDelete /></el-icon>
            关闭所有
          </el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useTagsViewsStore } from '@/stores/modules/tagsView'
import type { TabsPaneContext, TabPaneName } from 'element-plus'
import { useRoutesList } from '@/stores/modules/routesList'
import pinia from '@/stores'
import { useGlobalStore } from '@/stores/modules/global'

const globalStore = useGlobalStore(pinia)
const route = useRoute()
const router = useRouter()
const tagsViewsStore = useTagsViewsStore(pinia)
const storesRoutesList = useRoutesList(pinia)
const tabsMenuValue = ref(route.fullPath)
const tagsList = computed(() => tagsViewsStore.tagsList)

onMounted(() => {
  initTabs()
})

// 监听路由的变化
watch(
  () => route.fullPath,
  () => {
    if (route.meta.isFull) return
    tabsMenuValue.value = route.fullPath
    const tabsParams = {
      icon: route.meta.icon as string,
      title: route.meta.title as string,
      path: route.fullPath,
      name: route.name as string,
      close: !route.meta.isAffix,
      isKeepAlive: route.meta.isKeepAlive as boolean
    }
    tagsViewsStore.addTags(tabsParams)
  },
  { immediate: true }
)

// 初始化需要固定的 tabs
const initTabs = () => {
  storesRoutesList.routesList.forEach((item) => {
    if (item.meta.isAffix && !item.meta.isHide && !item.meta.isFull) {
      const tabsParams = {
        icon: item.meta.icon,
        title: item.meta.title,
        path: item.path,
        name: item.name,
        close: !item.meta.isAffix,
        isKeepAlive: item.meta.isKeepAlive
      }
      tagsViewsStore.addTags(tabsParams)
    }
  })
}

// Tab Click
const tabClick = (tabItem: TabsPaneContext) => {
  const fullPath = tabItem.props.name as string
  router.push(fullPath)
}

// Remove Tab
const tabRemove = (fullPath: TabPaneName) => {
  tagsViewsStore.removeTags(fullPath as string, fullPath == route.fullPath)
}

// 关闭当前
const closeCurrentTab = () => {
  if (route.meta.isAffix) return
  tabRemove(route.fullPath)
}
const closeOtherTab = () => {
  tagsViewsStore.closeMultipleTag(route.fullPath)
}
const closeAllTab = async () => {
  tagsViewsStore.closeMultipleTag()
  router.push('/')
}
</script>

<style scoped lang="scss">
.tags-box {
  background-color: var(--el-bg-color);
  padding: 0 10px;
  border-bottom: 1px solid #e4e7ed;
  display: flex;
  align-items: center;
  justify-content: space-between;
  .tags-menu {
    flex: 1;
    width: calc(100% - 70px);
  }
}

:deep(.el-tabs) {
  .el-tabs__header {
    margin: 0;
  }
  .el-tabs__nav-wrap::after {
    opacity: 0;
  }
  .is-icon-close {
    opacity: 0;
    transition: all 0.1s ease-out;
    transform: translateX(100%);
  }
  .el-tabs__item {
    padding-right: 0;
    --el-text-color-primary: #909399;
  }
  .el-tabs__item:hover,
  .el-tabs__item.is-active {
    .is-icon-close {
      opacity: 1;
      transform: translateX(0);
    }
  }
  .el-tabs__active-bar {
    height: 1.5px;
  }
}
</style>
