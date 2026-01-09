<!-- 特殊页面的筛选项 -->
<template>
  <div class="tab-box" v-if="tabMenuStore.show">
    <div class="tab-title" v-if="tabMenuStore.tabTitle">{{ tabMenuStore.tabTitle }}</div>
    <div class="tabs-menu">
      <el-tabs v-model="currentTab" @tab-click="tabClick">
        <el-tab-pane
          v-for="(item, index) in tabList"
          :key="index"
          :label="$t(item.title)"
          :name="item.name"
        >
          <template #label>
            {{ $t(item.title) }}
          </template>
        </el-tab-pane>
      </el-tabs>
    </div>
    <div>
      <!-- 传入的内容 -->
      <slot name="extraContent"></slot>
    </div>
  </div>
</template>
<script setup lang="ts">
import type { TabsPaneContext } from 'element-plus'
import { useTabMenuStore } from '@/stores/modules/tabMenu'
import pinia from '@/stores'

const tabMenuStore = useTabMenuStore(pinia)

const currentTab = ref()
const tabList = computed(() => tabMenuStore.tabList)

// 只要列表发生变化就重新获取 当前选中值
watch(
  () => tabList.value,
  () => {
    currentTab.value = tabMenuStore.curTab
  },
  { immediate: true }
)
const tabClick = (tabItem: TabsPaneContext) => {
  if (tabItem.props.name) {
    tabMenuStore.changeTab(tabItem.props.name as string)
  }
}
</script>

<style scoped lang="scss">
.tab-box {
  background-color: var(--el-bg-color);
  padding: 0 10px;
  border-bottom: 1px solid #e4e7ed;
  display: flex;
  align-items: center;
  justify-content: space-between;
  .tabs-menu {
    flex: 1;
    width: calc(100% - 70px);
  }
  .tab-title {
    border-right: 1px solid #e9ebf0;
    padding-right: 20px;
    margin-right: 20px;
    margin-left: 5px;
    color: rgb(55 65 81);
    font-weight: 600;
    font-size: 18px;
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
    --el-text-color-primary: #656f7d;
  }
  .el-tabs__item {
    // color: #2A2E34;
    --el-color-primary: #2a2e34;
    &.is-active {
      font-weight: 700;
    }
  }
  .el-tabs__item:hover,
  .el-tabs__item.is-active {
    .is-icon-close {
      opacity: 1;
      transform: translateX(0);
    }
  }
  .el-tabs__active-bar {
    height: 2px;
  }
}
</style>
