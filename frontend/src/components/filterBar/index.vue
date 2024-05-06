<template>
  <div class="filter-bar">
    <div class="tab-title" v-if="props.tabTitle">{{ $t(props.tabTitle) }}</div>
    <div class="tabs-menu">
      <el-tabs v-model="currentTab">
        <el-tab-pane
          v-for="(item, index) in props.tabList"
          :key="index"
          :label="item.title"
          :name="item.name"
        >
          <template #label>
            {{ $t(item.title) }}
          </template>
        </el-tab-pane>
      </el-tabs>
    </div>
    <div>
      <slot></slot>
    </div>
  </div>
  <div class="placeholder-bar"></div>
</template>

<script setup lang="ts">
const props = defineProps({
  tabTitle: {
    type: String,
    default: ''
  },
  tabList: {
    type: Array as PropType<{ title: string; name: string; value: string }[]>,
    default: () => []
  }
})

const currentTab = defineModel('currentTab', { type: String })
</script>

<style scoped lang="scss">
.filter-bar {
  background-color: var(--el-bg-color);
  padding: 0 5px;
  border-bottom: 1px solid #e4e7ed;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: absolute;
  right: 0;
  left: 0;
  top: -10px;
  z-index: 1;

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

.placeholder-bar {
  height: 46px;
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
