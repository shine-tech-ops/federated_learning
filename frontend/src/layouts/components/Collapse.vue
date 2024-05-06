<template>
  <el-icon class="collapse-icon" @click="toggleCollapse">
    <component :is="globalStore.isCollapse ? 'expand' : 'fold'"></component>
  </el-icon>
</template>

<script setup lang="ts">
import pinia from '@/stores'
import { useGlobalStore } from '@/stores/modules/global'

const globalStore = useGlobalStore(pinia)

const toggleCollapse = () => {
  globalStore.setGlobalState('isCollapse', !globalStore.isCollapse)
}
</script>

<style scoped lang="scss">
.collapse-icon {
  margin-right: 20px;
  font-size: 22px;
  color: var(--el-header-text-color);
  cursor: pointer;
}
</style>
