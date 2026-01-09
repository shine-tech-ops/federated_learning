<template>
  <div class="logo flex justify-start items-center">
    <el-image :src="globalStore.sysLogo" class="logo-img">
      <template #error>
        <img class="logo-img" :src="globalStore.defaultLogo" />
      </template>
    </el-image>
    <!-- 超出8个字就显示tooltip -->
    <el-tooltip
      :content="$t(globalStore.sysTitle)"
      :disabled="$t(globalStore.sysTitle).length <= 8"
    >
      <span class="logo-text truncate">{{ $t(globalStore.sysTitle) }}</span>
    </el-tooltip>
  </div>
</template>

<script setup lang="ts">
import { useGlobalStore } from '@/stores/modules/global'
import pinia from '@/stores'
const globalStore = useGlobalStore(pinia)
</script>
<style lang="scss" scoped>
.logo {
  flex-shrink: 0;
  width: 200px;
  margin-right: 16px;
  .logo-img {
    width: 20px;
    object-fit: contain;
    margin-right: 6px;
  }
  .logo-text {
    font-size: 16px;
    font-weight: bold;
    color: var(--el-header-logo-text-color);
    white-space: nowrap;
  }
}
</style>
