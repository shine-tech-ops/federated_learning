<template>
  <div class="not-container">
    <svg-icon name="no-content" class="no-img"></svg-icon>
    <div class="not-detail">
      <h2>404</h2>
      <h4>抱歉，您访问的页面不存在</h4>
      <el-button type="primary" @click="goHome()"> 返回首页 </el-button>
    </div>
  </div>
</template>

<script setup lang="ts" name="404">
import { goHome } from './common'
</script>

<style scoped lang="scss">
@import './index.scss';
</style>
