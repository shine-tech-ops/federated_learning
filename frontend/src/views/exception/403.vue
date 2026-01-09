<template>
  <div class="not-container">
    <svg-icon name="no-permission" class="no-img"></svg-icon>
    <div class="not-detail">
      <h2>403</h2>
      <h4>抱歉，您无权访问该页面</h4>
      <el-button type="primary" @click="goHome()"> 返回首页 </el-button>
    </div>
  </div>
</template>

<script setup lang="ts" name="403">
import { goHome } from './common'
</script>

<style scoped lang="scss">
@import './index.scss';
</style>
