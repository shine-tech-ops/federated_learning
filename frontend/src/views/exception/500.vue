<template>
  <div class="not-container">
    <svg-icon name="error" class="no-img"></svg-icon>
    <div class="not-detail">
      <h2>500</h2>
      <h4>抱歉，服务器出错了</h4>
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
