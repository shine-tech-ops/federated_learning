<template>
  <div class="message">
    <el-popover placement="bottom" :width="310" trigger="click">
      <template #reference>
        <el-badge :value="unRead" class="cursor-pointer">
          <el-icon><Bell /></el-icon>
        </el-badge>
      </template>
      <el-tabs v-model="activeName">
        <el-tab-pane :label="`通知(${messageList.length})`" name="first">
          <div class="list">
            <div class="item" v-for="(item, index) in messageList" :key="index">
              <el-icon><ChatLineRound /></el-icon>
              <div class="content">
                <span class="title">{{ item.title }}</span>
                <span class="date">{{ item.time }}</span>
              </div>
            </div>
          </div>
        </el-tab-pane>
        <el-tab-pane label="消息(0)" name="second">
          <el-empty description="暂无消息" />
        </el-tab-pane>
      </el-tabs>
    </el-popover>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
const activeName = ref('first')
const messageList = ref([
  { time: '一分钟前', title: '一个通知', type: 1 },
  { time: '一分钟前', title: '一个通知', type: 1 },
  { time: '一分钟前', title: '一个通知', type: 1 }
])

const unRead = ref(2)
</script>

<style scoped lang="scss">
.list {
  display: flex;
  flex-direction: column;
  .item {
    display: flex;
    align-items: center;
    padding: 20px 0;
    border-bottom: 1px solid var(--el-border-color-light);
    &:last-child {
      border: none;
    }
    .el-icon {
      font-size: 40px;
      margin: 0 20px 0 5px;
    }
    .content {
      display: flex;
      flex-direction: column;
      .title {
        margin-bottom: 5px;
      }
      .date {
        font-size: 12px;
        color: var(--el-text-color-secondary);
      }
    }
  }
}
</style>
