<template>
  <page-header title="日志与审计" />

  <el-form inline>
    <el-form-item label="关键字">
      <el-input v-model="filters.keyword" />
    </el-form-item>
    <el-form-item label="时间范围">
      <el-date-picker v-model="filters.dateRange" type="daterange" />
    </el-form-item>
    <el-button @click="searchLogs">搜索</el-button>
  </el-form>

  <el-table :data="logs">
    <el-table-column prop="time" label="时间" />
    <el-table-column prop="user" label="操作人" />
    <el-table-column prop="action" label="操作内容" />
    <el-table-column prop="level" label="日志等级" />
  </el-table>
</template>
