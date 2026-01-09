<template>
  <page-header title="节点状态监控" />

  <div class="grid grid-cols-3 gap-4 mb-4">
    <el-card>在线节点数：{{ stats.online }}</el-card>
    <el-card>离线节点数：{{ stats.offline }}</el-card>
    <el-card>低质量节点数：{{ stats.lowQuality }}</el-card>
  </div>

  <el-table :data="nodes">
    <el-table-column prop="ip" label="IP" />
    <el-table-column prop="cpu" label="CPU使用%" />
    <el-table-column prop="memory" label="内存使用%" />
    <el-table-column prop="gpu" label="GPU使用%" />
    <el-table-column prop="score" label="健康评分" />
  </el-table>
</template>
