<template>
  <page-header title="统计与报表" />

  <el-row :gutter="20">
    <el-col :span="12">
      <el-card>
        <h4>训练效果趋势图</h4>
        <!-- 可使用 ECharts 或其他图表库 -->
        <div id="accuracyChart" style="height: 300px;"></div>
      </el-card>
    </el-col>
    <el-col :span="12">
      <el-card>
        <h4>资源使用统计图</h4>
        <div id="resourceChart" style="height: 300px;"></div>
      </el-card>
    </el-col>
  </el-row>

  <el-button @click="generateReport">生成报告</el-button>
</template>
