<template>
  <page-header title="训练日志">
    <div style="display: flex; gap: 10px;">
      <el-select v-model="search.task_id" placeholder="选择任务" clearable style="width: 200px" @change="getList">
        <el-option
          v-for="task in tasks"
          :key="task.id"
          :label="task.name"
          :value="task.id"
        />
      </el-select>
      <el-select v-model="search.device_id" placeholder="选择设备" clearable style="width: 200px" @change="getList">
        <el-option
          v-for="device in devices"
          :key="device"
          :label="device"
          :value="device"
        />
      </el-select>
      <el-select v-model="search.round" placeholder="选择轮次" clearable style="width: 150px" @change="getList">
        <el-option
          v-for="round in rounds"
          :key="round"
          :label="`Round ${round}`"
          :value="round"
        />
      </el-select>
      <el-select v-model="search.level" placeholder="日志级别" clearable style="width: 150px" @change="getList">
        <el-option label="DEBUG" value="DEBUG" />
        <el-option label="INFO" value="INFO" />
        <el-option label="WARNING" value="WARNING" />
        <el-option label="ERROR" value="ERROR" />
      </el-select>
      <el-select v-model="search.phase" placeholder="阶段" clearable style="width: 150px" @change="getList">
        <el-option label="训练" value="train" />
        <el-option label="上传" value="upload" />
        <el-option label="聚合" value="aggregate" />
        <el-option label="评估" value="evaluate" />
        <el-option label="系统" value="system" />
      </el-select>
      <el-button type="primary" @click="getList">刷新</el-button>
      <el-button @click="showStats = !showStats">{{ showStats ? '隐藏统计' : '显示统计' }}</el-button>
    </div>
  </page-header>

  <!-- 统计信息卡片 -->
  <div v-if="showStats && stats" class="stats-cards" style="margin: 20px 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
    <el-card>
      <div class="stat-item">
        <div class="stat-label">总日志数</div>
        <div class="stat-value">{{ stats.total_stats?.total_logs || 0 }}</div>
      </div>
    </el-card>
    <el-card>
      <div class="stat-item">
        <div class="stat-label">总轮次</div>
        <div class="stat-value">{{ stats.total_stats?.total_rounds || 0 }}</div>
      </div>
    </el-card>
    <el-card>
      <div class="stat-item">
        <div class="stat-label">参与设备数</div>
        <div class="stat-value">{{ stats.total_stats?.unique_devices || 0 }}</div>
      </div>
    </el-card>
  </div>

  <!-- 图表区域 -->
  <div v-if="showStats && stats" class="charts-container" style="margin: 20px 0;">
    <el-row :gutter="20">
      <el-col :span="12">
        <el-card>
          <template #header>
            <span>训练指标趋势</span>
          </template>
          <div ref="metricsChartRef" style="width: 100%; height: 300px;"></div>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card>
          <template #header>
            <span>设备性能对比</span>
          </template>
          <div ref="deviceChartRef" style="width: 100%; height: 300px;"></div>
        </el-card>
      </el-col>
    </el-row>
  </div>

  <!-- 日志列表 -->
  <div class="mt-4">
    <el-table
      v-loading="loading"
      :data="tableData"
      border
      style="width: 100%"
      max-height="600"
    >
      <el-table-column prop="log_timestamp" label="时间" width="180" />
      <el-table-column prop="task_detail.name" label="任务" width="150" />
      <el-table-column prop="device_id" label="设备ID" width="150" />
      <el-table-column prop="round" label="轮次" width="80" align="center" />
      <el-table-column prop="phase" label="阶段" width="100">
        <template #default="{ row }">
          <el-tag :type="getPhaseTagType(row.phase)">{{ getPhaseLabel(row.phase) }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="level" label="级别" width="100">
        <template #default="{ row }">
          <el-tag :type="getLevelTagType(row.level)">{{ row.level }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="loss" label="Loss" width="100" align="right">
        <template #default="{ row }">
          <span v-if="row.loss !== null && row.loss !== undefined">{{ row.loss.toFixed(4) }}</span>
          <span v-else>-</span>
        </template>
      </el-table-column>
      <el-table-column prop="accuracy" label="Accuracy" width="120" align="right">
        <template #default="{ row }">
          <span v-if="row.accuracy !== null && row.accuracy !== undefined">{{ (row.accuracy * 100).toFixed(2) }}%</span>
          <span v-else>-</span>
        </template>
      </el-table-column>
      <el-table-column prop="num_examples" label="样本数" width="100" align="right" />
      <el-table-column prop="message" label="消息" min-width="200" show-overflow-tooltip />
      <el-table-column prop="error_message" label="错误信息" min-width="200" show-overflow-tooltip v-if="hasErrors" />
    </el-table>
  </div>

  <!-- 分页 -->
  <el-pagination
    v-model:current-page="page.page"
    v-model:page-size="page.pageSize"
    layout="total, prev, pager, next"
    :total="page.total"
    @current-change="handlePageChange"
    style="margin-top: 20px; text-align: right"
  />
</template>

<script lang="ts" setup>
import { ref, onMounted, watch, nextTick } from 'vue'
import { trainingLogModel } from '@/api/trainingLog'
import { federatedTaskModel } from '@/api/federatedTask'
import * as echarts from 'echarts'

const search = ref({
  task_id: null,
  device_id: null,
  round: null,
  level: null,
  phase: null
})

const tasks = ref([])
const devices = ref([])
const rounds = ref([])
const loading = ref(false)
const tableData = ref([])
const showStats = ref(false)
const stats = ref(null)

const page = ref({
  total: 0,
  page: 1,
  pageSize: 20
})

const metricsChartRef = ref(null)
const deviceChartRef = ref(null)
let metricsChart = null
let deviceChart = null

const hasErrors = ref(false)

// 兼容拦截器处理后的响应（默认返回 data），以及仍然包含 code/data 的旧格式
const normalizeResponse = (res: any) => {
  if (!res) return {}
  if (typeof res === 'object' && 'code' in res && 'data' in res) {
    return res.data
  }
  return res
}

// 获取任务列表
const getTasks = async () => {
  try {
    const res = await federatedTaskModel.getTasksApi()
    const data = normalizeResponse(res)
    tasks.value = data?.results || data?.list || data?.data || data || []
  } catch (error) {
    console.error('获取任务列表失败:', error)
  }
}

// 获取日志列表
const getList = async () => {
  loading.value = true
  try {
    const params = {
      page: page.value.page,
      page_size: page.value.pageSize,
      ...search.value
    }
    // 移除空值
    Object.keys(params).forEach(key => {
      if (params[key] === null || params[key] === undefined || params[key] === '') {
        delete params[key]
      }
    })

    const res = await trainingLogModel.getTrainingLogsApi(params)
    const data = normalizeResponse(res)

    // 兼容不同接口返回结构（list/results/data）
    const list =
      data?.list ||
      data?.results ||
      (Array.isArray(data?.data) ? data?.data : data?.data?.list) ||
      data?.data ||
      []
    tableData.value = list

    // 兼容不同总数字段
    page.value.total =
      data?.total ??
      data?.data?.total ??
      (Array.isArray(list) ? list.length : 0)

    // 如果后端返回当前页信息，覆盖本地页码
    if (data?.page) {
      page.value.page = data.page
    }
    
    // 提取设备和轮次列表
    const uniqueDevices = [...new Set(tableData.value.map(log => log.device_id).filter(Boolean))]
    devices.value = uniqueDevices
    
    const uniqueRounds = [...new Set(tableData.value.map(log => log.round).filter(r => r !== null && r !== undefined))].sort((a, b) => a - b)
    rounds.value = uniqueRounds
    
    // 检查是否有错误
    hasErrors.value = tableData.value.some(log => log.error_message)
  } catch (error) {
    console.error('获取日志列表失败:', error)
  } finally {
    loading.value = false
  }
}

// 获取统计信息
const getStats = async () => {
  if (!search.value.task_id) return
  
  try {
    const res = await trainingLogModel.getTrainingLogStatsApi({ task_id: search.value.task_id })
    const data = normalizeResponse(res)
    stats.value = data
    nextTick(() => {
      updateCharts()
    })
  } catch (error) {
    console.error('获取统计信息失败:', error)
  }
}

// 更新图表
const updateCharts = () => {
  if (!stats.value) return

  // 训练指标趋势图
  if (metricsChartRef.value && stats.value.round_stats) {
    if (!metricsChart) {
      metricsChart = echarts.init(metricsChartRef.value)
    }
    
    const roundStats = stats.value.round_stats
    const option = {
      title: { text: 'Loss & Accuracy 趋势' },
      tooltip: { trigger: 'axis' },
      legend: { data: ['Loss', 'Accuracy'] },
      xAxis: {
        type: 'category',
        data: roundStats.map(r => `Round ${r.round}`)
      },
      yAxis: [
        {
          type: 'value',
          name: 'Loss',
          position: 'left'
        },
        {
          type: 'value',
          name: 'Accuracy',
          position: 'right',
          axisLabel: {
            formatter: '{value}%'
          }
        }
      ],
      series: [
        {
          name: 'Loss',
          type: 'line',
          data: roundStats.map(r => r.avg_loss?.toFixed(4) || 0)
        },
        {
          name: 'Accuracy',
          type: 'line',
          yAxisIndex: 1,
          data: roundStats.map(r => (r.avg_accuracy * 100)?.toFixed(2) || 0)
        }
      ]
    }
    metricsChart.setOption(option)
  }

  // 设备性能对比图
  if (deviceChartRef.value && stats.value.device_stats) {
    if (!deviceChart) {
      deviceChart = echarts.init(deviceChartRef.value)
    }
    
    const deviceStats = stats.value.device_stats
    const option = {
      title: { text: '设备准确率对比' },
      tooltip: { trigger: 'axis' },
      xAxis: {
        type: 'category',
        data: deviceStats.map(d => d.device_id)
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: '{value}%'
        }
      },
      series: [
        {
          type: 'bar',
          data: deviceStats.map(d => ((d.avg_accuracy || 0) * 100).toFixed(2))
        }
      ]
    }
    deviceChart.setOption(option)
  }
}

const handlePageChange = () => {
  getList()
}

const getPhaseTagType = (phase) => {
  const map = {
    train: 'primary',
    upload: 'success',
    aggregate: 'warning',
    evaluate: 'info',
    system: ''
  }
  return map[phase] || ''
}

const getPhaseLabel = (phase) => {
  const map = {
    train: '训练',
    upload: '上传',
    aggregate: '聚合',
    evaluate: '评估',
    system: '系统'
  }
  return map[phase] || phase
}

const getLevelTagType = (level) => {
  const map = {
    DEBUG: 'info',
    INFO: '',
    WARNING: 'warning',
    ERROR: 'danger'
  }
  return map[level] || ''
}

// 监听统计显示状态
watch(showStats, (val) => {
  if (val && search.value.task_id) {
    getStats()
  }
})

// 监听任务变化
watch(() => search.value.task_id, (val) => {
  if (val && showStats.value) {
    getStats()
  }
})

onMounted(() => {
  getTasks()
  getList()
  
  // 窗口大小改变时重新调整图表
  window.addEventListener('resize', () => {
    if (metricsChart) metricsChart.resize()
    if (deviceChart) deviceChart.resize()
  })
})
</script>

<style scoped>
.stats-cards .stat-item {
  text-align: center;
}

.stats-cards .stat-label {
  font-size: 14px;
  color: #666;
  margin-bottom: 8px;
}

.stats-cards .stat-value {
  font-size: 24px;
  font-weight: bold;
  color: #409eff;
}
</style>

