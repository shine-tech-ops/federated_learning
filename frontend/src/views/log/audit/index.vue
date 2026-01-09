<template>
  <page-header title="操作日志管理">
    <el-button type="danger" @click="handleBatchDelete">批量删除</el-button>
    <el-button type="danger" @click="handleClearAll">清空日志</el-button>
  </page-header>

  <el-form :model="searchForm" label-width="100px" style="margin-bottom: 20px">
    <el-row :gutter="20">
      <el-col :span="6">
        <el-form-item label="用户">
          <el-input v-model="searchForm.user" placeholder="用户名" />
        </el-form-item>
      </el-col>
      <el-col :span="6">
        <el-form-item label="请求方法">
          <el-select v-model="searchForm.method" placeholder="请选择" style="width: 100%">
            <el-option label="GET" value="GET" />
            <el-option label="POST" value="POST" />
            <el-option label="PUT" value="PUT" />
            <el-option label="DELETE" value="DELETE" />
          </el-select>
        </el-form-item>
      </el-col>
      <el-col :span="6">
        <el-form-item label="请求路径">
          <el-input v-model="searchForm.path" placeholder="路径" />
        </el-form-item>
      </el-col>
      <el-col :span="6">
        <el-button type="primary" @click="handleSearch">搜索</el-button>
        <el-button @click="resetSearch">重置</el-button>
      </el-col>
    </el-row>
  </el-form>

  <el-table :data="logs" border style="width: 100%" @selection-change="handleSelectionChange">
    <el-table-column type="selection" width="50" />
    <el-table-column prop="created_at" label="时间" width="160" />
    <el-table-column prop="user.name" label="用户" width="120" />
    <el-table-column prop="ip" label="IP" width="130" />
    <el-table-column prop="method" label="方法" width="80" />
    <el-table-column prop="path" label="请求路径" />
    <el-table-column prop="response_code" label="状态码" width="90" />
<!--    <el-table-column prop="response_body" label="响应体" />-->
    <el-table-column label="操作" width="100">
      <template #default="{ row }">
        <el-popconfirm title="确定要删除这条日志吗？" @confirm="handleDelete([row.id])">
          <template #reference>
            <el-button size="small" type="danger">删除</el-button>
          </template>
        </el-popconfirm>
      </template>
    </el-table-column>
  </el-table>

  <el-pagination
    v-model:current-page="currentPage"
    v-model:page-size="pageSize"
    layout="total, prev, pager, next"
    :total="total"
    @current-change="handlePageChange"    style="margin-top: 20px; text-align: right"
  />
</template>
<script setup lang="ts">import { ref, onMounted } from 'vue'
import { getSystemLog, delSystemLog } from '@/api/system'
import { ElMessage, ElMessageBox } from 'element-plus'

const logs = ref([])
const selectedIds = ref<number[]>([])
const currentPage = ref(1)
const pageSize = ref(20)
const total = ref(0)

// 搜索表单
const searchForm = ref({
  user: '',
  method: '',
  path: ''
})

// 获取日志数据
const fetchLogs = async (page = 1) => {
  try {
    const res = await getSystemLog({
      page,
      page_size: pageSize.value,
      user: searchForm.value.user,
      method: searchForm.value.method,
      path: searchForm.value.path
    })
    logs.value = res.list || []
    total.value = res.total || 0
    currentPage.value = page
  } catch (error) {
    logs.value = []
    total.value = 0
  }
}

onMounted(() => {
  fetchLogs()
})

// 分页
const handlePageChange = async (page: number) => {
  await fetchLogs(page)
}

// 搜索
const handleSearch = () => {
  fetchLogs()
}

// 重置搜索
const resetSearch = () => {
  searchForm.value = {
    user: '',
    method: '',
    path: ''
  }
  fetchLogs()
}

// 表格选择
const handleSelectionChange = (selection: any[]) => {
  selectedIds.value = selection.map(item => item.id)
}

// 删除操作
const handleDelete = async (ids: number[]) => {
  try {
    await delSystemLog({ "ids": ids })
    fetchLogs(currentPage.value)
  } catch (error) {
    ElMessage.error('删除失败')
  }
}

// 批量删除
const handleBatchDelete = () => {
  if (selectedIds.value.length === 0) {
    ElMessage.warning('请先选择要删除的日志')
    return
  }
  handleDelete(selectedIds.value)
}

// 清空全部日志
const handleClearAll = () => {
  ElMessageBox.confirm(
    '确定要清空所有日志吗？',
    '警告',
    {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    }
  ).then(async () => {
    await handleDelete([])
  }).catch(() => {
    // 用户点击取消或关闭弹窗
    ElMessage.info('操作已取消')
  })
}
</script>
<style scoped>.dialog-footer button:first-child {
  margin-right: 10px;
}
</style>