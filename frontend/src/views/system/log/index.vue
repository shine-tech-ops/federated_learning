<template>
  <page-header title="系统日志">
    <div>
      <el-date-picker
        v-model="search.date"
        type="daterange"
        range-separator="To"
        class="mr-3"
        value-format="YYYY-MM-DD HH:mm:ss"
        start-placeholder="开始日期"
        end-placeholder="结束日期"
        :default-time="defaultTime"
        :clearable="false"
        @change="getList"
      >
      </el-date-picker>
      <el-button
        type="primary"
        icon="Upload"
        @click="exportExcel()"
        v-auth="['edit_user_management']"
        >导出日志</el-button
      >
    </div>
  </page-header>
  <div class="mt-4">
    <elx-table
      v-model:loading="loading"
      :columns="tableColumns"
      v-model:data="tableData"
      :pi="page.page"
      :ps="page.pageSize"
      :total="page.total"
      :page="{ show: true, showSize: false }"
      @change="handleChange"
    ></elx-table>
  </div>
</template>
<script lang="ts" setup name="systemLog">
import { formatDate } from '@/utils/formatDate'
import { sysModel } from '@/api'
import { exportToCsv } from '@/utils'
import { getDefaultDateRange } from '@/utils/index'

// 时分秒 按00:00:00 - 23:59:59
const defaultTime = ref<[Date, Date]>([
  new Date(2000, 1, 1, 0, 0, 0),
  new Date(2000, 2, 1, 23, 59, 59)
])

// 默认一周
const defaultValue = getDefaultDateRange()

const search = ref({
  date: defaultValue as any
})

const page = ref({
  total: 0,
  page: 1,
  pageSize: 20
})

const loading = ref(false)
const tableColumns: TableColumnType[] = [
  { prop: 'operation_time', label: '操作时间' },
  { prop: 'user_name', label: '用户名' },
  { prop: 'role_name', label: '角色名' },
  { prop: 'content', label: '操作内容' }
]

const tableData = ref<TableDataType[]>([])

const getDateParams = () => {
  const date = search.value.date
  return { operation_time_from: date[0], operation_time_to: date[1] }
}

const getList = () => {
  loading.value = true
  const dateParams = getDateParams()
  const params = { page: page.value.page, page_size: page.value.pageSize, ...dateParams }
  sysModel
    .getSystemLog(params)
    .then((res: any) => {
      tableData.value = res.list
      page.value.total = res.total
      page.value.page = res.page
    })
    .finally(() => {
      loading.value = false
    })
}

const handleChange = (p: ElxTable.Change) => {
  console.log('p', p)
  if (p.type === 'ps' && p.ps) {
    page.value.pageSize = p.ps
  }

  if (p.type === 'pi' && p.pi) {
    page.value.page = p.pi
  }

  getList()
}

const exportExcel = () => {
  const dateParams = getDateParams()
  const params = { page: page.value.page, page_size: page.value.pageSize, ...dateParams, export: 1 }
  sysModel.exportSystemLog(params).then((res: any) => {
    console.log(res)
    exportToCsv(res, `系统日志${params.operation_time_from}_${params.operation_time_to}`)
  })
}

onMounted(() => {
  getList()
})
</script>
