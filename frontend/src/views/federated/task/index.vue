
<template>
  <page-header title="联邦任务管理">
    <el-button type="primary" @click="createTask">新建任务</el-button>
  </page-header>

  <el-table :data="tasks" border style="width: 100%">
    <el-table-column prop="id" label="任务ID" />
    <el-table-column prop="name" label="任务名称" />
    <el-table-column prop="status" label="状态" />
    <el-table-column prop="model_info_detail.name" label="模型" />
    <el-table-column prop="model_version_detail.version" label="版本" />
    <el-table-column prop="region_node_detail.name" label="区域服务器" />
    <el-table-column prop="rounds" label="训练轮次" />
    <el-table-column prop="participation_rate" label="参与率" />
    <el-table-column prop="aggregation_method" label="聚合方式" />
    <el-table-column prop="created_at" label="创建时间" width="160" />
    <el-table-column prop="progress" label="进度">
      <template #default="{ row }">
        <el-progress :percentage="row.progress || 0" />
      </template>
    </el-table-column>
    <el-table-column label="操作">
      <template #default="{ row }">
        <el-button size="small" type="primary" @click="editTask(row)">编辑</el-button>
        <el-popconfirm title="确定要删除这个任务吗？" @confirm="deleteTask(row.id)">
          <template #reference>
            <el-button size="small" type="danger">删除</el-button>
          </template>
        </el-popconfirm>
        <el-button size="small" type="warning" @click="pauseTask(row)" v-if="row.status === 'running'">暂停</el-button>
        <el-button size="small" type="success" @click="resumeTask(row)" v-if="row.status === 'paused'">继续</el-button>
      </template>
    </el-table-column>
  </el-table>

  <!-- 分页 -->
  <el-pagination
    v-model:current-page="currentPage"
    v-model:page-size="pageSize"
    layout="total, prev, pager, next"
    :total="total"
    @current-change="handlePageChange"    style="margin-top: 20px; text-align: right"
  />

  <!-- 对话框 -->
  <el-dialog v-model="dialogVisible" :title="isEdit ? '编辑任务' : '新建任务'">
    <el-form :model="taskForm" label-width="120px" :rules="rules" ref="formRef">
      <el-form-item label="任务名称" prop="name">
        <el-input v-model="taskForm.name" />
      </el-form-item>
      <el-form-item label="任务描述">
        <el-input v-model="taskForm.description" type="textarea" />
      </el-form-item>

    <el-form-item label="模型" prop="model_info">
    <el-select v-model="taskForm.model_info" placeholder="请选择模型" style="width: 100%" @change="handleModelChange">
      <el-option
        v-for="model in modelInfos"
        :key="model.id"
        :label="model.name"
        :value="model.id"
      />
    </el-select>
  </el-form-item>

  <el-form-item label="模型版本" prop="model_version">
    <el-select v-model="taskForm.model_version" placeholder="请选择模型版本" style="width: 100%">
      <el-option
        v-for="version in modelVersions"
        :key="version.id"
        :label="version.version"
        :value="version.id"
      />
    </el-select>
  </el-form-item>

    <el-form-item label="区域服务器" prop="region_node">
      <el-select v-model="taskForm.region_node" placeholder="请选择区域服务器" style="width: 100%">
        <el-option
          v-for="node in regionNodes"
          :key="node.id"
          :label="node.name"
          :value="node.id"
        />
      </el-select>
    </el-form-item>
<!--      <el-form-item label="训练轮次" prop="rounds">-->
<!--        <el-input-number v-model="taskForm.rounds" :default="10" :min="1" />-->
<!--      </el-form-item>-->
<!--      <el-form-item label="聚合方式" prop="aggregation_method">-->
<!--        <el-select v-model="taskForm.aggregation_method" placeholder="请选择" style="width: 100%">-->
<!--          <el-option-->
<!--            v-for="method in aggregationMethods"-->
<!--            :key="method"-->
<!--            :label="method"-->
<!--            :value="method"-->
<!--          />-->
<!--        </el-select>-->
<!--      </el-form-item>-->
    </el-form>
    <template #footer>
      <span class="dialog-footer">
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="submitTask">确定</el-button>
      </span>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">import { ref, onMounted } from 'vue'
import { federatedTaskModel } from '@/api/federatedTask'
import { systemConfigModel } from '@/api/systemConfig'
import { regionNodeApi } from '@/api/nodeManagement'
import { modelInfoApi, modelManagementApi } from '@/api/modelManagement'
import { ElPopconfirm, ElMessage, type FormRules } from 'element-plus'

// 任务数据
const tasks = ref([])
const currentPage = ref(1)
const pageSize = ref(20)
const total = ref(0)

// 对话框相关
const dialogVisible = ref(false)
const isEdit = ref(false)
const formRef = ref()
defineExpose({ formRef })
// 新增字段
const modelInfos = ref([])
const modelVersions = ref([])
const regionNodes = ref([])

// 获取模型信息列表
const fetchModelInfos = async () => {
  try {
    const res = await modelInfoApi.fetchModelInfos({})
    modelInfos.value = res.list || []
  } catch (error) {
    ElMessage.error('获取模型信息失败')
  }
}

// 获取模型版本列表（根据模型）
const fetchModelVersions = async (modelId = null) => {
  try {
    const res = await modelManagementApi.fetchModelVersions({
      model_id: modelId
    })
    modelVersions.value = res.list || []
    return Promise.resolve()
  } catch (error) {
     ElMessage.error('获取模型版本失败')
     return Promise.reject()
  }
}

// 模型选择变化时，更新版本列表
const handleModelChange = (modelId: number | null) => {
  if (modelId) {
    fetchModelVersions(modelId)
  } else {
    modelVersions.value = []
  }
}

// 获取区域服务器列表
const fetchRegionNodes = async () => {
  try {
    const res = await regionNodeApi.fetchRegionNodes({})
    regionNodes.value = res.list || []
  } catch (error) {
    ElMessage.error('获取区域服务器失败')
  }
}
// 表单验证规则
const rules = ref<FormRules>({
  name: [
    { required: true, message: '任务名称不能为空', trigger: 'blur' },
    { min: 2, max: 50, message: '长度在2到50个字符之间', trigger: 'blur' }
  ],
  // rounds: [
  //   { required: true, message: '训练轮次必须大于0', trigger: 'change' },
  //   { type: 'number', min: 1, message: '轮次不能小于1', trigger: 'blur' }
  // ],
  model_info: [
    { required: true, message: '请选择模型', trigger: 'change' }
  ],
  model_version: [
    { required: true, message: '请选择模型版本', trigger: 'change' }
  ],
  region_node: [
    { required: true, message: '请选择区域服务器', trigger: 'change' }
  ]
})

// 任务表单数据
// 表单数据
const taskForm = ref({
  id: null,
  name: '',
  description: '',
  // rounds: 10,
  // aggregation_method: 'fedavg',
  model_info: null,
  model_version: null,
  region_node: null
})

// 可用的聚合方法列表
const aggregationMethods = ref<string[]>([])

// 获取任务列表（带分页）
const fetchTasks = async (page = 1) => {
  try {
    const res = await federatedTaskModel.getTasksApi({
      page,
      page_size: pageSize.value
    })

    tasks.value = res.list || []
    total.value = res.total || 0
    currentPage.value = page
  } catch (error) {
    tasks.value = []
    total.value = 0
  }
}

// 获取可用的聚合方法列表
const fetchAggregationMethods = async () => {
  try {
    const res = await systemConfigModel.getAggregationMethodApi({})
    if (Array.isArray(res)) {
      aggregationMethods.value = res
      // 如果聚合如果聚合方法列表不为空方法列表不为空，设置默认值
      if (aggregationMethods.value.length > 0) {
        taskForm.value.aggregation_method = aggregationMethods.value[0]
      }
    } else {
      // 如果后端返回的数据格式不符合预期，使用默认值
      aggregationMethods.value = ['fedavg', 'fedprox']
      taskForm.value.aggregation_method = 'fedavg'
    }
  } catch (error) {
    // 如果请求失败，使用默认值
    aggregationMethods.value = ['fedavg', 'fedprox']
    taskForm.value.aggregation_method = 'fedavg'
    ElMessage.warning('使用默认聚合方式，请检查网络连接')
  }
}

onMounted(async () => {
  await Promise.all([
    fetchTasks(),
    // fetchAggregationMethods(),
    fetchModelInfos(),
    fetchRegionNodes(),
    // fetchModelVersions(),
  ])
})

// 翻页处理
const handlePageChange = async (page: number) => {
  await fetchTasks(page)
}

// 新建任务
const createTask = () => {
  isEdit.value = false
  taskForm.value = {
    id: null,
    name: '',
    description: '',
    // rounds: 10,
    // aggregation_method: aggregationMethods.value.length > 0 ? aggregationMethods.value[0] : 'fedavg',
    model_info: null,
    model_version: null,
    region_node: null
  }
  dialogVisible.value = true
}

// 编辑任务
const editTask = (row) => {
  isEdit.value = true
  taskForm.value = { ...row }

  // 确保聚合方式是有效值
  // if (aggregationMethods.value.includes(row.aggregation_method)) {
  //   taskForm.value.aggregation_method = row.aggregation_method
  // } else if (aggregationMethods.value.length > 0) {
  //   taskForm.value.aggregation_method = aggregationMethods.value[0]
  // }

  // 如果有 model_info，加载模型版本并选中当前版本
  if (taskForm.value.model_info) {
    fetchModelVersions(taskForm.value.model_info).then(() => {
      taskForm.value.model_version = row.model_version
    })
  } else {
    modelVersions.value = []
    taskForm.value.model_version = null
  }

  dialogVisible.value = true
}

// 提交表单（新增/更新）
const submitTask = async () => {
  try {
    await formRef.value.validate()
    if (isEdit.value) {

      await federatedTaskModel.updateTaskApi(taskForm.value)
      ElMessage.success('任务编辑成功')
    } else {
      await federatedTaskModel.createTaskApi(taskForm.value)
      ElMessage.success('任务创建成功')
    }

    dialogVisible.value = false
    await fetchTasks(currentPage.value)
  } catch (error) {
    return
  }
}

// 删除任务
const deleteTask = async (taskId) => {
  try {
    await federatedTaskModel.deleteTaskApi({ id: taskId })
    ElMessage.success('任务删除成功')
    await fetchTasks(currentPage.value)
  } catch (error) {
    return
  }
}

// 暂停任务
const pauseTask = async (row) => {
  try {
    await federatedTaskModel.pauseTaskApi({ id: row.id })
    ElMessage.success('任务已暂停')
    await fetchTasks(currentPage.value)
  } catch (error) {
    return
  }
}

// 继续任务
const resumeTask = async (row) => {
  try {
    await federatedTaskModel.resumeTaskApi({ id: row.id })
    ElMessage.success('任务已继续')
    await fetchTasks(currentPage.value)
  } catch (error) {
    return
  }
}
</script>

<style scoped>.dialog-footer button:first-child {
  margin-right: 10px;
}
</style>
