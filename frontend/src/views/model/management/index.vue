<template>
  <page-header title="模型管理">
    <el-button type="primary" @click="showCreateModelDialog">新建模型</el-button>
  </page-header>

  <el-row :gutter="20">
    <!-- 左侧模型列表 -->
    <el-col :span="6">
      <el-card style="margin-bottom: 20px; background-color: #f9fafc;">
        <div slot="header">
          <span>模型列表</span>
        </div>
        <el-menu
          :default-active="selectedModelId ? String(selectedModelId) : undefined"
          @select="handleModelSelect"
          style="border-right: none;"
        >
          <el-menu-item v-for="model in modelInfos" :key="model.id" :index="String(model.id)">
            <div style="display: flex; justify-content: space-between; width: 100%;">
              {{ model.name }}
              <el-popconfirm title="确定要删除这个模型吗？" @confirm="deleteModel(model)">
                <template #reference>
                  <el-button size="small" type="danger">删除</el-button>
                </template>
              </el-popconfirm>
            </div>
          </el-menu-item>
        </el-menu>
        <el-pagination
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          layout="total, prev, pager, next"
          :total="totalModels"
          @current-change="handleCurrentChange"
        />
      </el-card>
    </el-col>

    <!-- 右侧版本管理 -->
    <el-col :span="18">
      <el-card  class="version-card">
        <div slot="header"  class="version-header">
          <span>版本管理 - {{ selectedModelName }}</span>
          <el-button type="primary" @click="showCreateVersionDialog">上传版本</el-button>
        </div>

        <el-table :data="modelVersions" border style="width: 100%;" @selection-change="handleSelectionChange">
          <el-table-column type="selection" width="55" />
          <el-table-column prop="version" label="版本号" />
          <el-table-column prop="accuracy" label="准确率" />
          <el-table-column prop="loss" label="损失值" />
          <el-table-column prop="is_deployed" label="是否部署">
            <template #default="{ row }">
              <el-switch v-model="row.is_deployed" @change="toggleDeploy(row)" />
            </template>
          </el-table-column>
          <el-table-column label="操作" width="150">
            <template #default="{ row }">
              <el-button size="small" type="primary" @click="downloadVersion(row)">下载</el-button>
              <el-popconfirm title="确定要删除这个版本吗？" @confirm="deleteVersion(row)"  :disabled="row.is_deployed">
                <template #reference>
                  <el-button size="small" type="danger"  :disabled="row.is_deployed">删除</el-button>
                </template>
              </el-popconfirm>
            </template>
          </el-table-column>
        </el-table>
        <el-pagination
          v-model:current-page="currentVersionPage"
          v-model:page-size="versionPageSize"
          layout="total, prev, pager, next"
          :total="totalVersions"
          @current-change="handleVersionCurrentChange"
        />
      </el-card>

      <el-card style="margin-top: 20px;">
        <div slot="header" class="version-header">
          <span>对话日志</span>
          <el-button type="primary" :disabled="!selectedModelId" @click="refreshChatLogs">刷新</el-button>
        </div>

        <el-form :inline="true" size="small" style="margin-bottom: 10px;">
          <el-form-item label="关键字">
            <el-input v-model="chatFilters.keyword" placeholder="输入/输出匹配" clearable />
          </el-form-item>
          <el-form-item label="时间范围">
            <el-date-picker
              v-model="chatFilters.dateRange"
              type="datetimerange"
              range-separator="至"
              start-placeholder="开始时间"
              end-placeholder="结束时间"
              value-format="YYYY-MM-DDTHH:mm:ssZ"
              style="width: 360px;"
            />
          </el-form-item>
          <el-form-item>
            <el-button type="primary" :disabled="!selectedModelId" @click="handleChatSearch">查询</el-button>
            <el-button @click="resetChatFilters">重置</el-button>
          </el-form-item>
        </el-form>

        <el-table :data="chatLogs" border style="width: 100%;" v-loading="chatLoading">
          <el-table-column prop="created_at" label="时间" width="180" />
          <el-table-column prop="model_version_detail.version" label="版本" width="120">
            <template #default="{ row }">
              {{ row.model_version_detail?.version || '-' }}
            </template>
          </el-table-column>
          <el-table-column prop="input_text" label="输入" show-overflow-tooltip>
            <template #default="{ row }">
              {{ row.input_text }}
            </template>
          </el-table-column>
          <el-table-column prop="output_text" label="输出" show-overflow-tooltip>
            <template #default="{ row }">
              {{ row.output_text || '-' }}
            </template>
          </el-table-column>
        </el-table>
        <el-pagination
          style="margin-top: 10px;"
          v-model:current-page="chatPage"
          v-model:page-size="chatPageSize"
          layout="total, prev, pager, next"
          :total="chatTotal"
          @current-change="handleChatPageChange"
        />
      </el-card>
    </el-col>
  </el-row>

  <!-- 新建模型对话框 -->
  <el-dialog v-model="dialogVisible" title="新建模型">
    <el-form :model="newModelForm" label-width="120px">
      <el-form-item label="模型名称">
        <el-input v-model="newModelForm.name" />
      </el-form-item>
      <el-form-item label="描述">
        <el-input v-model="newModelForm.description" type="textarea" />
      </el-form-item>
    </el-form>
    <template #footer>
      <el-button @click="dialogVisible = false">取消</el-button>
      <el-button type="primary" @click="createModel">确定</el-button>
    </template>
  </el-dialog>

  <!-- 新建版本对话框 -->
  <el-dialog v-model="versionDialogVisible" title="创建版本">
    <el-form :model="newVersionForm" label-width="120px">
      <el-form-item label="版本号">
        <el-input v-model="newVersionForm.version" placeholder="例如：v1.0.0" />
      </el-form-item>
      <el-form-item label="模型文件">
        <el-upload
          :action="'#'"
          :http-request="customUpload"
          :before-upload="beforeUpload"
          :limit="1"
          :on-exceed="handleExceed"
          :file-list="fileList"
          accept=".pt,.zip,.pth,.pkl"
        >
          <el-button type="primary">点击上传</el-button>
        </el-upload>
      </el-form-item>
    </el-form>
    <template #footer>
      <el-button @click="versionDialogVisible = false">取消</el-button>
      <el-button type="primary" @click="submitVersion">提交</el-button>
    </template>
  </el-dialog>
</template>
<style scoped>.version-card {
  background-color: #fff;
  padding: 20px;
}

.version-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px; /* 调整这个值来控制间距 */
}
</style>
<script setup lang="ts">
import { ref, onMounted } from 'vue'
import {
  modelManagementApi,
  modelInfoApi
} from '@/api/modelManagement'
import { modelChatLogApi } from '@/api/modelChatLog'
import { ElMessage } from 'element-plus'
import { nextTick } from 'vue'

type ModelInfo = { id: number; name: string; description?: string }
type ModelVersion = {
  id: number
  version: string
  accuracy?: number
  loss?: number
  is_deployed?: boolean
  model_file?: string
  [key: string]: any
}
type ChatLog = {
  id: number
  input_text: string
  output_text?: string
  created_at: string
  model_version_detail?: { version?: string }
}
// 模型数据
const modelInfos = ref<ModelInfo[]>([])
const modelVersions = ref<ModelVersion[]>([])
const selectedModelId = ref<number | null>(null)
const selectedModelName = ref('请选择模型')

onMounted(async () => {
  await fetchModelInfos()

  if (modelInfos.value.length > 0) {
    const firstModel = modelInfos.value[0]
    await handleModelSelect(firstModel.id)

    // 可选：滚动到顶部
    nextTick(() => {
      const elMenu = document.querySelector('.el-menu')
      if (elMenu) {
        elMenu.scrollTop = 0
      }
    })
  }
})

// 上传文件（使用新的模型专用上传接口）
const customUpload = async (options: { file: File }) => {
  const { file } = options

  try {
    const res = await modelManagementApi.uploadModelFile(file) // 使用模型专用上传接口
    // 手动处理成功回调（不使用 el-upload 的 on-success，避免重复触发）
    handleUploadSuccess(res, { raw: file })
    // 返回成功，让 el-upload 组件知道上传完成
    return res
  } catch (error) {
    ElMessage.error('文件上传失败')
    // 抛出错误，让 el-upload 组件知道上传失败
    throw error
  }
}

// 新建模型表单
const dialogVisible = ref(false)
const newModelForm = ref<{ name: string; description: string }>({
  name: '',
  description: ''
})

// 新建版本表单
const versionDialogVisible = ref(false)
const newVersionForm = ref<{ version: string; file: File | string | null; model_id: number | null }>({
  version: '',
  file: null,
  model_id: null
})
const fileList = ref<any[]>([])

// 分页相关
const currentPage = ref(1)
const pageSize = ref(10)
const totalModels = ref(0)
const currentVersionPage = ref(1)
const versionPageSize = ref(10)
const totalVersions = ref(0)

// 对话日志
const chatLogs = ref<ChatLog[]>([])
const chatPage = ref(1)
const chatPageSize = ref(10)
const chatTotal = ref(0)
const chatLoading = ref(false)
const chatFilters = ref<{ keyword: string; dateRange: string[] }>({
  keyword: '',
  dateRange: []
})

// 获取所有模型
const fetchModelInfos = async (page = 1) => {
  try {
    const res = await modelInfoApi.fetchModelInfos({ page, page_size: pageSize.value })
    modelInfos.value = res.list || []
    totalModels.value = res.total || 0
  } catch (error) {
    modelInfos.value = []
    totalModels.value = 0
  }
}

// 切换模型时加载对应版本
const handleModelSelect = async (modelId: number | string) => {
  selectedModelId.value = Number(modelId)
  const model = modelInfos.value.find(m => m.id === Number(modelId))
  selectedModelName.value = model?.name || '未知模型'

  // 加载模型版本
  try {
    const res = await modelManagementApi.fetchModelVersions({ model_id: modelId, page: 1, page_size: versionPageSize.value })
    modelVersions.value = res.list || []
    totalVersions.value = res.total || 0
  } catch (error) {
    modelVersions.value = []
    totalVersions.value = 0
  }

  await fetchChatLogs(1)
}

// 处理模型分页变化
const handleCurrentChange = async (val: number) => {
  currentPage.value = val
  await fetchModelInfos(val)
}

// 处理版本分页变化
const handleVersionCurrentChange = async (val: number) => {
  currentVersionPage.value = val
  if (selectedModelId.value) {
    const res = await modelManagementApi.fetchModelVersions({
      model_id: selectedModelId.value,
      page: val,
      page_size: versionPageSize.value
    })
    modelVersions.value = res.list || []
    totalVersions.value = res.total || 0
  }
}

// 加载对话日志
const fetchChatLogs = async (page = 1) => {
  if (!selectedModelId.value) {
    chatLogs.value = []
    chatTotal.value = 0
    return
  }

  chatLoading.value = true
  const params: any = {
    model_id: selectedModelId.value,
    page,
    page_size: chatPageSize.value
  }

  if (chatFilters.value.keyword) {
    params.keyword = chatFilters.value.keyword
  }
  if (chatFilters.value.dateRange && chatFilters.value.dateRange.length === 2) {
    params.start_time = chatFilters.value.dateRange[0]
    params.end_time = chatFilters.value.dateRange[1]
  }

  try {
    const res = await modelChatLogApi.fetchChatLogs(params)
    chatLogs.value = res.list || []
    chatTotal.value = res.total || 0
    chatPage.value = page
  } catch (error) {
    chatLogs.value = []
    chatTotal.value = 0
  } finally {
    chatLoading.value = false
  }
}

const handleChatSearch = async () => {
  await fetchChatLogs(1)
}

const handleChatPageChange = async (val: number) => {
  chatPage.value = val
  await fetchChatLogs(val)
}

const resetChatFilters = async () => {
  chatFilters.value = {
    keyword: '',
    dateRange: []
  }
  await fetchChatLogs(1)
}

const refreshChatLogs = async () => {
  await fetchChatLogs(chatPage.value)
}

// 显示新建模型对话框
const showCreateModelDialog = () => {
  newModelForm.value = { name: '', description: '' }
  dialogVisible.value = true
}

// 提交新建模型
const createModel = async () => {
  if (!newModelForm.value.name.trim()) {
    ElMessage.warning('请输入模型名称')
    return
  }

  try {
    await modelInfoApi.createModel(newModelForm.value)
    ElMessage.success('模型创建成功')
    dialogVisible.value = false
    await fetchModelInfos()
  } catch (error) {
    return
  }
}

// 删除模型
const deleteModel = async (model: ModelInfo) => {
  try {
    await modelInfoApi.deleteModel(model.id)
    ElMessage.success(`模型 ${model.name} 删除成功`)
    await fetchModelInfos()
  } catch (error) {
    return
  }
}

// 显示新建版本对话框
const showCreateVersionDialog = () => {
  if (!selectedModelId.value) {
    ElMessage.warning('请先选择一个模型')
    return
  }

  newVersionForm.value = {
    version: '',
    file: null,
    model_id: selectedModelId.value
  }
  fileList.value = []
  versionDialogVisible.value = true
}

// 文件上传前检查
const beforeUpload = (file: File) => {
  const isValid = file.type === 'application/zip' || file.name.endsWith('.pt')
  if (!isValid) {
    ElMessage.error('只能上传 .pt 或 zip 文件')
  }
  return isValid
}

// 上传成功回调
const handleUploadSuccess = (response: any, file: { raw: File }) => {
  console.log('Upload response:', response) // 调试信息
  
  // 响应拦截器会自动解包，最终返回的是 data 对象
  // 后端返回: { code: 200, msg: "上传成功", data: { file_path: "models/..." } }
  // 拦截器处理后: { file_path: "models/..." }
  let filePath = null
  
  if (response) {
    // 如果响应拦截器已经解包，response 就是 data 对象
    if (response.file_path) {
      filePath = response.file_path
    } 
  
 
  }
  console.log('filePath:', filePath)
  
  if (filePath) {
    newVersionForm.value.file = filePath
    ElMessage.success('文件上传成功')
  } else {
    ElMessage.error('获取文件路径失败，请查看控制台')
    console.error('无法从响应中提取 file_path，完整响应:', JSON.stringify(response, null, 2))
  }
}

// 文件超出限制提示
const handleExceed = () => {
  ElMessage.warning('最多上传一个文件')
}

// 提交新版本
const submitVersion = async () => {
  const { version, file, model_id } = newVersionForm.value

  if (!version || !file || !model_id) {
    ElMessage.warning('请填写完整信息')
    return
  }

  const formData = new FormData()
  formData.append('version', version)
  formData.append('model_info', String(model_id))
  formData.append('model_file', file as any)

  try {
    await modelManagementApi.createModelVersion(formData)
    ElMessage.success('版本创建成功')
    versionDialogVisible.value = false
    await handleModelSelect(model_id)
  } catch (error) {
    return
  }
}

// 上线/下线版本
const toggleDeploy = async (row: ModelVersion) => {
  try {
    await modelManagementApi.deployModel(row.id, !!row.is_deployed)
    ElMessage.success(`${row.is_deployed ? '上线' : '下线'}成功`)
  } catch (error) {
    row.is_deployed = !row.is_deployed
  }
}
// 选择的版本
const selectedVersions = ref<ModelVersion[]>([])

// 删除版本
const deleteVersion = async (row: ModelVersion) => {
  try {
    await modelManagementApi.deleteModelVersion(row.id)
    ElMessage.success(`版本 ${row.version} 删除成功`)
    if (selectedModelId.value !== null) {
      await handleModelSelect(selectedModelId.value)
    }
  } catch (error) {
    return
  }
}

// 表格多选
const handleSelectionChange = (selection: ModelVersion[]) => {
  selectedVersions.value = selection
}

// 下载模型版本（最小修改：直接把返回值当 Blob，不再读取 headers）
const downloadVersion = async (row: ModelVersion) => {
  try {
    const blob = await modelManagementApi.downloadModelVersion(row.id)
    console.log(blob)
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    const filename = (row?.model_file && String(row.model_file).split('/').pop()) || `${row.version || 'model'}.bin`
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
    ElMessage.success('下载开始')
  } catch (error) {
    ElMessage.error('下载失败')
    console.error('Download error:', error)
  }
}

</script>
