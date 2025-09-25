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
          :default-active="selectedModelId"
          @select="handleModelSelect"
          style="border-right: none;"
        >
          <el-menu-item v-for="model in modelInfos" :key="model.id" :index="String(model.id)">
            <div style="display: flex; justify-content: space-between; width: 100%;">
              {{ model.name }}
              <el-popconfirm title="确定要删除这个模型吗？" @confirm="deleteModel(model)">
                <template #reference>
                  <el-button size="mini" type="danger">删除</el-button>
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
          :on-success="handleUploadSuccess"
          :http-request="customUpload"
          :before-upload="beforeUpload"
          :limit="1"
          :on-exceed="handleExceed"
          :file-list="fileList"
          accept=".pt,.zip"
          :data="{ upload_to_path: 'models' }"
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
import { ElMessage } from 'element-plus'
import { uploadFile } from '@/api/system'
import { nextTick } from 'vue'
// 模型数据
const modelInfos = ref([])
const modelVersions = ref([])
const selectedModelId = ref(null)
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

// 上传文件
const customUpload = async (options) => {
  const { file } = options
  const formData = new FormData()
  formData.append('file', file)
  formData.append('upload_to_path', 'models')

  try {
    const res = await uploadFile(formData) // 使用封装好的 API 请求
    handleUploadSuccess(res, { raw: file }) // 手动触发 on-success
  } catch (error) {
    return
    // 可以在这里做错误处理或提示
  }
}

// 新建模型表单
const dialogVisible = ref(false)
const newModelForm = ref({
  name: '',
  description: ''
})

// 新建版本表单
const versionDialogVisible = ref(false)
const newVersionForm = ref({
  version: '',
  file: null as File | null,
  model_id: null as number | null
})
const fileList = ref([])

// 分页相关
const currentPage = ref(1)
const pageSize = ref(10)
const totalModels = ref(0)
const currentVersionPage = ref(1)
const versionPageSize = ref(10)
const totalVersions = ref(0)

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
const handleModelSelect = async (modelId) => {
  selectedModelId.value = modelId
  const model = modelInfos.value.find(m => m.id === parseInt(modelId))
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
}

// 处理模型分页变化
const handleCurrentChange = async (val) => {
  currentPage.value = val
  await fetchModelInfos(val)
}

// 处理版本分页变化
const handleVersionCurrentChange = async (val) => {
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
const deleteModel = async (model) => {
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
const beforeUpload = (file) => {
  const isValid = file.type === 'application/zip' || file.name.endsWith('.pt')
  if (!isValid) {
    ElMessage.error('只能上传 .pt 或 zip 文件')
  }
  return isValid
}

// 上传成功回调
const handleUploadSuccess = (response, file) => {
  newVersionForm.value.file = response.file_path
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
  formData.append('model_info', model_id)
  formData.append('model_file', file)

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
const toggleDeploy = async (row) => {
  try {
    await modelManagementApi.deployModel(row.id, row.is_deployed)
    ElMessage.success(`${row.is_deployed ? '上线' : '下线'}成功`)
  } catch (error) {
    row.is_deployed = !row.is_deployed
  }
}
// 选择的版本
const selectedVersions = ref([])

// 删除版本
const deleteVersion = async (row) => {
  try {
    await modelManagementApi.deleteModelVersion(row.id)
    ElMessage.success(`版本 ${row.version} 删除成功`)
    await handleModelSelect(selectedModelId.value)
  } catch (error) {
    return
  }
}

// 表格多选
const handleSelectionChange = (selection) => {
  selectedVersions.value = selection
}

// 下载模型版本（最小修改：直接把返回值当 Blob，不再读取 headers）
const downloadVersion = async (row) => {
  try {
    const blob = await modelManagementApi.downloadModelVersion(row.id)
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

onMounted(async () => {
  await fetchModelInfos()
})
</script>
