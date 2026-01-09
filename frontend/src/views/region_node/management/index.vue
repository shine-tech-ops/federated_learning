<template>
  <page-header title="区域节点管理">
    <el-button type="primary" @click="createRegionNode">新建区域节点</el-button>
  </page-header>

  <el-table :data="regionNodes" border style="width: 100%">
    <el-table-column prop="id" label="ID" />
    <el-table-column prop="name" label="名称" />
    <el-table-column prop="description" label="描述" />
    <el-table-column prop="ip_address" label="IP地址" />
    <el-table-column prop="created_at" label="创建时间" width="160" />
    <el-table-column label="操作">
      <template #default="{ row }">
        <el-button size="small" type="primary" @click="editRegionNode(row)">编辑</el-button>
        <el-popconfirm title="确定要删除这个区域节点吗？" @confirm="deleteRegionNode(row.id)">
          <template #reference>
            <el-button size="small" type="danger">删除</el-button>
          </template>
        </el-popconfirm>
      </template>
    </el-table-column>
  </el-table>

  <!-- 分页 -->
  <el-pagination
    v-model:current-page="currentPage"
    v-model:page-size="pageSize"
    layout="total, prev, pager, next"
    :total="total"
    @current-change="handlePageChange"
    style="margin-top: 20px; text-align: right"
  />

  <!-- 对话框 -->
  <el-dialog v-model="dialogVisible" :title="isEdit ? '编辑区域节点' : '新建区域节点'">
    <el-form :model="form" label-width="120px" :rules="rules" ref="formRef">
      <el-form-item label="名称" prop="name">
        <el-input v-model="form.name" />
      </el-form-item>
      <el-form-item label="描述">
        <el-input v-model="form.description" type="textarea" />
      </el-form-item>
      <el-form-item label="IP地址" prop="ip_address">
        <el-input v-model="form.ip_address" />
      </el-form-item>
    </el-form>
    <template #footer>
      <span class="dialog-footer">
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="submitRegionNode">确定</el-button>
      </span>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { regionNodeApi } from '@/api/nodeManagement'
import { ElMessage, ElPopconfirm, type FormRules } from 'element-plus'


const regionNodes = ref([])
const currentPage = ref(1)
const pageSize = ref(20)
const total = ref(0)

const dialogVisible = ref(false)
const isEdit = ref(false)
const form = ref({
  id: null,
  name: '',
  description: '',
  ip_address: ''
})

const rules = ref<FormRules>({
  name: [
    { required: true, message: '名称不能为空', trigger: 'blur' },
    { min: 2, max: 20, message: '名称长度在2到20个字符之间', trigger: 'blur' }
  ],
  ip_address: [
    { required: true, message: '请输入合法的IP地址', trigger: 'blur' },
    { pattern: /^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$/, message: 'IP地址格式不正确', trigger: 'blur' }
  ]
})

const fetchRegionNodes = async (page = 1) => {
  try {
    const res = await regionNodeApi.fetchRegionNodes({ page, page_size: pageSize.value })
    regionNodes.value = res.list || []
    total.value = res.total || 0
    currentPage.value = page
  } catch (error) {
    regionNodes.value = []
    total.value = 0
  }
}

onMounted(async () => {
  await fetchRegionNodes()
})

const handlePageChange = async (page: number) => {
  await fetchRegionNodes(page)
}

const createRegionNode = () => {
  isEdit.value = false
  form.value = {
    id: null,
    name: '',
    description: ''
  }
  dialogVisible.value = true
}

const editRegionNode = (row) => {
  isEdit.value = true
  form.value = { ...row }
  dialogVisible.value = true
}

const submitRegionNode = async () => {
  try {
    if (isEdit.value) {
      await regionNodeApi.updateRegionNode(form.value)
      ElMessage.success('区域节点更新成功')
    } else {
      await regionNodeApi.createRegionNode(form.value)
      ElMessage.success('区域节点创建成功')
    }

    dialogVisible.value = false
    await fetchRegionNodes(currentPage.value)
  } catch (error) {
    return
  }
}

const deleteRegionNode = async (id: number) => {
  try {
    await regionNodeApi.deleteRegionNode(id)
    ElMessage.success('区域节点删除成功')
    await fetchRegionNodes(currentPage.value)
  } catch (error) {
    return
  }
}
</script>
