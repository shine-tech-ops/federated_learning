<template>
  <page-header title="边缘节点管理">
    <el-button type="primary" @click="createEdgeNode">新建边缘节点</el-button>
  </page-header>

  <el-table :data="edgeNodes" border style="width: 100%">
    <el-table-column prop="id" label="ID" />
    <el-table-column prop="device_id" label="设备ID" />
    <el-table-column prop="region_node_detail.name" label="所属区域" />
    <el-table-column prop="last_heartbeat" label="最后心跳时间" width="160" />
    <el-table-column prop="status" label="状态" width="100">
      <template #default="{ row }">
        <el-tag v-if="row.status === 'online'" type="success">在线</el-tag>
        <el-tag v-else-if="row.status === 'offline'" type="danger">离线</el-tag>
        <el-tag v-else>维护</el-tag>
      </template>
    </el-table-column>
    <el-table-column prop="created_at" label="创建时间" width="160" />
<!--    <el-table-column label="操作">-->
<!--      <template #default="{ row }">-->
<!--        <el-button size="small" type="primary" @click="editEdgeNode(row)">编辑</el-button>-->
<!--        <el-popconfirm title="确定要删除这个边缘节点吗？" @confirm="deleteEdgeNode(row.id)">-->
<!--          <template #reference>-->
<!--            <el-button size="small" type="danger">删除</el-button>-->
<!--          </template>-->
<!--        </el-popconfirm>-->
<!--      </template>-->
<!--    </el-table-column>-->
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
  <el-dialog v-model="dialogVisible" :title="isEdit ? '编辑边缘节点' : '新建边缘节点'">
    <el-form :model="form" label-width="120px" ref="formRef">
      <el-form-item label="名称" prop="name">
        <el-input v-model="form.name" />
      </el-form-item>
      <el-form-item label="所属区域">
        <el-select v-model="form.region_node" placeholder="请选择区域" style="width: 100%">
          <el-option
            v-for="region in regionList"
            :key="region.id"
            :label="region.name"
            :value="region.id"
          />
        </el-select>
      </el-form-item>
    </el-form>
    <template #footer>
      <span class="dialog-footer">
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="submitEdgeNode">确定</el-button>
      </span>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { edgeNodeApi, regionNodeApi } from '@/api/nodeManagement'
import { ElMessage, ElPopconfirm } from 'element-plus'

const edgeNodes = ref([])
const regionList = ref([])
const currentPage = ref(1)
const pageSize = ref(20)
const total = ref(0)

const dialogVisible = ref(false)
const isEdit = ref(false)
const form = ref({
  id: null,
  name: '',
  region_id: null
})

const fetchEdgeNodes = async (page = 1) => {
  try {
    const res = await edgeNodeApi.fetchEdgeNodes({ page, page_size: pageSize.value })
    edgeNodes.value = res.list || []
    total.value = res.total || 0
    currentPage.value = page
  } catch (error) {
    edgeNodes.value = []
    total.value = 0
  }
}

const fetchRegions = async () => {
  try {
    const res = await regionNodeApi.fetchRegionNodes({})
    regionList.value = res.list || []
  } catch (error) {
    regionList.value = []
  }
}

onMounted(async () => {
  await Promise.all([fetchEdgeNodes(), fetchRegions()])
})

const handlePageChange = async (page: number) => {
  await fetchEdgeNodes(page)
}

const createEdgeNode = () => {
  isEdit.value = false
  form.value = {
    id: null,
    name: '',
    region_id: regionList.value.length > 0 ? regionList.value[0].id : null
  }
  dialogVisible.value = true
}

const editEdgeNode = (row) => {
  isEdit.value = true
  form.value = { ...row }
  dialogVisible.value = true
}

const submitEdgeNode = async () => {
  try {
    if (isEdit.value) {
      await edgeNodeApi.updateEdgeNode(form.value)
      ElMessage.success('边缘节点更新成功')
    } else {
      await edgeNodeApi.createEdgeNode(form.value)
      ElMessage.success('边缘节点创建成功')
    }

    dialogVisible.value = false
    await fetchEdgeNodes(currentPage.value)
  } catch (error) {
    return
  }
}

const deleteEdgeNode = async (id: number) => {
  try {
    await edgeNodeApi.deleteEdgeNode(id)
    ElMessage.success('边缘节点删除成功')
    await fetchEdgeNodes(currentPage.value)
  } catch (error) {
    return
  }
}
</script>