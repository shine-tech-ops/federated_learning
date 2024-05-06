
<template>
  <page-header title="系统配置管理">
    <el-button type="primary" @click="createConfig">新建配置</el-button>
  </page-header>

  <el-table :data="configs" border style="width: 100%">
    <el-table-column prop="name" label="配置名称" />
    <el-table-column prop="description" label="描述" />
    <el-table-column prop="created_at" label="创建时间" width="160" />
    <el-table-column prop="updated_at" label="最后更新时间" width="160" />
    <!-- 是否激活列 -->
    <el-table-column prop="is_active" label="是否激活">
      <template #default="{ row }">
        <el-tag :type="row.is_active ? 'success' : 'info'" size="small">
          {{ row.is_active ? '已激活' : '未激活' }}
        </el-tag>
      </template>
    </el-table-column>

    <el-table-column label="操作">
      <template #default="{ row }">
        <el-button size="small" type="primary" @click="editConfig(row)">编辑</el-button>
        <!-- 激活按钮 -->
        <el-button size="small" type="success" @click="activateConfig(row)" v-if="!row.is_active">激活</el-button>

        <!-- 删除按钮（未激活才能删除） -->
        <el-popconfirm title="确定要删除这个配置吗？" @confirm="deleteConfig(row.id)">
          <template #reference>
            <el-button size="small" type="danger" :disabled="row.is_active">删除</el-button>
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
    @current-change="handlePageChange"    style="margin-top: 20px; text-align: right"
  />

  <!-- 对话框 -->
  <el-dialog v-model="dialogVisible" :title="isEdit ? '编辑配置' : '新建配置'">
    <el-form :model="configForm" label-width="120px" ref="formRef">
      <el-form-item label="配置名称" prop="name">
        <el-input v-model="configForm.name" />
      </el-form-item>
      <el-form-item label="描述">
        <el-input v-model="configForm.description" type="textarea" />
      </el-form-item>

      <el-divider content-position="left">联邦学习参数</el-divider>
      <el-form-item label="训练轮次" prop="federated.rounds">
        <el-input-number v-model="configForm.config_data.federated.rounds" :min="1" />
      </el-form-item>
      <el-form-item label="参与比例">
        <el-slider v-model="configForm.config_data.federated.participationRate" :min="0" :max="100" />
      </el-form-item>
      <el-form-item label="聚合方式">
        <el-select v-model="configForm.config_data.federated.aggregation" placeholder="请选择" style="width: 100%">
          <el-option
            v-for="method in aggregationMethods"
            :key="method"
            :label="method"
            :value="method"
          />
        </el-select>
      </el-form-item>

      <el-divider content-position="left">下发策略</el-divider>
      <el-form-item label="类型">
        <el-radio-group v-model="configForm.config_data.strategy.type">
          <el-radio label="full">全量更新</el-radio>
          <el-radio label="delta">差量更新</el-radio>
        </el-radio-group>
      </el-form-item>

      <el-divider content-position="left">黑名单</el-divider>
      <el-table :data="configForm.config_data.blacklist.nodes">
        <el-table-column prop="ip" label="IP地址" />
        <el-table-column label="操作">
          <template #default="{ $index }">
            <el-button size="small" type="danger" @click="removeNode($index)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- 新增IP输入框 -->
      <div style="margin-top: 10px;">
        <el-input v-model="newBlacklistIp" placeholder="请输入要添加的IP" style="width: 300px; margin-right: 10px;" />
        <el-button type="primary" @click="addBlacklistIp">添加IP</el-button>
      </div>
    </el-form>

    <template #footer>
      <span class="dialog-footer">
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="submitConfig">确定</el-button>
      </span>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">import { ref, onMounted } from 'vue'
import { systemConfigModel } from '@/api/systemConfig'
import { ElMessage, ElPopconfirm } from 'element-plus'

// 数据
const configs = ref([])
const currentPage = ref(1)
const pageSize = ref(20)
const total = ref(0)

const dialogVisible = ref(false)
const isEdit = ref(false)
const formRef = ref()
defineExpose({ formRef })

// 可用的聚合方法列表
const aggregationMethods = ref<string[]>([])

const configForm = ref({
  id: null,
  name: '',
  description: '',
  config_data: {
    federated: {
      rounds: 10,
      participationRate: 80,
      aggregation: '' // 聚合方式初始化为空
    },
    strategy: {
      type: 'full'
    },
    blacklist: {
      nodes: []
    }
  }
})

// 获取配置列表
const fetchConfigs = async (page = 1) => {
  try {
    const res = await systemConfigModel.getConfigsApi({
      page,
      page_size: pageSize.value
    })
    configs.value = res.list || []
    total.value = res.total || 0
    currentPage.value = page
  } catch (error) {
    configs.value = []
    total.value = 0
  }
}

// 获取可用的聚合方法列表
const fetchAggregationMethods = async () => {
  try {
    const res = await systemConfigModel.getAggregationMethodApi({})
    if (Array.isArray(res)) {
      aggregationMethods.value = res
      // 如果聚合方法列表不为空，设置默认值
      if (aggregationMethods.value.length > 0) {
        configForm.value.config_data.federated.aggregation = aggregationMethods.value[0]
      }
    } else {
      // 如果后端返回的数据格式不符合预期，使用默认值
      aggregationMethods.value = ['fedavg', 'fedprox']
      configForm.value.config_data.federated.aggregation = 'fedavg'
    }
  } catch (error) {
    // 如果请求失败，使用默认值
    aggregationMethods.value = ['fedavg', 'fedprox']
    configForm.value.config_data.federated.aggregation = 'fedavg'
  }
}

onMounted(async () => {
  await Promise.all([
    fetchConfigs(),
    fetchAggregationMethods()
  ])
})

// 翻页处理
const handlePageChange = async (page: number) => {
  await fetchConfigs(page)
}

// 新建配置
const createConfig = () => {
  isEdit.value = false
  configForm.value = {
    id: null,
    name: '',
    description: '',
    config_data: {
      federated: {
        rounds: 10,
        participationRate: 80,
        aggregation: aggregationMethods.value.length > 0 ? aggregationMethods.value[0] : 'fedavg'
      },
      strategy: {
        type: 'full'
      },
      blacklist: {
        nodes: []
      }
    }
  }
  dialogVisible.value = true
}

// 编辑配置
const editConfig = (row: any) => {
  isEdit.value = true
  configForm.value = JSON.parse(JSON.stringify(row)) // 深拷贝防止响应式污染

  // 确保聚合方式是有效值
  if (aggregationMethods.value.includes(row.config_data.federated.aggregation)) {
    configForm.value.config_data.federated.aggregation = row.config_data.federated.aggregation
  } else if (aggregationMethods.value.length > 0) {
    configForm.value.config_data.federated.aggregation = aggregationMethods.value[0]
  }

  dialogVisible.value = true
}

// 提交表单
const submitConfig = async () => {
  try {
    if (isEdit.value) {
      await systemConfigModel.updateConfigApi(configForm.value)
      ElMessage.success('配置编辑成功')
    } else {
      await systemConfigModel.createConfigApi(configForm.value)
      ElMessage.success('配置创建成功')
    }

    dialogVisible.value = false
    await fetchConfigs(currentPage.value)
  } catch (error) {
    return
  }
}

// 激活配置
const activateConfig = async (row: any) => {
  try {
    await systemConfigModel.activateConfigApi({ id: row.id });
    ElMessage.success('配置激活成功');
    await fetchConfigs(currentPage.value);
  } catch (error) {
    return
  }
};

// 删除配置
const deleteConfig = async (id: number) => {
  try {
    await systemConfigModel.deleteConfigApi({ id })
    ElMessage.success('配置删除成功')
    await fetchConfigs(currentPage.value)
  } catch (error) {
    return
  }
}

const newBlacklistIp = ref('')

// 添加黑名单IP
function addBlacklistIp() {
  if (!newBlacklistIp.value.trim()) {
    ElMessage.warning("IP不能为空")
    return
  }
  if (!/^(25[0-5]|2[0-4]\d|[01]?\d\d?)\.(25[0-5]|2[0-4]\d|[01]?\d\d?)\.(25[0-5]|2[0-4]\d|[01]?\d\d?)\.(25[0-5]|2[0-4]\d|[01]?\d\d?)$/.test(newBlacklistIp.value.trim())) {
    ElMessage.warning("请输入正确的IP地址")
    return
  }
  if (configForm.value.config_data.blacklist.nodes.some((item: any) => item.ip === newBlacklistIp.value.trim())) {
    ElMessage.warning("该IP已存在")
    return
  }
  configForm.value.config_data.blacklist.nodes.push({ip: newBlacklistIp.value.trim()})
  newBlacklistIp.value = ''
}

// 删除IP节点
function removeNode(index: number) {
  configForm.value.config_data.blacklist.nodes.splice(index, 1)
}
</script>
