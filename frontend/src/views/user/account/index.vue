<template>
  <page-header title="账号管理">
    <el-button
      type="primary"
      icon="Plus"
      @click="handleModal('add')"
      v-auth="['edit_user_management']"
      >添加用户</el-button
    >
  </page-header>
  <div class="mt-4">
    <elx-table
      v-model:loading="loading"
      :columns="tableColumns"
      v-model:data="tableData"
    ></elx-table>
  </div>
  <add-user ref="addModal" @ok="addOk"></add-user>
</template>
<script lang="ts" setup name="systemUser">
// import { onMounted, ref } from 'vue'
import AddUser from './Add.vue'
import { ElMessage } from 'element-plus'
import { authModel } from '@/api'
import { useUserStore } from '@/stores/modules/user'
import pinia from '@/stores'

const userStore = useUserStore(pinia)

const loading = ref(false)
const tableColumns: TableColumnType[] = [
  { prop: 'name', label: '用户名' },
  {
    prop: 'role',
    label: '角色名',
    format: (row: any) => row.role[0]?.name
  },
  { prop: 'mobile', label: '联系电话' },
  { prop: 'email', label: '邮箱' },
  {
    prop: 'is_active',
    label: '是否启用',
    type: 'switch',
    auth: ['edit_user_management'],
    disabled: (row: any) => row.is_superuser,
    change: (row: any, val: boolean) => {
      const asyncRes = authModel.tmpUpdate({ ...row, is_active: val })
      asyncRes.then(() => {})
      return asyncRes
    }
  },
  {
    prop: 'operation',
    label: '操作',
    buttons: [
      {
        type: 'edit',
        label: '编辑',
        auth: ['edit_user_management'],
        disabled: (row: any) => row.is_superuser || row.admin,
        handle: (row: any) => {
          handleModal('edit', row)
        }
      },
      {
        type: 'delete',
        label: '删除',
        auth: ['edit_user_management'],
        disabled: (row: any) => row.is_superuser || row.admin || row.id === userStore.users.id,
        handle: (row: any) => {
          authModel.tmpUpdate({ id: row.id }).then(() => {
            ElMessage.success('删除成功')
            getList()
          })
        }
      }
    ]
  }
]

const tableData = ref<TableDataType[]>([])
const addModal = ref()

const getList = () => {
  loading.value = true
  authModel
    .getUserList()
    .then((res: any) => {
      tableData.value = res
    })
    .finally(() => {
      loading.value = false
    })
}
const handleModal = (type: DialogType, rowInfo?: any) => {
  addModal.value.open(type, rowInfo)
}

const addOk = (type: DialogType) => {
  addModal.value.close()
  if (type !== 'view') {
    ElMessage.success(`${type === 'add' ? '添加' : '保存'}成功`)
    getList()
  }
}
onMounted(() => {
  getList()
})
</script>
