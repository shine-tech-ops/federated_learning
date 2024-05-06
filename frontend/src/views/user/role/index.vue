<template>
  <page-header title="角色权限">
    <el-button
      type="primary"
      icon="Plus"
      @click="handleModal('add')"
      v-auth="['edit_permission_management']"
      >添加角色</el-button
    >
  </page-header>
  <div class="mt-4">
    <elx-table
      v-model:loading="loading"
      :columns="tableColumns"
      v-model:data="tableData"
    ></elx-table>
  </div>
  <add-role ref="addModal" @ok="addOk"></add-role>
</template>
<script lang="ts" setup name="systemRole">
import AddRole from './Add.vue'
import { ElNotification, ElMessage } from 'element-plus'
import { authModel } from '@/api'
import { usePermissionStore } from '@/stores/modules/permission'
import pinia from '@/stores'

const permissionState = usePermissionStore(pinia)
const loading = ref(false)
const tableColumns: TableColumnType[] = [
  { prop: 'name', label: '角色名称', width: '200' },
  {
    prop: 'permission',
    label: '权限',
    format: (row: any) => {
      if (row.permission && row.permission.length > 0) {
        const arr = row.permission.map(
          (item: { name_zh: any; id: number }) =>
            permissionState.getPermissionNameById(item.id) || item.name_zh
        )
        return arr.join('、')
      }
      return ''
    }
  },
  {
    prop: 'operation',
    label: '操作',
    buttons: [
      {
        type: 'edit',
        label: '编辑',
        auth: ['edit_permission_management'],
        disabled: (row: any) => row.role_type === 0,
        handle: (row: any) => {
          handleModal('edit', row)
        }
      },
      {
        type: 'delete',
        label: '删除',
        auth: ['edit_permission_management'],
        disabled: (row: any) => row.role_type === 0,
        handle: (row: any) => {
          authModel.delRole({ id: row.id }).then(() => {
            ElMessage.success('删除成功')
            getList()
          })
        }
      },
      {
        type: 'view',
        label: '详情',
        handle: (row: any) => {
          handleModal('view', row)
        }
      }
    ]
  }
]

const tableData = ref([])
const addModal = ref()

const getList = () => {
  loading.value = true
  authModel
    .getRoleList()
    .then((res: any) => {
      // 过滤掉超管角色
      const result = res.filter((item: { name: string }) => item.name !== '超级管理员')
      tableData.value = result
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
