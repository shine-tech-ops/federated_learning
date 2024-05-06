<template>
  <elx-dialog
    :title="title"
    :visible="dialogVisible"
    @close="handleClose"
    @submit="submitForm"
    :type="modalType"
    width="420px"
  >
    <crud-form
      :columns="columns"
      :disabled="disabled"
      :rules="rules"
      v-model:data="formData"
      ref="formRef"
    >
      <template #permission="scope">
        <el-tree
          class="overflow-y-auto max-h-80 w-full"
          :data="permissionData"
          show-checkbox
          default-expand-all
          node-key="id"
          :props="{ label: 'name_zh' }"
          ref="treeRef"
          check-on-click-node
          @check-change="handleCheckChange"
          highlight-current
        >
          <template v-slot="{ node }">
            <span>{{ node.label }}</span>
          </template>
        </el-tree>
      </template>
    </crud-form>
  </elx-dialog>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, computed, nextTick } from 'vue'
import { authModel } from '@/api'
import { usePermissionStore } from '@/stores/modules/permission'
import pinia from '@/stores'
import type { Permission } from '@/stores/interface'
import { cloneDeep } from 'lodash'
import { genRules } from '@/utils/validate'

const permissionState = usePermissionStore(pinia)

const dialogVisible = ref(false)
const formData = ref({
  name: '',
  permission: []
})
const modalType = ref<DialogType>('add')
const roleId = ref(null)
const permissionData = ref<Permission[]>([])
const treeRef = ref()
const formRef = ref()

const title = computed(
  () => ({ view: '查看角色', edit: '编辑角色', add: '新增角色' })[modalType.value]
)
const disabled = computed(() => modalType.value === 'view')

const columns = [
  { prop: 'name', label: '角色名称' },
  { prop: 'permission', label: '权限', type: 'custom' }
]

const record = ref<AnyObj>({})
const emit = defineEmits<{
  ok: [value: any]
}>()

const rules = {
  name: genRules(''),
  permission: genRules('permission')
}

const clear = () => {
  modalType.value = 'add'
  roleId.value = null
  treeRef.value.setCheckedKeys([])
  formRef.value?.resetFields()
  for (const key in formData.value) {
    // @ts-ignore
    delete formData.value[key]
  }
}

const open = (type: DialogType, rowInfo?: any) => {
  dialogVisible.value = true
  modalType.value = type
  if (rowInfo) {
    nextTick(() => {
      const { name, permission } = rowInfo
      if (permission) {
        treeRef.value.setCheckedNodes(permission)
      }
      formData.value.name = name
      record.value = rowInfo
    })
  }
}

const close = () => {
  dialogVisible.value = false
  clear()
}

const handleClose = () => {
  close()
}

// 初始化数据
const init = async () => {
  // 初始化权限列表
  permissionData.value = permissionState.oriTreeData
  // authModel.permissions().then((res: any) => {
  //   permissionData.value = res
  // })
}

const handleCheckChange = (_data: any, _checked: any, _indeterminate: any) => {
  formData.value.permission = treeRef.value.getCheckedNodes()
}

const submitForm = ({ close, changeLoading }: any) => {
  formRef.value.formValidate(() => {
    changeLoading(true)
    const fun = modalType.value === 'add' ? authModel.addRole : authModel.updateRole
    formData.value.permission
    const params = cloneDeep(formData.value)
    params.permission = params.permission.filter((item: { id: number }) => item.id !== 0)
    fun({ ...record.value, ...params })
      .then(() => {
        close()
        emit('ok', modalType.value)
      })
      .finally(() => {
        changeLoading(false)
      })
  })
}

onMounted(() => {
  init()
})

defineExpose({
  open,
  close
})
</script>
