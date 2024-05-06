<template>
  <elx-dialog
    title="账户设置"
    ok-text="完成"
    :visible="dialogVisible"
    @close="closeDialog"
    @submit="submitForm"
  >
    <crud-form :columns="columns" :rules="rules" v-model:data="formData" ref="formRef"></crud-form>
  </elx-dialog>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useUserStore } from '@/stores/modules/user'
import { ElMessage } from 'element-plus'
import { authModel } from '@/api'
import pinia from '@/stores'
import { type FormRules } from 'element-plus'
import type { UserState } from '@/stores/interface/index'
import { genRules } from '@/utils/validate'
import useRules from '@/hooks/useRules'
const dialogVisible = ref(false)
const userStore = useUserStore(pinia)
const formRef = ref()

const rules = useRules(() => {
  return {
    name: genRules(''),
    mobile: genRules(''),
    email: genRules('email'),
    password: genRules('password')
  }
})
const rules2 = {
  name: genRules(''),
  mobile: genRules(''),
  email: genRules('email'),
  password: genRules('password')
} as FormRules

const columns: Form.Column[] = [
  { prop: 'name', label: '用户名' },
  {
    prop: 'role',
    label: '角色',
    type: 'select',
    optConfig: { key: 'id', label: 'name', value: 'id' },
    disabled: () => true,
    options: () => {
      return new Promise((resolve) => {
        authModel
          .getRoleList()
          .then((res: any) => {
            resolve(res)
          })
          .catch(() => {
            resolve([])
          })
      })
    }
  },
  { prop: 'mobile', label: '联系电话' },
  { prop: 'email', label: '邮箱' }
  // {
  //   prop: 'password',
  //   label: '密码',
  //   bind: { 'show-password': true }
  // }
]

const formData = ref<Form.Data>({
  name: null,
  mobile: null,
  email: null,
  role: null
})

const closeDialog = () => {
  dialogVisible.value = false
  formRef.value?.resetFields()
}

const submitForm = ({ close, changeLoading }: any) => {
  formRef.value.formValidate(() => {
    changeLoading(true)
    console.log('params', formData.value)
    const params = { ...formData.value, role: [{ id: formData.value.role }] }
    authModel
      .updateUser(params)
      .then((data: UserState[]) => {
        closeDialog()
        ElMessage.success(`账户更新成功`)
        userStore.updateUsers(data[0])
      })
      .finally(() => {
        changeLoading(false)
      })
  })
}

// 账户设置
const open = () => {
  dialogVisible.value = true
  formData.value = {
    ...formData.value,
    ...userStore.users,
    role: userStore.users.role[0]?.id
  }
}

defineExpose({
  open
})
</script>
