<template>
  <elx-dialog
    title="站点设置"
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
import { genRules } from '@/utils/validate'
// 弃用
const dialogVisible = ref(false)
const userStore = useUserStore(pinia)
const formRef = ref()

const rules = {
  name: genRules(''),
  mobile: genRules(''),
  email: genRules('email'),
  password: genRules('password')
} as FormRules

const columns: Form.Column[] = [
  { prop: 'siteName', label: '站点名称' },
  { prop: 'address', label: '地址' },
  { prop: 'name', label: '联系人' },
  { prop: 'mobile', label: '联系电话' },
  { prop: 'email', label: '邮箱' },
  { prop: 'adminAccount', label: '管理员账号', disabled: () => true },
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
  }
]

const formData = ref<Form.Data>({
  siteName: null,
  address: null,
  name: null,
  mobile: null,
  email: null,
  role: null,
  adminAccount: null
})

const closeDialog = () => {
  dialogVisible.value = false
  formRef.value?.resetFields()
}

const submitForm = ({ close, changeLoading }: any) => {
  formRef.value.formValidate(() => {
    changeLoading(true)
    console.log('params', formData.value)
    const params = { ...formData.value }
    authModel
      .tmpUpdate(params)
      .then(() => {
        closeDialog()
      })
      .finally(() => {
        changeLoading(false)
      })
  })
}

// 账户设置
const open = () => {
  dialogVisible.value = true
  // userStore.users
  console.log('🚀 ~ openDialog ~ userStore.users:', userStore.users)
  formData.value = {
    ...formData.value,
    ...userStore.users
  }
}

defineExpose({
  open
})
</script>
