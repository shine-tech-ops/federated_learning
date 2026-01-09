<template>
  <elx-dialog
    title="修改密码"
    ok-text="完成"
    :visible="dialogVisible"
    @close="closeDialog"
    @submit="submitForm"
    :showClose="!force"
    :bind="bind"
  >
    <div v-if="force">为了保护您的账户安全，请设定您的个人密码：</div>
    <crud-form :columns="columns" :rules="rules" v-model:data="formData" ref="formRef"></crud-form>
  </elx-dialog>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useUserStore } from '@/stores/modules/user'
import { ElMessageBox } from 'element-plus'
import { authModel } from '@/api'
import pinia from '@/stores'
import { type FormRules } from 'element-plus'
import { REGEXP_PWD } from '@/utils/validate'
import { useI18n } from 'vue-i18n'

const { t: $t } = useI18n()
const dialogVisible = ref(false)
const userStore = useUserStore(pinia)
const formRef = ref()

const formData = ref<Form.Data>({
  password: null,
  password2: null
})

// 是否强制修改密码
const force = ref(false)

const bind = computed(() => ({
  closeOnClickModal: !force.value,
  showClose: !force.value,
  closeOnPressEscape: !force.value
}))

const validatePass1 = async (_rule: any, value: any, callback: any) => {
  if (value === '') {
    callback(new Error('请输入密码'))
  } else if (!REGEXP_PWD.test(value)) {
    callback(new Error('密码应为6-30位'))
  } else {
    if (formData.value.password2 !== '') {
      const _formRef = formRef.value.oriFormRef()
      const validateFields = _formRef.validateFields || _formRef.validateField
      validateFields(['password2'])
    }
    callback()
  }
}
const validatePass2 = async (_rule: any, value: any, callback: any) => {
  if (value === '') {
    callback(new Error('请再次输入密码'))
  } else if (value !== formData.value.password) {
    callback(new Error('两次输入密码不一致!'))
  } else {
    callback()
  }
}

const rules = {
  password: [{ validator: validatePass1, trigger: 'blur' }],
  password2: [{ validator: validatePass2, trigger: 'blur' }]
} as FormRules

const columns: Form.Column[] = [
  {
    prop: 'password',
    label: '新密码',
    bind: { 'show-password': true },
    placeholder: '请输入至少 6 位新密码'
  },
  {
    prop: 'password2',
    label: '确认密码',
    bind: { 'show-password': true },
    placeholder: '请再次输入密码'
  }
]

const closeDialog = () => {
  dialogVisible.value = false
  formRef.value?.resetFields()
}

const submitForm = ({ close, changeLoading }: any) => {
  formRef.value.formValidate(() => {
    console.log('params', formData.value)
    const params = { id: userStore.users.id, password: formData.value.password }
    changeLoading(true)
    authModel
      .updateUser(params)
      .then(() => {
        closeDialog()
        ElMessageBox.alert($t('login.changePasswordDone'), $t('message.tips'), {
          showClose: false,
          showCancelButton: false,
          confirmButtonText: $t('login.goToLogin'),
          type: 'success',
          callback: (arg: any) => {
            if (arg === 'confirm') {
              userStore.logout()
            }
          }
        })
      })
      .finally(() => {
        changeLoading(false)
      })
  })
}

// 账户设置
const open = (isForce = false) => {
  dialogVisible.value = true
  force.value = isForce
}

defineExpose({
  open
})
</script>
