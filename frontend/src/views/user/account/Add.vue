<template>
  <elx-dialog
    :title="title"
    :visible="dialogVisible"
    @close="close"
    @submit="submitForm"
    :type="modalType"
  >
    <crud-form
      :columns="columns()"
      :disabled="disabled"
      :rules="rules"
      v-model:data="formData"
      ref="formRef"
    ></crud-form>
  </elx-dialog>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, computed, nextTick } from 'vue'
import { authModel } from '@/api'
import { type FormRules } from 'element-plus'
import { genRules } from '@/utils/validate'

const dialogVisible = ref(false)
const formData = ref<Form.Data>({
  name: '',
  role: '',
  mobile: '',
  email: '',
  status: '',
  password: ''
})
const modalType = ref<DialogType>('add')
const formRef = ref()
const roleOptions = ref<any[]>([])
const record = ref<AnyObj>({})

const title = computed(
  () => ({ view: '查看用户', edit: '编辑用户', add: '新增用户' })[modalType.value]
)
const disabled = computed(() => modalType.value === 'view')
const isAdd = computed(() => modalType.value === 'add')

const emit = defineEmits<{ ok: [value: any] }>()

// const rules: FormRules = {
//   name: genRules(''),
//   mobile: genRules('mobile'),
//   email: genRules('email'),
//   password: genRules('password'),
// }

const getRules = () =>
  <FormRules>{
    name: genRules(''),
    mobile: genRules('mobile'),
    email: genRules('email'),
    role: genRules('select'),
    password: isAdd.value ? [] : genRules('password', true)
  }

const rules = ref({})

const columns = () => [
  { prop: 'name', label: '用户名称' },
  {
    prop: 'role',
    label: '角色',
    type: 'select',
    optConfig: { key: 'id', label: 'name', value: 'id' },
    options: () => {
      return new Promise((resolve) => {
        authModel
          .getRoleList()
          .then((res: any) => {
            // 过滤掉超管角色
            const result = res.filter((item: { name: string }) => item.name !== '超级管理员')
            resolve(result)
          })
          .catch(() => {
            resolve([])
          })
      })
    }
  },
  { prop: 'mobile', label: '手机号' },
  { prop: 'email', label: '邮箱' },
  { prop: 'status', label: '是否启用', type: 'switch' },
  {
    prop: 'password',
    label: '密码',
    bind: {
      'show-password': !isAdd.value,
      placeholder: isAdd.value ? '首次创建，密码为手机号后4位' : '请输入'
    },
    iif: () => modalType.value !== 'view',
    disabled: () => {
      return isAdd.value
    }
  }
]

watch(
  () => formData.value.mobile,
  (mobile) => {
    if (isAdd.value && mobile) {
      formData.value.password = mobile.slice(-4)
    }
  }
)

const clear = () => {
  for (const key in formData.value) {
    delete formData.value[key]
  }
  formRef.value?.resetFields()
}

const open = (type: DialogType, rowInfo?: any) => {
  dialogVisible.value = true
  if (modalType.value !== type || Object.keys(rules.value).length === 0) {
    modalType.value = type
    rules.value = getRules()
  }
  nextTick(() => {
    formData.value.password = ''
    formData.value.status = true
    if (rowInfo) {
      formData.value.name = rowInfo.name
      formData.value.email = rowInfo.email
      formData.value.role = rowInfo.role[0]?.id
      formData.value.mobile = rowInfo.mobile
      formData.value.email = rowInfo.email
      formData.value.status = rowInfo.is_active
      // formData.value.password = '*********'
      // 记录
      record.value = rowInfo
    } else {
      record.value = {}
    }
    setTimeout(() => {
      formRef.value.oriFormRef().clearValidate()
    }, 0)
  })
}

const close = () => {
  dialogVisible.value = false
  clear()
}

// 初始化数据
const init = async () => {
  authModel.getRoleList().then((res: any) => {
    roleOptions.value = res
  })
}

const submitForm = ({ close, changeLoading }: any) => {
  formRef.value.formValidate(() => {
    changeLoading(true)
    const params: any = { ...record.value, ...formData.value, role: [{ id: formData.value.role }] }
    if (modalType.value === 'edit') {
    }
    const fun = modalType.value === 'add' ? authModel.addUser : authModel.updateUser
    fun(params)
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
