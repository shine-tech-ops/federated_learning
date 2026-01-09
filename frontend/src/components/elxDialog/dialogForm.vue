<template>
  <elx-dialog
    :title="props.dialogTitle"
    :visible="dialogVisible"
    @close="close"
    @submit="submitForm"
  >
    <crud-form
      :columns="props.formColumns"
      :disabled="disabled"
      :rules="props.formRules"
      v-model:data="formData"
      ref="formRef"
    ></crud-form>
  </elx-dialog>
</template>

<script setup lang="ts">
import { ref, reactive, computed, nextTick } from 'vue'
import { type FormRules } from 'element-plus'

interface TypeProps {
  dialogCancelText?: string
  dialogOkText?: string
  dialogTitle: string
  dialogType: DialogType
  dialogWidth?: string
  formColumns: Form.Column[]
  formRules: FormRules
}

const props = withDefaults(defineProps<TypeProps>(), {
  dialogType: 'add',
  formColumns: () => [],
  formRules: () => ({})
})

const dialogVisible = ref(false)
const formData = ref<Form.Data>({})
const modalType = ref<DialogType>('add')
const formRef = ref()
const disabled = computed(() => modalType.value === 'view')
const emit = defineEmits<{ submit: [value: any] }>()

const clear = () => {
  modalType.value = 'add'
  formRef.value?.resetFields()
  for (const key in formData.value) {
    delete formData.value[key]
  }
}

const open = (type: DialogType, rowInfo?: any) => {
  dialogVisible.value = true
  modalType.value = type
  if (rowInfo) {
    nextTick(() => {
      Object.assign(formData.value, rowInfo)
    })
  }
}

const close = () => {
  dialogVisible.value = false
  clear()
}

const submitForm = ({ close, changeLoading }: any) => {
  console.log('formRef.value')
  formRef.value.formValidate(() => {
    changeLoading(true)
    const finallyFun = () => {
      changeLoading(false)
    }
    emit('submit', {
      formData: formData.value,
      type: modalType.value,
      close,
      finallyFun
    })
  })
}

defineExpose({
  open,
  close
})
</script>
