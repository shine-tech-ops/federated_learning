<template>
  <div>
    <el-switch
      v-model="value"
      :size="props.size"
      :before-change="beforeChange"
      :loading="loading"
      :disabled="props.disabled"
    ></el-switch>
  </div>
</template>
<script setup lang="ts">
import { ref } from 'vue'
import { isFunction } from '../../utils/is/index'
import { ElNotification, type ComponentSize, ElMessageBox } from 'element-plus'
import { useI18n } from 'vue-i18n'

const { t: $t } = useI18n()
interface TypeProps {
  size?: ComponentSize
  disabled: boolean
  popConfirm?: string
  beforeChange: (params: any) => Promise<any> | boolean
}

const props = withDefaults(defineProps<TypeProps>(), {
  size: 'small',
  disabled: false,
  // 更改前是否二次确认
  popConfirm: '',
  beforeChange: () => true
})

const value = defineModel('value', { type: Boolean })

const loading = ref(false)

const handleBeforeChange = () => {
  if (isFunction(props.beforeChange)) {
    loading.value = true
    const promiseRes = props.beforeChange(!value.value)
    if (typeof promiseRes !== 'boolean') {
      promiseRes
        .then(() => {
          ElNotification.success({ message: '切换成功', position: 'bottom-right' })
        })
        .finally(() => {
          loading.value = false
        })
    }
    return promiseRes
  }
  return true
}
const beforeChange = () => {
  if (props.popConfirm) {
    return new Promise<any>((resolve, reject) => {
      ElMessageBox.confirm(props.popConfirm, $t('message.tips'), {
        type: 'warning'
      })
        .then(() => {
          resolve(handleBeforeChange())
        })
        .catch(() => {
          reject()
        })
    })
  } else {
    return handleBeforeChange()
  }
}
</script>
