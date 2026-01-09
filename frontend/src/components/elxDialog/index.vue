<!-- 弹窗 -->
<template>
  <el-dialog
    :width="props.width"
    v-model="dialogVisible"
    :close-on-click-modal="false"
    :before-close="handleClose"
    :show-close="props.showClose"
    v-bind="props.bind"
  >
    <template #header>
      <span class="dialog-header">
        {{ props.title }}
        <!-- <page-header :title="props.title"></page-header> -->
      </span>
    </template>
    <slot></slot>
    <template #footer>
      <slot name="footer">
        <span class="dialog-footer 14" v-if="!disabled">
          <el-button @click="handleClose()" v-if="props.showClose" type="info">{{
            $t(props.cancelText)
          }}</el-button>
          <el-button
            type="primary"
            @click="submitForm()"
            :loading="submitting"
            v-if="props.showSubmit"
            >{{ $t(props.okText) }}</el-button
          >
        </span>
      </slot>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { ref, computed, nextTick } from 'vue'

interface TypeProps {
  width?: string
  cancelText?: string
  okText?: string
  title: string
  visible: boolean
  type?: DialogType
  bind?: AnyObj
  showClose?: boolean
  showSubmit?: boolean
}

const props = withDefaults(defineProps<TypeProps>(), {
  width: '630px',
  cancelText: 'tableForm.cancel',
  okText: 'tableForm.confirm',
  title: '',
  visible: false,
  type: 'add',
  showClose: true,
  showSubmit: true
})

const dialogVisible = ref(props.visible)
const submitting = ref(false)
const disabled = computed(() => props.type === 'view')

watch(
  () => props.visible,
  (visible) => {
    dialogVisible.value = visible
  }
)

const emit = defineEmits<{ submit: [value: any]; close: [value: any] }>()

const changeLoading = (val: boolean) => {
  submitting.value = val
}

const close = () => {
  dialogVisible.value = false
}

const handleClose = () => {
  close()
  emit('close', {})
}

const submitForm = () => {
  emit('submit', { close, changeLoading })
}
</script>
