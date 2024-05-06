<template>
  <el-form
    :inline="searchMode"
    :model="props.data"
    status-icon
    :rules="props.rules || {}"
    :label-position="searchMode ? 'right' : 'top'"
    ref="formRef"
    :class="{ 'search-form-inline': searchMode, 'inner-label': props.innerLabel }"
  >
    <template v-for="(col, index) in props.columns" :key="index">
      <el-form-item
        v-if="!col.iif || (col.iif && col.iif(props.data[col.prop], props.data))"
        :label="$t(col.label)"
        :prop="col.prop"
        :style="{ width: col.width || '' }"
      >
        <!-- 下拉框 -->
        <el-select
          v-if="col.type === 'select'"
          v-model="props.data[col.prop]"
          :value-key="col.optConfig?.key"
          v-bind="getBindProp(col, props.data)"
        >
          <el-option
            v-for="(item, index) in optionGroup[col.prop]"
            :key="col.prop + index"
            :label="$t(item[col.optConfig?.label || 'label'])"
            :value="item[col.optConfig?.value || 'value']"
          >
          </el-option>
        </el-select>
        <!-- 开关 -->
        <el-switch
          v-else-if="col.type === 'switch'"
          v-model="props.data[col.prop]"
          v-bind="getBindProp(col, props.data)"
        ></el-switch>
        <!-- 单选 -->
        <el-radio-group
          v-else-if="col.type === 'radio'"
          v-model="props.data[col.prop]"
          v-bind="getBindProp(col, props.data)"
        >
          <el-radio
            v-for="(item, index) in optionGroup[col.prop]"
            :key="col.prop + index"
            :label="item[col.optConfig?.value || 'value']"
            >{{ item[col.optConfig?.label || 'label'] }}</el-radio
          >
        </el-radio-group>
        <el-checkbox-group
          v-else-if="col.type === 'checkbox'"
          v-model="props.data[col.prop]"
          v-bind="getBindProp(col, props.data)"
        >
          <el-checkbox
            :border="true"
            v-for="(item, index) in optionGroup[col.prop]"
            :key="col.prop + index"
            :label="item[col.optConfig?.value || 'value']"
            >{{ item[col.optConfig?.label || 'label'] }}</el-checkbox
          >
        </el-checkbox-group>
        <el-date-picker
          v-else-if="col.type === 'date'"
          v-model="props.data[col.prop]"
          type="date"
          value-format="YYYY-MM-DD HH:mm:ss"
          v-bind="getBindProp(col, props.data)"
        >
        </el-date-picker>
        <!-- 日期范围 -->
        <el-date-picker
          v-else-if="col.type === 'daterange'"
          v-model="props.data[col.prop]"
          type="daterange"
          range-separator="To"
          value-format="YYYY-MM-DD HH:mm:ss"
          :default-time="defaultTime"
          v-bind="getBindProp(col, props.data)"
        >
          <!-- start-placeholder="开始日期"
          end-placeholder="结束日期" -->
        </el-date-picker>
        <!-- 自定义 -->
        <template v-else-if="col.type === 'custom'">
          <slot :name="col.prop" :data="props.data"></slot>
        </template>
        <el-input
          v-else-if="col.type === 'textarea'"
          v-model="props.data[col.prop]"
          :rows="2"
          v-bind="getBindProp(col, props.data)"
          type="textarea"
        />
        <el-input
          v-else-if="col.type === 'number'"
          v-model="props.data[col.prop]"
          v-bind="getBindProp(col, props.data)"
          type="number"
        >
        </el-input>
        <!-- 默认input -->
        <el-input v-else v-model="props.data[col.prop]" v-bind="getBindProp(col, props.data)">
        </el-input>
      </el-form-item>
    </template>
    <slot></slot>
  </el-form>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import type { FormInstance, FormRules } from 'element-plus'
import { isFunction } from '@/utils/is/index'
import { useI18n } from 'vue-i18n'

const { t: $t } = useI18n()
interface TypeProps {
  disabled?: boolean
  columns: Form.Column[]
  data: Form.Data
  rules?: FormRules
  mode?: Form.Mode
  // 是否允许清除，一般全局设置即可，默认true
  clearable?: boolean
  // label是否在input中
  innerLabel?: boolean
}

const props = withDefaults(defineProps<TypeProps>(), {
  disabled: false,
  columns: () => [],
  data: () => ({}),
  rules: () => ({}),
  mode: 'default',
  clearable: true,
  innerLabel: false
})

const formRef = ref<FormInstance>()
const optionGroup = ref<{ [key in string]: any[] }>({})

/**
 * 是否为查询模式
 * 查询模式表单为行内，label不显示
 */
const searchMode = computed(() => props.mode === 'search')

// 时分秒 按00:00:00 - 23:59:59
const defaultTime = ref<[Date, Date]>([
  new Date(2000, 1, 1, 0, 0, 0),
  new Date(2000, 2, 1, 23, 59, 59)
])

const getPlaceholder = (col: Form.Column) => {
  let text = col.placeholder || ''
  if (text) {
    return text
  }
  if (['input', 'number'].includes(col.type) || !col.type) {
    text = $t('validate.pleaseInput')
  } else {
    text = $t('validate.pleaseSelect')
  }

  // if (searchMode.value) {
  //   text += col.label
  // }
  return text
}

const getDisabled = (col: Form.Column, row: Form.Data) => {
  if (props.disabled) {
    return true
  }

  if (col.disabled) {
    return col.disabled(row)
  }

  return false
}

const getBindProp = (col: Form.Column, row: Form.Data) => {
  return {
    placeholder: getPlaceholder(col),
    disabled: getDisabled(col, props.data),
    clearable: props.clearable,
    ...col.bind
  }
}

watch(
  () => props.columns,
  (columns) => {
    columns.map((col) => {
      // 统一获取所有选项
      if (['select', 'radio', 'checkbox'].includes(col.type) && col.options) {
        if (isFunction(col.options)) {
          col.options().then((res) => {
            optionGroup.value[col.prop] = res
          })
        } else {
          optionGroup.value[col.prop] = col.options
        }
      }
    })
  },
  { immediate: true }
)

const formValidate = (callback: () => any, error?: () => any) => {
  formRef.value!.validate((valid) => {
    if (valid) {
      callback()
    } else {
      error && error()
    }
  })
}

const resetFields = () => {
  formRef.value?.resetFields()
}

const oriFormRef = () => {
  return formRef.value
}

defineExpose({
  formValidate,
  resetFields,
  oriFormRef
})
</script>
<style scoped lang="scss">
:deep(.el-form-item) {
  margin-bottom: 15px;
  .el-date-editor {
    --el-date-editor-width: 100%;
  }
}

.search-form-inline {
  .el-input {
    --el-input-width: 180px;
  }
  .el-select {
    --el-select-width: 180px;
  }
  &.el-form--inline .el-form-item {
    margin-right: 12px;
  }
  .el-date-editor .el-range-input {
    width: 110px;
  }
}
</style>
