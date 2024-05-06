<script lang="ts" setup>
import { inject, type Ref, watchEffect } from 'vue'
import { resetRouter } from '../../router/index'

import type { EditActions, FormModel, FormModelItem, FormProps } from './editTable.vue'

export interface Rule {}

interface ColumnScope {
  row?: any
  column: any
  $index: number
  [key: string]: any
}

const props = withDefaults(
  defineProps<{
    prop?: string
    label?: string
    labelEn?: string
    width?: string
    rules?: Rule[]
    enum?: { [key in string]: any }
  }>(),
  {
    prop: '',
    label: '',
    labelEn: ''
  }
)

const defaultEditActions: EditActions = {
  addRow: () => {},
  deleteRow: () => {},
  startEditable: () => {},
  cancelEditable: () => {},
  saveEditable: (index: number) =>
    new Promise<any>((resolve, reject) => {
      resolve(true)
    })
}

const editActions = inject<EditActions | undefined>('editActions')

const formModel = inject<Ref<FormModel | undefined>>('formModel')

const formProps = inject<Ref<FormProps | undefined>>('formProps')

watchEffect(() => {
  if (props.prop) {
    formProps?.value?.add(props.prop)
  }
})

const getEditModel = (index: number): FormModelItem => {
  if (!formModel || !formModel.value?.model) {
    return {
      isEditing: false,
      isNew: false,
      formData: {},
      data: {}
    }
  }
  return formModel.value.model[index]
}

const getEditRow = (index: number): any => getEditModel(index).formData

const isEditing = (index: number): boolean => getEditModel(index).isEditing ?? false

const calculateColumnDefaultValue = (scope: ColumnScope) => {
  const val = scope.row?.[props.prop]
  if (props.enum) {
    return props.enum[val]
  }

  return val
}
</script>

<template>
  <el-table-column v-bind="$attrs" :prop="prop" :label="labelEn" :width="width">
    <template #header>
      <!-- <div>{{ label }}</div> -->
      <!-- <div>{{ labelEn }}</div> -->
      <el-tooltip :content="label" placement="top-start">
        <div>{{ labelEn }}</div>
      </el-tooltip>
    </template>
    <template #default="scope">
      <el-form-item
        v-if="isEditing(scope.$index)"
        :prop="`model.${scope.$index}.formData.${prop}`"
        :rules="rules"
      >
        <slot
          name="edit"
          :$index="scope.$index"
          :row="getEditRow(scope.$index)"
          :column="scope.column"
          :actions="editActions ?? defaultEditActions"
        >
          {{ calculateColumnDefaultValue(scope) }}
        </slot>
      </el-form-item>
      <slot
        v-else
        :$index="scope.$index"
        :row="scope.row"
        :column="scope.column"
        :actions="editActions ?? defaultEditActions"
      >
        {{ calculateColumnDefaultValue(scope) }}
      </slot>
    </template>
  </el-table-column>
</template>
