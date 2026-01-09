<template>
  <el-table
    ref="tableRef"
    v-loading="loading"
    :data="props.data"
    :border="false"
    @cell-click="handleCellClick"
    v-bind="props.tableBind"
  >
    <template v-for="item in props.columns" :key="item">
      <template v-if="!item.iif || (item.iif && item.iif())">
        <!-- 操作 -->
        <elxOperationVue :item="item" v-if="item.prop === 'operation'"></elxOperationVue>
        <!-- 其他显示字段 -->
        <el-table-column v-else :prop="item.prop" :label="$t(item.label || '')" :width="item.width">
          <template #default="scope">
            <!-- 开关 -->
            <template v-if="item.type === 'switch'">
              <elx-switch
                :disabled="item.disabled ? item.disabled(scope.row) : false"
                v-if="userStore.hasPermission(item.auth)"
                v-model:value="scope.row[item.prop]"
                :before-change="
(                  val: boolean) => {
                    return item.change && item.change(scope.row, val)
                  }
                "
              ></elx-switch>
            </template>
            <!-- 带枚举项 -->
            <template v-else-if="item.enum">
              <renderEnum v-bind="{ scope, item }"></renderEnum>
            </template>
            <template v-else-if="item.type === 'pic'">
              <el-avatar
                :size="60"
                :src="item.format ? item.format(scope.row) : scope.row[item.prop]"
                shape="square"
                :fit="'contain'"
                class="avatar-darken"
              />
            </template>
            <!-- 需要转换的情况 -->
            <template v-else-if="item.format">
              {{ item.format(scope.row) }}
            </template>
            <!-- 自定义 -->
            <template v-else-if="item.type === 'custom'">
              <slot :name="item.prop" :data="scope.row"></slot>
            </template>
            <template v-else-if="item.render">
              <component :is="item.render" v-bind="scope" v-if="item.render" />
            </template>
          </template>
        </el-table-column>
      </template>
    </template>
    <!-- noData -->
    <template #empty>
      <el-empty :image-size="120" :description="$t('message.noData')" />
    </template>
  </el-table>
  <!-- 分页 -->
  <div v-if="props.page.show" class="mt-5 mb-10">
    <pagination
      :data="paginationData"
      :handle-size-change="pageSizeChange"
      :handle-current-change="currentPageChange"
      class="float-right"
    ></pagination>
  </div>
</template>
<script lang="tsx" setup>
import { tableButtonType } from '@/utils/dict'
import { useUserStore } from '@/stores/modules/user'
import pinia from '@/stores'
import elxOperationVue from './elxOperation.vue'
import { useI18n } from 'vue-i18n'

const { t: $t } = useI18n()
const userStore = useUserStore(pinia)
const tableRef = ref()

interface TypePage {
  show: boolean
  pageSizes?: number[]
  showSize?: boolean
}

interface TypeProps {
  columns: TableColumnType[]
  data: TableDataType[]
  loading: boolean
  // 分页配置
  page?: TypePage
  // 当前页码
  pi?: number
  // 一页有几条数据
  ps?: number
  // 共多少数据
  total?: number
  cellClick?: Function
  tableBind?: any
}

const paginationData = computed(() => ({
  page: props.pi,
  pageSize: props.ps,
  total: props.total,
  pageSizes: props.page.pageSizes,
  layout: props.page.showSize ? 'total, sizes, prev, pager, next' : 'total, prev, pager, next'
}))

const props = withDefaults(defineProps<TypeProps>(), {
  loading: false,
  columns: () => [],
  data: () => [],
  page: () => ({
    show: false,
    pageSizes: [10, 25, 50, 100],
    showSize: false
  }),
  pi: 1,
  ps: 20,
  total: 0
})

const renderEnum = ({ item, scope }: { item: TableColumnType; scope: any }) => {
  const val = scope.row[item.prop]
  const enumConfig = item.enum?.find((v) => v.value === val)
  const type = item.type
  let label = $t(enumConfig?.label || '')
  if (type === 'tag') {
    return (
      <>
        <el-tag type={enumConfig?.type}>{label}</el-tag>
      </>
    )
  }
  return (
    <>
      <el-text type={enumConfig?.type}>{label}</el-text>
    </>
  )
}

const emit = defineEmits<{ change: [value: ElxTable.Change] }>()

const pageSizeChange = (ev: any) => {
  emit('change', { type: 'ps', ps: ev })
}

const currentPageChange = (ev: any) => {
  emit('change', { type: 'pi', pi: ev })
}

// 单元格点击事件
const handleCellClick = (row: any, column: any, event: any) => {
  if (props.cellClick) {
    props.cellClick(row, column, event)
  }
}

defineExpose({
  table: tableRef
})
</script>
<style lang="scss" scoped>
.table-border {
  border: 1px solid #e9e5e5;
  border-bottom: 0;
}
</style>
