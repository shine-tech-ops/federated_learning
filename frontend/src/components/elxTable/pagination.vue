<template>
  <!-- 分页组件 -->
  <el-pagination
    background
    :current-page="data.page"
    :page-size="data.pageSize"
    :page-sizes="data.pageSizes || [10, 25, 50, 100]"
    :total="data.total"
    :layout="data.layout || 'total, prev, pager, next'"
    @size-change="handleSizeChange || defaultFun"
    @current-change="handleCurrentChange"
  ></el-pagination>
</template>

<script setup lang="ts" name="Pagination">
interface Data {
  page: number
  pageSize: number
  total: number
  pageSizes?: number[]
  layout?: string
}

const defaultFun = () => {}
interface PaginationProps {
  data: Data
  handleSizeChange?: (size: number) => void
  handleCurrentChange: (currentPage: number) => void
}

defineProps<PaginationProps>()
</script>
