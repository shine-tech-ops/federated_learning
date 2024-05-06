<template>
  <!-- 操作 -->
  <el-table-column
    :label="$t(item.label || 'tableForm.operation')"
    class-name="table-operation-cell"
    :width="item.width || 160"
  >
    <template #default="scope">
      <!-- 默认用下拉形式 -->
      <el-dropdown v-if="isFold">
        <span class="el-dropdown-link">
          <el-icon><More /></el-icon>
        </span>
        <template #dropdown>
          <el-dropdown-menu>
            <template v-for="btn in item.buttons" :key="btn">
              <el-dropdown-item
                v-if="isShow(btn, scope)"
                v-auth:if="btn.auth"
                @click="actionConfirm(btn, scope)"
                :disabled="btn.disabled ? btn.disabled(scope.row) : false"
              >
                <el-button
                  link
                  :disabled="btn.disabled ? btn.disabled(scope.row) : false"
                  size="default"
                  :type="tableButtonType[btn.type]"
                  >{{ $t(btn.label) }}</el-button
                >
              </el-dropdown-item>
            </template>
          </el-dropdown-menu>
        </template>
      </el-dropdown>
      <!-- 如果监测到实际显示的按钮数等于1，就直接显示 -->
      <template v-else>
        <template v-for="btn in item.buttons" :key="btn">
          <el-button
            v-if="isShow(btn, scope)"
            v-auth:if="btn.auth"
            link
            size="default"
            :type="tableButtonType[btn.type]"
            :disabled="btn.disabled ? btn.disabled(scope.row) : false"
            @click="actionConfirm(btn, scope)"
            >{{ $t(btn.label) }}</el-button
          >
        </template>
      </template>
    </template>
  </el-table-column>
</template>

<script setup lang="ts">
import { tableButtonType } from '@/utils/dict'
import { ElMessageBox } from 'element-plus'
import { useI18n } from 'vue-i18n'

const { t: $t } = useI18n()
const props = defineProps<{ item: TableColumnType }>()

const isFold = ref(true)

const isShow = (btn: TableButtonType, scope: any) => {
  return !btn.iif || (btn.iif && btn.iif(scope.row))
}

const actionConfirm = (btn: TableButtonType, scope: any) => {
  if (btn.type === 'delete' || !!btn.confirm) {
    const title = `${$t('message.confirmAction')}${$t(btn.label)}?`
    ElMessageBox.confirm(title, $t('message.tips'), {
      type: 'warning',
      beforeClose: (action, instance, done) => {
        if (action === 'confirm') {
          instance.confirmButtonLoading = true
          instance.confirmButtonText = 'Loading...'

          const result = btn.handle(scope.row)

          if (result instanceof Promise) {
            result
              .then(() => {
                instance.confirmButtonLoading = false
                done()
              })
              .catch(() => {
                // TODO i18n
                instance.confirmButtonText = 'Retry'
                instance.confirmButtonLoading = false
              })
          } else {
            setTimeout(() => {
              instance.confirmButtonLoading = false
              done()
            }, 1000)
          }
        } else {
          done()
        }
      }
    })
      .then(() => {})
      .catch(() => {})
  } else {
    btn.handle(scope.row)
  }
}
</script>
