<template>
  <el-dropdown trigger="click">
    <div class="flex items-center">
      <el-icon class="!text-lg"><User /></el-icon>
      <span class="text-sm ml-2">{{ userStore.users.name }}</span>
      <el-icon class="el-icon--right">
        <arrow-down />
      </el-icon>
      <!-- <svg-icon name="avatar" class="!text-2xl"></svg-icon> -->
    </div>
    <template #dropdown>
      <el-dropdown-menu>
        <template>
          <el-dropdown-item
            @click="openAccountDialog()"
            v-if="!userStore.authIs(['admin', 'superuser'])"
          >
            <el-icon><User /></el-icon>{{ $t('header.accountSetting') }}
          </el-dropdown-item>
        </template>
        <el-dropdown-item @click="changePassword()">
          <el-icon><Edit /></el-icon>{{ $t('header.changePassword') }}
        </el-dropdown-item>
        <el-dropdown-item @click="logout()">
          <el-icon><SwitchButton /></el-icon>{{ $t('header.logout') }}
        </el-dropdown-item>
      </el-dropdown-menu>
    </template>
  </el-dropdown>

  <account-set ref="accountDialogRef"></account-set>
  <!-- <site-set ref="siteDialogRef"></site-set> -->
  <!-- <change-password-vue ref="changePasswordRef"></change-password-vue> -->
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useUserStore } from '@/stores/modules/user'
import { ElMessageBox } from 'element-plus'
import { authModel } from '@/api'
import pinia from '@/stores'
import { Session } from '@/utils/storage'
import AccountSet from './AccountSet.vue'
import SiteSet from './SiteSet.vue'
// import ChangePasswordVue from './ChangePassword.vue'
import mittBus from '@/utils/mittBus'
import { useI18n } from 'vue-i18n'

const { t: $t } = useI18n()
const userStore = useUserStore(pinia)
const accountDialogRef = ref()
const siteDialogRef = ref()
const changePasswordRef = ref()

const openAccountDialog = () => {
  accountDialogRef.value.open()
}

const openSiteDialog = () => {
  siteDialogRef.value.open()
}

// 退出登录
const logout = () => {
  ElMessageBox.confirm($t('login.logoutConfirm'), $t('message.tips'), {
    type: 'warning',
    customClass: 'no-icon'
  }).then(async () => {
    userStore.logout()
  })
}

const changePassword = () => {
  changePasswordRef.value.open()
}

onMounted(() => {
  // 强制修改密码
  mittBus.on('force-change-password', () => {
    changePasswordRef.value.open(true)
  })
})
</script>
