<template>
  <div class="select-none">
    <!-- <img :src="bg" class="wave pointer-events-none" /> -->
    <div class="login-container" :style="`background-image: url(${bg})`">
      <!-- <div class="login-img">
        <svg-icon name="login"></svg-icon>
      </div> -->
      <div class="logo-box">
        <el-image :src="globalStore.sysLogo" class="h-[80px] inline-block">
          <template #error>
            <img :src="globalStore.defaultLogo" class="h-[80px] inline-block" alt="logo" />
          </template>
        </el-image>
      </div>
      <div class="mr-6 mt-6 absolute right-0 top-0">
        <Language></Language>
      </div>
      <div class="login-box">
        <div class="login-form">
          <!-- <el-image :src="globalStore.sysLogo" class="h-[80px] inline-block">
            <template #error>
              <img :src="globalStore.defaultLogo" class="h-[80px] inline-block" alt="logo" />
            </template>
          </el-image> -->
          <h2 class="outline-none">{{ $t(globalStore.sysTitle) }}</h2>

          <el-form ref="ruleFormRef" :model="ruleForm" :rules="rules" size="large">
            <el-form-item prop="username">
              <el-input
                clearable
                v-model="ruleForm.username"
                :placeholder="$t('login.account')"
                :prefix-icon="User"
              />
            </el-form-item>

            <el-form-item prop="password">
              <el-input
                clearable
                show-password
                v-model="ruleForm.password"
                :placeholder="$t('login.password')"
                :prefix-icon="Lock"
              />
            </el-form-item>

            <el-button
              class="w-full mt-4"
              size="default"
              type="primary"
              :loading="loading"
              @click.prevent="onLogin(ruleFormRef)"
            >
              {{ $t('login.login') }}
            </el-button>
          </el-form>
        </div>
      </div>
    </div>
  </div>
</template>
<script setup lang="ts">
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import type { FormInstance } from 'element-plus'
import { initRouter } from '@/router'
// import bg from '@/assets/images/bg.png'
import bg from '@/assets/logo.svg'
import { ref, reactive, onMounted, onBeforeUnmount } from 'vue'
import { Lock, User } from '@element-plus/icons-vue'
import { loginRules } from '@/utils/validate'
import { authModel } from '@/api'
import { Session } from '@/utils/storage'
import { usePermissionStore } from '@/stores/modules/permission'
import pinia from '@/stores'
import { useGlobalStore } from '@/stores/modules/global'
import { genRules } from '@/utils/validate'
import { type FormRules } from 'element-plus'
import Language from '@/layouts/components/Language.vue'
import useRules from '@/hooks/useRules'
// import { useI18n } from 'vue-i18n'

// const { t: $t } = useI18n()
const rules2: FormRules = {
  username: genRules(''),
  password: genRules('')
}

const rules = useRules(() => {
  return {
    username: genRules(''),
    password: genRules('')
  }
})
const globalStore = useGlobalStore(pinia)

defineOptions({
  name: 'Login'
})
const router = useRouter()
console.log('🚀 ~ router:', router)
const loading = ref(false)
const ruleFormRef = ref<FormInstance>()

// const title = import.meta.env.VITE_APP_TITLE

const ruleForm = reactive({
  username: '',
  password: ''
})

const onLogin = async (formEl: FormInstance | undefined) => {
  loading.value = true
  if (!formEl) return
  await formEl.validate((valid, fields) => {
    if (valid) {
      authModel
        .login({ name: ruleForm.username, password: ruleForm.password })
        .then((res: { access: any }) => {
          if (res.access) {
            Session.set('token', res.access)
            initRouter().then(() => {
              ElMessage.success('登录成功')
              router.push('/')
              // 初始化权限列表
              const permissionState = usePermissionStore(pinia)
              permissionState.initPermissions()
            })
          } else {
            ElMessage.error('登录失败，返回格式不对')
            loading.value = false
          }
        })
        .catch((err: any) => {
          console.log('err', err)
          loading.value = false
          // ElMessage.error(err.response.data.detail)
        })
    } else {
      loading.value = false
      return fields
    }
  })
}

/** 使用公共函数，避免`removeEventListener`失效 */
function onkeypress({ code }: KeyboardEvent) {
  if (code === 'Enter') {
    onLogin(ruleFormRef.value)
  }
}

onMounted(() => {
  window.document.addEventListener('keypress', onkeypress)
})

onBeforeUnmount(() => {
  window.document.removeEventListener('keypress', onkeypress)
})
</script>
<style scoped>
@import './index.scss';
</style>
