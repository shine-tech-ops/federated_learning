<template>
  <el-config-provider :locale="locale">
    <router-view />
  </el-config-provider>
</template>
<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { getBrowserLang } from '@/utils'
import { ElConfigProvider } from 'element-plus'
import type { LanguageType } from './stores/interface'
import pinia from '@/stores'
import { useGlobalStore } from '@/stores/modules/global'
import en from 'element-plus/es/locale/lang/en'
import zhCn from 'element-plus/es/locale/lang/zh-cn'
const globalStore = useGlobalStore(pinia)

// init language
const i18n = useI18n()
onMounted(() => {
  const language = globalStore.language ?? getBrowserLang()
  i18n.locale.value = language
  // globalStore.setGlobalState("language", language as LanguageType);
})

watch(
  () => [globalStore.sysTitle, globalStore.language],
  ([title, lang]) => {
    document.title = i18n.t(title as string)
  },
  {
    immediate: true
  }
)

// element language
const locale = computed(() => {
  if (globalStore.language == 'zh') return zhCn
  if (globalStore.language == 'en') return en
  return getBrowserLang() == 'zh' ? zhCn : en
})
</script>
