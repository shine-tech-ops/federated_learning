<template>
  <el-dropdown trigger="click" @command="changeLanguage">
    <div class="flex items-center">
      <svg-icon name="lang" class="!text-xl"></svg-icon>
    </div>
    <template #dropdown>
      <el-dropdown-menu>
        <el-dropdown-item
          v-for="item in languageList"
          :key="item.value"
          :command="item.value"
          :disabled="language === item.value"
        >
          {{ item.label }}
        </el-dropdown-item>
      </el-dropdown-menu>
    </template>
  </el-dropdown>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { computed } from 'vue'
import pinia from '@/stores'
import { useGlobalStore } from '@/stores/modules/global'
import type { LanguageType } from '@/stores/interface'
const i18n = useI18n()
const globalStore = useGlobalStore(pinia)
const language = computed(() => globalStore.language)
const languageList = [
  { label: '简体中文', value: 'zh' },
  { label: 'English', value: 'en' }
]

const changeLanguage = (lang: LanguageType) => {
  i18n.locale.value = lang as string
  globalStore.setGlobalState('language', lang as LanguageType)
}
</script>
