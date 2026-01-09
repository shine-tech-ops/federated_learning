import { defineStore } from 'pinia'
import { type GlobalState } from '@/stores/interface'
import * as config from '@/config'

let suffix = ''
// import { useI18n } from 'vue-i18n'

export const useGlobalStore = defineStore({
  id: 'app-global',
  // 修改默认值之后，需清除 localStorage 数据
  state: (): GlobalState => ({
    // 当前系统语言
    language: 'zh',
    // 深色模式
    isDark: false,
    // 折叠菜单
    isCollapse: false,
    // 是否显示tagsView的图标
    tagsViewIcon: false,
    // 站点类型 main中心站 sub 分站
    site: 'sub',
    defaultLogo: config.SYSTEM_LOGO,
    // 系统logo
    sysLogo: config.SYSTEM_LOGO,
    // 系统名称
    sysTitle: config.SYSTEM_TITLE,
    // 有缓存，不能用
    sysTitleSuffix: suffix,
    // 是否已初始化过系统信息
    sysLoaded: false,
    // 系统信息id
    siteBrandId: 0,
    // 站点UUID
    siteUUID: '',
    sysTheme: 'default',
    isFirstLogin: false
  }),
  // getters: {},
  actions: {
    async initSysInfo() {},
    setLanguage(val: 'zh' | 'en' | null) {
      this.language = val
    },
    updateSysInfo(res: { name: string; logo: string; id: number; uuid: string }) {},
    // Set GlobalState
    setGlobalState(...args: ObjToKeyValArray<GlobalState>) {
      this.$patch({ [args[0]]: args[1] })
    },
    // 重置系统配置
    resetSysInfo() {
      const { logo, title } = this.getDefaultSystem()
      this.sysLogo = logo
      this.sysTitle = title
    },
    // 更新 favicon.ico
    setFavicon() {
      const link = this.sysLogo || this.defaultLogo
      let $favicon = document.querySelector('link[rel="icon"]')
      // @ts-ignore
      if ($favicon !== null && $favicon.href != link) {
        // @ts-ignore
        $favicon.href = link
      } else {
        $favicon = document.createElement('link')
        // @ts-ignore
        $favicon.rel = 'icon'
        // @ts-ignore
        $favicon.href = link
        document.head.appendChild($favicon)
      }
    },
    // 初始化系统配置
    initSystem() {
      // document.title = $t(this.sysTitle) + suffix
      // document.title = this.sysTitle + suffix
      this.setFavicon()
    },
    getDefaultSystem() {
      return {
        logo: config.SYSTEM_LOGO,
        title: config.SYSTEM_TITLE,
        id: this.siteBrandId
      }
    },
    markFirstLogin(val: boolean) {
      this.isFirstLogin = val
    }
  },
  persist: true
})
