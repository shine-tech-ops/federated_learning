// import './assets/main.css'
import './styles/tailwind.scss'
// element css
import 'element-plus/dist/index.css'
// element dark css
// import 'element-plus/theme-chalk/dark/css-vars.css'
import './styles/themes.scss'
import './styles/transition.scss'
import './styles/index.scss'
// element icons
import * as Icons from '@element-plus/icons-vue'
import svgIcon from '@/components/svgIcon/index.vue'
import 'virtual:svg-icons-register'
import App from './App.vue'
import router from './router'
import pinia from '@/stores'
import directive from '@/directive/index'
// import { useGlobalStore } from '@/stores/modules/global'
import ElementPlus from 'element-plus'
import zhCn from 'element-plus/es/locale/lang/zh-cn'
import { createApp } from 'vue'
// import { createPinia } from 'pinia'
// vue i18n
import I18n from '@/languages/index'

const app = createApp(App)

// register the element Icons component
Object.keys(Icons).forEach((key) => {
  app.component(key, Icons[key as keyof typeof Icons])
})

// app.use(createPinia())
app.use(I18n).use(pinia).use(directive).use(router)
app.component('svg-icon', svgIcon)
app.use(ElementPlus)
// app.use(ElementPlus, { size: 'default', zIndex: 3000, locale: zhCn })

app.mount('#app')
// const globalStore = useGlobalStore(pinia)
// 每次刷新页面都要更新
// globalStore.initSystem()
