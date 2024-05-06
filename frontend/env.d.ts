/// <reference types="vite/client" />
declare module 'js-cookie'
declare module 'qs'
declare module 'axios'
declare module 'lodash'
declare module 'nprogress'
declare module 'mockjs'

declare module '*.vue' {
  import { ComponentOptions } from 'vue'
  // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/ban-types
  const componentOptions: ComponentOptions
  export default componentOptions
}
