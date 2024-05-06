const mode = import.meta.env.VITE_ROUTER_MODE
import * as _ from 'lodash'
import { formatDate } from '@/utils/formatDate'
import { sysModel } from '@/api'
import { Session, Local } from '@/utils/storage'

/**
 * @description 生成唯一 uuid
 * @returns {String}
 */
export function generateUUID() {
  let uuid = ''
  for (let i = 0; i < 32; i++) {
    const random = (Math.random() * 16) | 0
    if (i === 8 || i === 12 || i === 16 || i === 20) uuid += '-'
    uuid += (i === 12 ? 4 : i === 16 ? (random & 3) | 8 : random).toString(16)
  }
  return uuid
}

/**
 * @description 生成随机数
 * @param {Number} min 最小值
 * @param {Number} max 最大值
 * @returns {Number}
 */
export function randomNum(min: number, max: number): number {
  const num = Math.floor(Math.random() * (min - max) + max)
  return num
}

/**
 * @description 获取当前时间对应的提示语
 * @returns {String}
 */
export function getTimeState() {
  const timeNow = new Date()
  const hours = timeNow.getHours()
  if (hours >= 6 && hours <= 10) return `早上好 ⛅`
  if (hours >= 10 && hours <= 14) return `中午好 🌞`
  if (hours >= 14 && hours <= 18) return `下午好 🌞`
  if (hours >= 18 && hours <= 24) return `晚上好 🌛`
  if (hours >= 0 && hours <= 6) return `凌晨好 🌛`
}

/**
 * @description 获取不同路由模式所对应的 url + params
 * @returns {String}
 */
export function getUrlWithParams() {
  const url: { [key in string]: string } = {
    hash: location.hash.substring(1),
    history: location.pathname + location.search
  }
  return url[mode]
}

export const getLabelByValue = (arr: Form.EnumObj[], val: any) => {
  const res = arr.find((item) => item.value === val)
  return res?.label || ''
}

/**
 * 文件转base64 url
 * @param file
 * @returns
 */
export function convertToBase64(file: File) {
  return new Promise<string>(function (resolve, reject) {
    const reader = new FileReader()
    let base64 = ''
    reader.onload = (e) => {
      base64 = e.target?.result as string
    }
    reader.onerror = function (error) {
      reject(error)
    }
    reader.onloadend = function () {
      resolve(base64)
    }
    reader.readAsDataURL(file)
  })
}

/**
 * 导出csv
 * @param csvData
 * @param fileName
 */
export const exportToCsv = (csvData: string, fileName: string) => {
  const blob = new Blob([csvData], { type: 'text/csv;charset=utf-8' })
  const link = document.createElement('a')
  const url = window.URL.createObjectURL(blob)
  link.href = url
  link.download = fileName
  link.click()
  window.URL.revokeObjectURL(url)
}

/**
 * 导出
 * @param response
 * @param fileName 注意fileName中需要带后缀
 */
export const exportStream = (response: string, fileName: string) => {
  const blob = new Blob([response])
  const link = document.createElement('a')
  const url = window.URL.createObjectURL(blob)
  link.href = url
  link.download = fileName
  link.click()
  window.URL.revokeObjectURL(url)
}

/**
 * 根据比率得到对应的颜色
 * @param percentage 百分比
 * @param colors 颜色范围
 * @returns
 */
export function getColorByRate(percentage: number, colors: RATE.Color[]) {
  for (const colorItem of colors) {
    if (percentage <= colorItem.percentage) {
      return colorItem.color
    }
  }

  return '#5cb87a'
}

/**
 * 根据倍数获取对应的值
 * multiplier 倍数
 * ori = true 不四舍五入
 */
export const multiplierValue = {
  set: (value: string | number, multiplier: number, ori = false) => {
    const v = Number(value) / multiplier
    if (ori) return v
    return isNaN(v) ? '-' : _.round(v, 2)
  },
  get: (value: string | number, multiplier: number, ori = false) => {
    const v = Number(value) * multiplier
    if (ori) return v
    return isNaN(v) ? '-' : _.round(v, 2)
  }
}

/**
 * 获取指定天数的日期范围
 */
export function getDefaultDateRange(durationInDays = 7, format = 'YYYY-MM-DD HH:mm:ss') {
  const today = new Date()
  today.setHours(23, 59, 59, 999)
  const end = today.getTime()
  const startTime = end - 1000 * 3600 * 24 * durationInDays
  const start = new Date(startTime).setHours(0, 0, 0, 0)
  return [formatDate(start, format), formatDate(end, format)]
}

export function isValid(val: unknown) {
  return val !== null && val !== undefined && val !== ''
}

/**
 * @description 获取浏览器默认语言
 * @returns {String}
 */
export function getBrowserLang() {
  // @ts-ignore
  let browserLang = navigator.language ? navigator.language : navigator.browserLanguage
  let defaultBrowserLang = ''
  if (['cn', 'zh', 'zh-cn'].includes(browserLang.toLowerCase())) {
    defaultBrowserLang = 'zh'
  } else {
    defaultBrowserLang = 'en'
  }
  return defaultBrowserLang
}

/**
 * 向系统日志中添加一条记录
 * @param operation.page 日志所属页面的名称
 * @param operation.action 在页面上执行的动作
 * @param operation.content 动作的具体内容或描述
 */
export function addSysLogX(operation: { page?: string; action?: string; content: string }) {
  const { page, action, content } = operation
  let arr = [page, action, content]
  // 过滤掉空字符串
  let res = arr.filter((item) => item !== '').join(' - ')
  // let res = arr.join(' - ')
  sysModel.setSystemLog(res)
}

export function authorization() {
  return {
    get: Local.get,
    set: Local.set
  }
}
