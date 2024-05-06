// 系统全局字典
/**
 * 将枚举值转换为 Form.EnumObj 格式的数组
 * @param enumObj
 * @returns
 */
function enumToArray<T extends string | number | symbol>(enumObj: { [key in keyof T]: T[key] }) {
  return Object.keys(enumObj)
    .filter((k) => isNaN(Number(k)) && typeof enumObj[k as keyof typeof enumObj] !== 'function')
    .map((k) => ({
      value: enumObj[k as keyof typeof enumObj] as string,
      label: enumObj[k as keyof typeof enumObj] as string
    }))
}

/**
 * @description：用户状态
 */
export const userStatus = [
  { label: '启用', value: 1, type: 'success' },
  { label: '禁用', value: 0, type: 'danger' }
]

import type { ButtonType } from 'element-plus'
// table 按钮类型
export const tableButtonType: { [key in string]: ButtonType } = {
  // edit: 'primary',
  edit: 'info',
  view: 'info',
  delete: 'danger'
}

export const booleanEnumArr: Form.EnumObj[] = [
  { value: true, label: '是' },
  { value: false, label: '否' }
]

/**
 * 将枚举数组转换成字典对象
 */
export const enumArrayToDict = (arr: Form.EnumObj[]) => {
  const obj: { [key in string]: any } = {}
  arr.forEach((item) => {
    obj[item.value] = item.label
  })
  return obj
}

// 等级
// TODO i18n
export const level: Form.EnumObj[] = [
  { value: 1, label: '1级' },
  { value: 2, label: '2级' },
  { value: 3, label: '3级' }
]
