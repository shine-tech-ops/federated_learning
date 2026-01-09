// ? Element 常用表单校验规则
import { reactive } from 'vue'
import type { FormRules } from 'element-plus'
import { cloneDeep } from 'lodash'
import i18n from '@/languages/index'
const { t: $t } = i18n.global

export const REGEXP_MOBILE =
  /^(((13[0-9]{1})|(15[0-9]{1})|(16[0-9]{1})|(17[3-8]{1})|(18[0-9]{1})|(19[0-9]{1})|(14[5-7]{1}))+\d{8})$/

/**
 *  @rule 手机号
 */
export function checkPhoneNumber(rule: any, value: any, callback: any) {
  const regexp = REGEXP_MOBILE
  if (value === '') callback($t('validate.pleaseInput'))
  if (!regexp.test(value)) {
    callback(new Error($t('validate.enterValidMobileNumber')))
  } else {
    return callback()
  }
}

/** 密码正则（密码格式应为8-18位数字、字母、符号的任意两种组合） */
export const REGEXP_PWD_1 =
  /^(?![0-9]+$)(?![a-z]+$)(?![A-Z]+$)(?!([^(0-9a-zA-Z)]|[()])+$)(?!^.*[\u4E00-\u9FA5].*$)([^(0-9a-zA-Z)]|[()]|[a-z]|[A-Z]|[0-9]){8,18}$/

/** 密码 */
export const REGEXP_PWD = /^([a-zA-Z0-9!@#$%^&*()_+.,?/]){6,30}$/

/** 登录校验 */
export const loginRules = reactive(<FormRules>{
  password: [
    {
      validator: (rule, value, callback) => {
        if (value === '') {
          callback(new Error($t('validate.enterPassword')))
        } else if (!REGEXP_PWD.test(value)) {
          callback(new Error($t('validate.passwordLength')))
        } else {
          callback()
        }
      },
      trigger: 'blur'
    }
  ]
})

export const rules_email = [
  { required: true, message: $t('validate.enterEmailAddress'), trigger: 'blur' },
  { type: 'email', message: $t('validate.enterValidEmailAddress'), trigger: ['blur', 'change'] }
]

export const REGEXP_IPV4 =
  /^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/

const checkPermissions = (
  _rule: any,
  value: string | any[],
  callback: (arg0?: Error | undefined) => void
) => {
  if (value.length <= 0) {
    return callback(new Error($t('validate.permissionNotEmpty')))
  } else {
    callback()
  }
}

export const rules_ipv4 = [
  { required: true, message: $t('validate.enterIpAddress'), trigger: 'blur' },
  { type: 'ip_address', message: $t('validate.enterValidIpAddress'), trigger: ['blur', 'change'] }
]

const genRequiredRule = (type: string) => {
  return type === 'select'
    ? { required: true, message: $t('validate.pleaseSelect'), trigger: 'change' }
    : { required: true, message: $t('validate.pleaseInput'), trigger: 'blur' }
  // : { required: true, message: '请输入', trigger: 'blur' }
}

export const genRulesDict = (): { [key in string]: any } => ({
  ipv4: [
    { required: true, message: $t('validate.enterIpAddress'), trigger: 'blur' },
    {
      pattern: REGEXP_IPV4,
      message: $t('validate.enterValidIpAddress'),
      trigger: ['blur', 'change']
    }
  ],
  email: [
    { required: true, message: $t('validate.enterEmailAddress'), trigger: 'blur' },
    { type: 'email', message: $t('validate.enterValidEmailAddress'), trigger: ['blur', 'change'] }
  ],
  password: [
    {
      validator: (_rule: any, value: any, callback: any) => {
        if (value === '') {
          callback(new Error($t('validate.enterPassword')))
        } else if (!REGEXP_PWD.test(value)) {
          callback(new Error($t('validate.passwordLength')))
        } else {
          callback()
        }
      },
      trigger: 'blur'
    }
  ],
  permission: [genRequiredRule('select'), { validator: checkPermissions, trigger: 'blur' }],
  mobile: [
    genRequiredRule('input'),
    {
      pattern: REGEXP_MOBILE,
      message: $t('validate.enterValidMobileNumber'),
      trigger: ['blur', 'change']
    }
  ]
})

/*
export const rulesDict: { [key in string]: any } = {
  ipv4: [
    { required: true, message: $t('validate.enterIpAddress'), trigger: 'blur' },
    { pattern: REGEXP_IPV4, message: $t('validate.enterValidIpAddress'), trigger: ['blur', 'change'] }
  ],
  email: [
    { required: true, message: $t('validate.enterEmailAddress'), trigger: 'blur' },
    { type: 'email', message: $t('validate.enterValidEmailAddress'), trigger: ['blur', 'change'] }
  ],
  password: [
    {
      validator: (_rule: any, value: any, callback: any) => {
        if (value === '') {
          callback(new Error($t('validate.enterPassword')))
        } else if (!REGEXP_PWD.test(value)) {
          callback(new Error($t('validate.passwordLength')))
        } else {
          callback()
        }
      },
      trigger: 'blur'
    }
  ],
  permission: [genRequiredRule('select'), { validator: checkPermissions, trigger: 'blur' }],
  mobile: [
    genRequiredRule('input'),
    { pattern: REGEXP_MOBILE, message: $t('validate.enterValidMobileNumber'), trigger: ['blur', 'change'] }
  ]
}*/

/**
 * 获取表单验证规则
 * @param key 首先会从 rulesDict 中获取，若没有且required = true，就根据key是否为select 返回对应的必填校验规则
 * @param required 默认true，可不传
 * @returns
 */
export const genRules = (key: string, required: boolean = true) => {
  let res: any[] = []
  let rulesDict = genRulesDict()
  if (rulesDict[key]) {
    res = cloneDeep(rulesDict[key])
    if (!required) {
      res = res.filter((item) => !item.required)
    }
  }

  if (required && res.findIndex((v) => v.required) < 0) {
    res.push(genRequiredRule(key))
  }

  return res
}
