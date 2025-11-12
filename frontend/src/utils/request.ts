import axios, { type AxiosResponse, type AxiosError } from 'axios'

import qs from 'qs'
import _ from 'lodash'
import { ElMessage, ElMessageBox } from 'element-plus'
import router from '@/router'
import { Session } from '@/utils/storage'
// import { useI18n } from 'vue-i18n'
import i18n from '@/languages/index'
const { t: $t } = i18n.global

const NODE_ENV = import.meta.env.VITE_USER_NODE_ENV
const MOCK_URL = import.meta.env.VITE_APP_API_MOCK_URL
const LOCAL_URL = import.meta.env.VITE_APP_API_LOCAL_URL
// const APP_TITLE = import.meta.env.VITE_APP_TITLE
const PROD_URL = import.meta.env.VITE_APP_API_PROD

export interface ResponseData {
  success: boolean
  message?: string
  data?: any
}

/*
  @ 创建axios实例
*/
const service = axios.create({
  // baseUrl: import.meta.env.VITE_APP_API_BASE_URL,
  baseURL: '',
  headers: { 'Content-Type': 'application/json' },
  paramsSerializer: {
    serialize(params: any) {
      return qs.stringify(params, { allowDots: true })
    }
  }
})


/*
  @ request 拦截器
*/
service.interceptors.request.use(
(config: typeof AxiosResponse) => {
     console.log(config)
    // 本地mock统一加 mock 前缀
    if (config.mock === 'local') {
      config.url = `/mock${config.url}`
    }

    // mock 模式：本地和在线mock两种混合
    if (NODE_ENV === 'mock') {
      if (config.mockID) {
        if (!config.params) {
          config.params = {}
        }
        // 相同的接口必须加mock id区分
        config.params['apifoxApiId'] = config.mockID
      }
      if (config.mock === 'online') {
        config.url = `/mock-online${config.url}`
      }
    }
    // 开发模式：本地 mock 和 真实接口混合
    if (NODE_ENV === 'development' || NODE_ENV === 'production' || NODE_ENV === 'main') {
      // 已经配置过的接口
      if (config.mock === 'online') {
        // config.developing 表示接口状态为开发中
        if (config.prod) {
          config.url = PROD_URL + config.url
        } else {
          config.url = (config.developing ? MOCK_URL : LOCAL_URL) + config.url
        }
      }
    }

    // 如果是 FormData，删除 Content-Type，让浏览器自动设置（包含 boundary）
    if (config.data instanceof FormData) {
      delete config.headers!['Content-Type']
    }

    if (Session.get('token')) {
      config.headers!['Authorization'] = `Bearer ${Session.get('token')}`
    }
    return config
  },
  (error: typeof AxiosError) => {
    Promise.reject(error)
  }
)

const CODEMESSAGE: { [key: number]: string } = {
  200: '服务器成功返回请求的数据。',
  201: '新建或修改数据成功。',
  202: '一个请求已经进入后台排队（异步任务）。',
  204: '删除数据成功。',
  400: '发出的请求有错误，服务器没有进行新建或修改数据的操作。',
  401: '用户没有权限（令牌、用户名、密码错误）。',
  403: '用户得到授权，但是访问是被禁止的。',
  404: '发出的请求针对的是不存在的记录，服务器没有进行操作。',
  406: '请求的格式不可得。',
  410: '请求的资源被永久删除，且不会再得到的。',
  422: '当创建一个对象时，发生一个验证错误。',
  500: '服务器发生错误，请检查服务器。',
  502: '网关错误。',
  503: '服务不可用，服务器暂时过载或维护。',
  504: '网关超时。',
  0: '服务器故障'
}

function popErr(msg: any, handle: string) {
  try {
    // 如果设置了处理方式为 none 就不弹窗
    if (handle !== 'none') {
      // 按长度自动计算关闭时间（1s 看30个字符）
      const duration = Math.max(3, msg.length / 30) * 1000
      ElMessage({
        type: 'error',
        message: msg,
        duration,
        dangerouslyUseHTMLString: true
      })
    }
  } catch (error) {}
}

function checkStatus(
  _code: string | number,
  config: typeof AxiosResponse,
  resData: any,
  msg?: any
) {
  const resCheck = config.resCheck !== false
  const errHandle = config.errHandle
  const code = Number(_code)
  switch (code) {
    case 200:
      if (resCheck) {
        try {
          const { msg, code, data } = resData
          config.resCheck = false
          return checkStatus(code, config, data, msg)
          // return checkStatus(200, config, data)
        } catch (error) {
          console.log('处理响应值发生意外', error)
        }
      } else {
        return resData
      }
      break

    case 4001:
      const errMsg = CODEMESSAGE[code] || msg || '服务器故障'
      popErr(errMsg, errHandle)
      break
    // 传参格式不对的提示
    case 406:
    case 409:
    // 无权限
    case 401:
    case 400:
    default:
      /**
       * 各种错误情况：
       * 1. 登录接口，输入不存在的账号 --> { detail: '...' }
       */
      if (code === 401) {
        if (Session.get('token') && errHandle !== 'none') {
          onUnauthenticated()
          return
        }
      }

      console.log('resData:', resData, 'msg:', msg)

      // TODO 根据后端定义的格式弹出错误内容
      let errText = CODEMESSAGE[code] || '服务器故障'
      // 部分接口已更新为，面向用户的错误信息放在msg里，框架的提示信息放在resData
      // 旧接口默认返回的msg是 error，要排除
      if (msg && msg !== 'error') {
        errText = msg
      }
      // 后面的会逐渐被去掉，接口统计改为 将提示信息放在 msg 里
      else if (resData.code) {
        errText = (typeof resData.data === 'string' ? resData.data : resData.msg) || resData.msg
      } else {
        if (resData.detail) {
          errText = resData.detail
        } else {
          if (Array.isArray(resData)) {
            let text = ''
            resData.forEach((item: { msg: string }) => {
              text += `${item.msg}<br>`
            })
            errText = text
          } else if (
            typeof resData === 'string' &&
            !resData.includes('<!DOCTYPE html>') &&
            resData.length < 100
          ) {
            /**
             {"code":404,"msg":"error","data":"The target record not exists."}
             {"code":406,"msg":"error","data":"uuid错误, 请检查后重试"}
            */
            errText = resData || errText
          }
        }
      }
      // if (code === 406) {
      //   errText = errText + '\n<br>' + resData
      // }
      popErr(errText, errHandle)
      break
  }
  return false
}

service.interceptors.response.use(
  (response: typeof AxiosResponse, config: any) => {
    try {
      // const { msg, code, data } = response.data
      const resData = checkStatus(response.status, response.config, response.data)
      if (!resData) {
        return Promise.reject(service.interceptors.response)
      }
      return resData
      // if (code && code !== 200) {
      //   // `token` 过期或者账号已在别处登录
      //   if (code === 401 || code === 4001) {
      //     onUnauthenticated()
      //   } else {
      //     ElMessage.error((typeof data === 'string' ? data : msg) || msg)
      //   }
      //   return Promise.reject(service.interceptors.response)
      // } else {
      //   return data
      // }
    } catch (error) {
      return Promise.reject(service.interceptors.response)
    }
  },
  (error: typeof AxiosError) => {
    if (error.message.indexOf('timeout') != -1) {
      ElMessage.error('网络超时')
    } else if (error.message == 'Network Error') {
      ElMessage.error('网络连接错误')
    } else if (error.name.indexOf('CanceledError') != -1) {
      // 手动取消了请求
    } else {
      const code = _.get(error, 'response.status')
      checkStatus(code, error.config, error.response.data)
      // if (error.response.data) ElMessage.error(error.response.statusText)
      // else ElMessage.error('接口路径找不到')
    }
    // const code = _.get(error, 'response.status')
    // console.log('🚀 ~ code:', code)
    // // 认证信息已过期
    // if (code === 401) {
    //   onUnauthenticated()
    // }
    return Promise.reject(error)
  }
)

function onUnauthenticated() {
  const currentPath = router.currentRoute.value.path
  const isLoginPage = currentPath === '/login'
  if (isLoginPage) {
    return
  }
  ElMessageBox.confirm($t('login.forceLogout'), $t('login.confirmLogout'), {
    confirmButtonText: $t('login.logInAgain'),
    type: 'warning'
  }).then(() => {
    Session.clear()
    router.push({ path: '/' })
    location.reload()
  })
}

export default service
