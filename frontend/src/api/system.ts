import service from '@/utils/request'
import {systemConfigModel} from "@/api/systemConfig";
export const CONFIG = {
  mock: import.meta.env.VUE_APP_MOCK || 'online',
  developing: import.meta.env.VUE_APP_DEVELOPING === 'true' ? true : false
}
const mock = CONFIG.mock
// const mock = 'local'
const developing = CONFIG.developing
const method = {
  del: 'DELETE',
  put: 'PUT',
  post: 'POST',
  get: 'GET'
}

export const sysModel = {

  uploadFile: (data: any) => {
    return service({
      developing,
      mock,
      method: method.post,
      url: `/v1/common/upload/`,
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      data
    })
  },
  getSystemLog: (data: any) => {
    return service({
      developing,
      mock,
      method: method.get,
      url: `/v1/common_config/system_log/`,
      params: data,
      // 不对错误结果进行弹窗提示
      errHandle: 'none'
    })
  },
  exportSystemLog: (data: any) => {
    return service({
      developing,
      // mock: 'local',
      mock,
      method: method.get,
      url: `/v1/common_config/system_log/`,
      params: data,
      resCheck: false
    })
  },
  setSystemLog: (content: string) => {
    return service({
      developing,
      // mock: 'local',
      mock,
      method: method.post,
      url: `/v1/common_config/system_log/`,
      data: { content }
    })
  },
  systemLogExport: (data: any) => {
    return service({
      developing,
      // mock: 'local',
      mock,
      method: method.get,
      url: `/v1/system/log/export/`,
      params: data
    })
  }
}

export const {
    uploadFile,
    getSystemLog,
    exportSystemLog,
    setSystemLog,
    systemLogExport
} = sysModel
