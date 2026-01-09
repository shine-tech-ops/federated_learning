import service from '@/utils/request'
import { CONFIG } from './system'

const method = {
  del: 'DELETE',
  put: 'PUT',
  post: 'POST',
  get: 'GET'
}

export const modelInfoApi = {
  /**
   * 获取所有模型信息
   */
  fetchModelInfos: (params = {}) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.get,
      url: '/v1/learn_management/model_info/',
      params,
    })
  },

  /**
   * 创建模型
   */
  createModel: (data: any) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.post,
      url: '/v1/learn_management/model_info/',
      data
    })
  },

  /**
   * 删除模型
   */
  deleteModel: (id: number) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.del,
      url: '/v1/learn_management/model_info/',
      data: { id }
    })
  }
}

export const modelManagementApi = {
   /**
   * 创建模型版本
   */
  createModelVersion: (data: any) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.post,
      url: '/v1/learn_management/model_version/',
      data
    })
  },
  /**
   * 获取模型版本列表
   */
  fetchModelVersions: (params: any) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.get,
      url: '/v1/learn_management/model_version/',
      params
    })
  },

  /**
   * 部署/取消部署模型版本
   */
  deployModel: (id: number, is_deployed: boolean) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.post,
      url: '/v1/learn_management/model_version/deploy/',
      data: { id, is_deployed }
    })
  },

  /**
   * 删除模型版本
   */
  deleteModelVersion: (id: number) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.del,
      url: '/v1/learn_management/model_version/',
      data: { id }
    })
  },

  /**
   * 上传模型文件
   */
  uploadModelFile: (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.post,
      url: '/v1/learn_management/model_version/upload/',
      data: formData,
      resCheck: true  // 使用默认的响应检查，会自动解包 data
      // 注意：不要手动设置 Content-Type，让浏览器自动设置（包含 boundary）
    })
  },

  /**
   * 下载模型版本文件（blob）
   */
  downloadModelVersion: (id: number) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.get,
      url: `/v1/learn_management/model_version/${id}/download/`,
      responseType: 'blob',
      resCheck: false,      // 防止拦截器按 JSON 解包
      errHandle: 'none'
    })
  }
}

// 为了保持原有引用方式不变，继续导出单个 API
export const {
  fetchModelInfos,
  createModel,
  deleteModel
} = modelInfoApi

export const {
    createModelVersion,
    fetchModelVersions,
    deployModel,
    deleteModelVersion,
    uploadModelFile,
    downloadModelVersion
} = modelManagementApi

