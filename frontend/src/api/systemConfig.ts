import request from '@/utils/request'
import { CONFIG } from './system'

const method = {
  del: 'DELETE',
  put: 'PUT',
  post: 'POST',
  get: 'GET'
}

export const systemConfigModel = {
  /**
   * 获取系统配置列表（支持分页）
   */
  getConfigsApi: (params = {}) => {
    return request({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.get,
      url: '/v1/learn_management/system/config/',
      params
    })
  },

  /**
   * 获取单个系统配置详情
   */
  getConfigDetailApi: (id: number) => {
    return request({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.get,
      url: `/v1/learn_management/system/config/${id}/`,
    })
  },

  /**
   * 创建系统配置
   */
  createConfigApi: (configData: any) => {
    return request({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.post,
      url: '/v1/learn_management/system/config/',
      data: configData
    })
  },

  /**
   * 更新系统配置
   */
  updateConfigApi: (configData: any) => {
    return request({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.put,
      url: '/v1/learn_management/system/config/',
      data: configData
    })
  },

  /**
   * 删除系统配置
   */
  deleteConfigApi: (configData: any) => {
    return request({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.del,
      url: '/v1/learn_management/system/config/',
      data: configData
    })
  },
  /**
   * 激活系统配置
   */
  activateConfigApi: (configData: any) => {
      return request({
        developing: CONFIG.developing,
        mock: CONFIG.mock,
        method: method.post,
        url: '/v1/learn_management/system/config/activate/',
        data: configData
      })
  },
  /**
   * 获取联邦学习聚合策略映射
   */
  getAggregationMethodApi: (configData: any) => {
      return request({
        developing: CONFIG.developing,
        mock: CONFIG.mock,
        method: method.get,
        url: '/v1/learn_management/system/aggregation_method/',
        params: configData
      })
  }
}

// 为了保持原有引用方式不变，继续导出单个 API
export const {
  getConfigsApi,
  getConfigDetailApi,
  createConfigApi,
  updateConfigApi,
  deleteConfigApi,
  activateConfigApi,
  getAggregationMethodApi,
} = systemConfigModel
