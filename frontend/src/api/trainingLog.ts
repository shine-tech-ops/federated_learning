import service from '@/utils/request'
import { CONFIG } from './system'

const method = {
  del: 'DELETE',
  put: 'PUT',
  post: 'POST',
  get: 'GET'
}

export const trainingLogModel = {
  /**
   * 获取训练日志列表
   */
  getTrainingLogsApi: (params = {}) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.get,
      url: '/v1/learn_management/training_log/',
      params
    })
  },

  /**
   * 上传训练日志
   */
  uploadTrainingLogApi: (logData: any) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.post,
      url: '/v1/learn_management/training_log/',
      data: logData
    })
  },

  /**
   * 获取训练日志统计信息
   */
  getTrainingLogStatsApi: (params = {}) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.get,
      url: '/v1/learn_management/training_log/stats/',
      params
    })
  }
}

export const {
  getTrainingLogsApi,
  uploadTrainingLogApi,
  getTrainingLogStatsApi
} = trainingLogModel

