import service from '@/utils/request'
import { CONFIG } from './system'

const method = {
  del: 'DELETE',
  put: 'PUT',
  post: 'POST',
  get: 'GET'
}

export const modelChatLogApi = {
  /**
   * 获取模型对话日志
   */
  fetchChatLogs: (params = {}) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.get,
      url: '/v1/learn_management/model_chat_log/',
      params
    })
  },

  /**
   * 上传模型对话日志
   */
  uploadChatLog: (data: any) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.post,
      url: '/v1/learn_management/model_chat_log/',
      data
    })
  }
}

export const { fetchChatLogs, uploadChatLog } = modelChatLogApi


