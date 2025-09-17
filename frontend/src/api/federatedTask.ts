import service from '@/utils/request'
import { CONFIG } from './system'

const method = {
  del: 'DELETE',
  put: 'PUT',
  post: 'POST',
  get: 'GET'
}

export const federatedTaskModel = {
  /**
   * 创建联邦任务
   */
  createTaskApi: (taskData: any) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.post,
      url: '/v1/learn_management/federated_task/',
      data: taskData
    })
  },

  /**
   * 获取任务列表
   */
  getTasksApi: (params = {}) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.get,
      url: '/v1/learn_management/federated_task/',
      params // 添加 params 参数
    })
  },

  /**
   * 更新任务信息
   */
  updateTaskApi: (taskData: any) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.put,
      url: `/v1/learn_management/federated_task/`,
      data: taskData
    })
  },

  /**
   * 删除任务
   */
  deleteTaskApi: (taskData: any) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.del,
      url: `/v1/learn_management/federated_task/`,
      data: taskData
    })
  },
  /**
   * 开始任务
   */
  startTaskApi: (taskData: any) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.put,
      url: `/v1/learn_management/federated_task/start`,
      data: taskData
    })
  },

  /**
   * 暂停任务
   */
  pauseTaskApi: (taskData: any) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.put,
      url: `/v1/learn_management/federated_task/pause`,
      data: taskData
    })
  },

  /**
   * 恢复任务
   */
  resumeTaskApi: (taskData: any) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.put,
      url: `/v1/learn_management/federated_task/resume`,
      data: taskData
    })
  }
}

// 为了保持原有引用方式不变，继续导出单个 API
export const {
  createTaskApi,
  getTasksApi,
  updateTaskApi,
  pauseTaskApi,
  resumeTaskApi
} = federatedTaskModel
