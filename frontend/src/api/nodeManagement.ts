import service from '@/utils/request'
import { CONFIG } from './system'

const method = {
  del: 'DELETE',
  put: 'PUT',
  post: 'POST',
  get: 'GET'
}

export const regionNodeApi = {
  /**
   * 获取区域节点列表
   */
  fetchRegionNodes: (params = {}) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.get,
      url: '/v1/learn_management/region_nodes/',
      params
    })
  },

  /**
   * 创建区域节点
   */
  createRegionNode: (data: any) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.post,
      url: '/v1/learn_management/region_nodes/',
      data
    })
  },

  /**
   * 更新区域节点
   */
  updateRegionNode: (data: any) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.put,
      url: '/v1/learn_management/region_nodes/',
      data
    })
  },

  /**
   * 删除区域节点
   */
  deleteRegionNode: (id: number) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.del,
      url: '/v1/learn_management/region_nodes/',
      data: { id }
    })
  }
}

export const edgeNodeApi = {
  /**
   * 获取边缘节点列表
   */
  fetchEdgeNodes: (params = {}) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.get,
      url: '/v1/learn_management/edge_nodes/',
      params
    })
  },

  /**
   * 创建边缘节点
   */
  createEdgeNode: (data: any) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.post,
      url: '/v1/learn_management/edge_nodes/',
      data
    })
  },

  /**
   * 更新边缘节点
   */
  updateEdgeNode: (data: any) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.put,
      url: '/v1/learn_management/edge_nodes/',
      data
    })
  },

  /**
   * 删除边缘节点
   */
  deleteEdgeNode: (id: number) => {
    return service({
      developing: CONFIG.developing,
      mock: CONFIG.mock,
      method: method.del,
      url: '/v1/learn_management/edge_nodes/',
      data: { id }
    })
  }
}

export default {
  regionNodeApi,
  edgeNodeApi
}