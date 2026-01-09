import service from '@/utils/request'
import { CONFIG } from './system'
const mock = CONFIG.mock
const developing = CONFIG.developing
const method = {
  del: 'DELETE',
  put: 'PUT',
  post: 'POST',
  get: 'GET'
}

export const authModel = {
  login: (data: any) => {
    return service({
      developing,
      mock,
      method: method.post,
      url: `/v1/account/login/`,
      data,
      // 特殊接口不进行格式校验 默认true 会严格按照 { msg, data, code } 校验
      resCheck: false
    })
  },
  logout: () => {
    return service({
      developing,
      mock,
      method: method.post,
      url: `/auth/logout`
    })
  },
  // 获取当前登录用户信息
  getCurrentUser: () => {
    return service({
      developing,
      // mock: 'local',
      mock,
      method: method.get,
      url: `/v1/account/current_user/`,
      mockID: '145211388',
      resCheck: false
    })
  },
  // 权限列表
  permissions: () => {
    return service({
      developing,
      mock,
      method: method.get,
      url: `/v1/account/permission/`
    })
  },
  /** 用户管理 **/
  getUserList: () => {
    return service({
      developing,
      mock,
      method: method.get,
      url: `/v1/account/user/`,
      mockID: '144770555'
    })
  },
  addUser: (data: any) => {
    return service({
      developing,
      mock,
      method: method.post,
      url: `/v1/account/user/`,
      data,
      mockID: '144770824'
    })
  },
  updateUser: (data: any) => {
    return service({
      developing,
      mock,
      method: method.put,
      url: `/v1/account/user/`,
      data,
      mockID: '144770837'
    })
  },
  delUser: (data: any) => {
    return service({
      developing,
      mock,
      method: method.del,
      url: `/v1/account/user/`,
      data,
      mockID: '144770839'
    })
  },
  tmpUpdate: (data?: any) => {
    return service({
      developing,
      mock: 'local',
      method: method.post,
      url: `/tmp/update`,
      data
    })
  },
  /** 角色管理 **/
  // 新增角色和权限
  addRole: (data: any) => {
    return service({
      developing,
      mock,
      method: method.post,
      url: `/v1/account/user_role/`,
      mockID: '144629787',
      data
    })
  },
  // 更新角色和权限
  updateRole: (data: any) => {
    return service({
      developing,
      mock,
      method: method.put,
      url: `/v1/account/user_role/`,
      mockID: '144630078',
      data
    })
  },
  // 删除角色和权限
  delRole: (data: any) => {
    return service({
      developing,
      mock,
      method: method.del,
      url: `/v1/account/user_role/`,
      mockID: '144630206',
      data
    })
  },
  // 获取角色列表
  getRoleList: () => {
    return service({
      developing,
      mock,
      method: method.get,
      url: `/v1/account/user_role/`,
      mockID: '144628078'
    })
  }
}
