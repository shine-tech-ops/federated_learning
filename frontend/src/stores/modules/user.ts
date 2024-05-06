import { defineStore } from 'pinia'
import { type UserStates, type UserState } from '../interface'
// import { Session } from '@/utils/storage'
import { authModel } from '@/api'
import { Session } from '@/utils/storage'
import router from '@/router'
import mittBus from '@/utils/mittBus'
import pinia from '@/stores'
import { useGlobalStore } from '@/stores/modules/global'

/**
 * 用户信息
 * @methods setUserInfos 设置用户信息
 */
export const useUserStore = defineStore('app-user', {
  state: (): UserStates => ({
    users: {
      id: 0,
      role: [],
      name: '',
      mobile: '',
      email: '',
      is_active: true,
      is_superuser: false,
      is_admin: false
    },
    // 当前用户所有权限
    permissions: [],
    isSuperAdmin: false,
    isFirstLogin: false
    // 必须记住用户密码，方便对比是否为首次登录
    // 为了安全性，加密保存用户登录的密码
    // LOGIN_MARK: ''
  }),
  actions: {
    updateUsers(userInfo: UserState) {
      this.users = { ...userInfo }
      this.isSuperAdmin = userInfo.is_superuser
      this.setPermissions(userInfo.role[0]?.permission)
      const globalStore = useGlobalStore(pinia)
      this.isFirstLogin = globalStore.isFirstLogin
      if (this.isFirstLogin) {
        setTimeout(() => {
          mittBus.emit('force-change-password')
        }, 1000)
      }
    },
    async setUsers() {
      const users: any = await this.getApiUserInfo()
      const curUser = users.data[0] as UserState
      this.updateUsers(curUser)
    },
    async getApiUserInfo() {
      return new Promise((resolve, reject) => {
        authModel
          .getCurrentUser()
          .then((res: any) => {
            console.log('🚀 ~ .then ~ res:', res)
            resolve(res)
          })
          .catch(() => {
            window.localStorage.clear()
            Session.clear()
            router.replace('/403')
          })
      })
      // return authModel.getCurrentUser()
    },
    // 设置权限
    setPermissions(arr: { id: number; name_en: string; name_zh: string }[]) {
      if (arr) {
        this.permissions = arr.map((item) => item.name_en)
      }
    },
    // 判断是否有某个权限
    // 如果传入多个权限，只要满足一个就返回true
    hasPermission(_auth?: string | string[]): boolean {
      if (this.isSuperAdmin) {
        return true
      }
      let auth = Array.isArray(_auth) ? _auth : [_auth]
      auth = auth.filter((v: any) => !!v)
      // 没有配置就当有权限
      if (auth.length === 0) {
        return true
      }
      let flag = false
      this.permissions.map((val: string) => {
        if (auth.includes(val)) {
          flag = true
        }
      })
      return flag
    },
    /**
     * 判断用户类型
     * @param _roles 可传字符串或数组，支持 ['!superuser'] 表示不是超管
     * @param condition 默认 only 只要有一个满足就返回 true，all表示必须全部满足才返回 true
     * @returns
     */
    authIs(_roles: string | string[], condition: 'only' | 'all' = 'only'): boolean {
      const roles: string[] = Array.isArray(_roles) ? _roles : [_roles]
      let flag = false
      let flagAll = true
      roles.forEach((_role) => {
        let role = _role
        let reverse = false
        if (_role.startsWith('!')) {
          role = role.slice(1)
          reverse = true
        }
        const key = `is_${role}`
        if (Object.prototype.hasOwnProperty.call(this.users, key)) {
          //@ts-ignore
          let check = this.users[key]
          if (reverse) {
            check = !check
          }
          if (check) {
            flag = true
          } else {
            flagAll = false
          }
        }
      })
      return condition === 'only' ? flag : flagAll
    },
    /** 退出登录 */
    async logout() {
      window.localStorage.clear()
      // 退出登录不需要接口
      // await authModel.logout()
      Session.clear()
      window.location.reload()
    }
  }
})
