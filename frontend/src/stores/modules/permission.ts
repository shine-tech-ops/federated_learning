import { defineStore } from 'pinia'
import type { PermissionStates, Permission } from '../interface'
import { authModel } from '@/api'
import { cloneDeep } from 'lodash'

/**
 * 权限列表
 */
export const usePermissionStore = defineStore('app-permissions', {
  state: (): PermissionStates => ({
    dict: {},
    oriTreeData: [],
    treeData: []
  }),
  actions: {
    async initPermissions() {
      const data = await authModel.permissions()
      this.oriTreeData = cloneDeep(data)
      this.processData(data)
      this.treeData = data
      this.genDict(this.treeData)
    },
    // 处理权限
    processData(data: Permission[]) {
      data.map((item) => {
        if (item.children && item.children.length > 0) {
          item.children.map((child) => {
            // 补全 查看 / 编辑权限名称
            if (child.name_en.startsWith('view_') || child.name_en.startsWith('edit_')) {
              child.name_zh = `${child.name_zh} - ${item.name_zh}`
            }
          })
          this.processData(item.children)
        }
      })
    },
    // 把树形权限拍平，生成以 id 为 key的字典
    genDict(data: Permission[]) {
      data.map((item) => {
        this.dict[item.id] = item.name_zh
        if (item.children) {
          this.genDict(item.children)
        }
      })
    },
    // 根据id 获取权限名称
    getPermissionNameById(id: any) {
      return this.dict[id]
    }
  },
  persist: true
})
