/**
 * v-auth
 * 元素权限指令
 */
import { useUserStore } from '@/stores/modules/user'
import type { Directive, DirectiveBinding } from 'vue'
import pinia from '@/stores'

/**
 * v-auth=['user-add', 'user-view']
 * v-auth='user-add'
 * 多个权限满足一个就显示
 */
export const auth: Directive = {
  mounted(el: HTMLElement, binding: DirectiveBinding) {
    const { value, arg } = binding
    let values = Array.isArray(value) ? value : [value]
    if (arg === 'if') {
      values = values.filter((v: any) => !!v)
      if (values.length === 0) {
        return
      }
    }
    const userStore = useUserStore(pinia)
    let hasPermission = userStore.hasPermission(values)
    if (!hasPermission) el.remove()
  }
}

/**
 * 判断是否为 超管或管理员
 * v-auth-is="['superuser', 'admin']" 只要是超管或管理员
 * v-auth-is.all="['superuser', 'admin']" 必须是超管并且也是管理员
 */
export const authIs: Directive = {
  mounted(el: HTMLElement, binding: DirectiveBinding) {
    const { value, arg } = binding
    if (arg === 'all') {
    }
    const condition = arg === 'all' ? 'all' : 'only'
    const userStore = useUserStore(pinia)
    let hasPermission = userStore.authIs(value, condition)
    if (!hasPermission) el.remove()
  }
}
