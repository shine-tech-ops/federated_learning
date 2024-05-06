import { type RouteRecordRaw } from 'vue-router'
import { dynamicRoutes } from './dynamicRoutes'
/**
 * 路由meta对象参数说明
 * meta: {
 *      title:          菜单栏及 tagsView 栏、菜单搜索名称（国际化）
 *      isLink：        是否超链接菜单，开启外链条件，`1、isLink: 链接地址不为空`
 *      isHide：        是否隐藏此路由
 *      isKeepAlive：   是否缓存组件状态
 *      isAffix：       是否固定在 tagsView 栏上
 *      roles：         当前路由权限标识，取角色管理。控制路由显示、隐藏。超级管理员：admin 普通角色：common
 *      icon：          菜单、tagsView 图标，阿里：加 `iconfont xxx`，fontawesome：加 `fa xxx`
 * }
 */

/**
 * 404、401界面
 */
export const notFoundAndNoPower = [
  {
    path: '/:path(.*)*',
    name: 'notFound',
    component: () => import('@/views/exception/404.vue'),
    meta: {
      title: '未找到页面',
      isHide: true
    }
  },
  {
    path: '/403',
    name: 'noPermission',
    component: () => import('@/views/exception/403.vue'),
    meta: {
      title: '没有访问权限',
      isHide: true
    }
  },
  {
    path: '/500',
    name: 'serverError',
    component: () => import('@/views/exception/500.vue'),
    meta: {
      title: '没有访问权限',
      isHide: true
    }
  }
]

/**
 * 定义静态路由（默认路由）
 * @returns 返回路由菜单数据
 */
export const staticRoutes: Array<RouteRecordRaw> = [
  {
    path: '/login',
    name: 'login',
    component: () => import('@/views/login/index.vue'),
    meta: {
      title: '登录'
    }
  },
  {
    path: '/',
    redirect: '',
    name: 'root'
  },
  {
    path: '/layout',
    name: 'layout',
    component: () => import('@/layouts/index.vue'),
    children: []
  }
]

export { dynamicRoutes }
