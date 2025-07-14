import { type RouteRecordRaw } from 'vue-router'

/**
 * 定义动态路由
 * @returns 返回路由菜单数据
 */
export const dynamicRoutes: Array<RouteRecordRaw> = [
  {
    path: '/system/config',
    name: 'systemConfig',
    component: () => import('@/views/system/config/index.vue'),
    meta: {
      title: 'menu.systemConfig',
      icon: 'Setting',
      isKeepAlive: true
    }
  },
  {
    path: '/federated/tasks',
    name: 'federatedTasks',
    component: () => import('@/views/federated/task/index.vue'),
    meta: {
      title: 'menu.federatedTasks',
      icon: 'Document',
      isKeepAlive: true
    }
  },
  {
    path: '/model/management',
    name: 'modelManagement',
    component: () => import('@/views/model/management/index.vue'),
    meta: {
      title: 'menu.modelManagement',
      icon: 'Box',
      isKeepAlive: true
    }
  },
  {
    path: '/region/node/management',
    name: 'regionNodeManagement',
    component: () => import('@/views/region_node/management/index.vue'),
    meta: {
      title: 'menu.regionNodeManagement',
      icon: 'Monitor',
      isKeepAlive: true
    }
  },
  {
    path: '/edge/node/management',
    name: 'edgeNodeManagement',
    component: () => import('@/views/edge_node/management/index.vue'),
    meta: {
      title: 'menu.edgeNodeManagement',
      icon: 'Monitor',
      isKeepAlive: true
    }
  },
  {
    path: '/log/audit',
    name: 'logAudit',
    component: () => import('@/views/log/audit/index.vue'),
    meta: {
      title: 'menu.logAudit',
      icon: 'DocumentCopy',
      isKeepAlive: true
    }
  },
  {
    path: '/statistics/report',
    name: 'statisticsReport',
    component: () => import('@/views/statistics/report/index.vue'),
    meta: {
      title: 'menu.statisticsReport',
      icon: 'DataLine',
      isKeepAlive: true
    }
  },
    {
    path: '/account-permissions',
    name: 'accountPermissions',
    meta: {
      icon: 'menu-auth',
      customIcon: true,
      title: 'menu.accountPermissions',
      isLink: '',
      isHide: false,
      isFull: false,
      isKeepAlive: true
    },
    children: [
      {
        path: '/account-permissions/role-permissions',
        name: 'rolePermissions',
        component: () => import('@/views/user/role/index.vue'),
        meta: {
          title: 'menu.rolePermissions',
          isLink: '',
          isHide: false,
          isFull: false,
          isKeepAlive: true,
          auth: ['']
        }
      },
      {
        path: '/account-permissions/account-management',
        name: 'accountManagement',
        component: () => import('@/views/user/account/index.vue'),
        meta: {
          title: 'menu.accountManagement',
          isLink: '',
          isHide: false,
          isFull: false,
          isKeepAlive: true,
          auth: ['']
        }
      }
    ]
  }
]
