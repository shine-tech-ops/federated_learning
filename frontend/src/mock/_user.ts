import { genResponse, getPageResponse } from './utils'

export default [
  {
    url: '/mock/v1/account/login/',
    method: 'POST',
    statusCode: 200,
    response: (opt: any) => {
      // return {
      //   code: 400,
      //   msg: '测试',
      //   data: []
      // }
      return {
        refresh:
          'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTcwNjM0NDA5OCwiaWF0IjoxNzA2MjU3Njk4LCJqdGkiOiIwNGJiYzczYWQ0OTc0NGZiYTIyMjg3NTIwNTg1OTM1YyIsInVzZXJfaWQiOjR9.U-ZQ_k3aCjFSoXnIrqt-DxJf0-ukEeJP4KJIXiDAXAE',
        access:
          'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzA2MjU5NDk4LCJpYXQiOjE3MDYyNTc2OTgsImp0aSI6ImFiNTI4ZjFjMjExMzQ0ZTk5OTExY2ExMWYwOTY3MTM1IiwidXNlcl9pZCI6NH0.t8EwFgQ_NdPZDsvWYBAhFeHeRCWCzuppXFX1yKVmBWU'
      }
    }
  },
  {
    url: '/mock/auth/logout',
    method: 'POST',
    statusCode: 200,
    response: () => {
      return genResponse({})
    }
  },
  {
    url: '/mock/v1/account/current_user/',
    method: 'get',
    statusCode: 200,
    response: (opt: any) => {
      return genResponse([
        {
          id: 3,
          role: [
            {
              id: 2,
              name: 'admin'
            }
          ],
          name: 'admin',
          mobile: '123456',
          email: 'f@163.com',
          is_active: true,
          is_superuser: true,
          is_admin: false
        }
        // {
        //   id: 3,
        //   role: [
        //     {
        //       id: 2,
        //       name: '本地测试'
        //     }
        //   ],
        //   name: 'bbb',
        //   mobile: '1124123',
        //   email: 'tes1t@123.com',
        //   is_active: true,
        //   is_superuser: true,
        //   is_admin: false
        // }
      ])
    }
  },
  {
    url: '/mock/v1/account/permission/',
    method: 'get',
    statusCode: 200,
    response: (opt: any) => {
      return genResponse([
        {
          id: 0,
          name_en: 'all',
          name_zh: '全部',
          children: [
            {
              id: 1,
              name_en: 'home',
              name_zh: '首页',
              parent: 0,
              children: []
            },
            {
              id: 2,
              name_en: 'device_management',
              name_zh: '设备管理',
              parent: 0,
              children: [
                {
                  id: 3,
                  name_en: 'basic_device_monitoring',
                  name_zh: '动力设备',
                  parent: 2,
                  children: [
                    {
                      id: 4,
                      name_en: 'view_basic_device_monitoring',
                      name_zh: '查看',
                      parent: 3
                    },
                    {
                      id: 5,
                      name_en: 'edit_basic_device_monitoring',
                      name_zh: '编辑',
                      parent: 3
                    }
                  ]
                },
                {
                  id: 6,
                  name_en: 'environmental_device_monitoring',
                  name_zh: '环境设备',
                  parent: 2,
                  children: [
                    {
                      id: 7,
                      name_en: 'view_environmental_device_monitoring',
                      name_zh: '查看',
                      parent: 6
                    },
                    {
                      id: 8,
                      name_en: 'edit_environmental_device_monitoring',
                      name_zh: '编辑',
                      parent: 6
                    }
                  ]
                },
                {
                  id: 9,
                  name_en: 'non_monitoring_device',
                  name_zh: '非监控设备',
                  parent: 2,
                  children: [
                    {
                      id: 10,
                      name_en: 'view_non_monitoring_device',
                      name_zh: '查看',
                      parent: 9
                    },
                    {
                      id: 11,
                      name_en: 'edit_non_monitoring_device',
                      name_zh: '编辑',
                      parent: 9
                    }
                  ]
                },
                {
                  id: 12,
                  name_en: 'protocol_converter_setting',
                  name_zh: '收敛主机设置',
                  parent: 2,
                  children: [
                    {
                      id: 13,
                      name_en: 'view_protocol_converter_setting',
                      name_zh: '查看',
                      parent: 12
                    },
                    {
                      id: 14,
                      name_en: 'edit_protocol_converter_setting',
                      name_zh: '编辑',
                      parent: 12
                    }
                  ]
                },
                {
                  id: 63,
                  name_en: 'ipvs_setting',
                  name_zh: 'IPVS',
                  parent: 2,
                  children: [
                    {
                      id: 64,
                      name_en: 'view_ipvs_setting',
                      name_zh: '查看',
                      parent: 63
                    },
                    {
                      id: 65,
                      name_en: 'edit_ipvs_setting',
                      name_zh: '编辑',
                      parent: 63
                    }
                  ]
                }
              ]
            }
          ]
        }
      ])
    }
  },
  {
    url: '/mock/tmp/update',
    method: 'POST',
    statusCode: 200,
    timeout: 1200,
    response: () => {
      return genResponse({})
    }
  },
  {
    url: '/mock/v1/account/user/',
    method: 'GET',
    statusCode: 200,
    response: (opt: any) => {
      const res = []
      for (let i = 0; i < 23; i++) {
        res.push({
          email: 'email_' + i,
          id: i,
          is_active: Math.random() >= 0.5,
          is_admin: Math.random() >= 0.5,
          is_superuser: i === 0,
          mobile: 'mobile_' + i,
          name: 'name_' + i,
          role: [
            {
              id: 1,
              permission: [
                {
                  id: 1,
                  name_en: 'overview_screen',
                  name_zh: '首页'
                },
                {
                  id: 2,
                  name_en: 'device_management',
                  name_zh: '设备管理'
                }
              ],
              name: 'role_name_' + i
            }
          ]
        })
      }
      return getPageResponse(res, opt)
    }
  },
  {
    url: '/mock/v1/account/user_role/',
    method: 'GET',
    statusCode: 200,
    response: (opt: any) => {
      const permission = [
        {
          id: 1,
          name_en: 'overview_screen',
          name_zh: '总览大屏'
        },
        {
          id: 2,
          name_en: 'device_management',
          name_zh: '设备管理'
        },
        {
          id: 3,
          name_en: 'basic_device_monitoring',
          name_zh: '动力设备'
        },
        {
          id: 4,
          name_en: 'view_basic_device_monitoring',
          name_zh: '查看'
        },
        {
          id: 5,
          name_en: 'edit_basic_device_monitoring',
          name_zh: '编辑'
        }
      ]

      const res = []
      for (let i = 0; i < 6; i++) {
        res.push({
          id: i,
          name: 'name_' + i,
          permission
        })
      }

      return getPageResponse(res, opt)
    }
  }
]
