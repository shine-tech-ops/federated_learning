import { genResponse, getPageResponse } from './utils'

export default [
  {
    url: '/mock/v1/common_config/system_log/',
    method: 'GET',
    statusCode: 200,
    response: (opt: any) => {
      const res = []
      let now = new Date().getTime() - 1000 * 60
      for (let i = 0; i < 100; i++) {
        now += 1000 * 60 * 60 * 1
        res.push({
          operation_time: new Date(now).toLocaleString(),
          user_name: 'user_' + i,
          role_name: 'role_' + i,
          content: 'content_' + i
        })
      }
      return getPageResponse(res, opt)
    }
  }
]
