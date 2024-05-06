import systemList from './_system'
import userList from './_user'
export default [...systemList, ...userList]

// import { type MockMethod } from 'vite-plugin-mock'
// // 获取所有 mock 文件
// const modules: Record<string, any> = import.meta.glob('./_*.ts', { eager: true })
// let mockList: MockMethod[] = []
// for (const path in modules) {
//   mockList = mockList.concat(modules[path].default)
// }

// export default mockList
