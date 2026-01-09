export const genResponse = (data: any, msg = '成功') => {
  return {
    code: 200,
    data,
    msg
  }
}

export function getPageData(array: any[], page: number, page_size: number) {
  // 确保页码和每页大小是有效的数字
  if (typeof page !== 'number' || isNaN(page) || page < 1) {
    page = 1
  }
  if (typeof page_size !== 'number' || isNaN(page_size) || page_size < 1) {
    page_size = 10 // 默认每页大小
  }

  // 计算起始索引（注意：JavaScript的数组索引从0开始）
  const startIndex = (page - 1) * page_size

  // 确保起始索引不超过数组长度
  if (startIndex >= array.length) {
    // 如果没有足够的数据，返回空数组或根据需要处理
    return []
  }

  // 计算结束索引（注意：不包括endIndex，所以加1）
  const endIndex = Math.min(startIndex + page_size, array.length)

  // 使用slice方法获取数据
  return array.slice(startIndex, endIndex)
}

export function getPageResponse(array: any[], opt: any) {
  const { page, page_size } = opt.query
  if (page_size) {
    let _page = Number(page)
    let _page_size = Number(page_size)
    const list = getPageData(array, _page, _page_size)
    return genResponse({
      list,
      total: array.length,
      page: _page
    })
  } else {
    return genResponse(array)
  }
}
