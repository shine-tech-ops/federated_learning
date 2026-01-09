/**
 * 时间日期转换
 * @param date 当前时间，new Date() 格式
 * @param format 需要转换的时间格式字符串
 * @description format 字符串随意，如 `YYYY-mm、YYYY-mm-DD`
 * @description format 季度："YYYY-MM-DD HH:mm:ss QQQQ"
 * @description format 星期："YYYY-MM-DD HH:mm:ss WWW"
 * @description format 几周："YYYY-MM-DD HH:mm:ss ZZZ"
 * @description format 季度 + 星期 + 几周："YYYY-MM-DD HH:mm:ss WWW QQQQ ZZZ"
 * @returns 返回拼接后的时间字符串
 */
export function formatDate(_date: Date | number, format: string): string {
  if (!_date) {
    return ''
  }
  // const date = typeof _date === 'number' ? new Date(_date) : _date
  const date = new Date(_date)
  let we = date.getDay() // 星期
  let z = getWeek(date) // 周
  let qut = Math.floor((date.getMonth() + 3) / 3).toString() // 季度
  const opt: { [key: string]: string } = {
    'Y+': date.getFullYear().toString(), // 年
    'M+': (date.getMonth() + 1).toString(), // 月(月份从0开始，要+1)
    'D+': date.getDate().toString(), // 日
    'H+': date.getHours().toString(), // 时
    'm+': date.getMinutes().toString(), // 分
    's+': date.getSeconds().toString(), // 秒
    'q+': qut // 季度
  }
  // 中文数字 (星期)
  const week: { [key: string]: string } = {
    '0': '日',
    '1': '一',
    '2': '二',
    '3': '三',
    '4': '四',
    '5': '五',
    '6': '六'
  }
  // 中文数字（季度）
  const quarter: { [key: string]: string } = {
    '1': '一',
    '2': '二',
    '3': '三',
    '4': '四'
  }
  if (/(W+)/.test(format))
    format = format.replace(
      RegExp.$1,
      RegExp.$1.length > 1 ? (RegExp.$1.length > 2 ? '星期' + week[we] : '周' + week[we]) : week[we]
    )
  if (/(Q+)/.test(format))
    format = format.replace(
      RegExp.$1,
      RegExp.$1.length == 4 ? '第' + quarter[qut] + '季度' : quarter[qut]
    )
  if (/(Z+)/.test(format))
    format = format.replace(RegExp.$1, RegExp.$1.length == 3 ? '第' + z + '周' : z + '')
  for (let k in opt) {
    let r = new RegExp('(' + k + ')').exec(format)
    // 若输入的长度不为1，则前面补零
    if (r)
      format = format.replace(
        r[1],
        RegExp.$1.length == 1 ? opt[k] : opt[k].padStart(RegExp.$1.length, '0')
      )
  }
  return format
}

/**
 * 获取当前日期是第几周
 * @param dateTime 当前传入的日期值
 * @returns 返回第几周数字值
 */
export function getWeek(dateTime: Date): number {
  let temptTime = new Date(dateTime.getTime())
  // 周几
  let weekday = temptTime.getDay() || 7
  // 周1+5天=周六
  temptTime.setDate(temptTime.getDate() - weekday + 1 + 5)
  let firstDay = new Date(temptTime.getFullYear(), 0, 1)
  let dayOfWeek = firstDay.getDay()
  let spendDay = 1
  if (dayOfWeek != 0) spendDay = 7 - dayOfWeek + 1
  firstDay = new Date(temptTime.getFullYear(), 0, 1 + spendDay)
  let d = Math.ceil((temptTime.valueOf() - firstDay.valueOf()) / 86400000)
  let result = Math.ceil(d / 7)
  return result
}

export function completeDate(dateString: string | number | Date, isEndOfDay = false) {
  // 解析日期字符串
  const date = new Date(dateString)

  // 补全月份的日期
  const year = date.getFullYear()
  const month = date.getMonth() + 1
  const day = isEndOfDay ? new Date(year, month, 0).getDate() : 1

  // 补全小时、分钟和秒
  const hours = isEndOfDay ? 23 : 0
  const minutes = isEndOfDay ? 59 : 0
  const seconds = isEndOfDay ? 59 : 0

  // 构建新的日期对象
  const completedDate = new Date(year, month - 1, day, hours, minutes, seconds)

  return completedDate
}

// 自动补全日期范围
export function completeDateArr(
  dateArr: (string | number | Date)[],
  format = 'YYYY-MM-DD HH:mm:ss'
) {
  const startDate = completeDate(dateArr[0])
  const endDate = completeDate(dateArr[1], true)
  const start = formatDate(startDate, format)
  const end = formatDate(endDate, format)
  return [start, end]
}
