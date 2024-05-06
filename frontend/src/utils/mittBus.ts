import mitt from 'mitt'

const mittBus = mitt()

/**
 * device-oid-completed 设备oid获取完成
 * force-change-password 强制修改密码
 * alarm-type-completed 告警类型获取完成
 * main-scroll-to 主区域滚动事件
 */
export default mittBus
