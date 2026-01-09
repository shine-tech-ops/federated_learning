// import { type DEVICE_TYPE } from '@/stores/interface/index'

// 分站点
export type STATION = {
  address: string
  city: string
  id: string
  ip: string
  name: string
  province: string
  status: string
  uuid: string
  contact: string
  contact_email: string
  contact_mobile: string
  ticket_count: number | null
  safe_uptime: number | null
  health_rate: number | null
  asset?: { [x: string]: string }
}

// 协议转换器
export type PROTOCOL_CONVERTOR = {
  auth_password: string
  auth_protocol: string
  community: string
  id: number
  ip: string
  name: string
  port: number
  priv_password: string
  priv_protocol: string
  security_level: number
  security_user: string
  snmp_engine_id: number
  station: STATION
  version: string
}

// 设备
export type DEVICE = {
  created_at: string
  deleted_at: string
  description: string
  device_altitude: string
  device_boot_time: string
  device_cleanliness_level: number
  device_cleanliness_update_time: string
  device_failure_rate: number | string
  device_health_rate: number | string
  device_manufacturer_name: string
  device_oid_position: string
  device_sn: string
  device_type: DEVICE_TYPE.Datum
  device_uptime: string
  device_warranty_expiration_time: string
  id: string
  name: string
  operator_by: string
  position: string
  protocol_convertor: PROTOCOL_CONVERTOR
  station: STATION
  status: string
  updated_at: string
  uuid: string
  bak_position?: string
}

// 设备健康度
export type DEVICE_HEALTH_RATE = {
  altitude: string
  clean_level: number
  clean_time: string
  device_type: string
  health_rate: string
  loads: any[]
  normal_life: string
  standard_voltage: string
  t: number
  used_time: string
  voltages: any[]
}

// 设备故障率
export type DEVICE_FAULT_RATE = {
  fault_rate: number
  total_fault_hours: number
  total_run_hours: number
}

// 告警
export type ALARM = {
  id: number
  created_at: string
  device_name: string
  description: string
  alarm_value: string
  alarm_oid: string
  threshold: number
  threshold_low: number
  threshold_high: number
  unit: string
  status: number
  alarm_rule_title: string
  alarm_type_name: string
  ticket: string
  device: DEVICE
}

// 系统设置
export type COMMON_CONFIG = {
  config_data: {
    host: string
    client_id: string
    password: string
    username: string
    port_ws: string
    port: string
    is_main_station: boolean
  }
  config_type: string
  id: number
}

// 站点信息
export type SITE_BRAND = {
  theme: string
  uuid: string
  logo: string
  name: string
  id: number
  asset?: { [x: string]: string }
}
