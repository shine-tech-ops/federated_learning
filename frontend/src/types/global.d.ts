declare namespace Menu {
  interface MenuOptions {
    path: string
    name: string
    component?: string | (() => Promise<unknown>)
    redirect?: string
    meta: MetaProps
    children?: MenuOptions[]
  }
  interface MetaProps {
    icon: string
    // 是否为自定义icon，自定义icon在 assets/icon 中以svg文件形式存在
    customIcon?: boolean
    title: string
    activeMenu?: string
    isLink?: string
    isHide: boolean
    isFull: boolean
    isAffix: boolean
    isKeepAlive: boolean
  }
}

type ObjToKeyValArray<T> = {
  [K in keyof T]: [K, T[K]]
}[keyof T]

// 路由列表
declare interface RoutesListState<T = any> {
  routesList: T[]
  homeUrl: string
}

/**
 * 弹窗组件
 */
declare type DialogType = 'edit' | 'add' | 'view'
declare interface DialogTitleDictType {
  view: string
  edit: string
  add: string
}

/**
 * 表格组件
 */
declare interface TableButtonType {
  type: string
  label: string
  confirm?: boolean
  auth?: string[]
  disabled?: (row: any) => boolean
  handle: (row: any) => void | Promise<any>
  iif?: (row: any) => boolean
}

declare interface TableColumnType {
  format?: (row: any) => any
  disabled?: (row: any) => boolean
  // custom switch pic tag text
  type?: string
  change?: (row: any, val: boolean) => any
  buttons?: TableButtonType[]
  prop: string
  label?: string
  auth?: string[]
  width?: string
  iif?: () => boolean
  enum?: ElxTable.Enum[]
  cellClick?: any
  render?: (scope: RenderScope<T>) => VNode | string // 自定义单元格内容渲染（tsx语法）
}

declare interface TableDataType {
  [x: string]: any
  [key in string]: any
}

declare interface AnyObj {
  [x: string]: any
  [key in string]: any
}

declare type ElementTextType = 'success' | 'warning' | 'info' | 'primary' | 'danger'

/**
 * 表单组件
 */
declare namespace Form {
  // 表单模式
  type Mode = 'search' | 'default'
  type ColType = 'select' | 'input' | 'switch'
  type EnumObj = {
    label: string
    value: any
    type?: ElementTextType
    id?: number
  }
  interface SelectOption {
    [key in string]: any
  }
  ;[]

  type Data = {
    [key in string]: any
  }

  interface Column {
    prop: string
    label: string
    type?: FormColType
    placeholder?: string
    disabled?: (row: any) => boolean
    optConfig?: {
      key: string
      label: string
      value: string
    }
    options?: (() => Promise<any>) | EnumObj[]
    bind?: AnyObj
    iif?: (row: any, data: any) => boolean
    change?: (row: any) => void
    // el-form-item 的宽度
    width?: string
  }
}

declare namespace ElxTable {
  interface Change {
    type: 'ps' | 'pi'
    ps?: number
    pi?: number
  }

  interface Enum {
    label: string
    value: any
    type?: ElementTextType
  }
}

declare namespace RATE {
  interface Color {
    color: string
    percentage: number
  }
}
