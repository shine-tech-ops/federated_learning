export type LanguageType = 'zh' | 'en' | null

/* GlobalState */
export interface GlobalState {
  language: LanguageType
  isDark: boolean
  isCollapse: boolean
  tagsViewIcon: boolean
  site: 'main' | 'sub'
  sysLogo: string
  sysTitle: string
  sysTitleSuffix: string
  defaultLogo: string
  sysLoaded: boolean
  siteBrandId: number
  sysTheme: string
  isFirstLogin: boolean
  siteUUID: string
}

/* tagsListProps */
export interface tagsListProps {
  icon: string
  title: string
  path: string
  name: string
  close: boolean
  isKeepAlive: boolean
}

/* TagViewState */
export interface TagsViewState {
  tagsList: tagsListProps[]
}

/* KeepAliveState */
export interface KeepAliveState {
  keepAliveName: string[]
}

export interface UserState {
  id: number
  role: any[]
  name: string
  mobile: string
  email: string
  is_active: boolean
  is_superuser: boolean
  is_admin: boolean
}

export interface UserStates {
  users: UserState
  permissions: string[]
  isSuperAdmin: boolean
  isFirstLogin: boolean
}

export interface Permission {
  id: number
  name_en: string
  name_zh: string
  parent?: number
  children?: Permission[]
}

export interface PermissionStates {
  oriTreeData: Permission[]
  dict: { [key in number]: any }
  treeData: Permission[]
}

/** 选项卡筛选 - 三级菜单 */
export interface TabListProps {
  name: string
  title: string
  value?: any
}
export interface TabMenuStates {
  tabList: TabListProps[]
  curTab: any
  // 是否显示组件
  show: boolean
  tabTitle: string
}

/**
 * 字典相关
 */

export interface DictStates {}

/**
 * 面包屑
 */
export interface BreadcrumbStates {
  pathTitle: { [path: string]: string }
}
