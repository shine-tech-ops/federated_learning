# tableColumns

- 说明: 表格配置

```json
{
  // 转换显示内容 row 当前行的数据
  format?: (row: any) => any
  // 是否禁用 有行内编辑操作的时候生效
  disabled?: (row: any) => boolean
  /*
    目前支持
    switch
  */
  type?: string
  // 配合 type=switch使用 val: 切换的值
  change?: (row: any, val: boolean) => any
  // 按钮配置项，当 prop = operation时生效
  buttons?: TableButtonType[]
  // 字段
  prop: string
  // 表头名
  label?: string
  // 权限，若无编辑权限 disabled = true
  auth?: string[]
  // 表格宽度
  width?: string

}
```
