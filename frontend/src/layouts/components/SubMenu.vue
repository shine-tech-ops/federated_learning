<template>
  <template v-for="subItem in menuList" :key="subItem.path">
    <el-sub-menu v-if="subItem.children?.length" :index="subItem.path">
      <template #title>
        <template v-if="subItem.meta.icon">
          <el-icon v-if="!subItem.meta.customIcon">
            <component :is="subItem.meta.icon"></component>
          </el-icon>
          <div v-else class="el-icon">
            <svg-icon :name="subItem.meta.icon"></svg-icon>
          </div>
        </template>
        <span class="ellipsis" :title="$t(subItem.meta.title)">{{ $t(subItem.meta.title) }}</span>
      </template>
      <SubMenu :menu-list="subItem.children" />
    </el-sub-menu>
    <el-menu-item v-else :index="subItem.path" @click="handleClickMenu(subItem)">
      <template v-if="subItem.meta.icon">
        <el-icon v-if="!subItem.meta.customIcon">
          <component :is="subItem.meta.icon"></component>
        </el-icon>
        <div v-else class="el-icon">
          <svg-icon :name="subItem.meta.icon"></svg-icon>
        </div>
      </template>
      <template #title>
        <span class="ellipsis" :title="$t(subItem.meta.title)">{{ $t(subItem.meta.title) }}</span>
      </template>
    </el-menu-item>
  </template>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router'
defineProps<{ menuList: Menu.MenuOptions[] }>()

const router = useRouter()
const handleClickMenu = (subItem: Menu.MenuOptions) => {
  if (subItem.meta.isLink) return window.open(subItem.meta.isLink, '_blank')
  router.push(subItem.path)
}
</script>

<style lang="scss">
.el-sub-menu .el-sub-menu__title:hover {
  color: var(--el-menu-hover-text-color) !important;
  background-color: transparent !important;
  border-radius: 8px;
}
.el-menu--collapse {
  .is-active {
    .el-sub-menu__title {
      // color: #ffffff !important;
      // background-color: var(--el-color-primary) !important;
      background-color: var(--el-menu-active-bg-color) !important;
      border-radius: 8px;
    }
  }
}
.el-menu-item {
  border-radius: 8px;
  &:hover {
    color: var(--el-menu-hover-text-color);
  }
  &.is-active {
    color: var(--el-menu-active-color) !important;
    background-color: var(--el-menu-active-bg-color) !important;
    // &::before {
    //   position: absolute;
    //   top: 0;
    //   bottom: 0;
    //   width: 4px;
    //   content: '';
    //   background-color: var(--el-color-primary);
    // }
  }
}
.el-menu-item {
  &.is-active {
    &::before {
      left: 0;
    }
  }
}
.columns {
  .el-menu-item {
    &.is-active {
      &::before {
        right: 0;
      }
    }
  }
}

.el-menu-item [class^='svg-icon'],
.el-sub-menu .svg-icon {
  font-size: 18px;
}

.el-menu-item,
.el-sub-menu {
  .el-icon {
    margin-right: 10px;
  }
}

.el-sub-menu .el-menu-item {
  margin: 10px 0;
}

.el-menu-item,
.el-sub-menu {
  margin: 10px 0;
}
</style>
