<template>
  <div class="flex items-center justify-center overflow-hidden whitespace-nowrap">
    <Logo></Logo>
    <Collapse id="collapseIcon" />
    <Breadcrumb></Breadcrumb>
  </div>
  <div class="flex items-center justify-center whitespace-nowrap">
    <!-- <Notify class="mr-6"></Notify> -->
    <Language class="mr-6"></Language>
    <Avatar></Avatar>
  </div>
</template>

<script setup lang="ts">
import Collapse from './Collapse.vue'
import Breadcrumb from './Breadcrumb.vue'
import Avatar from './Avatar.vue'
import Notify from './Notify.vue'
import Language from './Language.vue'
import Logo from './Logo.vue'
</script>
