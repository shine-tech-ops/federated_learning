<template>
  <div class="flex w-full items-center justify-between p-2 px-3" :class="styleClass">
    <div class="flex flex-row gap-4 w-full items-center flex-1 whitespace-nowrap">
      <slot name="title" :titleClass="titleClass">
        <div :class="titleClass">
          {{ props.title }}
        </div>
      </slot>
    </div>
    <slot></slot>
  </div>
</template>
<script setup lang="ts">
const props = defineProps({
  title: {
    type: String,
    default: ''
  },
  theme: {
    type: String,
    // modal, page
    default: 'page'
  }
})

const styleClass = computed(() => {
  if (props.theme === 'modal') {
    return 'bg-[#F7F8F9] modal-border'
  }
  return ''
})

const titleClass = 'text-[16px] leading-[22px] text-gray-700 font-semibold'
</script>
