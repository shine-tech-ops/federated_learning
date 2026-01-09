<template>
  <svg aria-hidden="true" class="svg-icon inline-block" :width="props.size" :height="props.size">
    <use
      :xlink:href="symbolId"
      :fill="props.color"
      :stroke="props.stroke"
      :stroke-width="props.strokeWidth"
    />
  </svg>
</template>

<script setup lang="ts">
import { computed } from 'vue'
const props = defineProps({
  prefix: {
    type: String,
    default: 'icon'
  },
  name: {
    type: String,
    required: true
  },
  color: {
    type: String,
    default: 'currentColor'
  },
  size: {
    type: String,
    default: '1em'
  },
  stroke: {
    type: String,
    default: ''
  },
  strokeWidth: {
    type: Number,
    default: 2
  }
})

const symbolId = computed(() => `#${props.prefix}-${props.name}`)
</script>

<style lang="scss">
.svg-icon {
  font-size: 1.5rem;
}
</style>
