/*
	一个拖拽指令，可在父元素区域任意拖拽元素。
  支持被拖拽元素自身设置 margin 偏移

	使用：在 Dom 上加上 v-draggable 即可
	<div class="dialog-model" v-draggable></div>
  
  使用: v-draggable="dragEndCallback" 监听拖拽结束事件
  <div v-draggable="dragEndCallback({x:number,y:number})"></div>
*/
import type { Directive, ObjectDirective } from 'vue'
interface ElType extends HTMLElement {
  parentNode: any
}
interface DraggableBinding extends ObjectDirective {
  value?: (event: MouseEvent) => void
}
const draggable: Directive<ElType, DraggableBinding> = {
  mounted: function (el: ElType, binding: any) {
    el.style.cursor = 'move'
    el.style.position = 'absolute'
    el.onmousedown = function (e) {
      const marginLeft = parseFloat(window.getComputedStyle(el).marginLeft) || 0
      const marginTop = parseFloat(window.getComputedStyle(el).marginTop) || 0

      let disX = e.pageX - el.offsetLeft + marginLeft
      let disY = e.pageY - el.offsetTop + marginTop
      document.onmousemove = function (e) {
        let x = e.pageX - disX
        let y = e.pageY - disY
        let maxX = el.parentNode.offsetWidth - el.offsetWidth
        let maxY = el.parentNode.offsetHeight - el.offsetHeight
        if (x < 0) {
          x = 0
        } else if (x > maxX) {
          x = maxX
        }

        if (y < 0) {
          y = 0
        } else if (y > maxY) {
          y = maxY
        }
        el.style.left = x + 'px'
        el.style.top = y + 'px'
      }
      document.onmouseup = function () {
        document.onmousemove = document.onmouseup = null
        if (binding.value && typeof binding.value === 'function') {
          const x = parseFloat(el.style.left)
          const y = parseFloat(el.style.top)
          binding.value({ x, y }, e)
        }
      }
    }
  }
}
export default draggable
