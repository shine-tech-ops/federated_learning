import { ref, onMounted, onUnmounted } from 'vue'

interface IdleTimerOptions {
  timeout: number // 指定的超时时间，单位为毫秒
  onIdle: () => void // 在指定的超时时间内没有操作时执行的回调函数
}

export default function useIdleTimer(options: IdleTimerOptions) {
  const { timeout, onIdle } = options
  let timer: ReturnType<typeof setTimeout> | null = null
  const lastActiveTime = ref(Date.now())

  // 重置计时器
  function resetTimer() {
    if (timer) clearTimeout(timer)
    timer = setTimeout(() => {
      onIdle()
    }, timeout)
  }

  // 监听用户操作
  function handleActivity() {
    lastActiveTime.value = Date.now()
    resetTimer()
  }

  // 监听页面加载完成和卸载事件
  onMounted(() => {
    window.addEventListener('mousemove', handleActivity)
    window.addEventListener('mousedown', handleActivity)
    window.addEventListener('keypress', handleActivity)
    resetTimer()
  })

  onUnmounted(() => {
    window.removeEventListener('mousemove', handleActivity)
    window.removeEventListener('mousedown', handleActivity)
    window.removeEventListener('keypress', handleActivity)
    if (timer) clearTimeout(timer)
  })

  return {
    lastActiveTime
  }
}
