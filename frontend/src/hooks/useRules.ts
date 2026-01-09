import { type FormRules } from 'element-plus'
import pinia from '@/stores'
import { useGlobalStore } from '@/stores/modules/global'
const globalStore = useGlobalStore(pinia)
export default function useRules(ruleFun: Function) {
  const rules = ref<FormRules>({})
  const initRules = () => {
    rules.value = ruleFun()
  }

  watch(
    () => globalStore.language,
    () => {
      initRules()
    },
    {
      immediate: true
    }
  )

  return rules
}
