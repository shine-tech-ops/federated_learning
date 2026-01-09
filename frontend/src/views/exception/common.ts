import { useRoutesList } from '@/stores/modules/routesList'
import pinia from '@/stores'
import { computed } from 'vue'
import router from '@/router'
const storesRoutesList = useRoutesList(pinia)
const homeUrl = computed(() => storesRoutesList.homeUrl)

export function goHome() {
  router.push(homeUrl.value)
}
