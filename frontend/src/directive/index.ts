import type { App, Directive } from 'vue'
import { auth, authIs } from './modules/auth'
import draggable from './modules/draggable'

const directivesList: { [key: string]: Directive } = {
  auth,
  authIs,
  draggable
}

const directives = {
  install: function (app: App<Element>) {
    Object.keys(directivesList).forEach((key) => {
      app.directive(key, directivesList[key])
    })
  }
}

export default directives
