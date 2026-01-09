import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueJsx from '@vitejs/plugin-vue-jsx'
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'
import { createSvgIconsPlugin } from 'vite-plugin-svg-icons'
import vueSetupExtend from 'vite-plugin-vue-setup-extend'

import path from 'path'
import { resolve } from 'path'
import { viteMockServe } from 'vite-plugin-mock'

import tailwindcss from 'tailwindcss'
// import autoprefixer from 'autoprefixer'
export default ({ mode }: any) => {
  const VITE_API_HOST: string = loadEnv(mode, process.cwd()).VITE_APP_API_BASE_URL
  const VITE_API_MOCK: string = loadEnv(mode, process.cwd()).VITE_APP_API_MOCK_URL

  // https://vitejs.dev/config/
  return defineConfig({
    build: {
      outDir: 'dist'
    },
    plugins: [
      vue(),
      vueJsx(),
      // MOCK 服务
      viteMockServe({
        mockPath: './src/mock',
        logger: true
      }),
      createSvgIconsPlugin({
        // 指定需要缓存的图标文件夹
        iconDirs: [path.resolve(process.cwd(), 'src/assets/icons')],
        // 指定symbolId格式
        symbolId: 'icon-[dir]-[name]'

        /**
         * 自定义插入位置
         * @default: body-last
         */
        // inject?: 'body-last' | 'body-first'

        /**
         * custom dom id
         * @default: __svg__icons__dom__
         */
        // customDomId: '__svg__icons__dom__',
      }),
      vueSetupExtend(),
      AutoImport({
        dts: 'src/auto-import.d.ts',
        // 自动导入vue3的配置
        imports: ['vue', 'vue-router'],
        eslintrc: {
          enabled: true
        },
        resolvers: [ElementPlusResolver()]
        // resolvers: []
      }),
      Components({
        resolvers: [ElementPlusResolver()]
      })
    ],
    resolve: {
      alias: {
        // '@': fileURLToPath(new URL('./src', import.meta.url))
        '@': resolve(__dirname, './src')
      }
    },
    css: {
      postcss: {
        plugins: [tailwindcss /*, autoprefixer*/]
      }
    },
    server: {
      host: '0.0.0.0',
      proxy: {
        '/mock-online': {
          target: VITE_API_MOCK,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/mock-online/, '')
        },
        '/mock': {
          target: VITE_API_HOST,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/mock/, '')
        }
      }
    }
  })
}
