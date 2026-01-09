/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg_color: "var(--el-bg-color)",
        primary: "var(--el-color-primary)",
        text_color_primary: "var(--el-text-color-primary)",
        text_color_regular: "var(--el-text-color-regular)"
      },
      height: {
        // 主区域高度，用法 h-main
        main: 'calc(100vh - 110px)',
      },
      minHeight: {
        main: 'calc(100vh - 110px)',
        mainSub: 'calc(100vh - 110px - 40px)'
      }
    },
  },
  plugins: [],
}