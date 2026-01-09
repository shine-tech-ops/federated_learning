FROM node:18-alpine AS builder


# 安装 pnpm
RUN npm install -g pnpm

WORKDIR /app

# pnpm
COPY frontend/package.json frontend/pnpm-lock.yaml ./

RUN pnpm install

COPY frontend/. .

RUN pnpm run build-only

FROM nginx:alpine
COPY dockers/default.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80
EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]
