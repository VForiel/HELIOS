FROM node:18-alpine as build

WORKDIR /app

COPY web/frontend/package.json .
COPY web/frontend/vite.config.js .
# tailwind files
COPY web/frontend/postcss.config.js .
COPY web/frontend/tailwind.config.js .
COPY web/frontend/index.html .

RUN npm install

COPY web/frontend/src/ ./src/

RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY web/frontend/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
