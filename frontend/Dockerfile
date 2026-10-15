# # Build stage
# FROM node:20-alpine AS builder
# WORKDIR /app

# COPY package.json package-lock.json* ./
# COPY tsconfig.json tsconfig.app.json tsconfig.node.json ./
# COPY index.html ./
# COPY vite.config.ts ./
# COPY public ./public
# COPY src ./src

# RUN npm install
# RUN npm run build

# # Production stage
# FROM nginx:stable-alpine
# COPY --from=builder /app/dist /usr/share/nginx/html

# # Remove default nginx config and use a minimal one for SPA routing
# RUN rm /etc/nginx/conf.d/default.conf
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# EXPOSE 80
# CMD ["nginx", "-g", "daemon off;"]



# Build stage
FROM node:20-alpine AS builder
WORKDIR /app

COPY package.json package-lock.json* ./
COPY tsconfig.json tsconfig.app.json tsconfig.node.json ./
COPY index.html ./
COPY vite.config.ts ./
COPY public ./public
COPY src ./src

# Backend API URL baked in at build time (no nginx proxy anymore,
# so the browser must be able to reach this URL directly)
ARG VITE_API_BASE=http://localhost:8000
ENV VITE_API_BASE=$VITE_API_BASE

RUN npm install
RUN npm run build

# Production stage - plain static server, no nginx
FROM node:20-alpine
WORKDIR /app

RUN npm install -g serve

COPY --from=builder /app/dist ./dist

EXPOSE 3000
CMD ["serve", "-s", "dist", "-l", "3000"]