# ══════════════════════════════════════════════════════════════
#  OptiQuant Frontend — Production Nginx Build
#  Optimized for: security, caching, compression, size
# ══════════════════════════════════════════════════════════════

FROM nginx:1.27-alpine AS production

LABEL maintainer="OptiQuant Team" \
      version="2.0.0" \
      description="OptiQuant Frontend — Nginx SPA Server"

# Remove default config & content
RUN rm -rf /usr/share/nginx/html/* /etc/nginx/conf.d/default.conf

# Copy frontend assets
COPY frontend/index.html frontend/login.html frontend/50x.html /usr/share/nginx/html/
COPY frontend/styles.css frontend/premium.css /usr/share/nginx/html/
COPY frontend/app.js frontend/premium-motion.js /usr/share/nginx/html/
COPY frontend/charts /usr/share/nginx/html/charts/

# Copy production nginx config
COPY infra/docker/nginx.prod.conf /etc/nginx/conf.d/default.conf

# Security: disable server tokens, create cache dirs
RUN echo 'server_tokens off;' > /etc/nginx/conf.d/security.conf && \
    mkdir -p /var/cache/nginx/proxy_cache && \
    chown -R nginx:nginx /var/cache/nginx

EXPOSE 80 443

HEALTHCHECK --interval=10s --timeout=3s --retries=3 --start-period=5s \
    CMD wget --no-verbose --tries=1 --spider http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
