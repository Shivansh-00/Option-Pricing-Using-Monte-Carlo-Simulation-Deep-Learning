# ══════════════════════════════════════════════════════════════
#  OptionQuant Frontend — Optimized Nginx Build
# ══════════════════════════════════════════════════════════════

FROM nginx:1.27-alpine

# Remove default nginx content
RUN rm -rf /usr/share/nginx/html/*

# Copy all frontend assets
COPY frontend/index.html frontend/login.html frontend/50x.html /usr/share/nginx/html/
COPY frontend/styles.css frontend/premium.css /usr/share/nginx/html/
COPY frontend/app.js frontend/premium-motion.js /usr/share/nginx/html/
COPY frontend/charts /usr/share/nginx/html/charts/

# Nginx configuration
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Security headers config
RUN echo 'server_tokens off;' > /etc/nginx/conf.d/security.conf

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
