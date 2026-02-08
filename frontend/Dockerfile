FROM nginx:1.27-alpine

# Remove default nginx welcome page
RUN rm -rf /usr/share/nginx/html/*

# Copy frontend files directly to nginx html root
COPY frontend/index.html frontend/login.html frontend/styles.css frontend/app.js /usr/share/nginx/html/
COPY frontend/charts /usr/share/nginx/html/charts/

# Replace default nginx config
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
