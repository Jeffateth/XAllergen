# Use the latest secure Node.js Alpine image
FROM node:18-alpine AS build

WORKDIR /app

# Copy only package files to leverage Docker layer caching
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy the rest of the application code
COPY . .

# Build the application
RUN npm run build

# Use the latest secure Nginx Alpine image
FROM nginx:stable-alpine

# Copy the built application from the build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Configure Nginx to handle SPA routing
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose the default HTTP port
EXPOSE 80

# Run Nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]