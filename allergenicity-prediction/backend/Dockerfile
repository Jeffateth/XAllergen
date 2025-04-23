FROM python:3.10-slim-bullseye

# Update the base image to ensure it has the latest security patches
RUN apt-get update && apt-get upgrade -y && apt-get clean

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port the app will run on
EXPOSE 5000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]