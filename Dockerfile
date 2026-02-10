# Use official Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 10000

# Run app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
