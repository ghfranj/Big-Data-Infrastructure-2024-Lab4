# Specify the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the source code to the working directory
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Define the command to run the application
CMD ["python3", "main.py"]
