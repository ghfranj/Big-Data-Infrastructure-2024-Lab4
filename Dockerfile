# Spedocker run -p 8081:8080 -p 50000:50000 jenkins/jenkins:ltscify the base image
FROM python:3.9-slim
#FROM ghfranjabour/lab1_big_data_infrastructure:latest
# Set the working directory
WORKDIR /app

# Copy the source code to the working directory
COPY . .

# Install dependencies
#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

# Define the command to run the application
CMD ["python3", "main.py"]
