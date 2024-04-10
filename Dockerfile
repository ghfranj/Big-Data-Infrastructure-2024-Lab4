FROM python:3.9-slim

WORKDIR /app

# Copy the source code to the working directory
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Run the application
CMD ["python3", "main.py"]
