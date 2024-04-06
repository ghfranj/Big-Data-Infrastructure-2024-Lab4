FROM python:latest

# Set working directory
WORKDIR /app

ENV PYTHONUNBUFFERED 1

# Copy project files
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Set command to run when the container starts
CMD ["python", "main.py"]
