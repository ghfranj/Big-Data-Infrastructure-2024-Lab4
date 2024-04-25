FROM ghfranjabour/lab2_big_data_infrastructure:v2.0

# Install system dependencies
RUN apt-get update && apt-get install -y  postgresql-client

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the source code into the container
COPY . .

CMD ["python3", "main.py"]