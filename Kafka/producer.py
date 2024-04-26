from kafka import KafkaProducer
import json

# Function to send message to Kafka topic
def send_to_kafka(message):
    try:
        producer = KafkaProducer(bootstrap_servers='kafka:9093')
    except:
        producer = KafkaProducer(bootstrap_servers='localhost:9092')

    producer.send('model_results', json.dumps(message).encode('utf-8'))
    producer.flush()
    producer.close()
