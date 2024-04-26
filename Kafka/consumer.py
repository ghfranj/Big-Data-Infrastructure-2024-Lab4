from kafka import KafkaConsumer
import json

# Function to receive message from Kafka topic
def receive_from_kafka():
    consumer = KafkaConsumer('model_results', bootstrap_servers='localhost:9092', group_id='my_consumer_group')

    for message in consumer:
        result = json.loads(message.value.decode('utf-8'))
        print("Received message:", result)
        break

    consumer.close()