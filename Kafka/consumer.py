from kafka import KafkaConsumer
import json

# Function to receive message from Kafka topic
def receive_from_kafka():
    try:
        consumer = KafkaConsumer('model_results', bootstrap_servers='kafka:9093', group_id='my_consumer_group')
    except:
        consumer = KafkaConsumer('model_results', bootstrap_servers='localhost:9092', group_id='my_consumer_group')

    for message in consumer:
        result = json.loads(message.value.decode('utf-8'))
        print(f"finished training the model with {result}")
        break

    consumer.close()