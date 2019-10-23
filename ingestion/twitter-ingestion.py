from time import sleep
from json import dumps, loads
from kafka import KafkaProducer

kafka_running = False

while kafka_running == False:
    try:
        producer = KafkaProducer(bootstrap_servers=['kafka-1:9092'],
                                value_serializer=lambda x: dumps(x).encode('utf-8'))
    except:
        print('No kafka brokers are running yet')
        sleep(3)
    else:
        kafka_running = True
        print("Connected to kafka broker!")

# Send hello world message every 10 seconds
while True:
    producer.send('twitter', value={'hello': 'world'})
    sleep(10)