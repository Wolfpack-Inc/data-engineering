# Adjusted from https://github.com/ekhtiar/streaming-data-pipeline

import requests
from time import sleep
from json import dumps, loads
from kafka import KafkaProducer
from datetime import datetime

api_key = 'b01ab56d38341be9485d53ebfe5c945d6c35f8d3f3759a4c6bcdfc67f2eee4db'

# Wait three seconds for the kafka broker to start
sleep(3)

producer = KafkaProducer(bootstrap_servers=['kafka:9092'],
                                value_serializer=lambda x: dumps(x).encode('utf-8'))

while True:
    # Format the url for the api to get a OK response
    url = 'https://min-api.cryptocompare.com/data/price?fsym=BTC&tsyms=EUR'
    price = loads(requests.get(url).text)['EUR']

    print(price)

    # Get the current time
    current_time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')

    # Prepare the value and timestamp for sending to kafka
    data = {'timestamp': current_time, 'price': price}

    # Send data to kafka
    producer.send('crypto', value=data)

    # Wait 5 seconds till we send the next update
    sleep(5)

