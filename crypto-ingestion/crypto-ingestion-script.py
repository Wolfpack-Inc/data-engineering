import requests
from time import sleep
from json import dumps, loads
from kafka import KafkaProducer
from datetime import datetime

api_key = 'b01ab56d38341be9485d53ebfe5c945d6c35f8d3f3759a4c6bcdfc67f2eee4db'

#connect to kafka as a producer
# producer = KafkaProducer(bootstrap_servers=['kafka-1:9092'],
#                          value_serializer=lambda x:
#                          dumps(x).encode('utf-8'))

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

# list of crypto currencies we are interested in
crypto_coins = ['BTC'] #, 'ETH', 'XRP', 'BCH', 'LTC', 'EOS', 'XLM', 'LINK', 'DASH', 'XTZ']

# go on a never-ending loop
while True:
    # loop over every coin we are interested in
    for coin in crypto_coins:
        # format the url for the api to get a OK response
        url = 'https://min-api.cryptocompare.com/data/price?fsym={}&tsyms=USD'.format(coin)
        usd_value = loads(requests.get(url).text)['USD']
        time_now = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        # prepare the value and timestamp for sending to kafka
        data = {'timestamp': time_now, 'usd_value': usd_value}
        # send data to kafka
        producer.send(coin, value=data)
        # a small sleep to not overwhelm the API
        sleep(1)
    # wait 30 seconds till we send the next update
    sleep(30)

