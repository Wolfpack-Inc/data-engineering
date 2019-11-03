# Adjusted from https://github.com/ekhtiar/streaming-data-pipeline

import requests
from pymongo import MongoClient
from time import sleep
from json import dumps, loads
from datetime import datetime

# Mongo client
client = MongoClient('mongodb://mongo:27017')
db = client.raw
crypto = db.crypto

while True:
    # Format the url for the api to get a OK response
    url = 'https://min-api.cryptocompare.com/data/price?fsym=BTC&tsyms=EUR'
    price = loads(requests.get(url).text)['EUR']

    # Get the current time
    current_time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')

    # Prepare the value and timestamp for sending to kafka
    data = {'timestamp': current_time, 'price': price}

    # Store in the database
    crypto.insert_one(data)

    # Wait 10 seconds till we send the next update
    sleep(60)
