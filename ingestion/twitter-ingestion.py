import pprint
import json
from tweepy import OAuthHandler, Stream, StreamListener
from time import sleep
from json import dumps, loads
from kafka import KafkaProducer
import tweepy

# Twitter keys
consumer_key = "I8Qvxs0xU5ifPHhfg98kxJVc4"
consumer_secret = "43F9vXQIpDbJsB6TS8lVnKdBs7HjCpyLliM58oL8zCStAWD86J"
access_token = "233923293-45rYG7fgTQb5HWJMY12YO32oXV6RlvQXFDsH5a0L"
access_token_secret = "LMesjv8I1Ke85yVpRxFDblkVAOct7grAjnpc8uU8L9kjh"

# Flag to check if kafka brokers are running
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

class StreamToKafka(StreamListener):
    """ 
    A listener handles tweets that are received from the stream.
    """

    def on_data(self, data):
        tweet = json.loads(data)
        producer.send('twitter', value=tweet)
        return True

    def on_error(self, status):
        print(status)

to_kafka = StreamToKafka()
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

stream = Stream(auth, to_kafka)
stream.filter(track=['bitcoin'], languages=['en'])
