import pprint
import json
from tweepy import OAuthHandler, Stream, StreamListener
from time import sleep
from json import dumps, loads
from kafka import KafkaProducer
from datetime import datetime
import tweepy
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import nltk
nltk.download('vader_lexicon')

# Twitter keys
consumer_key = "I8Qvxs0xU5ifPHhfg98kxJVc4"
consumer_secret = "43F9vXQIpDbJsB6TS8lVnKdBs7HjCpyLliM58oL8zCStAWD86J"
access_token = "233923293-45rYG7fgTQb5HWJMY12YO32oXV6RlvQXFDsH5a0L"
access_token_secret = "LMesjv8I1Ke85yVpRxFDblkVAOct7grAjnpc8uU8L9kjh"

# Flag to check if kafka brokers are running
kafka_running = False

while kafka_running == False:
    try:
        producer = KafkaProducer(bootstrap_servers=['localhost:19092'], value_serializer=lambda x: dumps(x).encode('utf-8'))
    except:
        print('No kafka brokers are running yet')
        sleep(1)
    else:
        kafka_running = True
        print("Connected to kafka broker!")

# Create the vader sentiment analyser
analyzer = SentimentIntensityAnalyzer()

class StreamToKafka(StreamListener):
    """ 
    A listener handles tweets that are received from the stream.
    """

    def on_data(self, data):
        tweet = json.loads(data)

        # If there is no text
        if 'text' not in tweet:
            return True

        # Get the sentiment of the tweet using vader
        sentiment = analyzer.polarity_scores(tweet['text'])['compound']

        # Print the tweet and the sentiment
        # print(round(sentiment, 2), '|', tweet['text'].replace('\n', ''))

        # Parse the date
        date = datetime.strptime(tweet['created_at'], '%a %b %d %X %z %Y')
        date_str = date.strftime("%d-%m-%Y %H:%M:%S")

        # Get the location of the tweet
        # print(tweet.keys())
        # if 'coordinates' in tweet:
        #     if (tweet['coordinates'] != None) and ('coordinates' in tweet['coordinates']):
        #         location = tweet['coordinates']['coordinates']
        #         print(location)

        # Data to send
        data = {
            'timestamp': date_str,
            'text': tweet['text'],
            'sentiment': sentiment
        }

        producer.send('twitter', value=data)

        return True

    def on_error(self, status):
        print(status)

        if status == 420:
            print('Rate limited')
            sleep(30)

to_kafka = StreamToKafka()
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

stream = Stream(auth, to_kafka)
stream.filter(track=['bitcoin'], languages=['en'])
