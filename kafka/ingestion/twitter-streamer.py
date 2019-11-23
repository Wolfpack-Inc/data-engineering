from tweepy import OAuthHandler, Stream, StreamListener
import json
import pprint

pp = pprint.PrettyPrinter(indent=4)

# Go to http://apps.twitter.com and create an app.
# The consumer key and secret will be generated for you after
consumer_key = "I8Qvxs0xU5ifPHhfg98kxJVc4"
consumer_secret = "43F9vXQIpDbJsB6TS8lVnKdBs7HjCpyLliM58oL8zCStAWD86J"

# After the step above, you will be redirected to your app's page.
# Create an access token under the the "Your access token" section
access_token = "233923293-45rYG7fgTQb5HWJMY12YO32oXV6RlvQXFDsH5a0L"
access_token_secret = "LMesjv8I1Ke85yVpRxFDblkVAOct7grAjnpc8uU8L9kjh"


class StdOutListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """

    def on_data(self, data):
        tweet = json.loads(data)
        print(tweet['text'])
        print('-----------------------')
        # pp.pprint(tweet['text'])
        return True

    def on_error(self, status):
        print(status)


if __name__ == '__main__':
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    stream = Stream(auth, l)
    stream.filter(track=['btc', 'bitcoin', 'xbt', 'satoshi'], languages=['en'])
