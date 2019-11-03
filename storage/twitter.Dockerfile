FROM python:3.6.9-slim-buster

ADD twitter-storage.py /

RUN pip install kafka-python tweepy nltk pymongo

CMD [ "python", "-u", "./twitter-storage.py"]