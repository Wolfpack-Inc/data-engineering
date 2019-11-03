FROM python:3.6.9-slim-buster

ADD twitter-ingestion.py /

RUN pip install kafka-python tweepy nltk

CMD [ "python", "-u", "./stream-storage.py"]