FROM python:3.6.9-slim-buster

ADD twitter-ingestion.py /

RUN pip install kafka-python

CMD [ "python", "-u", "./twitter-ingestion.py" ]