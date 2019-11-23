FROM python:3.6.9-slim-buster

ADD crypto-ingestion.py /

RUN pip install kafka-python requests

CMD [ "python","-u","./crypto-ingestion.py" ]