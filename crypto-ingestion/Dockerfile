FROM mrkcse/docker-python-librdkafka

ADD crypto-ingestion-script.py /

RUN pip install kafka-python \
  && pip install requests

CMD [ "python","-u","./crypto-ingestion-script.py" ]