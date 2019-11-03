FROM python:3.6.9-slim-buster

ADD crypto-storage.py /

RUN pip install requests pymongo

CMD [ "python", "-u", "./crypto-storage.py"]