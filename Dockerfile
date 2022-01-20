
FROM python:3.9

WORKDIR /code

COPY requirements.txt .
COPY data/ /code/data/

RUN pip install -r requirements.txt
RUN pip install network

COPY network/ /code/network/


COPY src/ .


CMD [ "python", "./main.py" ]