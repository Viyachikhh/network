
FROM python:3.9

WORKDIR /code

COPY requirements.txt .
COPY data/ /code/data/

RUN pip install -r requirements.txt

COPY network/ /code/network/


COPY src/ .


CMD [ "python", "./main.py" ]