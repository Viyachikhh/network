FROM python:3.9

WORKDIR /code

COPY requirements.txt .
COPY data/ /code/data/

RUN pip install setup.py

COPY src/ .


CMD [ "python", "./main.py" ]