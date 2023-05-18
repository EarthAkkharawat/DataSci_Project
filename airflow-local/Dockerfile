FROM python:3.10-slim-buster
RUN apk add -u zlib-dev jpeg-dev gcc musl-dev

COPY requirements.txt .

RUN pip install -r requirements.txt

