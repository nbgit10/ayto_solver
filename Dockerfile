FROM --platform=linux/amd64 python:3.10

COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt
