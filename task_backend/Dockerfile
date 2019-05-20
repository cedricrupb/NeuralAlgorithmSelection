FROM ubuntu:18.04

RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y python3 python3-pip unzip openjdk-8-jre

COPY pesco.zip /app/pesco.zip
RUN unzip /app/pesco.zip

COPY . /app/python/task_backend
RUN cd /app/python/task_backend \
   && pip3 install -r requirements.txt \
   && cd /

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONPATH="$PYTHONPATH:/app/python"
