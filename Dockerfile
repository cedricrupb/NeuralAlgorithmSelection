FROM ubuntu:18.04

EXPOSE 5000

RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y unzip \
 && apt-get install -y python3 python3-pip openjdk-8-jre

COPY . /app
RUN pip3 install -r /app/requirements.txt \
    && unzip /app/pesco.zip \
    && mkdir upload \
    && cd /app

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV FLASK_APP=/app/rest-api.py

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]
