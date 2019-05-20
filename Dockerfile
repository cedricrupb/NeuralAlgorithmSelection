FROM cedricrupb/task-backend:latest

RUN apt-get install -y unzip openjdk-8-jre

COPY pesco.zip /app
RUN unzip /app/pesco.zip

COPY tasks/ /app/python
