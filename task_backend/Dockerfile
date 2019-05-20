FROM ubuntu:18.04

RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y unzip \
 && apt-get install -y python3 python3-pip

COPY . /app/python/task_backend
RUN cd /app/python/task_backend \
   && pip3 install -r requirements.txt \
   && cd /

ENV PYTHONPATH="$PYTHONPATH:/app/python"

CMD ["python3", "-m", "task_backend"]
