FROM cedricrupb/nas-base:latest

WORKDIR /app/python/

COPY task_backend/ ./task_backend/
COPY tasks/ ./
COPY requirements.txt ./requirements.txt

RUN pip3 install -r ./requirements.txt

CMD ["python", "-m", "task_backend"]
