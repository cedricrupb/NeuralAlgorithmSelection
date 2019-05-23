FROM cedricrupb/nas-base:latest

WORKDIR /app/python/

COPY requirements.txt ./
RUN pip install -r ./requirements.txt

COPY taskflow/ ./taskflow/
COPY tasks/ ./tasks/

CMD ["python3", "-m", "taskflow"]
