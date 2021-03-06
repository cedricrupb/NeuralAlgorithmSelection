FROM cedricrupb/nas-base:latest

WORKDIR /app/python/

COPY requirements.txt ./
RUN pip3 install torch==1.1.0 && pip3 install -r ./requirements.txt

COPY taskflow/ ./taskflow/
COPY tasks/ ./tasks/

CMD ["python3", "-m", "taskflow"]
