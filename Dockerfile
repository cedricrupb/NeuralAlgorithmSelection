FROM cedricrupb/nas-base:latest

COPY tasks/ /app/python
COPY tests/ /app/python
