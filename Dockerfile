FROM python:3-onbuild

EXPOSE 5000

CMD ["export", "FLASK_APP=rest-api.py"]
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
