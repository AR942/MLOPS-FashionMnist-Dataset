#Dockerfile to build a flask app

FROM python:3.8-slim-buster

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --upgrade wheel
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "-m", "flask", "app.py"]