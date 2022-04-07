FROM python:3.7.9-slim-buster

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["flask", "run", "--host", "0.0.0.0", "-p", "1234"]
#ENV FLASK_APP=server.py
#CMD ["python", "-m", "flask", "run"]
