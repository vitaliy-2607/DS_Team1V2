FROM python:3.10.9


WORKDIR /app

COPY . .



RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "DS_Team1V2/cifar10_web_app/app.py", "0.0.0.0:8080"]