FROM python:3.10.9


WORKDIR /app

COPY . .



RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

CMD ["python", "cifar10_web_app", "app.py", "0.0.0.0:8000"]