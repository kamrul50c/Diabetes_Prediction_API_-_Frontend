FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app
COPY ./src ./src

COPY ./diabetes_model.pkl ./diabetes_model.pkl
COPY ./metrics.json ./metrics.json

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
