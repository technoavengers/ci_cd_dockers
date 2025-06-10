FROM python:3.9-slim

WORKDIR /app

COPY serving/serving.py serving/preprocessing.py requirements.txt ./artifacts/random_forest_model.pkl ./artifacts/scaler.pkl /app/

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "serving:app", "--host", "0.0.0.0", "--port", "8000"]