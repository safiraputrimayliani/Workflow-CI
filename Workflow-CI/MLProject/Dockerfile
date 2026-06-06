FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    pandas==2.2.2 \
    scikit-learn==1.4.2 \
    matplotlib==3.8.4

COPY . .

EXPOSE 5001

CMD ["mlflow", "models", "serve", "-m", "/app/model", "--host", "0.0.0.0", "--port", "5001", "--no-conda"]
