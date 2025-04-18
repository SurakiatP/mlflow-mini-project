FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY test/ test/

EXPOSE 8000
EXPOSE 5000

CMD ["bash", "-c", "mlflow ui --host 0.0.0.0 --port 5000 & uvicorn src.serve:app --host 0.0.0.0 --port 8000"]
