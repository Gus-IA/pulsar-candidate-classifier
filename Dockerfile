FROM python:3.13-slim

WORKDIR /app

# XGBoost needs the OpenMP runtime, which slim images do not include
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY predict.py model.pkl ./

EXPOSE 9696

CMD ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]