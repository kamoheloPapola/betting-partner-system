FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# libgomp1 = LightGBM requirement, curl = model download
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN chmod +x /app/scripts/download_models.sh
RUN mkdir -p /app/data

CMD ["/bin/bash", "-c", "/app/scripts/download_models.sh && python scripts/fetch_fresh_data.py && uvicorn src.api.main:app --host 0.0.0.0 --port 8000"]
