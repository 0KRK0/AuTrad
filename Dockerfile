FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY argo argo
COPY risk risk
COPY execution execution
COPY audit audit
COPY marketdata marketdata
COPY backtest backtest
COPY intel intel
COPY api api
RUN useradd -r -u 10001 autrad && mkdir -p /var/log/autrad && chown autrad /var/log/autrad
USER autrad
EXPOSE 8000
