
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-api.txt .

RUN python -m pip install --no-cache-dir \
    torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN python -m pip install --no-cache-dir \
    -r requirements-api.txt

COPY src ./src

RUN useradd --create-home --uid 10001 appuser

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
