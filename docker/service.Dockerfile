FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ARG SERVICE_MODULE=agents.supervisor.app:app
ENV SERVICE_MODULE=${SERVICE_MODULE}

CMD ["sh", "-c", "uvicorn ${SERVICE_MODULE} --host 0.0.0.0 --port 8000"]
