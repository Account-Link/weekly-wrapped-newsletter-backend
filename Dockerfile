FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
COPY . /app

RUN uv sync

CMD ["uv", "run", "gunicorn", "-c", "gunicorn_conf.py", "app.main:app"]
