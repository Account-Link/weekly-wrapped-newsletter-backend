import multiprocessing
import os
from dotenv import load_dotenv

# Auto-load .env so PORT and other settings are picked up.
load_dotenv()

bind = f"0.0.0.0:{os.getenv('PORT', '8080')}"
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() // 2 or 2))
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "uvicorn.workers.UvicornWorker")
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))
# Default to stdout/stderr so container logs can be shipped by the host/agent.
accesslog = os.getenv("GUNICORN_ACCESS_LOG", "-")
errorlog = os.getenv("GUNICORN_ERROR_LOG", "-")
loglevel = os.getenv("LOG_LEVEL", "info").lower()

# TLS (optional)
certfile = os.getenv("GUNICORN_CERTFILE")
keyfile = os.getenv("GUNICORN_KEYFILE")
if os.getenv("HTTPS_ON", "").lower() in {"true", "1", "yes"}:
    # Only enable TLS if cert/key paths exist inside the container.
    if certfile and keyfile and os.path.exists(certfile) and os.path.exists(keyfile):
        pass  # gunicorn picks up certfile/keyfile from globals
    else:
        certfile = None
        keyfile = None
