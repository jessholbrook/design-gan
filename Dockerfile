# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers \
    DESIGN_GAN_RUNS_DIR=/data

WORKDIR /app

# Metadata + sources first so pip's editable install can find the package.
COPY pyproject.toml README.md ./
COPY src ./src

# Install the project, then Chromium + its Linux deps. `--with-deps` pulls in
# the apt packages the browser needs (fonts, libnss, etc.).
RUN pip install --upgrade pip \
 && pip install . \
 && python -m playwright install --with-deps chromium \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# /data is where we persist the SQLite DB and per-iteration artifacts.
# Fly.io mounts a volume here; see fly.toml.
VOLUME ["/data"]

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "design_gan.viewer:app", \
     "--host", "0.0.0.0", "--port", "8080"]
