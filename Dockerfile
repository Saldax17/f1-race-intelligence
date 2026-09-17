# Batch image for the F1 Race Intelligence pipeline (M2-M6).
#
# Build:  docker build -t f1-race-intelligence:dev .
# Run:    docker run --rm f1-race-intelligence:dev --stage m4 --year 2024 --session-key 9472
#
# The image holds code and configuration only: no datasets, no .env, no
# credentials. Data comes from a mounted volume (local runs) or from S3, with
# credentials supplied by the IAM role of the task that runs the container.

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Unprivileged user with a fixed UID, so file ownership on mounted volumes is
# predictable.
RUN groupadd --system --gid 10001 f1 \
    && useradd --system --uid 10001 --gid f1 --home-dir /app --shell /usr/sbin/nologin f1

# Dependencies first, from the exact lock, so code changes do not invalidate
# this layer.
COPY requirements/runtime-lock.txt requirements/runtime-lock.txt
RUN pip install --requirement requirements/runtime-lock.txt

# Then the project itself, without letting pip change any pinned version.
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-deps . \
    && pip check \
    && mkdir -p /app/data \
    && chown -R f1:f1 /app/data

# The installed package cannot locate configs/ relative to its own source
# file, so the config path is explicit. Storage and environment are chosen
# with F1_ENVIRONMENT / F1_STORAGE_* at run time.
ENV F1_CONFIG_PATH=/app/configs/config.yaml \
    F1_ENVIRONMENT=local

USER f1

ENTRYPOINT ["python", "-m", "f1_race_intelligence"]
CMD ["--help"]
