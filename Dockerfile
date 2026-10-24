# Batch image for the F1 Race Intelligence pipeline (M2-M6).
#
# Build:  docker build -t f1-race-intelligence:dev .
# Run:    docker run --rm f1-race-intelligence:dev --stage m4 --year 2024 --session-key 9472
#
# The image holds code and configuration only: no datasets, no .env, no
# credentials. Data comes from a mounted volume (local runs) or from S3, with
# credentials supplied by the IAM role of the task that runs the container.

# Base pinned by digest (the multi-arch index of python:3.12-slim-bookworm,
# Python 3.12.15), so a rebuild of the same commit gets the same base and an
# immutable ECR tag always means the same image. The tag is kept for readers;
# the digest decides. To take base-image security fixes, bump it on purpose:
#   docker buildx imagetools inspect python:3.12-slim-bookworm   (Digest: line)
# then rebuild and rerun pip check, --help, id and an M3-M5 smoke run.
FROM python:3.12-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70

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

# Hardening: no setuid/setgid executables. The pipeline never needs them, and
# they are the local privilege-escalation path of the base image's unpatched
# util-linux (mount) and glibc (ld.so AT_SECURE) findings. -xdev keeps find on
# the image filesystem; a failing chmod fails the build. The second find proves
# nothing is left.
RUN find / -xdev -type f -perm /6000 -exec chmod ug-s {} + \
    && test -z "$(find / -xdev -type f -perm /6000)"

# The installed package cannot locate configs/ relative to its own source
# file, so the config path is explicit. Storage and environment are chosen
# with F1_ENVIRONMENT / F1_STORAGE_* at run time.
ENV F1_CONFIG_PATH=/app/configs/config.yaml \
    F1_ENVIRONMENT=local

USER f1

ENTRYPOINT ["python", "-m", "f1_race_intelligence"]
CMD ["--help"]
