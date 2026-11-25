# Batch image for the F1 Race Intelligence pipeline (M2-M6).
#
# Build (traceable; refuses uncommitted changes; full recipe in infra/README.md):
#   test -z "$(git status --porcelain --untracked-files=all)" && docker build --platform linux/amd64 --build-arg F1_GIT_COMMIT="$(git rev-parse HEAD)" --build-arg F1_GIT_DIRTY=false -t f1-race-intelligence:dev .
# Run:    docker run --rm f1-race-intelligence:dev --stage m4 --year 2024 --session-key 9472
#
# The image holds code and configuration only: no datasets, no .env, no
# credentials. Data comes from a mounted volume (local runs) or from S3, with
# credentials supplied by the IAM role of the task that runs the container.

# Base pinned by digest (the multi-arch index of python:3.12-slim-trixie,
# Python 3.12.16, published 2026-11-24), so a rebuild of the same commit gets the
# same base and an immutable ECR tag always means the same image. The tag is kept
# for readers; the digest decides. To take base-image security fixes, bump it on
# purpose:
#   docker buildx imagetools inspect python:3.12-slim-trixie   (Digest: line)
# then rebuild and rerun pip check, --help, id and an M3-M5 smoke run.
#
# CVE-2026-124851 (zstd 1.5.1-1.5.7, fixed upstream in 1.6.0) has two copies to
# remove from this image, and patching one does not remove the other:
#   1. The operating system's libzstd1. Debian trixie fixed it in
#      1.5.7+dfsg-1+deb13u1 (DSA-6657-1), which the base image does not carry
#      yet, so the final stage installs exactly that version.
#   2. The zstd 1.5.7 that every pyarrow wheel on PyPI links statically into
#      libarrow, invisible to apt. The first stage builds pyarrow from the
#      official Arrow source release with ARROW_WITH_ZSTD=OFF; the final stage
#      installs that wheel in place of the locked one and asserts at build time
#      that no zstd is left. Parquet keeps SNAPPY, which the pipeline uses.

# Stage 1 (discarded): pyarrow without zstd. Build-only toolchain; nothing from
# here reaches the final image except the wheel.
FROM python:3.12-slim-trixie@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b AS pyarrow-build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/arrow-build-lock.txt requirements/arrow-build-lock.txt
RUN pip install --requirement requirements/arrow-build-lock.txt

COPY docker/build-pyarrow-without-zstd.sh /usr/local/bin/build-pyarrow-without-zstd.sh
RUN bash /usr/local/bin/build-pyarrow-without-zstd.sh

# Stage 2: the runtime image.
FROM python:3.12-slim-trixie@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Supply these at build time when building from a source checkout. Without a
# commit argument, the runtime records no commit rather than inventing one.
ARG F1_GIT_COMMIT
ARG F1_GIT_DIRTY

# Unprivileged user with a fixed UID, so file ownership on mounted volumes is
# predictable.
RUN groupadd --system --gid 10001 f1 \
    && useradd --system --uid 10001 --gid f1 --home-dir /app --shell /usr/sbin/nologin f1

# Surface 1 of CVE-2026-124851: the operating system's copy. The version is
# pinned exactly and checked, so the build fails (instead of silently keeping
# the vulnerable 1.5.7+dfsg-1 of the base image) if Debian stops serving it.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libzstd1=1.5.7+dfsg-1+deb13u1 \
    && test "$(dpkg-query --show --showformat='${Version}' libzstd1)" = "1.5.7+dfsg-1+deb13u1" \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, from the exact lock, so code changes do not invalidate
# this layer. pyarrow is the one entry left out of the lock install: it comes
# from the zstd-free wheel built in stage 1, at the same version. The lock file
# itself stays untouched (it is what the M6 reports hash), and the build fails if
# the lock stops pinning exactly pyarrow==25.0.1.
COPY requirements/runtime-lock.txt requirements/runtime-lock.txt
COPY --from=pyarrow-build /wheelhouse/ /wheelhouse/
RUN grep -v -E '^pyarrow==25\.0\.1[[:space:]]*$' requirements/runtime-lock.txt > /tmp/runtime-lock-without-pyarrow.txt \
    && ! grep -q '^pyarrow' /tmp/runtime-lock-without-pyarrow.txt \
    && pip install --requirement /tmp/runtime-lock-without-pyarrow.txt \
    && pip install --no-deps /wheelhouse/pyarrow-25.0.1-*.whl \
    && rm -rf /wheelhouse /tmp/runtime-lock-without-pyarrow.txt

# Surface 2 of CVE-2026-124851: fail the build if any zstd is left in pyarrow
# (the codec must be unavailable and no zstd symbol may remain in its binaries)
# or if Parquet lost SNAPPY, the codec the pipeline writes.
RUN python -c "import pyarrow as pa, pyarrow.parquet; assert pa.__version__ == '25.0.1'; assert not pa.Codec.is_available('zstd'); assert pa.Codec.is_available('snappy')" \
    && ! grep -rlE --include='*.so*' 'ZSTD_decompressStream|ZSTD_compressStream|ZSTD_versionString' "$(python -c 'import os, pyarrow; print(os.path.dirname(pyarrow.__file__))')"

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
    F1_ENVIRONMENT=local \
    F1_GIT_COMMIT=${F1_GIT_COMMIT} \
    F1_GIT_DIRTY=${F1_GIT_DIRTY} \
    F1_DEPENDENCY_LOCK_PATH=/app/requirements/runtime-lock.txt

USER f1

ENTRYPOINT ["python", "-m", "f1_race_intelligence"]
CMD ["--help"]
