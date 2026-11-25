#!/usr/bin/env bash
# Build PyArrow 25.0.1 from the official Apache Arrow source release with the
# zstd codec compiled out, for CVE-2026-124851.
#
# Why: the pyarrow wheels on PyPI (25.0.x and 26.0.x alike) statically link
# zstd 1.5.7 into libarrow, which is vulnerable. That copy is invisible to the
# operating system's package manager, so patching libzstd1 does not remove it.
# Arrow compiles zstd in only when ARROW_WITH_ZSTD is ON (cpp/cmake_modules/
# DefineOptions.cmake); this script leaves it OFF. The pipeline reads and writes
# Parquet with SNAPPY, which stays on.
#
# Runs inside a throwaway build stage (see Dockerfile and Dockerfile.forecast),
# which must provide: a C++20 compiler, make, curl, tar, and the tools of
# requirements/arrow-build-lock.txt (cmake, ninja, Cython, scikit-build-core...)
# already installed with pip. Output: one wheel in
# $WHEELHOUSE (default /wheelhouse). The same script serves both images, so the
# batch image and the Lambda image are built identically from their own base.
set -euo pipefail

ARROW_VERSION="25.0.1"
# SHA-256 of apache-arrow-25.0.1.tar.gz. Its SHA-512 matches the one Apache
# publishes next to the release in archive.apache.org/dist/arrow/arrow-25.0.1/.
ARROW_SHA256="43d5de0a581f43cf63a2c06b4dcf13b9ff6fcd800f023324596e5781093bc500"
ARROW_URL="https://archive.apache.org/dist/arrow/arrow-${ARROW_VERSION}/apache-arrow-${ARROW_VERSION}.tar.gz"

WORK="${WORK:-/tmp/arrow-work}"
ARROW_HOME="${ARROW_HOME:-/tmp/arrow-dist}"
WHEELHOUSE="${WHEELHOUSE:-/wheelhouse}"
JOBS="${JOBS:-6}"

mkdir -p "${WORK}" "${WHEELHOUSE}"
cd "${WORK}"

echo "=== Fetching and verifying Apache Arrow ${ARROW_VERSION} ==="
curl --fail --silent --show-error --location --retry 3 -o arrow.tar.gz "${ARROW_URL}"
echo "${ARROW_SHA256}  arrow.tar.gz" | sha256sum --check -
tar -xzf arrow.tar.gz
SRC="${WORK}/apache-arrow-${ARROW_VERSION}"

echo "=== Building Arrow C++ (zstd OFF) ==="
# Only what pandas.read_parquet(engine="pyarrow") needs: Parquet, which brings
# Compute, Dataset and Acero with it in pyarrow, plus the CSV/JSON/filesystem
# modules pyarrow imports unconditionally. Every dependency is built from the
# source archives pinned (with SHA-256) in cpp/thirdparty/versions.txt.
# Parquet encryption, Flight, S3/GCS/Azure/HDFS, ORC, Gandiva, Substrait and the
# brotli/bz2/lz4/zstd codecs are not needed and stay out.
cmake -S "${SRC}/cpp" -B "${WORK}/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${ARROW_HOME}" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DARROW_BUILD_SHARED=ON \
    -DARROW_BUILD_STATIC=OFF \
    -DARROW_BUILD_TESTS=OFF \
    -DARROW_BUILD_UTILITIES=OFF \
    -DARROW_DEPENDENCY_SOURCE=BUNDLED \
    -DARROW_DEPENDENCY_USE_SHARED=OFF \
    -DARROW_RPATH_ORIGIN=ON \
    -DARROW_COMPUTE=ON \
    -DARROW_CSV=ON \
    -DARROW_JSON=ON \
    -DARROW_FILESYSTEM=ON \
    -DARROW_ACERO=ON \
    -DARROW_DATASET=ON \
    -DARROW_PARQUET=ON \
    -DPARQUET_REQUIRE_ENCRYPTION=OFF \
    -DARROW_JEMALLOC=OFF \
    -DARROW_MIMALLOC=ON \
    -DARROW_WITH_SNAPPY=ON \
    -DARROW_WITH_ZLIB=ON \
    -DARROW_WITH_BROTLI=OFF \
    -DARROW_WITH_BZ2=OFF \
    -DARROW_WITH_LZ4=OFF \
    -DARROW_WITH_ZSTD=OFF
cmake --build "${WORK}/build" --target install --parallel "${JOBS}"

echo "=== Building the pyarrow wheel against it ==="
# Take the bundled libraries from the install above and ship them in the wheel,
# like the official wheels do. Features Arrow was built without are off here too.
export ARROW_HOME
export CMAKE_PREFIX_PATH="${ARROW_HOME}"
export PYARROW_BUNDLE_ARROW_CPP=ON
export PYARROW_WITH_ACERO=ON
export PYARROW_WITH_DATASET=ON
export PYARROW_WITH_PARQUET=ON
export PYARROW_WITH_PARQUET_ENCRYPTION=OFF
export PYARROW_WITH_FLIGHT=OFF
export PYARROW_WITH_GANDIVA=OFF
export PYARROW_WITH_ORC=OFF
export PYARROW_WITH_SUBSTRAIT=OFF
export PYARROW_WITH_S3=OFF
export PYARROW_WITH_GCS=OFF
export PYARROW_WITH_AZURE=OFF
export PYARROW_WITH_CUDA=OFF
# No git checkout in a release tarball: state the version instead of guessing it.
export SETUPTOOLS_SCM_PRETEND_VERSION="${ARROW_VERSION}"
cd "${SRC}/python"
python -m build --wheel --no-isolation --outdir "${WHEELHOUSE}" \
    -C build.verbose=true \
    -C cmake.build-type=Release \
    -C build.tool-args="-j${JOBS}"

ls -l "${WHEELHOUSE}"
