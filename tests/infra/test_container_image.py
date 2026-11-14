"""Static checks of the batch image definition (Dockerfile, .dockerignore).

They do not build anything; they pin the decisions a later edit could lose
without any test noticing: a reproducible base, the config the code needs,
a non-root user, and nothing private in the build context.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
DOCKERIGNORE = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")


def instructions(name):
    """Arguments of every ``name`` instruction, with ``\\`` continuations joined."""
    joined = re.sub(r"\\\n\s*", " ", DOCKERFILE)
    return re.findall(rf"^{name}\s+(.+)$", joined, flags=re.M)


def test_the_base_image_is_pinned_by_a_full_digest() -> None:
    (base,) = instructions("FROM")
    # An immutable ECR tag must always mean the same image.
    assert re.fullmatch(r"python:3\.12-slim-bookworm@sha256:[0-9a-f]{64}", base), base


def test_the_config_the_code_needs_is_copied_where_f1_config_path_points() -> None:
    match = re.search(r"F1_CONFIG_PATH=(\S+)", DOCKERFILE)
    assert match and match.group(1) == "/app/configs/config.yaml"
    assert "configs ./configs" in instructions("COPY")
    assert (REPO_ROOT / "configs" / "config.yaml").is_file()
    assert (REPO_ROOT / "configs" / "config.aws.yaml").is_file()


def test_dependencies_come_only_from_the_lock() -> None:
    runs = " ".join(instructions("RUN"))
    assert "pip install --requirement requirements/runtime-lock.txt" in runs
    assert "pip install --no-deps ." in runs and "pip check" in runs
    # The S3 backend's optional dependency must be in the image anyway.
    assert re.search(r"^boto3==", (REPO_ROOT / "requirements" / "runtime-lock.txt").read_text(encoding="utf-8"), flags=re.M)


# Roots certifi dropped after the eMudhra/emSign incident: G3 in 2026.10.19
# (GHSA-6q48-6c7w-9gw2), C1 and ECC C3 in 2026.10.23 (certifi b60466c).
# httpx (OpenF1) and botocore (S3) verify TLS against certifi's bundle.
COMPROMISED_EMSIGN_ROOTS = ("emSign ECC Root CA - G3", "emSign Root CA - C1", "emSign ECC Root CA - C3")


def test_the_lock_pins_a_certifi_without_the_compromised_emsign_roots() -> None:
    lock = (REPO_ROOT / "requirements" / "runtime-lock.txt").read_text(encoding="utf-8")
    match = re.search(r"^certifi==(\d+)\.(\d+)\.(\d+)$", lock, flags=re.M)
    assert match, "certifi must stay pinned in the lock"
    assert tuple(int(part) for part in match.groups()) >= (2026, 10, 23)


def test_the_installed_ca_bundle_excludes_the_compromised_emsign_roots() -> None:
    # The bundle actually used at run time, not the version number: run where
    # the lock is installed (the test environment, or the image).
    import certifi
    import httpx

    labels = re.findall(r'^# Label: "([^"]+)"', Path(certifi.where()).read_text(encoding="utf-8"), flags=re.M)
    assert len(labels) > 100, f"unexpectedly small CA bundle at {certifi.where()}"
    assert not set(COMPROMISED_EMSIGN_ROOTS) & set(labels)

    trusted = {dict(item[0] for item in cert["subject"]).get("commonName") for cert in httpx.create_ssl_context().get_ca_certs()}
    assert not set(COMPROMISED_EMSIGN_ROOTS) & trusted


def test_setuid_and_setgid_bits_are_stripped_and_verified_at_build_time() -> None:
    runs = instructions("RUN")
    hardening = next((run for run in runs if "-perm /6000" in run), None)
    assert hardening, "the image must strip setuid/setgid executables"
    assert "chmod ug-s" in hardening and "-xdev" in hardening
    assert 'test -z "$(find / -xdev -type f -perm /6000)"' in hardening, "the build must prove none are left"
    assert "|| true" not in hardening, "a failing chmod must fail the build"
    # Done as root, before switching to the unprivileged user.
    assert DOCKERFILE.index("-perm /6000") < DOCKERFILE.index("USER f1")


def test_the_container_runs_unprivileged_through_the_cli() -> None:
    assert instructions("USER") == ["f1"]
    assert "--uid 10001" in DOCKERFILE
    assert instructions("ENTRYPOINT") == ['["python", "-m", "f1_race_intelligence"]']


@pytest.mark.parametrize("pattern", ["data/", ".env", ".aws/", "tests/", ".git/", ".venv/", "*.parquet"])
def test_private_or_bulky_paths_never_reach_the_build_context(pattern) -> None:
    assert pattern in DOCKERIGNORE.splitlines()
