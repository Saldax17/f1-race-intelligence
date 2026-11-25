"""Static checks of the batch image definition (Dockerfile, .dockerignore).

They do not build anything; they pin the decisions a later edit could lose
without any test noticing: a reproducible base, the config the code needs,
a non-root user, nothing private in the build context, and documented build
recipes that record the exact commit and refuse uncommitted changes.
"""

import posixpath
import re
import shlex
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
DOCKERIGNORE = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")


# The image is built in two stages: a throwaway one that compiles pyarrow
# without zstd (CVE-2026-124851) and the runtime one. What the runtime image
# does (user, entrypoint, hardening, ...) is asserted on the last stage only.
STAGES = [stage for stage in re.split(r"(?m)^(?=FROM\s)", DOCKERFILE) if stage.startswith("FROM")]
BUILD_STAGE, FINAL_STAGE = STAGES[0], STAGES[-1]


def instructions(name, text=None):
    """Arguments of every ``name`` instruction of the runtime stage (or of ``text``), ``\\`` continuations joined."""
    joined = re.sub(r"\\\n\s*", " ", FINAL_STAGE if text is None else text)
    return re.findall(rf"^{name}\s+(.+)$", joined, flags=re.M)


def test_both_stages_are_pinned_to_the_same_full_digest() -> None:
    assert len(STAGES) == 2
    bases = [arguments.split()[0] for arguments in instructions("FROM", DOCKERFILE)]
    # An immutable ECR tag must always mean the same image.
    for base in bases:
        assert re.fullmatch(r"python:3\.12-slim-trixie@sha256:[0-9a-f]{64}", base), base
    assert len(set(bases)) == 1, "the build stage and the runtime stage must start from the same base"


def test_the_config_the_code_needs_is_copied_where_f1_config_path_points() -> None:
    match = re.search(r"F1_CONFIG_PATH=(\S+)", DOCKERFILE)
    assert match and match.group(1) == "/app/configs/config.yaml"
    assert "configs ./configs" in instructions("COPY")
    assert (REPO_ROOT / "configs" / "config.yaml").is_file()
    assert (REPO_ROOT / "configs" / "config.aws.yaml").is_file()


def test_dependencies_come_only_from_the_lock() -> None:
    runs = " ".join(instructions("RUN"))
    # The lock is installed whole except its pyarrow line, which the zstd-free
    # wheel replaces at the very same version (see the CVE-2026-124851 tests).
    # (tolerant of CRLF: a Windows checkout of the lock ends its lines with \r)
    assert "grep -v -E '^pyarrow==25\\.0\\.1[[:space:]]*$' requirements/runtime-lock.txt > /tmp/runtime-lock-without-pyarrow.txt" in runs
    assert "pip install --requirement /tmp/runtime-lock-without-pyarrow.txt" in runs
    assert "pip install --no-deps /wheelhouse/pyarrow-25.0.1-*.whl" in runs
    assert re.search(r"^pyarrow==25\.0\.1$", (REPO_ROOT / "requirements" / "runtime-lock.txt").read_text(encoding="utf-8"), flags=re.M)
    assert "pip install --no-deps ." in runs and "pip check" in runs
    # The S3 backend's optional dependency must be in the image anyway.
    assert re.search(r"^boto3==", (REPO_ROOT / "requirements" / "runtime-lock.txt").read_text(encoding="utf-8"), flags=re.M)


def test_m7_dependencies_are_optional_and_locked_separately() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    base_dependencies = [item.lower() for item in project["dependencies"]]
    training_extra = [item.lower() for item in project["optional-dependencies"]["training"]]
    runtime_lock = (REPO_ROOT / "requirements" / "runtime-lock.txt").read_text(encoding="utf-8").lower()
    training_lock = (REPO_ROOT / "requirements" / "training-lock.txt").read_text(encoding="utf-8")

    assert not any(item.startswith(("mlflow", "sqlalchemy")) for item in base_dependencies)
    assert training_extra == ["mlflow==3.18.0", "sqlalchemy==2.0.57"]
    assert not re.search(r"^(?:mlflow(?:-[a-z]+)?|sqlalchemy)==", runtime_lock, flags=re.M)
    assert re.search(r"^mlflow==3\.18\.0$", training_lock, flags=re.M)
    assert re.search(r"^sqlalchemy==2\.0\.57$", training_lock, flags=re.M)
    assert re.search(r'^gunicorn==23\.0\.0 ; platform_system != "Windows"$', training_lock, flags=re.M)
    assert re.search(r'^waitress==3\.0\.2 ; platform_system == "Windows"$', training_lock, flags=re.M)
    assert re.search(r'^pywin32==312 ; platform_system == "Windows"$', training_lock, flags=re.M)
    requirements = [
        line.strip()
        for line in training_lock.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert all(re.match(r"^[A-Za-z0-9_.-]+==[^ ;]+(?:\s*;\s*.+)?$", line) for line in requirements)


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


# -- .dockerignore semantics: what actually stays out of the build context


def _dockerignore_regex(pattern):
    """Docker's matching (moby/patternmatcher): anchored at the context root,
    ``*`` and ``?`` stop at ``/``, and only ``**`` crosses directories."""
    regex, i = "^", 0
    while i < len(pattern):
        char = pattern[i]
        if pattern.startswith("**/", i):
            regex, i = regex + "(.*/)?", i + 3
            continue
        if pattern.startswith("**", i):
            regex, i = regex + ".*", i + 2
            continue
        if char == "*":
            regex += "[^/]*"
        elif char == "?":
            regex += "[^/]"
        elif char == "[":
            close = pattern.index("]", i)
            regex, i = regex + pattern[i : close + 1], close
        else:
            regex += re.escape(char)
        i += 1
    return re.compile(regex + "$")


def dockerignore_excludes(path, lines=None):
    """Whether Docker leaves ``path`` (relative, ``/``-separated) out of the context.

    As in Docker: a pattern excludes a path when it matches the path or any of
    its parent directories, and the last matching pattern wins (``!`` re-includes).
    """
    lines = DOCKERIGNORE.splitlines() if lines is None else lines
    parts = path.split("/")
    candidates = ["/".join(parts[: n + 1]) for n in range(len(parts))]
    excluded = False
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        pattern = posixpath.normpath(line.lstrip("!").strip("/"))
        regex = _dockerignore_regex(pattern)
        if any(regex.match(candidate) for candidate in candidates):
            excluded = not negated
    return excluded


def test_the_matcher_is_root_anchored_like_docker() -> None:
    # Without this, the nested-path checks below could pass with a matcher
    # more permissive than Docker: the old patterns only covered the root.
    assert dockerignore_excludes("__pycache__/x.pyc", ["__pycache__/"])
    assert not dockerignore_excludes("src/pkg/__pycache__/x.pyc", ["__pycache__/"])
    assert not dockerignore_excludes("src/pkg.egg-info/PKG-INFO", ["*.egg-info/"])
    assert dockerignore_excludes("src/pkg.egg-info/PKG-INFO", ["**/*.egg-info/"])
    assert not dockerignore_excludes(".env.example", [".env.*", "!.env.example"])


@pytest.mark.parametrize(
    "path",
    [
        "src/f1_race_intelligence.egg-info/PKG-INFO",
        "src/f1_race_intelligence.egg-info/SOURCES.txt",
        "src/f1_race_intelligence/__pycache__/cli.cpython-312.pyc",
        "src/f1_race_intelligence/storage/backends/__pycache__/s3.cpython-312.pyc",
        "src/f1_race_intelligence/stray.pyc",
        "configs/__pycache__/anything.pyc",
        "__pycache__/root.cpython-312.pyc",
        "f1_race_intelligence.egg-info/PKG-INFO",
    ],
)
def test_nested_bytecode_and_egg_info_stay_out_of_the_build_context(path) -> None:
    assert dockerignore_excludes(path), path


@pytest.mark.parametrize(
    "path",
    [
        "src/f1_race_intelligence/__init__.py",
        "src/f1_race_intelligence/storage/backends/s3.py",
        "configs/config.yaml",
        "configs/config.aws.yaml",
        "requirements/runtime-lock.txt",
        "pyproject.toml",
        "README.md",
    ],
)
def test_what_the_image_copies_still_reaches_the_build_context(path) -> None:
    assert not dockerignore_excludes(path), path


def copied_sources():
    """Source paths of the Dockerfile's COPY instructions (every argument but the destination)."""
    # Every stage's COPY from the build context; ``COPY --from=<stage>`` copies from another stage, not from the context.
    return [
        source
        for arguments in instructions("COPY", DOCKERFILE)
        if not arguments.startswith("--from=")
        for source in arguments.split()[:-1]
    ]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_nothing_git_ignores_under_the_copied_paths_reaches_the_build_context() -> None:
    # Ignored files are invisible to the clean-tree guard, so the context must
    # drop them: whatever this checkout holds (editable-install egg-info,
    # bytecode written by the test run itself).
    listed = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "--", *copied_sources()],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if listed.returncode != 0:
        pytest.skip("not a git checkout")
    leaked = [path for path in listed.stdout.splitlines() if not dockerignore_excludes(path)]
    assert leaked == []


# -- log files: excluded at the root and at any depth to match .gitignore

LOG_FILE_CASES = [
    "debug.log",
    "src/debug.log",
    "configs/nested/server.log",
]


@pytest.mark.parametrize("path", LOG_FILE_CASES)
def test_log_files_stay_out_of_the_build_context_at_any_depth(path) -> None:
    assert dockerignore_excludes(path), path


def test_log_exclusions_are_kept_at_the_root_and_recursively() -> None:
    lines = {line.strip() for line in DOCKERIGNORE.splitlines()}
    assert {"*.log", "**/*.log"} <= lines


def test_dropping_recursive_log_pattern_leaks_nested_logs() -> None:
    lines = [line for line in DOCKERIGNORE.splitlines() if line.strip() != "**/*.log"]
    assert not dockerignore_excludes("src/debug.log", lines)
    assert not dockerignore_excludes("configs/nested/server.log", lines)
    assert dockerignore_excludes("debug.log", lines)


def test_dropping_both_log_patterns_leaks_root_and_nested_logs() -> None:
    lines = [line for line in DOCKERIGNORE.splitlines() if line.strip() not in {"*.log", "**/*.log"}]
    assert not dockerignore_excludes("debug.log", lines)
    assert not dockerignore_excludes("src/debug.log", lines)
    assert not dockerignore_excludes("configs/nested/server.log", lines)


# -- sensitive files: out of the build context at the root and at any depth
#
# Hypothetical paths only: these tests match names against the patterns and
# never create or read such files. Git ignores most of these names at any
# depth (.env, .aws/, *.pem, virtualenvs), so a nested one is invisible to the
# clean-tree guard; only .dockerignore keeps COPY src / COPY configs from
# baking it, or a committed .env.* / *.key, into the image.

NESTED_SENSITIVE_PATHS = [
    "configs/nested/.env",
    "configs/.env",
    "src/nested/.env.local",
    "src/nested/.aws/credentials",
    "configs/.aws/config",
    "src/pkg/cert.pem",
    "configs/tls/server.key",
    "src/nested/.venv",
    "src/nested/.venv/pyvenv.cfg",
    "src/nested/venv/bin/activate",
]
ROOT_SENSITIVE_PATHS = [
    ".env",
    ".env.local",
    ".aws/credentials",
    "cert.pem",
    "server.key",
    ".venv/pyvenv.cfg",
    "venv/bin/activate",
]
# The secret and environment patterns before they became recursive (4d7a4a9).
ROOT_ONLY_SENSITIVE_PATTERNS = [".env", ".env.*", "!.env.example", ".aws/", "*.pem", "*.key", ".venv/", "venv/"]
RECURSIVE_SENSITIVE_PATTERNS = ["**/.env", "**/.env.*", "**/.aws/", "**/*.pem", "**/*.key", "**/.venv/", "**/venv/"]


@pytest.mark.parametrize("path", NESTED_SENSITIVE_PATHS + ROOT_SENSITIVE_PATHS)
def test_sensitive_files_stay_out_of_the_build_context_at_any_depth(path) -> None:
    assert dockerignore_excludes(path), path


def test_the_root_exclusions_are_kept_next_to_the_recursive_ones() -> None:
    lines = {line.strip() for line in DOCKERIGNORE.splitlines()}
    assert set(ROOT_ONLY_SENSITIVE_PATTERNS) <= lines
    assert set(RECURSIVE_SENSITIVE_PATTERNS) <= lines


@pytest.mark.parametrize("path", NESTED_SENSITIVE_PATHS)
def test_the_previous_root_only_patterns_leak_nested_sensitive_files(path) -> None:
    # The defect fixed here: each nested case got through the old patterns...
    assert not dockerignore_excludes(path, ROOT_ONLY_SENSITIVE_PATTERNS), path
    # ...and the current file keeps it out.
    assert dockerignore_excludes(path), path


@pytest.mark.parametrize("path", ROOT_SENSITIVE_PATHS)
def test_the_previous_patterns_already_covered_the_root(path) -> None:
    assert dockerignore_excludes(path, ROOT_ONLY_SENSITIVE_PATTERNS), path


@pytest.mark.parametrize("pattern", RECURSIVE_SENSITIVE_PATTERNS)
def test_dropping_any_recursive_sensitive_pattern_leaks_a_nested_file(pattern) -> None:
    lines = [line for line in DOCKERIGNORE.splitlines() if line.strip() != pattern]
    leaked = [path for path in NESTED_SENSITIVE_PATHS if not dockerignore_excludes(path, lines)]
    assert leaked, f"without {pattern} nothing leaks: the nested cases do not exercise it"


# Keystores, SSH private keys and netrc credentials: per root pattern, a root
# case and a nested one (hypothetical paths, never created or read).
CREDENTIAL_FILE_CASES = {
    "*.p12": ("client.p12", "configs/certs/client.p12"),
    "*.pfx": ("client.pfx", "src/pkg/client.pfx"),
    "id_rsa": ("id_rsa", "configs/ssh/id_rsa"),
    "id_ed25519": ("id_ed25519", "src/nested/.ssh/id_ed25519"),
    "id_ecdsa": ("id_ecdsa", "configs/id_ecdsa"),
    ".netrc": (".netrc", "src/nested/.netrc"),
}
CREDENTIAL_PATHS = [path for cases in CREDENTIAL_FILE_CASES.values() for path in cases]


@pytest.mark.parametrize("path", CREDENTIAL_PATHS)
def test_credential_files_stay_out_of_the_build_context_at_any_depth(path) -> None:
    assert dockerignore_excludes(path), path


@pytest.mark.parametrize("pattern", CREDENTIAL_FILE_CASES)
def test_each_credential_pattern_is_kept_at_the_root_and_recursively(pattern) -> None:
    lines = {line.strip() for line in DOCKERIGNORE.splitlines()}
    assert {pattern, f"**/{pattern}"} <= lines


@pytest.mark.parametrize("pattern", CREDENTIAL_FILE_CASES)
def test_dropping_a_credential_pattern_leaks_exactly_its_own_cases(pattern) -> None:
    root_case, nested_case = CREDENTIAL_FILE_CASES[pattern]
    lines = DOCKERIGNORE.splitlines()

    def leaked(without):
        kept = [line for line in lines if line.strip() not in without]
        return {path for path in CREDENTIAL_PATHS if not dockerignore_excludes(path, kept)}

    # Without the recursive form only the nested case gets in (the root line still holds)...
    assert leaked({f"**/{pattern}"}) == {nested_case}
    # ...and without both, both cases do, and nothing that belongs to another pattern.
    assert leaked({pattern, f"**/{pattern}"}) == {root_case, nested_case}


def test_the_only_exception_reincludes_just_the_root_env_template() -> None:
    # The last matching pattern wins, so an exception after the sensitive
    # patterns could bring a secret back. The only one allowed is the root
    # template, which the image never copies.
    negations = [line.strip() for line in DOCKERIGNORE.splitlines() if line.strip().startswith("!")]
    assert negations == ["!.env.example"]
    assert not dockerignore_excludes(".env.example")
    assert dockerignore_excludes("configs/.env.example")
    assert dockerignore_excludes("src/nested/.env.example")


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_every_tracked_file_the_image_copies_still_reaches_the_build_context() -> None:
    # Recursive patterns must not over-exclude: everything versioned under the
    # COPY sources has to reach the build.
    listed = subprocess.run(["git", "ls-files", "--", *copied_sources()], cwd=REPO_ROOT, capture_output=True, text=True)
    if listed.returncode != 0:
        pytest.skip("not a git checkout")
    tracked = listed.stdout.splitlines()
    assert any(path.startswith("src/") for path in tracked) and "configs/config.yaml" in tracked
    assert [path for path in tracked if dockerignore_excludes(path)] == []


# -- traceable builds: the documented recipes record the commit they come from

# The image's architecture is the task definition's (X86_64); the commit is
# the full SHA that traceability.source_revision accepts (a short one is
# discarded); "false" may only be claimed behind the clean-tree guard.
TRACEABLE_BUILD_ARGS = (
    "--platform linux/amd64",
    '--build-arg F1_GIT_COMMIT="$(git rev-parse HEAD)"',
    "--build-arg F1_GIT_DIRTY=false",
)
# --untracked-files=all: git status honours status.showUntrackedFiles, and
# "no" would make a tree with new files look clean.
GUARD_COMMAND = "git status --porcelain --untracked-files=all"
CLEAN_TREE_GUARD = f'if [ -n "$({GUARD_COMMAND})" ]; then'
DOCKERFILE_GUARD = f'test -z "$({GUARD_COMMAND})" && docker build'
# Commands that produce or publish an image: only allowed in the clean branch.
IMAGE_COMMAND = re.compile(r"\bdocker\s+(?:buildx\s+)?(?:build|push|image\s+push|tag)\b")


def join_continuations(text):
    return re.sub(r"\\\n\s*", " ", text)


def documented_build_blocks():
    """``(source, bash block)`` of every documented ``docker build``, continuations joined."""
    blocks = []
    for source in ("README.md", "infra/README.md"):
        markdown = (REPO_ROOT / source).read_text(encoding="utf-8")
        for language, body in re.findall(r"^```(\w*)\n(.*?)^```", markdown, flags=re.S | re.M):
            if "docker build" in body:
                assert language == "bash", f"{source}: docker build documented in a ```{language} block"
                blocks.append((source, join_continuations(body)))
    return blocks


def dockerfile_build_comment():
    """The build command shown in the Dockerfile's header comment."""
    lines = [line.lstrip("#").strip() for line in DOCKERFILE.splitlines() if line.startswith("#")]
    (command,) = [line for line in lines if "docker build " in line]  # not "docker buildx"
    return command


def test_every_documented_build_records_the_full_commit_and_the_task_platform() -> None:
    blocks = documented_build_blocks()
    # Both READMEs document a build; neither may silently drop out of this check.
    assert {source for source, _ in blocks} == {"README.md", "infra/README.md"}

    commands = [(source, line) for source, block in blocks for line in block.splitlines() if "docker build" in line]
    commands.append(("Dockerfile", dockerfile_build_comment()))
    for source, command in commands:
        for argument in TRACEABLE_BUILD_ARGS:
            assert argument in command, f"{source}: {argument!r} missing from {command!r}"
        assert "--short" not in command, f"{source}: the image must record the full SHA"


def shell_lines(text):
    """Non-empty, non-comment lines of a (continuation-joined) shell fragment."""
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def guard_violations(block):
    """Why a documented build block could build or publish from a dirty tree (empty: it cannot).

    The shape required: only comments before the guard; a refusal branch that
    reports and fails; every image-producing command inside the clean branch;
    after ``fi``, nothing but ``docker run`` (the local smoke test).
    """
    if CLEAN_TREE_GUARD not in block:
        return ["no clean-tree guard"]
    guard = block.index(CLEAN_TREE_GUARD)
    otherwise = block.find("\nelse\n", guard)
    end = block.find("\nfi\n", otherwise)
    if otherwise < 0 or end < 0:
        return ["the guard has no else ... fi"]

    before, refusal, clean = block[:guard], block[guard + len(CLEAN_TREE_GUARD) : otherwise], block[otherwise:end]
    after = block[end + len("\nfi\n") :]
    problems = [f"before the guard: {line!r}" for line in shell_lines(before)]
    if "docker" in refusal or ">&2" not in refusal or shell_lines(refusal)[-1:] != ["false"]:
        problems.append("the dirty branch must only report to stderr and end in false")
    problems += [
        f"{match.group(0)!r} outside the clean branch"
        for match in IMAGE_COMMAND.finditer(block)
        if not otherwise < match.start() < end
    ]
    problems += [f"after fi: {line!r}" for line in shell_lines(after) if not line.startswith("docker run ")]
    if "docker build" not in clean:
        problems.append("no docker build in the clean branch")
    if "docker push" in clean and not re.search(r"-t \"\$REPO:\$TAG\" \. &&\s*docker push", clean):
        # A failed build must not push whatever image already has that tag.
        problems.append("docker push does not depend on the build succeeding")
    return problems


def test_documented_builds_and_pushes_only_run_from_a_clean_tree() -> None:
    for source, block in documented_build_blocks():
        assert guard_violations(block) == [], source

    assert dockerfile_build_comment().startswith(DOCKERFILE_GUARD)
    # The one-liner has a single build and nothing chained after it.
    assert len(IMAGE_COMMAND.findall(dockerfile_build_comment())) == 1


def publish_block():
    (block,) = [block for source, block in documented_build_blocks() if source == "infra/README.md"]
    return block


@pytest.mark.parametrize(
    "mutate, problem",
    [
        (lambda b: b + 'docker push "$REPO:$TAG"\n', "after fi"),
        (lambda b: b + "docker build -t other .\n", "after fi"),
        (lambda b: b + "docker tag f1:dev other:latest\n", "outside the clean branch"),
        (lambda b: b + "aws ecr batch-delete-image\n", "after fi"),
        (lambda b: "docker build -t early .\n" + b, "before the guard"),
        (lambda b: b.replace("  false\n", '  docker push "$REPO:$TAG"\n'), "dirty branch"),
        (lambda b: b.replace("-t \"$REPO:$TAG\" . &&", "-t \"$REPO:$TAG\" .;"), "depend on the build"),
    ],
    ids=["push-after-fi", "build-after-fi", "tag-after-fi", "other-command-after-fi", "build-before-guard", "push-when-dirty", "push-after-failed-build"],
)
def test_the_guard_check_rejects_unguarded_image_commands(mutate, problem) -> None:
    mutated = mutate(publish_block())
    assert any(problem in violation for violation in guard_violations(mutated)), guard_violations(mutated)


def test_the_guard_check_allows_a_smoke_run_after_the_build() -> None:
    assert guard_violations(publish_block() + 'docker run --rm "$REPO:$TAG" --help\n') == []


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_the_guard_sees_untracked_files_even_when_git_status_hides_them(tmp_path) -> None:
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True).stdout

    git("init", "-q")
    git("config", "status.showUntrackedFiles", "no")
    (tmp_path / "untracked.txt").write_text("new\n", encoding="utf-8")
    # The setting is in effect: the plain porcelain status calls this tree clean...
    assert git("status", "--porcelain") == ""
    # ...while the command the recipes run still reports the new file.
    assert "?? untracked.txt" in git(*shlex.split(GUARD_COMMAND)[1:])


def test_the_dockerfile_forwards_the_commit_build_args_to_the_runtime() -> None:
    assert {"F1_GIT_COMMIT", "F1_GIT_DIRTY"} <= set(instructions("ARG"))
    env = " ".join(instructions("ENV"))
    assert "F1_GIT_COMMIT=${F1_GIT_COMMIT}" in env and "F1_GIT_DIRTY=${F1_GIT_DIRTY}" in env
    assert DOCKERFILE.index("ARG F1_GIT_COMMIT") < DOCKERFILE.index("F1_GIT_COMMIT=${F1_GIT_COMMIT}")


def test_the_injected_values_are_what_the_m6_reports_record(monkeypatch) -> None:
    from f1_race_intelligence.modeling.traceability import source_revision

    full_sha = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setenv("F1_GIT_COMMIT", full_sha)
    monkeypatch.setenv("F1_GIT_DIRTY", "false")
    assert source_revision() == (full_sha, False)

    # Why the recipe uses the full SHA: a short one is not recorded at all.
    monkeypatch.setenv("F1_GIT_COMMIT", full_sha[:7])
    assert source_revision() == (None, None)


# -- CVE-2026-124851: no vulnerable zstd (1.5.1-1.5.7) in either image
#
# Two copies hid in the old image: the operating system's libzstd and the zstd
# 1.5.7 that every pyarrow wheel on PyPI links statically into libarrow. These
# tests pin the decisions that remove both; the images are verified by building
# them (the Dockerfiles also assert it at build time).

FORECAST_DOCKERFILE = (REPO_ROOT / "Dockerfile.forecast").read_text(encoding="utf-8")
PYARROW_BUILD_SCRIPT = (REPO_ROOT / "docker" / "build-pyarrow-without-zstd.sh").read_text(encoding="utf-8")
# SHA-256 of apache-arrow-25.0.1.tar.gz; its SHA-512 matches the one Apache publishes.
ARROW_25_0_1_SHA256 = "43d5de0a581f43cf63a2c06b4dcf13b9ff6fcd800f023324596e5781093bc500"


def test_the_batch_image_installs_the_fixed_debian_libzstd_and_checks_it() -> None:
    runs = " ".join(instructions("RUN"))
    # 1.5.7+dfsg-1 (what the base ships) is still vulnerable; the fix is the +deb13u1 build.
    assert "apt-get install --yes --no-install-recommends libzstd1=1.5.7+dfsg-1+deb13u1" in runs
    assert "dpkg-query --show --showformat='${Version}' libzstd1" in runs
    assert '= "1.5.7+dfsg-1+deb13u1"' in runs


def test_pyarrow_is_built_from_the_pinned_arrow_release_with_zstd_off() -> None:
    assert 'ARROW_VERSION="25.0.1"' in PYARROW_BUILD_SCRIPT
    assert f'ARROW_SHA256="{ARROW_25_0_1_SHA256}"' in PYARROW_BUILD_SCRIPT
    assert 'sha256sum --check' in PYARROW_BUILD_SCRIPT
    assert "-DARROW_WITH_ZSTD=OFF" in PYARROW_BUILD_SCRIPT
    assert "ARROW_WITH_ZSTD=ON" not in PYARROW_BUILD_SCRIPT
    # What the pipeline needs stays: Parquet with SNAPPY, and the bundled libraries in the wheel.
    for flag in ("-DARROW_PARQUET=ON", "-DARROW_WITH_SNAPPY=ON", "-DARROW_DATASET=ON", "-DARROW_COMPUTE=ON"):
        assert flag in PYARROW_BUILD_SCRIPT, flag
    assert "export PYARROW_BUNDLE_ARROW_CPP=ON" in PYARROW_BUILD_SCRIPT
    # Nothing that would pull zstd back in through another feature (ORC links it).
    for flag in ("-DARROW_ORC=ON", "-DARROW_FLIGHT=ON", "-DARROW_S3=ON"):
        assert flag not in PYARROW_BUILD_SCRIPT, flag


def test_the_pyarrow_build_toolchain_is_fully_pinned() -> None:
    lines = [
        line.strip()
        for line in (REPO_ROOT / "requirements" / "arrow-build-lock.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert lines and all(re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][^ ;]*", line) for line in lines), lines
    # Arrow 25 needs CMake >= 3.25; CMake 4 breaks the policies of some bundled dependencies.
    assert any(re.fullmatch(r"cmake==3\.(2[5-9]|3\d)\.\d+", line) for line in lines)
    # Built against the NumPy the image runs with.
    numpy = re.search(r"^numpy==(\S+)$", (REPO_ROOT / "requirements" / "runtime-lock.txt").read_text(encoding="utf-8"), flags=re.M)
    assert numpy and f"numpy=={numpy.group(1)}" in lines


def test_the_batch_build_fails_if_pyarrow_keeps_zstd_or_loses_snappy() -> None:
    runs = " ".join(instructions("RUN"))
    assert "assert not pa.Codec.is_available('zstd')" in runs
    assert "assert pa.Codec.is_available('snappy')" in runs
    assert "ZSTD_decompressStream" in runs and "--include='*.so*'" in runs
    # The wheel comes from the build stage, not from PyPI.
    assert "--from=pyarrow-build /wheelhouse/ /wheelhouse/" in instructions("COPY")
    assert "build-pyarrow-without-zstd.sh" in " ".join(instructions("RUN", BUILD_STAGE))


def forecast_stages():
    return [stage for stage in re.split(r"(?m)^(?=FROM\s)", FORECAST_DOCKERFILE) if stage.startswith("FROM")]


def test_the_forecast_image_builds_its_own_pyarrow_on_its_own_base() -> None:
    stages = forecast_stages()
    assert len(stages) == 2
    bases = [arguments.split()[0] for arguments in instructions("FROM", FORECAST_DOCKERFILE)]
    for base in bases:
        assert re.fullmatch(r"public\.ecr\.aws/lambda/python:3\.12@sha256:[0-9a-f]{64}", base), base
    assert len(set(bases)) == 1
    # The wheel is compiled on Amazon Linux 2023 itself; the Debian one is never reused (glibc/libstdc++ ABI).
    assert "trixie" not in " ".join(instructions("FROM", FORECAST_DOCKERFILE))
    build_runs = " ".join(instructions("RUN", stages[0]))
    assert "build-pyarrow-without-zstd.sh" in build_runs and "gcc-c++" in build_runs
    final_runs = " ".join(instructions("RUN", stages[-1]))
    assert "pip install --no-deps /wheelhouse/pyarrow-25.0.1-*.whl" in final_runs
    assert r"grep -v -E '^pyarrow==25\.0\.1[[:space:]]*$' requirements/runtime-lock.txt" in final_runs
    assert "--from=pyarrow-build" in " ".join(instructions("COPY", stages[-1]))


def test_the_forecast_build_removes_amazon_linuxs_libzstd_and_proves_no_zstd_is_left() -> None:
    # AL2023's repositories stop at libzstd 1.5.5 (no fixed package exists), and nothing the Lambda runs links it.
    final_runs = " ".join(instructions("RUN", forecast_stages()[-1]))
    assert "assert not pa.Codec.is_available('zstd')" in final_runs
    assert "assert pa.Codec.is_available('snappy')" in final_runs
    assert "rpm --erase --nodeps libzstd" in final_runs
    # The Lambda base has no ``find``: the file check walks the tree with Python.
    assert "f.startswith('libzstd')" in final_runs and "sys.exit(1 if hits else 0)" in final_runs
    # pyarrow built on this base links libatomic, absent from the Lambda image; it must be
    # installed while the package manager still works, i.e. before libzstd is removed.
    assert "microdnf install --assumeyes --setopt=install_weak_deps=0 libatomic" in final_runs
    assert final_runs.index("libatomic") < final_runs.index("rpm --erase")
    assert "'Src size is incorrect'" in final_runs
    # The removal is the last step that can run a package manager.
    assert final_runs.index("rpm --erase") > final_runs.index("pip install --no-deps .")
