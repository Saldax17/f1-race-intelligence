"""Make the infrastructure sources importable and parseable from tests.

The Lambda lives under ``infra/functions`` (it is deployed on its own, not as
part of the Python package), so its directory is put on ``sys.path`` here.
"""

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
INFRA = REPO_ROOT / "infra"

sys.path.insert(0, str(INFRA / "functions" / "availability_checker"))


class _CloudFormationLoader(yaml.SafeLoader):
    """SafeLoader that keeps intrinsic functions (!Ref, !Sub, ...) as ``{"Fn": value}``."""


def _intrinsic(loader: yaml.SafeLoader, tag_suffix: str, node: yaml.Node):
    name = "Ref" if tag_suffix == "Ref" else f"Fn::{tag_suffix}"
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    return {name: value}


_CloudFormationLoader.add_multi_constructor("!", _intrinsic)


@pytest.fixture(scope="session")
def template() -> dict:
    with (INFRA / "template.yaml").open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=_CloudFormationLoader)


@pytest.fixture(scope="session")
def asl_text() -> str:
    return (INFRA / "statemachine" / "pipeline.asl.json").read_text(encoding="utf-8")
