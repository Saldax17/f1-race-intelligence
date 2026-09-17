"""The pipeline state machine definition: structure and branching.

A tiny interpreter walks the Pass/Choice/Succeed/Fail/Map states with the
exact substitutions CloudFormation applies, so each branch can be followed
for a given availability result without AWS.
"""

import json
import re
from typing import Any, Dict, List, Tuple

import pytest

SUBSTITUTIONS = {
    "Environment": "dev",
    "DataBucketName": "f1-race-intelligence-dev-123456789012-us-east-1",
    "AlertTopicArn": "arn:aws:sns:us-east-1:123456789012:f1-race-intelligence-dev-critical-alerts",
}
PIPELINE_STATES = ["M2_Extraction", "M3_Validation", "M4_Consolidation", "M5_Features", "M6_ModelingDataset"]


def definition(asl_text: str, pipeline_enabled: str = "false") -> Dict[str, Any]:
    values = {**SUBSTITUTIONS, "PipelineEnabled": pipeline_enabled}
    rendered = re.sub(r"\$\{(\w+)\}", lambda match: values[match.group(1)], asl_text)
    return json.loads(rendered)


def get_path(document: Any, path: str) -> Any:
    assert path.startswith("$")
    value = document
    for part in [item for item in path[1:].split(".") if item]:
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


_MISSING = object()


def matches(rule: Dict[str, Any], data: Any) -> bool:
    value = get_path(data, rule["Variable"])
    if "IsBoolean" in rule:
        return isinstance(value, bool) is rule["IsBoolean"]
    if "BooleanEquals" in rule:
        return isinstance(value, bool) and value is rule["BooleanEquals"]
    if "StringEquals" in rule:
        return value == rule["StringEquals"]
    raise NotImplementedError(rule)


def set_path(data: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
    parts = [item for item in path[1:].split(".") if item]
    target = data
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value
    return data


def run(states: Dict[str, Any], start: str, data: Any) -> Tuple[List[str], str, Any]:
    """Follow the definition; returns (visited states, terminal type, final data)."""
    visited: List[str] = []
    name = start
    while True:
        state = states[name]
        visited.append(name)
        kind = state["Type"]
        if kind in ("Succeed", "Fail"):
            return visited, kind, data
        if kind == "Pass":
            if "Parameters" in state:
                data = {
                    key.removesuffix(".$"): (get_path(data, value) if key.endswith(".$") and value.startswith("$.") or value == "$" else ("execution-id" if key.endswith(".$") else value))
                    for key, value in state["Parameters"].items()
                }
            elif "Result" in state:
                data = set_path(data, state["ResultPath"], state["Result"])
            name = state["Next"]
        elif kind == "Choice":
            name = next((rule["Next"] for rule in state["Choices"] if matches(rule, data)), state["Default"])
        elif kind == "Map":
            items = get_path(data, state["ItemsPath"])
            processor = state["ItemProcessor"]
            for item in items:
                inner, terminal, _ = run(processor["States"], processor["StartAt"], {"session": item})
                visited.extend(inner)
            name = state["Next"]
        else:
            raise NotImplementedError(kind)


def availability(available: Any, reason=None) -> Dict[str, Any]:
    return {"available": available, "status_code": 200 if available else 401, "reason": reason, "source": "openf1", "checked_at": "2026-09-16T12:00:00+00:00"}


# -- structure --------------------------------------------------------------------


def test_every_placeholder_is_rendered_by_the_template_substitutions(asl_text, template) -> None:
    placeholders = set(re.findall(r"\$\{(\w+)\}", asl_text))
    substitutions = template["Resources"]["PipelineStateMachine"]["Properties"]["DefinitionSubstitutions"]

    assert placeholders == set(substitutions)
    definition(asl_text)  # valid JSON once rendered


def test_the_definition_is_valid_json_even_before_substitution(asl_text) -> None:
    json.loads(asl_text)


def test_every_transition_points_at_an_existing_state(asl_text) -> None:
    def check(states: Dict[str, Any]) -> None:
        for name, state in states.items():
            targets = [state.get("Next"), state.get("Default")]
            targets += [rule["Next"] for rule in state.get("Choices", [])]
            targets += [catch["Next"] for catch in state.get("Catch", [])]
            for target in filter(None, targets):
                assert target in states, f"{name} -> {target}"
            if state["Type"] == "Map":
                check(state["ItemProcessor"]["States"])

    document = definition(asl_text)
    assert document["StartAt"] in document["States"]
    check(document["States"])


def test_m2_to_m6_are_clearly_identified_and_are_placeholders_only(asl_text) -> None:
    document = definition(asl_text)
    per_session = document["States"]["PerSessionPipeline"]["ItemProcessor"]["States"]
    all_states = {**document["States"], **per_session}

    for name in PIPELINE_STATES:
        assert name in all_states
        assert all_states[name]["Type"] == "Pass", "no job runs until the batch jobs are deployed"
        assert "PLACEHOLDER" in all_states[name]["Comment"]
    assert [s for s in per_session] == PIPELINE_STATES[:4]
    assert not any(state.get("Resource", "").startswith("arn:aws:states:::ecs") for state in all_states.values())


def test_the_only_service_integration_is_the_alert(asl_text) -> None:
    resources = [state["Resource"] for state in definition(asl_text)["States"].values() if "Resource" in state]

    assert resources == ["arn:aws:states:::sns:publish"]


# -- branching ------------------------------------------------------------------------


@pytest.mark.parametrize("reason", ["openf1_live_restriction", "rate_limited", "openf1_error", "network_error"])
def test_unavailable_ends_successfully_without_the_pipeline(asl_text, reason) -> None:
    document = definition(asl_text, pipeline_enabled="true")

    visited, terminal, _ = run(document["States"], document["StartAt"], availability(False, reason))

    assert terminal == "Succeed"
    assert visited[-1] == "OpenF1Unavailable"
    assert not set(visited) & set(PIPELINE_STATES)


def test_available_but_disabled_stops_before_the_pipeline(asl_text) -> None:
    document = definition(asl_text, pipeline_enabled="false")

    visited, terminal, _ = run(document["States"], document["StartAt"], availability(True))

    assert (terminal, visited[-1]) == ("Succeed", "PipelineDisabled")
    assert not set(visited) & set(PIPELINE_STATES)


def test_available_and_enabled_walks_the_prepared_structure_doing_nothing(asl_text) -> None:
    document = definition(asl_text, pipeline_enabled="true")

    visited, terminal, data = run(document["States"], document["StartAt"], availability(True))

    assert terminal == "Succeed"
    assert visited == ["LoadContext", "IsAvailabilityResultValid", "IsOpenF1Available", "IsPipelineEnabled",
                       "PlanSessions", "PerSessionPipeline", "M6_ModelingDataset", "PipelineSucceeded"]
    assert data["plan"]["sessions"] == []


@pytest.mark.parametrize("bad_input", [{}, {"available": "true"}, {"status_code": 200}])
def test_a_malformed_input_fails_instead_of_passing_as_unavailable(asl_text, bad_input) -> None:
    document = definition(asl_text)

    visited, terminal, _ = run(document["States"], document["StartAt"], bad_input)

    assert (terminal, visited[-1]) == ("Fail", "InvalidAvailabilityResult")


def test_failures_in_the_pipeline_are_caught_and_alerted(asl_text) -> None:
    states = definition(asl_text)["States"]

    for name in ("PlanSessions", "PerSessionPipeline", "M6_ModelingDataset"):
        assert states[name]["Catch"][0]["Next"] == "NotifyPipelineFailure"
    assert states["NotifyPipelineFailure"]["Next"] == "PipelineFailed"
    assert states["PerSessionPipeline"]["MaxConcurrency"] == 1
