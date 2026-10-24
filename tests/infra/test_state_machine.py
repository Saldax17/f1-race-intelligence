"""The pipeline state machine definition: structure, branching and the task contract.

A small interpreter walks the definition with the exact substitutions
CloudFormation applies. ECS tasks, the plan read from S3 and the concurrency
check are simulated, so every branch -- and every command a task would run --
can be followed without AWS. The commands are then fed to the real CLI parser:
the state machine can only ask a task for something the image accepts.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pytest

from f1_race_intelligence import cli
from f1_race_intelligence.config.settings import DEFAULT_CONFIG_PATH, load_settings

SUBSTITUTIONS = {
    "Environment": "dev",
    "DataBucketName": "f1-race-intelligence-dev-123456789012-us-east-1",
    "AlertTopicArn": "arn:aws:sns:us-east-1:123456789012:f1-race-intelligence-dev-critical-alerts",
    "ClusterArn": "arn:aws:ecs:us-east-1:123456789012:cluster/f1-race-intelligence-dev-pipeline",
    "TaskDefinitionArn": "arn:aws:ecs:us-east-1:123456789012:task-definition/f1-race-intelligence-dev-pipeline:1",
    "SubnetA": "subnet-aaaaaaaaaaaaaaaaa",
    "SubnetB": "subnet-bbbbbbbbbbbbbbbbb",
    "TaskSecurityGroupId": "sg-0123456789abcdef0",
}
EXECUTION_NAME = "exec-1"
PIPELINE_STATES = ["M2_Extraction", "M3_Validation", "M4_Consolidation", "M5_Features", "M6_ModelingDataset"]
ECS_RUN_TASK_SYNC = "arn:aws:states:::ecs:runTask.sync"
SESSIONS = [
    {"year": 2023, "meeting_key": 1141, "session_key": 7953, "session_name": "Race"},
    {"year": 2024, "meeting_key": 1229, "session_key": 9472, "session_name": "Race"},
]


def definition(asl_text: str, pipeline_enabled: str = "false") -> Dict[str, Any]:
    values = {**SUBSTITUTIONS, "PipelineEnabled": pipeline_enabled}
    rendered = re.sub(r"\$\{(\w+)\}", lambda match: values[match.group(1)], asl_text)
    return json.loads(rendered)


# -- JSONPath and intrinsic functions (the subset the definition uses) --------------

_MISSING = object()


def get_path(document: Any, path: str, context: Optional[Dict[str, Any]] = None) -> Any:
    if path.startswith("$$"):
        document, path = context or {}, path[1:]
    assert path.startswith("$"), path
    value = document
    for name, index in re.findall(r"\.([A-Za-z_][\w]*)|\[(\d+)\]", path[1:]):
        if name:
            if not isinstance(value, dict) or name not in value:
                return _MISSING
            value = value[name]
        else:
            if not isinstance(value, list) or int(index) >= len(value):
                return _MISSING
            value = value[int(index)]
    return value


def evaluate(expression: str, data: Any, context: Dict[str, Any]) -> Any:
    """Evaluate a ``.$`` value: a path or an intrinsic function call."""
    value, rest = _parse(expression.strip(), data, context)
    assert not rest.strip(), f"unparsed: {rest!r}"
    return value


def _parse(text: str, data: Any, context: Dict[str, Any]) -> Tuple[Any, str]:
    if text.startswith("'"):
        end = text.index("'", 1)
        return text[1:end], text[end + 1 :]
    if text.startswith("$"):
        match = re.match(r"\$\$?(?:\.[A-Za-z_]\w*|\[\d+\])*", text)
        path = match.group(0)
        value = get_path(data, path, context)
        assert value is not _MISSING, f"path not found: {path}"
        return value, text[len(path) :]
    match = re.match(r"States\.(\w+)\(", text)
    assert match, f"unsupported expression: {text!r}"
    name, rest, args = match.group(1), text[match.end() :], []
    while True:
        rest = rest.lstrip()
        if rest.startswith(")"):
            rest = rest[1:]
            break
        value, rest = _parse(rest, data, context)
        args.append(value)
        rest = rest.lstrip()
        if rest.startswith(","):
            rest = rest[1:]
    if name == "Array":
        return list(args), rest
    if name == "Format":
        template, *values = args
        for value in values:
            template = template.replace("{}", json.dumps(value) if not isinstance(value, str) else value, 1)
        return template, rest
    if name == "ArrayLength":
        return len(args[0]), rest
    if name == "StringToJson":
        return json.loads(args[0]), rest
    if name == "JsonToString":
        return json.dumps(args[0]), rest
    raise NotImplementedError(name)


def render(template: Any, data: Any, context: Dict[str, Any]) -> Any:
    """Apply Parameters / ResultSelector / ItemSelector to ``data``."""
    if isinstance(template, dict):
        rendered = {}
        for key, value in template.items():
            if key.endswith(".$"):
                rendered[key[:-2]] = evaluate(value, data, context)
            else:
                rendered[key] = render(value, data, context)
        return rendered
    if isinstance(template, list):
        return [render(item, data, context) for item in template]
    return template


def set_path(data: Dict[str, Any], path: Optional[str], value: Any) -> Dict[str, Any]:
    if path is None:
        return data
    if path == "$":
        return value
    parts = [item for item in path[1:].split(".") if item]
    target = data
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value
    return data


def matches(rule: Dict[str, Any], data: Any) -> bool:
    value = get_path(data, rule["Variable"])
    if "IsBoolean" in rule:
        return isinstance(value, bool) is rule["IsBoolean"]
    if "BooleanEquals" in rule:
        return isinstance(value, bool) and value is rule["BooleanEquals"]
    if "StringEquals" in rule:
        return value == rule["StringEquals"]
    if "NumericEquals" in rule:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value == rule["NumericEquals"]
    if "NumericGreaterThan" in rule:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value > rule["NumericGreaterThan"]
    raise NotImplementedError(rule)


# -- the simulated world ------------------------------------------------------------


class TaskFailed(Exception):
    """What runTask.sync raises if Step Functions fails the task itself."""

    def __init__(self, error: str = "States.TaskFailed") -> None:
        super().__init__(error)
        self.error = error


class World:
    """Records the tasks the definition starts and answers the service calls."""

    def __init__(self, sessions=None, running: int = 1, exit_codes: Optional[Dict[str, int]] = None,
                 failures: Optional[Dict[str, str]] = None) -> None:
        self.sessions = SESSIONS if sessions is None else sessions
        self.running = running
        self.exit_codes = exit_codes or {}
        self.failures = failures or {}
        self.tasks: List[Tuple[str, Dict[str, Any]]] = []
        self.objects: Dict[str, str] = {}
        self.alerts: List[Dict[str, Any]] = []

    def call(self, name: str, state: Dict[str, Any], parameters: Dict[str, Any]) -> Any:
        resource = state["Resource"]
        if resource == ECS_RUN_TASK_SYNC:
            self.tasks.append((name, parameters))
            if name in self.failures:
                raise TaskFailed(self.failures[name])
            command = parameters["Overrides"]["ContainerOverrides"][0]["Command"]
            if command[:2] == ["--stage", "plan"]:  # the plan task writes the work units
                self.objects[command[command.index("--plan-output") + 1]] = json.dumps(self.sessions)
            return {"TaskArn": f"arn:aws:ecs:task/{len(self.tasks)}", "Containers": [{"ExitCode": self.exit_codes.get(name, 0)}]}
        if resource == "arn:aws:states:::aws-sdk:s3:getObject":
            return {"Body": self.objects[f"s3://{parameters['Bucket']}/{parameters['Key']}"]}
        if resource == "arn:aws:states:::aws-sdk:sfn:listExecutions":
            return {"Executions": [{"name": f"run-{i}"} for i in range(min(self.running, parameters["MaxResults"]))]}
        if resource == "arn:aws:states:::sns:publish":
            self.alerts.append(parameters)
            return {"MessageId": "m-1"}
        raise NotImplementedError(resource)


def run(states: Dict[str, Any], start: str, data: Any, world: World, context: Dict[str, Any]) -> Tuple[List[str], str, Any]:
    """Follow the definition; returns (visited states, terminal type, final data)."""
    visited: List[str] = []
    name = start
    while True:
        state = states[name]
        visited.append(name)
        kind = state["Type"]
        if kind == "Succeed":
            return visited, kind, data
        if kind == "Fail":
            return visited, kind, {"Error": state.get("Error"), "Cause": state.get("Cause")}
        if kind == "Pass":
            if "Parameters" in state:
                data = render(state["Parameters"], data, context)
            elif "Result" in state:
                data = set_path(data, state.get("ResultPath", "$"), state["Result"])
            name = state["Next"]
        elif kind == "Choice":
            name = next((rule["Next"] for rule in state["Choices"] if matches(rule, data)), state["Default"])
        elif kind == "Task":
            try:
                result = world.call(name, state, render(state["Parameters"], data, context))
            except TaskFailed as failure:
                catch = next((c for c in state.get("Catch", []) if set(c["ErrorEquals"]) & {failure.error, "States.ALL"}), None)
                if catch is None:  # no Catch here: the failure propagates (e.g. to the Map)
                    return visited, "Fail", {"Error": failure.error}
                data = set_path(data, catch["ResultPath"], {"Error": failure.error})
                name = catch["Next"]
                continue
            if "ResultSelector" in state:
                result = render(state["ResultSelector"], result, context)
            data = set_path(data, state.get("ResultPath", "$"), result)
            name = state["Next"]
        elif kind == "Map":
            processor = state["ItemProcessor"]
            failed = None
            for item in get_path(data, state["ItemsPath"]):
                item_context = {**context, "Map": {"Item": {"Value": item}}}
                inner, terminal, outcome = run(processor["States"], processor["StartAt"], render(state["ItemSelector"], data, item_context), world, item_context)
                visited.extend(inner)
                if terminal == "Fail":
                    failed = outcome
                    break
            if failed is not None:
                catch = state["Catch"][0]
                data = set_path(data, catch["ResultPath"], failed)
                name = catch["Next"]
                continue
            data = set_path(data, state.get("ResultPath", "$"), None)
            name = state["Next"]
        else:
            raise NotImplementedError(kind)


def execute(asl_text: str, payload: Any, pipeline_enabled: str = "true", world: Optional[World] = None):
    document = definition(asl_text, pipeline_enabled=pipeline_enabled)
    world = world or World()
    context = {"Execution": {"Name": EXECUTION_NAME, "Id": f"arn:aws:states:execution:{EXECUTION_NAME}"},
               "StateMachine": {"Id": "arn:aws:states:us-east-1:123456789012:stateMachine:f1-race-intelligence-dev-pipeline"}}
    visited, terminal, data = run(document["States"], document["StartAt"], payload, world, context)
    return visited, terminal, data, world


def availability(available: Any, reason=None) -> Dict[str, Any]:
    return {"available": available, "status_code": 200 if available else 401, "reason": reason, "source": "openf1", "checked_at": "2026-09-16T12:00:00+00:00"}


def all_states(document: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    states = dict(document["States"])
    for state in document["States"].values():
        if state["Type"] == "Map":
            states.update(state["ItemProcessor"]["States"])
    return states


def task_commands(world: World) -> List[Tuple[str, List[str]]]:
    return [(name, params["Overrides"]["ContainerOverrides"][0]["Command"]) for name, params in world.tasks]


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


def test_m2_to_m6_and_the_plan_are_fargate_tasks_that_wait_for_completion(asl_text, template) -> None:
    states = all_states(definition(asl_text))
    container = template["Resources"]["PipelineTaskDefinition"]["Properties"]["ContainerDefinitions"][0]["Name"]

    for name in ["PlanSessions", *PIPELINE_STATES]:
        state = states[name]
        assert (state["Type"], state["Resource"]) == ("Task", ECS_RUN_TASK_SYNC), name
        parameters = state["Parameters"]
        assert parameters["LaunchType"] == "FARGATE"
        assert parameters["Cluster"] == SUBSTITUTIONS["ClusterArn"]
        assert parameters["TaskDefinition"] == SUBSTITUTIONS["TaskDefinitionArn"]
        network = parameters["NetworkConfiguration"]["AwsvpcConfiguration"]
        assert network == {"Subnets": [SUBSTITUTIONS["SubnetA"], SUBSTITUTIONS["SubnetB"]],
                           "SecurityGroups": [SUBSTITUTIONS["TaskSecurityGroupId"]], "AssignPublicIp": "ENABLED"}
        assert parameters["Overrides"]["ContainerOverrides"][0]["Name"] == container
        assert 0 < state["TimeoutSeconds"] <= 1800, f"{name}: a hung task must not bill forever"


def test_every_task_retries_launch_failures_and_checks_its_exit_code(asl_text) -> None:
    states = all_states(definition(asl_text))

    for name in ["PlanSessions", *PIPELINE_STATES]:
        state = states[name]
        launch, run_failure = state["Retry"]
        # Launch problems (capacity, a non-empty RunTask Failures list) first:
        # States.TaskFailed matches every error and would swallow them.
        assert "AmazonECS.Unknown" in launch["ErrorEquals"], name
        assert run_failure["ErrorEquals"] == ["States.TaskFailed"], name
        assert state["ResultSelector"]["exit_code.$"] == "$.Containers[0].ExitCode"
        check = states[state["Next"]]
        assert check["Type"] == "Choice", f"{name}: whether a non-zero exit fails runTask.sync is not relied on"
        assert check["Choices"][0] == {"Variable": f"{state['ResultPath']}.exit_code", "NumericEquals": 0, "Next": check["Choices"][0]["Next"]}


def test_only_m2_retries_a_failed_run_more_than_once(asl_text) -> None:
    states = all_states(definition(asl_text))

    # M2 exits 1 on retryable OpenF1 failures; the other stages are
    # deterministic over S3 data, so one retry only covers transient issues.
    assert states["M2_Extraction"]["Retry"][1]["MaxAttempts"] == 2
    for name in ["PlanSessions", "M3_Validation", "M4_Consolidation", "M5_Features", "M6_ModelingDataset"]:
        assert states[name]["Retry"][1]["MaxAttempts"] == 1, name


def test_the_map_is_sequential_and_keeps_no_per_session_results(asl_text) -> None:
    pipeline = definition(asl_text)["States"]["PerSessionPipeline"]

    assert pipeline["MaxConcurrency"] == 1, "every M2 shares OpenF1's rate limit"
    assert pipeline["ResultPath"] is None, "results of every session would outgrow the 256 KB payload"
    assert list(pipeline["ItemProcessor"]["States"])[0] == "M2_Extraction"


def test_the_whole_execution_has_a_timeout(asl_text) -> None:
    assert 0 < definition(asl_text)["TimeoutSeconds"] <= 24 * 3600


# -- the contract with the image's CLI -----------------------------------------------


def test_every_command_the_state_machine_sends_is_accepted_by_the_cli(asl_text) -> None:
    _, terminal, _, world = execute(asl_text, availability(True))
    settings = load_settings(DEFAULT_CONFIG_PATH)

    assert terminal == "Succeed"
    for name, command in task_commands(world):
        assert all(isinstance(part, str) for part in command), f"{name}: ECS commands are strings"
        args = cli.build_parser().parse_args(command)
        stages = cli.parse_stages(args.stage)
        if args.plan_output:
            cli.parse_plan_destination(args.plan_output)
        if "m2" in stages and args.session_key is not None:
            assert args.year is not None, "m2 for one session needs --year"
        cli.check_stage_inputs(stages, settings, args.year)  # nothing to do = usage error


def test_stages_run_in_order_with_the_scope_of_each_session(asl_text) -> None:
    _, _, _, world = execute(asl_text, availability(True))
    commands = task_commands(world)

    expected = [("PlanSessions", ["--stage", "plan", "--plan-output", f"s3://{SUBSTITUTIONS['DataBucketName']}/manifests/plan/{EXECUTION_NAME}.json"])]
    for session in SESSIONS:
        year, meeting, key = str(session["year"]), str(session["meeting_key"]), str(session["session_key"])
        expected += [
            ("M2_Extraction", ["--stage", "m2", "--year", year, "--meeting-key", meeting, "--session-key", key]),
            ("M3_Validation", ["--stage", "m3", "--year", year, "--session-key", key]),
            ("M4_Consolidation", ["--stage", "m4", "--year", year, "--session-key", key]),
            ("M5_Features", ["--stage", "m5", "--year", year, "--session-key", key]),
        ]
    expected.append(("M6_ModelingDataset", ["--stage", "m6"]))
    assert commands == expected


def test_the_plan_is_read_back_from_where_the_plan_task_wrote_it(asl_text, template) -> None:
    states = definition(asl_text)["States"]
    _, _, data, world = execute(asl_text, availability(True))

    written = next(command for name, command in task_commands(world) if name == "PlanSessions")
    location = written[written.index("--plan-output") + 1]
    bucket, key = cli.parse_plan_destination(location)
    assert (bucket, key) == (SUBSTITUTIONS["DataBucketName"], f"manifests/plan/{EXECUTION_NAME}.json")
    assert states["LoadPlan"]["Parameters"]["Bucket"] == bucket
    # The task role may write it (manifests/*) and the state machine may read it.
    task_policy = json.dumps(template["Resources"]["DataProcessingTaskRole"]["Properties"]["Policies"])
    states_policy = json.dumps(template["Resources"]["PipelineStateMachineRole"]["Properties"]["Policies"])
    assert "/manifests/*" in task_policy and "/manifests/plan/*" in states_policy


# -- branching ------------------------------------------------------------------------


@pytest.mark.parametrize("reason", ["openf1_live_restriction", "rate_limited", "openf1_error", "network_error"])
def test_unavailable_ends_successfully_without_the_pipeline(asl_text, reason) -> None:
    visited, terminal, _, world = execute(asl_text, availability(False, reason))

    assert terminal == "Succeed"
    assert visited[-1] == "OpenF1Unavailable"
    assert world.tasks == []


def test_available_but_disabled_stops_before_any_task(asl_text) -> None:
    visited, terminal, _, world = execute(asl_text, availability(True), pipeline_enabled="false")

    assert (terminal, visited[-1]) == ("Succeed", "PipelineDisabled")
    assert world.tasks == [] and "CheckForConcurrentRun" not in visited


def test_available_and_enabled_runs_every_stage_of_every_session(asl_text) -> None:
    visited, terminal, _, world = execute(asl_text, availability(True))

    assert terminal == "Succeed" and visited[-1] == "PipelineSucceeded"
    assert visited[:6] == ["LoadContext", "IsAvailabilityResultValid", "IsOpenF1Available", "IsPipelineEnabled",
                           "CheckForConcurrentRun", "IsAnotherRunInProgress"]
    assert [name for name, _ in world.tasks].count("M2_Extraction") == len(SESSIONS)
    assert world.alerts == []


def test_a_concurrent_run_stops_before_starting_any_task(asl_text) -> None:
    visited, terminal, _, world = execute(asl_text, availability(True), world=World(running=2))

    assert (terminal, visited[-1]) == ("Succeed", "AnotherRunInProgress")
    assert world.tasks == []


def test_an_empty_plan_still_builds_the_modelling_dataset(asl_text) -> None:
    _, terminal, _, world = execute(asl_text, availability(True), world=World(sessions=[]))

    assert terminal == "Succeed"
    assert [name for name, _ in world.tasks] == ["PlanSessions", "M6_ModelingDataset"]


@pytest.mark.parametrize("stage", ["M2_Extraction", "M3_Validation", "M4_Consolidation", "M5_Features"])
def test_a_non_zero_exit_in_a_session_stops_the_run_and_alerts(asl_text, stage) -> None:
    visited, terminal, data, world = execute(asl_text, availability(True), world=World(exit_codes={stage: 1}))

    assert (terminal, visited[-1]) == ("Fail", "PipelineFailed")
    assert "NotifyPipelineFailure" in visited and len(world.alerts) == 1
    assert data["Error"] == "PipelineFailed"
    following = PIPELINE_STATES[PIPELINE_STATES.index(stage) + 1 :]
    assert not [name for name, _ in world.tasks if name in following]


@pytest.mark.parametrize("stage", ["PlanSessions", "M6_ModelingDataset"])
def test_a_non_zero_exit_in_plan_or_m6_alerts(asl_text, stage) -> None:
    visited, terminal, _, world = execute(asl_text, availability(True), world=World(exit_codes={stage: 2}))

    assert (terminal, visited[-1]) == ("Fail", "PipelineFailed")
    assert len(world.alerts) == 1
    assert json.loads(world.alerts[0]["Message"])["error"]["Error"] == "StageExitedNonZero"


@pytest.mark.parametrize("stage", ["PlanSessions", "M2_Extraction", "M6_ModelingDataset"])
def test_a_task_failed_by_step_functions_is_caught_and_alerted(asl_text, stage) -> None:
    visited, terminal, _, world = execute(asl_text, availability(True), world=World(failures={stage: "States.TaskFailed"}))

    assert (terminal, visited[-1]) == ("Fail", "PipelineFailed")
    assert len(world.alerts) == 1


@pytest.mark.parametrize("bad_input", [{}, {"available": "true"}, {"status_code": 200}])
def test_a_malformed_input_fails_instead_of_passing_as_unavailable(asl_text, bad_input) -> None:
    visited, terminal, _, world = execute(asl_text, bad_input)

    assert (terminal, visited[-1]) == ("Fail", "InvalidAvailabilityResult")
    assert world.tasks == []


def test_failures_in_the_pipeline_are_caught_and_alerted(asl_text) -> None:
    states = definition(asl_text)["States"]

    for name in ("CheckForConcurrentRun", "PlanSessions", "LoadPlan", "PerSessionPipeline", "M6_ModelingDataset"):
        assert states[name]["Catch"][0] == {"ErrorEquals": ["States.ALL"], "ResultPath": "$.error", "Next": "NotifyPipelineFailure"}, name
    assert states["NotifyPipelineFailure"]["Next"] == "PipelineFailed"


# Step Functions rejects Retry/Catch on any other state type (SCHEMA_VALIDATION_FAILED).
ERROR_HANDLING_STATE_TYPES = {"Task", "Map", "Parallel"}


def test_retry_and_catch_only_appear_where_step_functions_allows_them(asl_text) -> None:
    for name, state in all_states(definition(asl_text)).items():
        if state["Type"] not in ERROR_HANDLING_STATE_TYPES:
            assert "Catch" not in state, f"{name} ({state['Type']}) cannot have Catch"
            assert "Retry" not in state, f"{name} ({state['Type']}) cannot have Retry"
