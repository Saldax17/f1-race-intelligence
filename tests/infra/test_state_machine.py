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
from f1_race_intelligence.scope import SessionScope

SUBSTITUTIONS = {
    "Environment": "dev",
    "DataBucketName": "f1-race-intelligence-dev-123456789012-us-east-1",
    "DefaultExtractionYears": "",
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


def walk_states(states: Dict[str, Any]):
    for state in states.values():
        yield state
        if state.get("Type") == "Map":
            yield from walk_states(state["ItemProcessor"]["States"])


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
    if "And" in rule:
        return all(matches(item, data) for item in rule["And"])
    value = get_path(data, rule["Variable"])
    if "IsPresent" in rule:
        return (value is not _MISSING) is rule["IsPresent"]
    if "IsString" in rule:
        return isinstance(value, str) is rule["IsString"]
    if "IsNumeric" in rule:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)) is rule["IsNumeric"]
    if value is _MISSING:  # every other comparison on an absent path is false
        return False
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
                 failures: Optional[Dict[str, str]] = None, results: Optional[Dict[str, Any]] = None) -> None:
        self.results = results or {}  # raw ECS results to return instead of a well-formed one
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
                self.objects[command[command.index("--plan-output") + 1]] = json.dumps(
                    {"sessions": self.sessions, "decisions": []}
                )
            if name in self.results:
                return self.results[name]
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
                rendered = render(state["Parameters"], data, context)
                data = set_path(data, state["ResultPath"], rendered) if "ResultPath" in state else rendered
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
        assert "ResultSelector" not in state, f"{name}: the ECS result must be stored whole, not dereferenced"
        check = states[state["Next"]]
        assert check["Type"] == "Choice", f"{name}: whether a non-zero exit fails runTask.sync is not relied on"
        variable = f"{state['ResultPath']}.Containers[0].ExitCode"
        assert check["Choices"][0]["And"] == [
            {"Variable": variable, "IsPresent": True},
            {"Variable": variable, "IsNumeric": True},
            {"Variable": variable, "NumericEquals": 0},
        ], name
        assert check["Default"].endswith("ResultMalformed"), name


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
            ("M3_Validation", ["--stage", "m3", "--year", year, "--meeting-key", meeting, "--session-key", key]),
            ("M4_Consolidation", ["--stage", "m4", "--year", year, "--meeting-key", meeting, "--session-key", key, "--overwrite-derived"]),
            ("M5_Features", ["--stage", "m5", "--year", year, "--meeting-key", meeting, "--session-key", key, "--overwrite-derived"]),
        ]
    expected.append(("M6_ModelingDataset", ["--stage", "m6", "--require-complete-history"]))
    assert commands == expected


PER_SESSION_STAGES = ["M2_Extraction", "M3_Validation", "M4_Consolidation", "M5_Features"]


def test_m2_to_m5_receive_year_meeting_and_session_of_the_same_session(asl_text) -> None:
    """The incremental provenance and completion records need the whole scope, so
    every per-session stage must be given the same year, meeting and session."""
    _, _, _, world = execute(asl_text, availability(True))
    per_session = [(name, command) for name, command in task_commands(world) if name in PER_SESSION_STAGES]
    assert len(per_session) == len(SESSIONS) * len(PER_SESSION_STAGES)

    for index, session in enumerate(SESSIONS):
        chunk = per_session[index * len(PER_SESSION_STAGES) : (index + 1) * len(PER_SESSION_STAGES)]
        assert [name for name, _ in chunk] == PER_SESSION_STAGES
        expected = SessionScope(year=session["year"], meeting_key=session["meeting_key"], session_key=session["session_key"])
        for name, command in chunk:
            args = cli.build_parser().parse_args(command)
            scope = SessionScope(year=args.year, meeting_key=args.meeting_key, session_key=args.session_key)
            assert scope == expected, f"{name} got {scope}, expected the scope of session {session['session_key']}"
            assert None not in (scope.year, scope.meeting_key, scope.session_key), f"{name} has an incomplete scope"


def test_the_effective_arguments_are_exactly_the_options_the_cli_defines(asl_text) -> None:
    """argparse accepts abbreviations (``--meeting`` for ``--meeting-key``) and would hide a
    typo, so every option the definition emits must be one the CLI registers verbatim."""
    _, _, _, world = execute(asl_text, availability(True))
    registered = set(cli.build_parser()._option_string_actions)  # noqa: SLF001 - exact option names

    for name, command in task_commands(world):
        options = [part for part in command if part.startswith("--")]
        assert options, name
        assert set(options) <= registered, f"{name}: {sorted(set(options) - registered)} are not CLI options"
        assert len(options) == len(set(options)), f"{name} repeats an option"

    effective = dict(task_commands(world)[1:5])  # the first session: M2..M5
    expected_flags = {
        "M2_Extraction": {"overwrite": False, "overwrite_derived": False, "strict": False},
        "M3_Validation": {"overwrite": False, "overwrite_derived": False, "strict": False},
        "M4_Consolidation": {"overwrite": False, "overwrite_derived": True, "strict": False},
        "M5_Features": {"overwrite": False, "overwrite_derived": True, "strict": False},
    }
    for name, flags in expected_flags.items():
        args = cli.build_parser().parse_args(effective[name])
        assert {key: getattr(args, key) for key in flags} == flags, name


def test_the_plan_is_read_back_from_where_the_plan_task_wrote_it(asl_text, template) -> None:
    states = definition(asl_text)["States"]
    _, _, data, world = execute(asl_text, availability(True))

    written = next(command for name, command in task_commands(world) if name == "PlanSessions")
    location = written[written.index("--plan-output") + 1]
    bucket, key = cli.parse_plan_destination(location)
    assert (bucket, key) == (SUBSTITUTIONS["DataBucketName"], f"manifests/plan/{EXECUTION_NAME}.json")
    assert states["LoadPlan"]["Parameters"]["Bucket"] == bucket
    # The task role may write the plan prefix; the state machine may read it.
    task_policy = json.dumps(template["Resources"]["DataProcessingTaskRole"]["Properties"]["Policies"])
    states_policy = json.dumps(template["Resources"]["PipelineStateMachineRole"]["Properties"]["Policies"])
    assert "/manifests/plan/*" in task_policy and "/manifests/plan/*" in states_policy
    assert "/manifests/control/" not in task_policy


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


def all_task_states(asl: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    states = dict(asl["States"])
    for state in asl["States"].values():
        if state.get("Type") == "Map":
            states.update(state["ItemProcessor"]["States"])
    return {name: state for name, state in states.items() if state.get("Resource") == ECS_RUN_TASK_SYNC}


def test_every_fargate_task_receives_the_pipeline_execution_id(asl_text) -> None:
    asl = json.loads(asl_text)
    tasks = [
        state
        for state in walk_states(asl["States"])
        if state.get("Resource") == ECS_RUN_TASK_SYNC
    ]

    assert len(tasks) == 6
    for name, task in ((n, s) for n, s in all_task_states(asl).items()):
        environment = task["Parameters"]["Overrides"]["ContainerOverrides"][0]["Environment"]
        expected = [{"Name": "F1_PIPELINE_EXECUTION_ID", "Value.$": "$$.Execution.Id"}]
        if name == "PlanSessions":  # only planning is scoped by season
            expected.append({"Name": "F1_EXTRACTION_YEARS", "Value.$": "$.scope.value"})
        assert environment == expected, name


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


# -- R2: backfill by season ------------------------------------------------------


def plan_environment(world: World) -> Dict[str, str]:
    _, params = next((n, p) for n, p in world.tasks if n == "PlanSessions")
    return {item["Name"]: item["Value"] for item in params["Overrides"]["ContainerOverrides"][0]["Environment"]}


def test_default_execution_passes_the_stack_default_seasons_to_the_plan_task(asl_text) -> None:
    visited, terminal, _, world = execute(asl_text, availability(True))

    assert terminal == "Succeed"
    assert "UseConfiguredYears" in visited and "UseRequestedYears" not in visited
    assert plan_environment(world)["F1_EXTRACTION_YEARS"] == ""  # empty keeps the configured seasons


def test_the_stack_parameter_is_the_default_scope(asl_text) -> None:
    substituted = asl_text.replace("${DefaultExtractionYears}", "2023,2024")
    _, _, _, world = execute(substituted.replace("${DefaultExtractionYears}", ""), availability(True))
    document = definition(substituted.replace("${Environment}", "dev"))  # still valid JSON
    assert document["States"]["UseConfiguredYears"]["Result"] == {"value": "2023,2024"}
    assert plan_environment(world)["F1_EXTRACTION_YEARS"] == "2023,2024"


@pytest.mark.parametrize("years", ["2023", "2023,2024"])
def test_requested_seasons_reach_the_plan_task_environment(asl_text, years) -> None:
    visited, terminal, _, world = execute(asl_text, {**availability(True), "extraction_years": years})

    assert terminal == "Succeed"
    assert "UseRequestedYears" in visited and "UseConfiguredYears" not in visited
    assert plan_environment(world) == {"F1_PIPELINE_EXECUTION_ID": f"arn:aws:states:execution:{EXECUTION_NAME}", "F1_EXTRACTION_YEARS": years}


@pytest.mark.parametrize("bad", [2023, ["2023"], None, True])
def test_a_non_string_scope_is_rejected_before_any_task(asl_text, bad) -> None:
    visited, terminal, data, world = execute(asl_text, {**availability(True), "extraction_years": bad})

    assert terminal == "Fail" and data["Error"] == "InvalidExtractionScope"
    assert world.tasks == []


def test_m6_exit_3_defers_modelling_without_reporting_a_complete_pipeline(asl_text) -> None:
    visited, terminal, _, world = execute(asl_text, availability(True), world=World(exit_codes={"M6_ModelingDataset": 3}))

    assert terminal == "Succeed"
    assert visited[-1] == "ModelingDeferredHistoryIncomplete" and "PipelineSucceeded" not in visited
    assert world.alerts == []  # a deferral is not a failure


def test_m6_exit_0_is_still_the_only_path_to_pipeline_succeeded(asl_text) -> None:
    visited, _, _, _ = execute(asl_text, availability(True))
    assert visited[-1] == "PipelineSucceeded"
    visited, terminal, _, world = execute(asl_text, availability(True), world=World(exit_codes={"M6_ModelingDataset": 1}))
    assert terminal == "Fail" and len(world.alerts) == 1


def test_the_m6_task_always_asks_for_the_history_check(asl_text) -> None:
    _, _, _, world = execute(asl_text, availability(True))
    command = dict(task_commands(world))["M6_ModelingDataset"]
    assert command == ["--stage", "m6", "--require-complete-history"]


# -- R3: a bad ECS result is a controlled failure, never States.Runtime ---------------------

MALFORMED_RESULTS = {
    "no Containers": {"TaskArn": "t"},
    "empty Containers": {"TaskArn": "t", "Containers": []},
    "container without ExitCode": {"TaskArn": "t", "Containers": [{"Name": "pipeline"}]},
    "null ExitCode": {"TaskArn": "t", "Containers": [{"ExitCode": None}]},
    "string ExitCode": {"TaskArn": "t", "Containers": [{"ExitCode": "0"}]},
    "boolean ExitCode": {"TaskArn": "t", "Containers": [{"ExitCode": False}]},
    "list ExitCode": {"TaskArn": "t", "Containers": [{"ExitCode": [0]}]},
    "Containers not a list": {"TaskArn": "t", "Containers": {"ExitCode": 0}},
    "empty result": {},
}
TASK_STATES = ["PlanSessions", "M2_Extraction", "M3_Validation", "M4_Consolidation", "M5_Features", "M6_ModelingDataset"]


@pytest.mark.parametrize("label", sorted(MALFORMED_RESULTS))
@pytest.mark.parametrize("stage", TASK_STATES)
def test_a_malformed_ecs_result_alerts_and_fails_instead_of_raising_runtime(asl_text, stage, label) -> None:
    world = World(results={stage: MALFORMED_RESULTS[label]})
    visited, terminal, data, world = execute(asl_text, availability(True), world=world)

    assert terminal == "Fail", (stage, label)
    assert "NotifyPipelineFailure" in visited and "PipelineSucceeded" not in visited
    assert len(world.alerts) == 1
    assert json.loads(world.alerts[0]["Message"])["error"]["Error"] == "StageResultMalformed", (stage, label)


@pytest.mark.parametrize("stage", TASK_STATES)
@pytest.mark.parametrize("code", [1, 2, 137, 255, -1])
def test_any_other_numeric_exit_code_is_a_failed_stage_with_one_alert(asl_text, stage, code) -> None:
    visited, terminal, data, world = execute(asl_text, availability(True), world=World(exit_codes={stage: code}))

    assert terminal == "Fail" and len(world.alerts) == 1
    assert json.loads(world.alerts[0]["Message"])["error"]["Error"] == "StageExitedNonZero"
    assert "PipelineSucceeded" not in visited


def test_a_numeric_zero_is_the_only_path_to_success(asl_text) -> None:
    visited, terminal, _, world = execute(asl_text, availability(True))
    assert terminal == "Succeed" and visited[-1] == "PipelineSucceeded" and world.alerts == []

    zero_float = {"TaskArn": "t", "Containers": [{"ExitCode": 0.0}]}
    world = World(results={name: zero_float for name in TASK_STATES if name != "PlanSessions"})
    visited, _, _, _ = execute(asl_text, availability(True), world=world)
    assert visited[-1] == "PipelineSucceeded"


def test_ecs_errors_raised_by_the_task_state_still_use_the_existing_catch(asl_text) -> None:
    for stage in ("PlanSessions", "M6_ModelingDataset"):
        visited, terminal, data, world = execute(asl_text, availability(True), world=World(failures={stage: "States.TaskFailed"}))
        assert terminal == "Fail" and len(world.alerts) == 1, stage
        assert json.loads(world.alerts[0]["Message"])["error"]["Error"] == "States.TaskFailed"
    # inside the Map the failure propagates to the Map's Catch
    visited, terminal, _, world = execute(asl_text, availability(True), world=World(failures={"M3_Validation": "States.TaskFailed"}))
    assert terminal == "Fail" and len(world.alerts) == 1


def test_m6_exit_3_stays_a_controlled_deferral_and_is_not_a_malformed_result(asl_text) -> None:
    visited, terminal, _, world = execute(asl_text, availability(True), world=World(exit_codes={"M6_ModelingDataset": 3}))
    assert terminal == "Succeed" and visited[-1] == "ModelingDeferredHistoryIncomplete"
    assert world.alerts == [] and "PipelineSucceeded" not in visited and "M6ExitedNonZero" not in visited
    # 3 only means "deferred" for M6: any other stage exiting 3 is a plain failure
    for stage in ("PlanSessions", "M2_Extraction", "M5_Features"):
        visited, terminal, _, world = execute(asl_text, availability(True), world=World(exit_codes={stage: 3}))
        assert terminal == "Fail" and len(world.alerts) == 1, stage
    # a deferral needs a numeric 3: "3" as a string or no container is malformed, not deferred
    for bad in ({"TaskArn": "t", "Containers": [{"ExitCode": "3"}]}, {"TaskArn": "t", "Containers": []}):
        visited, terminal, _, world = execute(asl_text, availability(True), world=World(results={"M6_ModelingDataset": bad}))
        assert terminal == "Fail" and "ModelingDeferredHistoryIncomplete" not in visited


def _strings(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key != "Comment":
                yield from _strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _strings(item)
    elif isinstance(node, str):
        yield node


def test_no_state_dereferences_the_ecs_exit_code_outside_a_choice(asl_text) -> None:
    for name, state in all_states(definition(asl_text)).items():
        if state["Type"] in ("Choice", "Map"):
            continue
        for text in _strings(state):
            if text.startswith("$") or "$." in text:  # a path or an intrinsic function over a path
                assert "Containers" not in text and "ExitCode" not in text, f"{name}: {text}"
        if state.get("Resource") == ECS_RUN_TASK_SYNC:
            assert "ResultSelector" not in state, name


def test_the_state_machine_is_structurally_sound_after_r3(asl_text) -> None:
    document = definition(asl_text)
    inner = document["States"]["PerSessionPipeline"]["ItemProcessor"]
    for states, start in ((document["States"], document["StartAt"]), (inner["States"], inner["StartAt"])):
        reachable, pending = set(), [start]
        while pending:
            name = pending.pop()
            if name in reachable:
                continue
            assert name in states, f"dangling transition to {name}"
            reachable.add(name)
            state = states[name]
            pending.extend(state[k] for k in ("Next", "Default") if k in state)
            pending.extend(rule["Next"] for rule in state.get("Choices", []))
            pending.extend(catch["Next"] for catch in state.get("Catch", []))
        assert reachable == set(states), f"unreachable states: {sorted(set(states) - reachable)}"
        for name, state in states.items():
            if state["Type"] in ("Succeed", "Fail"):
                assert not {"Next", "End"} & set(state), name
            elif state["Type"] != "Choice":
                assert ("Next" in state) != bool(state.get("End")), name
