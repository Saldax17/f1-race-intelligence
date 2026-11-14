"""Static checks of infra/template.yaml: the properties reviewers must not lose.

These do not replace ``sam validate`` / ``cfn-lint``; they pin the decisions
(least privilege, bucket protection, gated pipeline) so a later edit that
weakens one fails loudly.
"""

import json
import re
import shlex

import pytest

REQUIRED_PARAMETERS = {"Environment", "ProjectName", "S3BucketName", "AvailabilitySchedule", "LogRetentionDays"}
REQUIRED_OUTPUTS = {
    "DataBucketName",
    "AvailabilityCheckerFunctionArn",
    "PipelineStateMachineArn",
    "AvailabilityScheduleRuleName",
    "AlertTopicArn",
}


def resources_of(template, kind):
    return {name: body for name, body in template["Resources"].items() if body["Type"] == kind}


def policy_statements(role):
    for policy in role["Properties"].get("Policies", []):
        for statement in policy["PolicyDocument"]["Statement"]:
            yield policy["PolicyName"], statement


def as_list(value):
    return value if isinstance(value, list) else [value]


def test_required_parameters_and_outputs_exist(template) -> None:
    assert REQUIRED_PARAMETERS <= set(template["Parameters"])
    assert REQUIRED_OUTPUTS <= set(template["Outputs"])


def test_schedule_defaults_to_15_minutes_and_rejects_aggressive_rates(template) -> None:
    schedule = template["Parameters"]["AvailabilitySchedule"]

    assert schedule["Default"] == "rate(15 minutes)"
    for value in schedule["AllowedValues"]:
        minutes = re.fullmatch(r"rate\((\d+) minutes\)", value)
        assert not minutes or int(minutes.group(1)) >= 15, value


def test_pipeline_is_disabled_by_default(template) -> None:
    assert template["Parameters"]["PipelineEnabled"]["Default"] == "false"


# -- deploy-time configuration: the schedule must never come back on by accident


DEPLOY_ENVIRONMENTS = ["dev", "prod"]

# One parameter override as SAM CLI reads it: items are split on spaces, so a
# value may contain one only inside quotes ('Key="rate(15 minutes)"'). An
# unquoted space makes SAM silently keep the first word.
_OVERRIDE = re.compile(r"""(?:ParameterKey=)?(?P<key>[A-Za-z0-9]+)(?:=|,ParameterValue=)(?P<value>"[^"]*"|'[^']*'|\S*)""")


def parse_overrides(items):
    parameters = {}
    for item in items:
        match = _OVERRIDE.fullmatch(item.strip())
        assert match, f"SAM would split or misread this parameter override: {item!r}"
        value = match["value"]
        if value[:1] in {'"', "'"}:
            value = value[1:-1]
        parameters[match["key"]] = value
    return parameters


def samconfig_overrides(samconfig, environment):
    return parse_overrides(samconfig[environment]["deploy"]["parameters"]["parameter_overrides"])


# Line continuation per shell: a trailing backslash in Bash, a backtick in PowerShell.
_CONTINUATION = {"bash": "\\", "sh": "\\", "": "\\", "powershell": "`"}


def code_blocks(readme):
    """``(language, body)`` of every fenced block, so one language cannot hide another."""
    return re.findall(r"^```(\w*)\n(.*?)^```", readme, flags=re.S | re.M)


def documented_deploy_commands(readme):
    """``(language, argv)`` of every ``sam deploy`` command in the README, continuations joined."""
    commands = []
    for language, block in code_blocks(readme):
        assert language in _CONTINUATION, f"no deploy check for ```{language} blocks"
        for line in block.replace(_CONTINUATION[language] + "\n", " ").splitlines():
            line = line.strip()
            if line.startswith("sam deploy"):
                commands.append((language, shlex.split(line, comments=True)))
    return commands


def test_schedule_is_disabled_by_default(template) -> None:
    state = template["Parameters"]["AvailabilityScheduleState"]

    assert state["Default"] == "DISABLED"
    assert set(state["AllowedValues"]) == {"ENABLED", "DISABLED"}
    # The rule follows the parameter; a hard-coded state would bypass it.
    rule = template["Resources"]["AvailabilityScheduleRule"]["Properties"]
    assert rule["State"] == {"Ref": "AvailabilityScheduleState"}


@pytest.mark.parametrize("environment", DEPLOY_ENVIRONMENTS)
def test_samconfig_keeps_the_schedule_and_the_pipeline_off(samconfig, environment) -> None:
    overrides = samconfig_overrides(samconfig, environment)

    assert overrides["AvailabilityScheduleState"] == "DISABLED"
    assert overrides["PipelineEnabled"] == "false"


@pytest.mark.parametrize("environment", DEPLOY_ENVIRONMENTS)
def test_samconfig_overrides_are_valid_template_parameters(template, samconfig, environment) -> None:
    # Guards the "rate(15" truncation: every value must be one the template accepts.
    for key, value in samconfig_overrides(samconfig, environment).items():
        assert key in template["Parameters"], key
        allowed = template["Parameters"][key].get("AllowedValues")
        assert allowed is None or value in [str(item) for item in allowed], (key, value)


def test_samconfig_commits_no_alert_address(samconfig, samconfig_text) -> None:
    for environment in DEPLOY_ENVIRONMENTS:
        assert "AlertEmail" not in samconfig_overrides(samconfig, environment)
    assert "@" not in samconfig_text


def test_documented_deploys_cannot_turn_the_schedule_on(infra_readme, samconfig) -> None:
    commands = documented_deploy_commands(infra_readme)
    assert {language for language, _ in commands} >= {"bash", "powershell"}, "deploys documented for both shells"

    for _, command in commands:
        assert "--config-env" in command, command
        if "--parameter-overrides" not in command:
            continue  # parameters come from samconfig.toml, checked above
        environment = command[command.index("--config-env") + 1]
        start = command.index("--parameter-overrides") + 1
        overrides = parse_overrides([arg for arg in command[start:] if not arg.startswith("--")])

        # A command-line override replaces samconfig's list, and a parameter
        # left out reverts to the template default: all of them must be there.
        assert set(samconfig_overrides(samconfig, environment)) <= set(overrides), command
        assert overrides["AvailabilityScheduleState"] == "DISABLED", command
        assert overrides["PipelineEnabled"] == "false", command
        # The address is supplied at deploy time, never written in the docs.
        assert overrides.get("AlertEmail", "").startswith("$"), command


def test_powershell_deploy_refuses_to_run_without_an_address(infra_readme) -> None:
    for language, block in code_blocks(infra_readme):
        if language != "powershell" or "sam deploy" not in block:
            continue
        check = block.index("IsNullOrWhiteSpace($env:ALERT_EMAIL)")
        # SAM only runs in the else branch of that check, so a missing
        # address stops everything even when the block is pasted line by line.
        assert check < block.index("} else {") < block.index("sam build") < block.index("sam deploy")


# -- alert e-mail: optional, never invented, isolated from everything else


def test_alert_email_is_optional_and_validated(template) -> None:
    parameter = template["Parameters"]["AlertEmail"]
    pattern = re.compile(parameter["AllowedPattern"])

    assert parameter["Default"] == ""
    assert pattern.fullmatch("")
    assert pattern.fullmatch("alerts@example.org")
    assert not pattern.fullmatch("not-an-address")


def test_alert_subscription_exists_only_when_an_address_is_given(template) -> None:
    assert template["Conditions"]["HasAlertEmail"] == {"Fn::Not": [{"Fn::Equals": [{"Ref": "AlertEmail"}, ""]}]}

    subscriptions = resources_of(template, "AWS::SNS::Subscription")
    assert list(subscriptions) == ["AlertEmailSubscription"]
    subscription = subscriptions["AlertEmailSubscription"]
    assert subscription["Condition"] == "HasAlertEmail"
    assert subscription["Properties"] == {
        "TopicArn": {"Ref": "AlertTopic"},
        "Protocol": "email",
        "Endpoint": {"Ref": "AlertEmail"},
    }


def test_alert_email_reaches_nothing_but_its_subscription(template) -> None:
    # Roles, bucket, state machine and rules must not depend on the address.
    for name, body in template["Resources"].items():
        if name != "AlertEmailSubscription":
            assert '{"Ref": "AlertEmail"}' not in json.dumps(body), name


def test_bucket_is_versioned_encrypted_private_retained_and_tagged(template) -> None:
    bucket = template["Resources"]["DataBucket"]
    properties = bucket["Properties"]

    assert bucket["DeletionPolicy"] == "RetainExceptOnCreate"
    assert bucket["UpdateReplacePolicy"] == "Retain"
    assert properties["VersioningConfiguration"]["Status"] == "Enabled"
    assert properties["BucketEncryption"]["ServerSideEncryptionConfiguration"][0]["ServerSideEncryptionByDefault"]["SSEAlgorithm"]
    assert all(properties["PublicAccessBlockConfiguration"].values())
    rule_ids = {rule["Id"] for rule in properties["LifecycleConfiguration"]["Rules"]}
    assert {"abort-incomplete-multipart-uploads", "expire-noncurrent-versions"} <= rule_ids
    # No rule may ever expire current datasets.
    for rule in properties["LifecycleConfiguration"]["Rules"]:
        if "ExpirationInDays" in rule:
            assert rule["Prefix"] == "logs/"
    assert {tag["Key"] for tag in properties["Tags"]} >= {"Project", "Environment"}


def test_image_repository_is_retained_immutable_and_scanned(template) -> None:
    repository = template["Resources"]["PipelineImageRepository"]
    properties = repository["Properties"]

    assert repository["DeletionPolicy"] == "RetainExceptOnCreate"
    assert repository["UpdateReplacePolicy"] == "Retain"
    assert properties["ImageTagMutability"] == "IMMUTABLE"
    assert properties["ImageScanningConfiguration"]["ScanOnPush"] is True


def test_bucket_policy_denies_insecure_transport(template) -> None:
    statement = template["Resources"]["DataBucketPolicy"]["Properties"]["PolicyDocument"]["Statement"][0]

    assert statement["Effect"] == "Deny"
    assert statement["Condition"]["Bool"]["aws:SecureTransport"] == "false"


def test_no_s3_objects_or_data_are_created_by_the_stack(template) -> None:
    kinds = {body["Type"] for body in template["Resources"].values()}

    assert not any(kind.startswith("Custom::") or kind == "AWS::CloudFormation::CustomResource" for kind in kinds)


def test_roles_are_separated(template) -> None:
    roles = resources_of(template, "AWS::IAM::Role")
    principals = {name: role["Properties"]["AssumeRolePolicyDocument"]["Statement"][0]["Principal"]["Service"] for name, role in roles.items()}

    assert principals["AvailabilityCheckerRole"] == "lambda.amazonaws.com"
    assert principals["PipelineStateMachineRole"] == "states.amazonaws.com"
    assert principals["DataProcessingTaskRole"] == "ecs-tasks.amazonaws.com"
    assert template["Resources"]["AvailabilityCheckerFunction"]["Properties"]["Role"] == {"Fn::GetAtt": "AvailabilityCheckerRole.Arn"}
    assert template["Resources"]["PipelineStateMachine"]["Properties"]["Role"] == {"Fn::GetAtt": "PipelineStateMachineRole.Arn"}


def test_lambda_role_uses_the_standard_trust_and_the_others_keep_source_account(template) -> None:
    roles = resources_of(template, "AWS::IAM::Role")

    for name, role in roles.items():
        statements = role["Properties"]["AssumeRolePolicyDocument"]["Statement"]
        assert len(statements) == 1 and statements[0]["Action"] == "sts:AssumeRole", name
        if name == "AvailabilityCheckerRole":
            assert statements[0] == {"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}
        else:
            assert statements[0]["Condition"]["StringEquals"]["aws:SourceAccount"] == {"Ref": "AWS::AccountId"}, name


def test_no_administrator_access_or_wildcard_actions(template) -> None:
    text = json.dumps(template)
    assert "AdministratorAccess" not in text
    assert "PowerUserAccess" not in text

    for name, role in resources_of(template, "AWS::IAM::Role").items():
        assert "ManagedPolicyArns" not in role["Properties"], name
        for policy, statement in policy_statements(role):
            for action in as_list(statement["Action"]):
                assert "*" not in action, f"{name}/{policy}: {action}"


# Actions AWS does not allow to scope to a resource.
RESOURCE_STAR_ALLOWED = {
    "ecr:GetAuthorizationToken",
    "logs:CreateLogDelivery", "logs:GetLogDelivery", "logs:UpdateLogDelivery", "logs:DeleteLogDelivery",
    "logs:ListLogDeliveries", "logs:PutResourcePolicy", "logs:DescribeResourcePolicies", "logs:DescribeLogGroups",
}


def test_resource_star_is_only_used_where_aws_requires_it(template) -> None:
    for name, role in resources_of(template, "AWS::IAM::Role").items():
        for policy, statement in policy_statements(role):
            if "*" in as_list(statement["Resource"]):
                assert set(as_list(statement["Action"])) <= RESOURCE_STAR_ALLOWED, f"{name}/{policy}"


def test_s3_permissions_are_limited_to_the_project_bucket(template) -> None:
    for name, role in resources_of(template, "AWS::IAM::Role").items():
        for policy, statement in policy_statements(role):
            if any(action.startswith("s3:") for action in as_list(statement["Action"])):
                assert "DataBucket" in json.dumps(statement["Resource"]), f"{name}/{policy}"


def test_only_the_data_processing_role_touches_s3_and_it_cannot_delete(template) -> None:
    roles = resources_of(template, "AWS::IAM::Role")
    s3_roles = {
        name
        for name, role in roles.items()
        for _, statement in policy_statements(role)
        if any(action.startswith("s3:") for action in as_list(statement["Action"]))
    }

    assert s3_roles == {"DataProcessingTaskRole", "PipelineStateMachineRole"}
    actions = {action for _, s in policy_statements(roles["DataProcessingTaskRole"]) for action in as_list(s["Action"])}
    assert not {action for action in actions if "Delete" in action}
    policies = json.dumps(roles["DataProcessingTaskRole"]["Properties"]["Policies"])
    assert "models/" not in policies, "models/ belongs to the future training role"
    # The state machine only reads the session plan the plan task wrote.
    states_s3 = [s for _, s in policy_statements(roles["PipelineStateMachineRole"]) if any(a.startswith("s3:") for a in as_list(s["Action"]))]
    assert states_s3 == [{"Effect": "Allow", "Action": "s3:GetObject", "Resource": {"Fn::Sub": "${DataBucket.Arn}/manifests/plan/*"}}]


def test_log_groups_use_the_retention_parameter(template) -> None:
    groups = resources_of(template, "AWS::Logs::LogGroup")

    assert len(groups) >= 3
    for name, group in groups.items():
        assert group["Properties"]["RetentionInDays"] == {"Ref": "LogRetentionDays"}, name


def test_state_machine_is_standard_and_logs(template) -> None:
    properties = template["Resources"]["PipelineStateMachine"]["Properties"]

    assert properties["Type"] == "STANDARD"
    assert properties["Logging"]["Destinations"]


def test_availability_results_reach_the_state_machine_unmodified(template) -> None:
    function = template["Resources"]["AvailabilityCheckerFunction"]["Properties"]
    rule = template["Resources"]["AvailabilityResultRule"]["Properties"]

    assert function["EventInvokeConfig"]["DestinationConfig"]["OnSuccess"]["Type"] == "EventBridge"
    assert function["EventInvokeConfig"]["DestinationConfig"]["OnFailure"] == {"Type": "SNS", "Destination": {"Ref": "AlertTopic"}}
    assert rule["Targets"][0]["InputPath"] == "$.detail.responsePayload"
    assert rule["Targets"][0]["Arn"] == {"Ref": "PipelineStateMachine"}


@pytest.mark.parametrize("alarm", [
    "AvailabilityCheckerErrorsAlarm",
    "PipelineExecutionsFailedAlarm",
    "OrchestrationRulesFailedInvocationsAlarm",
])
def test_critical_alarms_notify_the_alert_topic(template, alarm) -> None:
    assert template["Resources"][alarm]["Properties"]["AlarmActions"] == [{"Ref": "AlertTopic"}]


def test_no_model_hosting_or_training_resources(template) -> None:
    kinds = {body["Type"] for body in template["Resources"].values()}

    assert not any(kind.startswith("AWS::SageMaker::") for kind in kinds)
    # The batch tasks (Phase 1B) are data processing only: one task definition.
    assert list(resources_of(template, "AWS::ECS::TaskDefinition")) == ["PipelineTaskDefinition"]


# -- Phase 1B: hardened Fargate tasks, their network and the state machine's rights


def task_definition(template):
    return template["Resources"]["PipelineTaskDefinition"]


def pipeline_container(template):
    (container,) = task_definition(template)["Properties"]["ContainerDefinitions"]
    return container


def test_the_task_runs_hardened_and_unprivileged(template) -> None:
    container = pipeline_container(template)

    assert container["User"] == "10001:10001", "the image's unprivileged f1 user, never root"
    assert container["ReadonlyRootFilesystem"] is True
    assert container["LinuxParameters"]["Capabilities"] == {"Drop": ["ALL"]}
    assert "Add" not in container["LinuxParameters"]["Capabilities"]
    assert container.get("Privileged") is not True
    assert "PortMappings" not in container, "the batch container serves nothing"
    # Verified locally: every stage runs with no writable path (and tmpfs is
    # not available on Fargate), so no volume is mounted.
    assert "MountPoints" not in container and "Volumes" not in task_definition(template)["Properties"]
    assert "Tmpfs" not in json.dumps(container)


def test_the_task_definition_is_fargate_x86_with_explicit_size_and_logs(template) -> None:
    properties = task_definition(template)["Properties"]
    container = pipeline_container(template)

    assert properties["RequiresCompatibilities"] == ["FARGATE"] and properties["NetworkMode"] == "awsvpc"
    assert properties["RuntimePlatform"] == {"CpuArchitecture": "X86_64", "OperatingSystemFamily": "LINUX"}
    assert (properties["Cpu"], properties["Memory"]) == ("512", "1024")
    assert properties["ExecutionRoleArn"] == {"Fn::GetAtt": "DataProcessingExecutionRole.Arn"}
    assert properties["TaskRoleArn"] == {"Fn::GetAtt": "DataProcessingTaskRole.Arn"}
    assert container["LogConfiguration"]["LogDriver"] == "awslogs"
    assert container["LogConfiguration"]["Options"]["awslogs-group"] == {"Ref": "DataProcessingLogGroup"}


def test_the_task_environment_selects_s3_and_holds_no_secret(template) -> None:
    environment = {item["Name"]: item["Value"] for item in pipeline_container(template)["Environment"]}

    assert environment["F1_ENVIRONMENT"] == "aws"
    assert environment["F1_CONFIG_PATH"] == "/app/configs/config.yaml"
    assert environment["F1_STORAGE_BUCKET"] == {"Ref": "DataBucket"}
    # The task role only grants the lake's root prefixes: no storage prefix.
    assert "F1_STORAGE_PREFIX" not in environment
    assert not [name for name in environment if re.search(r"KEY|SECRET|TOKEN|PASSWORD", name)]
    assert "Secrets" not in pipeline_container(template)


def test_the_image_is_pinned_by_digest_and_never_a_tag(template) -> None:
    parameter = template["Parameters"]["PipelineImageUri"]
    pattern = re.compile(parameter["AllowedPattern"])
    repository = "793264560810.dkr.ecr.us-east-1.amazonaws.com/f1-race-intelligence-dev-pipeline"

    assert pipeline_container(template)["Image"] == {"Ref": "PipelineImageUri"}
    assert parameter["Default"] == ""
    assert pattern.fullmatch("")
    assert pattern.fullmatch(f"{repository}@sha256:{'a' * 64}")
    for rejected in (f"{repository}:latest", f"{repository}:79798a4", f"{repository}@sha256:abc", "python:3.12", f"{repository}@sha256:{'A' * 64}"):
        assert not pattern.fullmatch(rejected), rejected


def test_without_an_image_no_task_definition_exists_and_the_pipeline_cannot_be_enabled(template) -> None:
    assert template["Conditions"]["HasPipelineImage"] == {"Fn::Not": [{"Fn::Equals": [{"Ref": "PipelineImageUri"}, ""]}]}
    assert task_definition(template)["Condition"] == "HasPipelineImage"

    rule = template["Rules"]["PipelineNeedsAnImage"]
    assert rule["RuleCondition"] == {"Fn::Equals": [{"Ref": "PipelineEnabled"}, "true"]}
    assert rule["Assertions"][0]["Assert"] == {"Fn::Not": [{"Fn::Equals": [{"Ref": "PipelineImageUri"}, ""]}]}


def test_the_pipeline_and_schedule_stay_off_in_the_template_and_every_environment(template, samconfig) -> None:
    assert template["Parameters"]["PipelineEnabled"]["Default"] == "false"
    assert template["Parameters"]["AvailabilityScheduleState"]["Default"] == "DISABLED"
    for environment in DEPLOY_ENVIRONMENTS:
        overrides = samconfig_overrides(samconfig, environment)
        assert (overrides["PipelineEnabled"], overrides["AvailabilityScheduleState"]) == ("false", "DISABLED"), environment
        # No image is committed: it is supplied, by digest, at deploy time.
        assert "PipelineImageUri" not in overrides


def test_the_tasks_accept_no_inbound_traffic_and_only_leave_over_https(template) -> None:
    group = template["Resources"]["PipelineTaskSecurityGroup"]["Properties"]

    assert "SecurityGroupIngress" not in group
    assert not resources_of(template, "AWS::EC2::SecurityGroupIngress")
    assert group["SecurityGroupEgress"] == [
        {"Description": "HTTPS to OpenF1 and AWS service endpoints", "IpProtocol": "tcp", "FromPort": 443, "ToPort": 443, "CidrIp": "0.0.0.0/0"}
    ]
    assert group["VpcId"] == {"Ref": "PipelineVpc"}


def test_the_network_has_no_hourly_billed_resources(template) -> None:
    kinds = {body["Type"] for body in template["Resources"].values()}

    # Outbound access is a public IP per running task through an Internet
    # Gateway: no NAT Gateway, interface endpoint or Elastic IP billing by the hour.
    assert not kinds & {"AWS::EC2::NatGateway", "AWS::EC2::VPCEndpoint", "AWS::EC2::EIP", "AWS::EC2::TransitGateway"}
    route = template["Resources"]["PipelineDefaultRoute"]["Properties"]
    assert route["GatewayId"] == {"Ref": "PipelineInternetGateway"} and route["DestinationCidrBlock"] == "0.0.0.0/0"
    for name in ("PipelineSubnetA", "PipelineSubnetB"):
        assert template["Resources"][name]["Properties"]["MapPublicIpOnLaunch"] is False, name


def test_the_cluster_does_not_enable_container_insights(template) -> None:
    settings = template["Resources"]["PipelineCluster"]["Properties"]["ClusterSettings"]

    assert settings == [{"Name": "containerInsights", "Value": "disabled"}]


def state_machine_statements(template):
    return {statement.get("Sid", policy): statement for policy, statement in policy_statements(template["Resources"]["PipelineStateMachineRole"])}


def test_the_state_machine_runs_only_the_pipeline_task_on_its_cluster(template) -> None:
    run_task = state_machine_statements(template)["RunPipelineTaskDefinitionOnTheClusterOnly"]

    assert run_task["Action"] == "ecs:RunTask"
    assert run_task["Resource"]["Fn::Sub"].endswith(":task-definition/${ProjectName}-${Environment}-pipeline:*")
    assert run_task["Condition"] == {"ArnEquals": {"ecs:cluster": {"Fn::GetAtt": "PipelineCluster.Arn"}}}


def test_the_state_machine_watches_only_the_cluster_tasks_and_the_managed_rule(template) -> None:
    statements = state_machine_statements(template)

    watch = statements["WatchAndStopPipelineTasks"]
    assert sorted(watch["Action"]) == ["ecs:DescribeTasks", "ecs:StopTask"]
    assert watch["Resource"]["Fn::Sub"].endswith(":task/${ProjectName}-${Environment}-pipeline/*")
    events = statements["TaskCompletionEvents"]
    assert sorted(events["Action"]) == ["events:DescribeRule", "events:PutRule", "events:PutTargets"]
    assert events["Resource"]["Fn::Sub"].endswith(":rule/StepFunctionsGetEventsForECSTaskRule")


def test_pass_role_is_limited_to_the_two_task_roles_and_to_ecs(template) -> None:
    pass_role = state_machine_statements(template)["PassOnlyThePipelineTaskRoles"]

    assert pass_role["Action"] == "iam:PassRole"
    assert pass_role["Resource"] == [{"Fn::GetAtt": "DataProcessingTaskRole.Arn"}, {"Fn::GetAtt": "DataProcessingExecutionRole.Arn"}]
    assert pass_role["Condition"] == {"StringEquals": {"iam:PassedToService": "ecs-tasks.amazonaws.com"}}
    passes = [s for _, role in resources_of(template, "AWS::IAM::Role").items() for _, s in policy_statements(role) if "iam:PassRole" in as_list(s["Action"])]
    assert passes == [pass_role]


def test_the_concurrency_check_reads_only_this_state_machine(template) -> None:
    statement = next(s for _, s in policy_statements(template["Resources"]["PipelineStateMachineRole"]) if s["Action"] == "states:ListExecutions")

    assert statement["Resource"]["Fn::Sub"].endswith(":stateMachine:${ProjectName}-${Environment}-pipeline")
    assert template["Resources"]["PipelineStateMachine"]["Properties"]["Name"] == {"Fn::Sub": "${ProjectName}-${Environment}-pipeline"}


def test_the_task_role_keeps_its_scoped_s3_access(template) -> None:
    statements = {s["Sid"]: s for _, s in policy_statements(template["Resources"]["DataProcessingTaskRole"])}

    # ListBucket stays bucket-wide on purpose (see the template): with an
    # s3:prefix condition, HeadObject on a missing key answers 403 instead of
    # 404 and the idempotency checks ("does this file exist?") break.
    assert statements["ListProjectBucket"] == {"Sid": "ListProjectBucket", "Effect": "Allow", "Action": "s3:ListBucket", "Resource": {"Fn::GetAtt": "DataBucket.Arn"}}
    assert statements["ReadWritePipelineObjects"]["Action"] == ["s3:GetObject", "s3:PutObject"]
    prefixes = {item["Fn::Sub"].split("/", 1)[1] for item in statements["ReadWritePipelineObjects"]["Resource"]}
    assert prefixes == {"raw/*", "validated/*", "consolidated/*", "features/*", "modeling/*", "manifests/*"}


def test_the_state_machine_is_wired_to_the_task_resources(template) -> None:
    substitutions = template["Resources"]["PipelineStateMachine"]["Properties"]["DefinitionSubstitutions"]

    assert substitutions["ClusterArn"] == {"Fn::GetAtt": "PipelineCluster.Arn"}
    assert substitutions["TaskDefinitionArn"] == {"Fn::If": ["HasPipelineImage", {"Ref": "PipelineTaskDefinition"}, "not-configured"]}
    assert (substitutions["SubnetA"], substitutions["SubnetB"]) == ({"Ref": "PipelineSubnetA"}, {"Ref": "PipelineSubnetB"})
    assert substitutions["TaskSecurityGroupId"] == {"Ref": "PipelineTaskSecurityGroup"}
