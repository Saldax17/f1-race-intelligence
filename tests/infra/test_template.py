"""Static checks of infra/template.yaml: the properties reviewers must not lose.

These do not replace ``sam validate`` / ``cfn-lint``; they pin the decisions
(least privilege, bucket protection, gated pipeline) so a later edit that
weakens one fails loudly.
"""

import json
import re

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


def test_bucket_is_versioned_encrypted_private_retained_and_tagged(template) -> None:
    bucket = template["Resources"]["DataBucket"]
    properties = bucket["Properties"]

    assert bucket["DeletionPolicy"] == "Retain"
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

    assert s3_roles == {"DataProcessingTaskRole"}
    actions = {action for _, s in policy_statements(roles["DataProcessingTaskRole"]) for action in as_list(s["Action"])}
    assert not {action for action in actions if "Delete" in action}
    policies = json.dumps(roles["DataProcessingTaskRole"]["Properties"]["Policies"])
    assert "models/" not in policies, "models/ belongs to the future training role"


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
    assert "AWS::ECS::TaskDefinition" not in kinds, "batch jobs are deployed in a later step"
