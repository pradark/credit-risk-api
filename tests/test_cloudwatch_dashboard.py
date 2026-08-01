"""Tests for CloudWatch dashboard and alarm definitions."""

import json
from unittest.mock import MagicMock, Mock

import pytest

from app.monitoring.cloudwatch_dashboard import (
    DASHBOARD_NAME,
    build_alarm_definitions,
    build_dashboard_body,
    put_alarms,
    put_dashboard,
)


# ---------------------------------------------------------------------------
# build_dashboard_body
# ---------------------------------------------------------------------------

def test_build_dashboard_body_returns_valid_json() -> None:
    """Dashboard body must be valid JSON."""

    body = build_dashboard_body(
        model_version="v1",
        environment="test",
    )

    parsed = json.loads(body)

    assert "widgets" in parsed

    assert isinstance(
        parsed[
            "widgets"
        ],
        list,
    )

    assert len(
        parsed[
            "widgets"
        ]
    ) > 0


def test_build_dashboard_body_contains_auc_widget() -> None:
    """Dashboard must include an AUC trend widget."""

    body = build_dashboard_body()

    parsed = json.loads(body)

    titles = [
        w.get(
            "properties",
            {},
        ).get(
            "title",
            "",
        )
        for w in parsed[
            "widgets"
        ]
    ]

    assert any(
        "AUC" in t
        for t in titles
    )


def test_build_dashboard_body_contains_psi_widget() -> None:
    """Dashboard must include a PSI widget."""

    body = build_dashboard_body()

    parsed = json.loads(body)

    titles = [
        w.get(
            "properties",
            {},
        ).get(
            "title",
            "",
        )
        for w in parsed[
            "widgets"
        ]
    ]

    assert any(
        "PSI" in t
        for t in titles
    )


def test_build_dashboard_body_contains_error_widget() -> None:
    """Dashboard must include a prediction/explanation error widget."""

    body = build_dashboard_body()

    parsed = json.loads(body)

    titles = [
        w.get(
            "properties",
            {},
        ).get(
            "title",
            "",
        )
        for w in parsed[
            "widgets"
        ]
    ]

    assert any(
        "Error" in t
        for t in titles
    )


def test_build_dashboard_body_contains_pipeline_failure_widget() -> None:
    """Dashboard must include a pipeline failure widget."""

    body = build_dashboard_body()

    parsed = json.loads(body)

    titles = [
        w.get(
            "properties",
            {},
        ).get(
            "title",
            "",
        )
        for w in parsed[
            "widgets"
        ]
    ]

    assert any(
        "Pipeline" in t
        for t in titles
    )


def test_build_dashboard_body_does_not_contain_html() -> None:
    """Dashboard body must not reference HTML files."""

    body = build_dashboard_body()

    assert ".html" not in body.lower()
    assert "generate_html" not in body.lower()


# ---------------------------------------------------------------------------
# put_dashboard
# ---------------------------------------------------------------------------

def test_put_dashboard_calls_put_dashboard_api() -> None:
    """put_dashboard should call the CloudWatch API once."""

    client = Mock()

    result = put_dashboard(
        dashboard_name="TestDashboard",
        cloudwatch_client=client,
    )

    client.put_dashboard.assert_called_once()

    call_kwargs = client.put_dashboard.call_args.kwargs

    assert call_kwargs[
        "DashboardName"
    ] == "TestDashboard"

    assert "DashboardBody" in call_kwargs

    assert result[
        "dashboard_name"
    ] == "TestDashboard"


# ---------------------------------------------------------------------------
# build_alarm_definitions
# ---------------------------------------------------------------------------

def test_build_alarm_definitions_returns_list() -> None:
    """Alarm definitions must be a non-empty list."""

    alarms = build_alarm_definitions(
        model_version="v1",
        environment="test",
    )

    assert isinstance(
        alarms,
        list,
    )

    assert len(
        alarms
    ) > 0


def test_build_alarm_definitions_contains_auc_alarms() -> None:
    """Alarm list must contain AUC warning and alert alarms."""

    alarms = build_alarm_definitions(
        model_version="v1",
        environment="test",
    )

    alarm_names = [
        a[
            "AlarmName"
        ]
        for a in alarms
    ]

    assert any(
        "AUC" in name and "Warning" in name
        for name in alarm_names
    )

    assert any(
        "AUC" in name and "Alert" in name
        for name in alarm_names
    )


def test_build_alarm_definitions_contains_ks_alarms() -> None:
    """Alarm list must contain KS alarms."""

    alarms = build_alarm_definitions()

    alarm_names = [
        a[
            "AlarmName"
        ]
        for a in alarms
    ]

    assert any(
        "KS" in name
        for name in alarm_names
    )


def test_build_alarm_definitions_contains_ece_alarms() -> None:
    """Alarm list must contain ECE alarms."""

    alarms = build_alarm_definitions()

    alarm_names = [
        a[
            "AlarmName"
        ]
        for a in alarms
    ]

    assert any(
        "ECE" in name
        for name in alarm_names
    )


def test_build_alarm_definitions_contains_psi_alarms() -> None:
    """Alarm list must contain PSI alarms."""

    alarms = build_alarm_definitions()

    alarm_names = [
        a[
            "AlarmName"
        ]
        for a in alarms
    ]

    assert any(
        "PSI" in name
        for name in alarm_names
    )


def test_build_alarm_definitions_contains_pipeline_failure_alarm() -> None:
    """Alarm list must contain a pipeline failure alarm."""

    alarms = build_alarm_definitions()

    alarm_names = [
        a[
            "AlarmName"
        ]
        for a in alarms
    ]

    assert any(
        "Pipeline" in name
        for name in alarm_names
    )


def test_build_alarm_definitions_no_auto_actions_description() -> None:
    """No alarm description should suggest automatic model changes."""

    alarms = build_alarm_definitions()

    for alarm in alarms:
        description = alarm.get(
            "AlarmDescription",
            "",
        ).lower()

        assert "automatically retrain" not in description or \
               "do not automatically" in description

        assert "auto-deploy" not in description
        assert "auto-recalibrate" not in description


def test_build_alarm_definitions_with_sns_arn() -> None:
    """Alarms should include SNS ARN when provided."""

    alarms = build_alarm_definitions(
        sns_topic_arn="arn:aws:sns:us-east-1:123456789:alerts",
    )

    for alarm in alarms:
        if "AlarmActions" in alarm:
            assert (
                "arn:aws:sns:us-east-1:123456789:alerts"
                in alarm[
                    "AlarmActions"
                ]
            )


def test_build_alarm_definitions_without_sns_arn() -> None:
    """Without SNS ARN, AlarmActions should be empty."""

    alarms = build_alarm_definitions(
        sns_topic_arn=None,
    )

    for alarm in alarms:
        if "AlarmActions" in alarm:
            assert alarm[
                "AlarmActions"
            ] == []


# ---------------------------------------------------------------------------
# put_alarms
# ---------------------------------------------------------------------------

def test_put_alarms_creates_all_alarms() -> None:
    """put_alarms should call put_metric_alarm for each alarm."""

    client = Mock()

    result = put_alarms(
        model_version="v1",
        environment="test",
        cloudwatch_client=client,
    )

    expected_alarm_count = len(
        build_alarm_definitions(
            model_version="v1",
            environment="test",
        )
    )

    assert client.put_metric_alarm.call_count == expected_alarm_count

    assert result[
        "alarm_count"
    ] == expected_alarm_count

    assert len(
        result[
            "alarm_names"
        ]
    ) == expected_alarm_count


def test_put_alarms_does_not_pass_empty_alarm_actions() -> None:
    """AlarmActions should not be passed to API when empty."""

    client = Mock()

    put_alarms(
        model_version="v1",
        environment="test",
        sns_topic_arn=None,
        cloudwatch_client=client,
    )

    for call in client.put_metric_alarm.call_args_list:
        kwargs = call.kwargs

        if "AlarmActions" in kwargs:
            assert kwargs[
                "AlarmActions"
            ]  # Should not be empty if present
