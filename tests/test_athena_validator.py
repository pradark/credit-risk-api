"""Tests for Athena query execution and validation utilities."""

from unittest.mock import MagicMock, call

import pytest

from app.monitoring.athena_validator import (
    EXAMPLE_QUERIES,
    poll_query_status,
    run_latest_partition_check,
    run_query,
    run_row_count_check,
    start_query,
    validate_table_exists,
)


@pytest.fixture
def mock_athena_client():
    """Return a mock Athena client."""

    client = MagicMock()

    return client


def _make_execution_response(
    state: str,
    execution_id: str = "query-001",
    reason: str | None = None,
) -> dict:
    """Build a GetQueryExecution response dict."""

    status: dict = {"State": state}

    if reason:
        status[
            "StateChangeReason"
        ] = reason

    return {
        "QueryExecution": {
            "QueryExecutionId": execution_id,
            "Status": status,
        }
    }


# ---------------------------------------------------------------------------
# start_query
# ---------------------------------------------------------------------------

def test_start_query_returns_execution_id(
    mock_athena_client: MagicMock,
) -> None:
    """start_query should return the query execution ID."""

    mock_athena_client.start_query_execution.return_value = {
        "QueryExecutionId": "query-abc-123",
    }

    result = start_query(
        query="SELECT 1",
        database="test_db",
        output_location="s3://bucket/output/",
        workgroup="primary",
        athena_client=mock_athena_client,
    )

    assert result == "query-abc-123"

    mock_athena_client.start_query_execution.assert_called_once()


# ---------------------------------------------------------------------------
# poll_query_status
# ---------------------------------------------------------------------------

def test_poll_query_status_returns_on_success(
    mock_athena_client: MagicMock,
) -> None:
    """SUCCEEDED state should return immediately."""

    mock_athena_client.get_query_execution.return_value = (
        _make_execution_response(
            "SUCCEEDED",
            "query-001",
        )
    )

    result = poll_query_status(
        execution_id="query-001",
        poll_interval=0,
        athena_client=mock_athena_client,
    )

    assert (
        result[
            "QueryExecution"
        ][
            "Status"
        ][
            "State"
        ]
        == "SUCCEEDED"
    )


def test_poll_query_status_raises_on_failed(
    mock_athena_client: MagicMock,
) -> None:
    """FAILED state should raise RuntimeError."""

    mock_athena_client.get_query_execution.return_value = (
        _make_execution_response(
            "FAILED",
            reason="Table not found",
        )
    )

    with pytest.raises(
        RuntimeError,
        match="FAILED",
    ):
        poll_query_status(
            execution_id="query-001",
            poll_interval=0,
            athena_client=mock_athena_client,
        )


def test_poll_query_status_raises_on_cancelled(
    mock_athena_client: MagicMock,
) -> None:
    """CANCELLED state should raise RuntimeError."""

    mock_athena_client.get_query_execution.return_value = (
        _make_execution_response(
            "CANCELLED",
        )
    )

    with pytest.raises(
        RuntimeError,
        match="CANCELLED",
    ):
        poll_query_status(
            execution_id="query-001",
            poll_interval=0,
            athena_client=mock_athena_client,
        )


def test_poll_query_status_raises_on_timeout(
    mock_athena_client: MagicMock,
) -> None:
    """Unresolved query should raise TimeoutError after timeout."""

    mock_athena_client.get_query_execution.return_value = (
        _make_execution_response(
            "RUNNING",
        )
    )

    with pytest.raises(
        TimeoutError,
    ):
        poll_query_status(
            execution_id="query-001",
            poll_interval=0,
            timeout=0,
            athena_client=mock_athena_client,
        )


def test_poll_query_status_polls_until_complete(
    mock_athena_client: MagicMock,
) -> None:
    """Poller should keep polling until terminal state."""

    mock_athena_client.get_query_execution.side_effect = [
        _make_execution_response("RUNNING"),
        _make_execution_response("RUNNING"),
        _make_execution_response("SUCCEEDED"),
    ]

    result = poll_query_status(
        execution_id="q",
        poll_interval=0,
        timeout=60,
        athena_client=mock_athena_client,
    )

    assert (
        result[
            "QueryExecution"
        ][
            "Status"
        ][
            "State"
        ]
        == "SUCCEEDED"
    )

    assert mock_athena_client.get_query_execution.call_count == 3


# ---------------------------------------------------------------------------
# run_query
# ---------------------------------------------------------------------------

def test_run_query_combines_start_and_poll(
    mock_athena_client: MagicMock,
) -> None:
    """run_query should start a query and poll until completion."""

    mock_athena_client.start_query_execution.return_value = {
        "QueryExecutionId": "q-1",
    }

    mock_athena_client.get_query_execution.return_value = (
        _make_execution_response(
            "SUCCEEDED",
            "q-1",
        )
    )

    result = run_query(
        query="SELECT COUNT(*) FROM performance_metrics",
        athena_client=mock_athena_client,
    )

    assert (
        result[
            "QueryExecution"
        ][
            "QueryExecutionId"
        ]
        == "q-1"
    )


# ---------------------------------------------------------------------------
# validate_table_exists
# ---------------------------------------------------------------------------

def test_validate_table_exists_returns_true_on_success(
    mock_athena_client: MagicMock,
) -> None:
    """Successful query should indicate table exists."""

    mock_athena_client.start_query_execution.return_value = {
        "QueryExecutionId": "q-1",
    }

    mock_athena_client.get_query_execution.return_value = (
        _make_execution_response(
            "SUCCEEDED",
        )
    )

    result = validate_table_exists(
        table_name="performance_metrics",
        athena_client=mock_athena_client,
    )

    assert result[
        "exists"
    ] is True

    assert result[
        "table_name"
    ] == "performance_metrics"


def test_validate_table_exists_returns_false_on_failure(
    mock_athena_client: MagicMock,
) -> None:
    """Query failure should indicate table does not exist."""

    mock_athena_client.start_query_execution.side_effect = Exception(
        "Table not found"
    )

    result = validate_table_exists(
        table_name="missing_table",
        athena_client=mock_athena_client,
    )

    assert result[
        "exists"
    ] is False

    assert "error" in result


# ---------------------------------------------------------------------------
# run_row_count_check
# ---------------------------------------------------------------------------

def test_run_row_count_check_succeeds(
    mock_athena_client: MagicMock,
) -> None:
    """Row count check should return execution_id on success."""

    mock_athena_client.start_query_execution.return_value = {
        "QueryExecutionId": "q-count",
    }

    mock_athena_client.get_query_execution.return_value = (
        _make_execution_response(
            "SUCCEEDED",
            "q-count",
        )
    )

    result = run_row_count_check(
        table_name="performance_metrics",
        athena_client=mock_athena_client,
    )

    assert result[
        "status"
    ] == "succeeded"

    assert result[
        "execution_id"
    ] == "q-count"


# ---------------------------------------------------------------------------
# run_latest_partition_check
# ---------------------------------------------------------------------------

def test_run_latest_partition_check_succeeds(
    mock_athena_client: MagicMock,
) -> None:
    """Latest partition check should return execution_id on success."""

    mock_athena_client.start_query_execution.return_value = {
        "QueryExecutionId": "q-part",
    }

    mock_athena_client.get_query_execution.return_value = (
        _make_execution_response(
            "SUCCEEDED",
            "q-part",
        )
    )

    result = run_latest_partition_check(
        table_name="calibration_metrics",
        athena_client=mock_athena_client,
    )

    assert result[
        "status"
    ] == "succeeded"


# ---------------------------------------------------------------------------
# EXAMPLE_QUERIES
# ---------------------------------------------------------------------------

def test_example_queries_are_defined() -> None:
    """All expected example query keys should be present."""

    expected_keys = {
        "performance_count",
        "performance_trend",
        "calibration_by_band",
        "psi_latest",
        "segment_performance",
        "adverse_reason_frequency",
        "pipeline_history",
    }

    assert expected_keys.issubset(
        set(
            EXAMPLE_QUERIES
        )
    )


def test_example_queries_are_strings() -> None:
    """All example queries should be non-empty strings."""

    for key, query in EXAMPLE_QUERIES.items():
        assert isinstance(
            query,
            str,
        ), f"Query {key} is not a string"

        assert len(
            query.strip()
        ) > 0, f"Query {key} is empty"
