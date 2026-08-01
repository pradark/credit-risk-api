"""Tests for Glue Data Catalog definitions."""

from unittest.mock import MagicMock, Mock

import pytest

from app.monitoring.glue_catalog import (
    TABLE_REGISTRY,
    add_partition,
    build_table_input,
    create_all_tables,
    create_glue_database,
    create_or_update_table,
)


@pytest.fixture
def mock_glue_client():
    """Return a mock Glue client."""

    client = MagicMock()

    # Simulate EntityNotFoundException for "does not exist" cases
    client.exceptions.EntityNotFoundException = (
        type(
            "EntityNotFoundException",
            (Exception,),
            {},
        )
    )

    client.exceptions.AlreadyExistsException = (
        type(
            "AlreadyExistsException",
            (Exception,),
            {},
        )
    )

    return client


# ---------------------------------------------------------------------------
# create_glue_database
# ---------------------------------------------------------------------------

def test_create_glue_database_creates_when_not_exists(
    mock_glue_client: MagicMock,
) -> None:
    """Database should be created when it does not exist."""

    mock_glue_client.get_database.side_effect = (
        mock_glue_client.exceptions.EntityNotFoundException()
    )

    result = create_glue_database(
        database_name="test_db",
        glue_client=mock_glue_client,
    )

    assert result[
        "status"
    ] == "created"

    mock_glue_client.create_database.assert_called_once()


def test_create_glue_database_returns_existing(
    mock_glue_client: MagicMock,
) -> None:
    """When database already exists status should be already_exists."""

    mock_glue_client.get_database.return_value = {
        "Database": {"Name": "test_db"}
    }

    result = create_glue_database(
        database_name="test_db",
        glue_client=mock_glue_client,
    )

    assert result[
        "status"
    ] == "already_exists"

    mock_glue_client.create_database.assert_not_called()


# ---------------------------------------------------------------------------
# build_table_input
# ---------------------------------------------------------------------------

def test_build_table_input_structure() -> None:
    """Table input should include StorageDescriptor and PartitionKeys."""

    table_input = build_table_input(
        table_name="performance_metrics",
        columns=[
            ("auc", "double"),
            ("record_count", "int"),
        ],
        s3_location="s3://bucket/prefix/",
    )

    assert table_input[
        "Name"
    ] == "performance_metrics"

    sd = table_input[
        "StorageDescriptor"
    ]

    assert len(
        sd[
            "Columns"
        ]
    ) == 2

    assert sd[
        "Location"
    ] == "s3://bucket/prefix/"

    assert (
        "ParquetHiveSerDe"
        in sd[
            "SerdeInfo"
        ][
            "SerializationLibrary"
        ]
    )

    assert table_input[
        "TableType"
    ] == "EXTERNAL_TABLE"

    assert len(
        table_input[
            "PartitionKeys"
        ]
    ) == 2


def test_build_table_input_partition_keys() -> None:
    """Partition keys should be model_version and report_date."""

    table_input = build_table_input(
        table_name="t",
        columns=[],
        s3_location="s3://b/",
    )

    partition_names = [
        p[
            "Name"
        ]
        for p in table_input[
            "PartitionKeys"
        ]
    ]

    assert "model_version" in partition_names
    assert "report_date" in partition_names


# ---------------------------------------------------------------------------
# create_or_update_table
# ---------------------------------------------------------------------------

def test_create_or_update_table_creates_when_not_exists(
    mock_glue_client: MagicMock,
) -> None:
    """Table should be created when it does not exist."""

    mock_glue_client.get_table.side_effect = (
        mock_glue_client.exceptions.EntityNotFoundException()
    )

    result = create_or_update_table(
        table_name="performance_metrics",
        columns=[("auc", "double")],
        s3_location="s3://b/",
        glue_client=mock_glue_client,
    )

    assert result[
        "status"
    ] == "created"

    mock_glue_client.create_table.assert_called_once()


def test_create_or_update_table_updates_when_exists(
    mock_glue_client: MagicMock,
) -> None:
    """Table should be updated when it already exists."""

    mock_glue_client.get_table.return_value = {
        "Table": {"Name": "performance_metrics"}
    }

    result = create_or_update_table(
        table_name="performance_metrics",
        columns=[("auc", "double")],
        s3_location="s3://b/",
        glue_client=mock_glue_client,
    )

    assert result[
        "status"
    ] == "updated"

    mock_glue_client.update_table.assert_called_once()
    mock_glue_client.create_table.assert_not_called()


# ---------------------------------------------------------------------------
# add_partition
# ---------------------------------------------------------------------------

def test_add_partition_creates_new_partition(
    mock_glue_client: MagicMock,
) -> None:
    """A new partition should be registered in the Glue catalog."""

    mock_glue_client.batch_create_partition.return_value = {
        "Errors": []
    }

    result = add_partition(
        table_name="performance_metrics",
        model_version="v1",
        report_date="2026-08-01",
        glue_client=mock_glue_client,
    )

    assert result[
        "status"
    ] == "created"

    mock_glue_client.batch_create_partition.assert_called_once()


def test_add_partition_handles_existing_partition(
    mock_glue_client: MagicMock,
) -> None:
    """AlreadyExistsException should be handled gracefully."""

    mock_glue_client.batch_create_partition.side_effect = (
        mock_glue_client.exceptions.AlreadyExistsException()
    )

    result = add_partition(
        table_name="performance_metrics",
        model_version="v1",
        report_date="2026-08-01",
        glue_client=mock_glue_client,
    )

    assert result[
        "status"
    ] == "already_exists"


# ---------------------------------------------------------------------------
# create_all_tables
# ---------------------------------------------------------------------------

def test_create_all_tables_creates_all_six_tables(
    mock_glue_client: MagicMock,
) -> None:
    """All six monitoring tables should be created."""

    mock_glue_client.get_database.side_effect = (
        mock_glue_client.exceptions.EntityNotFoundException()
    )

    mock_glue_client.get_table.side_effect = (
        mock_glue_client.exceptions.EntityNotFoundException()
    )

    result = create_all_tables(
        database_name="test_db",
        bucket="test-bucket",
        analytics_prefix="analytics",
        glue_client=mock_glue_client,
    )

    assert result[
        "table_count"
    ] == 6

    assert set(
        result[
            "table_results"
        ]
    ) == {
        "performance_metrics",
        "calibration_metrics",
        "psi_metrics",
        "segment_metrics",
        "adverse_reason_summary",
        "pipeline_runs",
    }


def test_table_registry_covers_all_expected_tables() -> None:
    """The table registry should contain all six expected tables."""

    assert set(
        TABLE_REGISTRY
    ) == {
        "performance_metrics",
        "calibration_metrics",
        "psi_metrics",
        "segment_metrics",
        "adverse_reason_summary",
        "pipeline_runs",
    }
