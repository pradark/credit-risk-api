"""Tests for QuickSight asset definitions."""

import pytest

from app.monitoring.quicksight_assets import (
    DASHBOARD_SPECS,
    DATASET_QUERIES,
    build_all_dataset_inputs,
    build_athena_data_source_input,
    build_dataset_input,
    create_athena_data_source,
    get_dashboard_specifications,
)


# ---------------------------------------------------------------------------
# build_athena_data_source_input
# ---------------------------------------------------------------------------

def test_build_athena_data_source_input_structure() -> None:
    """Data source input should contain required QuickSight fields."""

    result = build_athena_data_source_input(
        data_source_id="test-source",
        database_name="test_db",
        workgroup="primary",
    )

    assert result[
        "DataSourceId"
    ] == "test-source"

    assert result[
        "Type"
    ] == "ATHENA"

    assert (
        "AthenaParameters"
        in result[
            "DataSourceParameters"
        ]
    )


# ---------------------------------------------------------------------------
# build_dataset_input
# ---------------------------------------------------------------------------

def test_build_dataset_input_structure() -> None:
    """Dataset input should contain DataSetId, Name, and ImportMode."""

    result = build_dataset_input(
        dataset_suffix="performance_metrics",
        sql_query="SELECT * FROM {database}.performance_metrics",
        database_name="test_db",
    )

    assert "DataSetId" in result
    assert "Name" in result

    assert result[
        "ImportMode"
    ] == "DIRECT_QUERY"

    assert "PhysicalTableMap" in result


def test_build_dataset_input_substitutes_database_name() -> None:
    """The {database} placeholder in SQL should be substituted."""

    result = build_dataset_input(
        dataset_suffix="psi_metrics",
        sql_query="SELECT * FROM {database}.psi_metrics",
        database_name="my_monitoring_db",
    )

    physical = list(
        result[
            "PhysicalTableMap"
        ].values()
    )[0]

    sql = physical[
        "CustomSql"
    ][
        "SqlQuery"
    ]

    assert "my_monitoring_db" in sql
    assert "{database}" not in sql


def test_build_dataset_input_with_spice_mode() -> None:
    """Import mode should be configurable."""

    result = build_dataset_input(
        dataset_suffix="performance_metrics",
        sql_query="SELECT * FROM {database}.performance_metrics",
        import_mode="SPICE",
    )

    assert result[
        "ImportMode"
    ] == "SPICE"


# ---------------------------------------------------------------------------
# build_all_dataset_inputs
# ---------------------------------------------------------------------------

def test_build_all_dataset_inputs_covers_all_tables() -> None:
    """Dataset inputs should be created for all six monitoring tables."""

    results = build_all_dataset_inputs(
        database_name="test_db",
    )

    assert len(
        results
    ) == len(
        DATASET_QUERIES
    )

    dataset_ids = {
        r[
            "DataSetId"
        ]
        for r in results
    }

    assert len(
        dataset_ids
    ) == len(
        results
    ), "Dataset IDs must be unique"


def test_build_all_dataset_inputs_no_html() -> None:
    """No dataset SQL query should reference HTML artifacts."""

    results = build_all_dataset_inputs()

    for dataset in results:
        physical_map = dataset.get(
            "PhysicalTableMap",
            {},
        )

        for table in physical_map.values():
            sql = (
                table.get(
                    "CustomSql",
                    {},
                )
                .get(
                    "SqlQuery",
                    "",
                )
                .lower()
            )

            assert ".html" not in sql
            assert "generate_html" not in sql


# ---------------------------------------------------------------------------
# DATASET_QUERIES
# ---------------------------------------------------------------------------

def test_dataset_queries_are_defined_for_all_tables() -> None:
    """DATASET_QUERIES must cover all six monitoring tables."""

    expected = {
        "performance_metrics",
        "calibration_metrics",
        "psi_metrics",
        "segment_metrics",
        "adverse_reason_summary",
        "pipeline_runs",
    }

    assert expected == set(
        DATASET_QUERIES
    )


# ---------------------------------------------------------------------------
# get_dashboard_specifications
# ---------------------------------------------------------------------------

def test_get_dashboard_specifications_returns_six_dashboards() -> None:
    """Spec catalogue should contain six dashboards."""

    specs = get_dashboard_specifications()

    assert len(
        specs
    ) == 6

    expected_keys = {
        "executive_model_health",
        "calibration",
        "drift",
        "segment_performance",
        "adverse_reasons",
        "governance",
    }

    assert expected_keys == set(
        specs
    )


def test_dashboard_specs_have_required_fields() -> None:
    """Each dashboard spec must contain name, description, and visuals."""

    specs = get_dashboard_specifications()

    for key, spec in specs.items():
        assert "name" in spec, f"{key} missing name"
        assert "description" in spec, f"{key} missing description"
        assert "visuals" in spec, f"{key} missing visuals"
        assert len(spec["visuals"]) > 0, f"{key} has empty visuals"


def test_dashboard_specs_reference_no_html() -> None:
    """No dashboard spec should reference HTML files."""

    specs = get_dashboard_specifications()

    for key, spec in specs.items():
        spec_str = str(spec).lower()

        assert (
            "generate_html_report" not in spec_str
        ), f"{key} references generate_html_report"

        assert (
            "monitoring_report.html" not in spec_str
        ), f"{key} references monitoring_report.html"


def test_governance_dashboard_includes_governance_note() -> None:
    """Governance dashboard spec should include a governance note."""

    specs = get_dashboard_specifications()

    gov_spec = specs[
        "governance"
    ]

    assert "governance_note" in gov_spec

    note = gov_spec[
        "governance_note"
    ].lower()

    assert "governance" in note


def test_calibration_dashboard_includes_governance_note() -> None:
    """Calibration spec should state no automatic recalibration."""

    specs = get_dashboard_specifications()

    cal_spec = specs[
        "calibration"
    ]

    assert "governance_note" in cal_spec


def test_adverse_reasons_dashboard_includes_compliance_note() -> None:
    """Adverse reasons spec should include a compliance note."""

    specs = get_dashboard_specifications()

    ar_spec = specs[
        "adverse_reasons"
    ]

    assert "compliance_note" in ar_spec


# ---------------------------------------------------------------------------
# create_athena_data_source
# ---------------------------------------------------------------------------

def test_create_athena_data_source_rejects_missing_account_id() -> None:
    """Missing AWS account ID must raise ValueError."""

    with pytest.raises(
        ValueError,
        match="AWS_ACCOUNT_ID",
    ):
        create_athena_data_source(
            account_id="",
        )


def test_create_athena_data_source_calls_api(monkeypatch) -> None:
    """create_athena_data_source should call QuickSight CreateDataSource."""

    from unittest.mock import Mock

    client = Mock()

    client.exceptions.ResourceExistsException = type(
        "ResourceExistsException",
        (Exception,),
        {},
    )

    client.create_data_source.return_value = {
        "Arn": "arn:aws:quicksight:...",
        "Status": 201,
    }

    result = create_athena_data_source(
        account_id="123456789012",
        data_source_id="test-source",
        quicksight_client=client,
    )

    client.create_data_source.assert_called_once()

    assert result[
        "status"
    ] == "created"


def test_create_athena_data_source_updates_when_exists() -> None:
    """Existing data source should be updated instead of recreated."""

    from unittest.mock import Mock

    client = Mock()

    ResourceExistsException = type(
        "ResourceExistsException",
        (Exception,),
        {},
    )

    client.exceptions.ResourceExistsException = (
        ResourceExistsException
    )

    client.create_data_source.side_effect = (
        ResourceExistsException()
    )

    result = create_athena_data_source(
        account_id="123456789012",
        data_source_id="existing-source",
        quicksight_client=client,
    )

    client.update_data_source.assert_called_once()

    assert result[
        "status"
    ] == "updated"
