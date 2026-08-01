"""Tests confirming HTML is not present in the production monitoring workflow.

These tests verify that:
- scripts/generate_monitoring_report.py does not call generate_html_report().
- scripts/run_monitoring_pipeline.py does not call generate_html_report().
- The monitoring pipeline module does not call generate_html_report().
- The BI dataset writer does not write HTML files.
- The monitoring CLI does not mention HTML report generation.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


def _collect_function_calls(
    source: str,
) -> list[str]:
    """Return all function call names found in Python source code."""

    tree = ast.parse(source)

    calls = []

    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.Call,
        ):
            if isinstance(
                node.func,
                ast.Name,
            ):
                calls.append(
                    node.func.id
                )

            elif isinstance(
                node.func,
                ast.Attribute,
            ):
                calls.append(
                    node.func.attr
                )

    return calls


# ---------------------------------------------------------------------------
# scripts/generate_monitoring_report.py
# ---------------------------------------------------------------------------

def test_generate_monitoring_report_script_does_not_call_html() -> None:
    """generate_monitoring_report.py must not call generate_html_report."""

    script_path = (
        PROJECT_ROOT
        / "scripts"
        / "generate_monitoring_report.py"
    )

    source = script_path.read_text(
        encoding="utf-8",
    )

    calls = _collect_function_calls(
        source
    )

    assert "generate_html_report" not in calls, (
        "scripts/generate_monitoring_report.py calls "
        "generate_html_report() — HTML must not be part of the "
        "production workflow."
    )


def test_generate_monitoring_report_script_does_not_import_html_function() -> None:
    """generate_monitoring_report.py must not import generate_html_report."""

    script_path = (
        PROJECT_ROOT
        / "scripts"
        / "generate_monitoring_report.py"
    )

    source = script_path.read_text(
        encoding="utf-8",
    )

    tree = ast.parse(source)

    imported_names = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.append(alias.name)

    assert "generate_html_report" not in imported_names, (
        "scripts/generate_monitoring_report.py imports "
        "generate_html_report()."
    )


# ---------------------------------------------------------------------------
# scripts/run_monitoring_pipeline.py
# ---------------------------------------------------------------------------

def test_run_monitoring_pipeline_script_does_not_call_html() -> None:
    """run_monitoring_pipeline.py must not call generate_html_report."""

    script_path = (
        PROJECT_ROOT
        / "scripts"
        / "run_monitoring_pipeline.py"
    )

    source = script_path.read_text(
        encoding="utf-8",
    )

    calls = _collect_function_calls(
        source
    )

    assert "generate_html_report" not in calls


def test_run_monitoring_pipeline_script_does_not_reference_html_report() -> None:
    """run_monitoring_pipeline.py must not reference monitoring_report.html."""

    script_path = (
        PROJECT_ROOT
        / "scripts"
        / "run_monitoring_pipeline.py"
    )

    source = script_path.read_text(
        encoding="utf-8",
    )

    assert "monitoring_report.html" not in source


# ---------------------------------------------------------------------------
# app/monitoring/pipeline.py
# ---------------------------------------------------------------------------

def test_pipeline_module_does_not_call_generate_html_report() -> None:
    """pipeline.py must not call generate_html_report."""

    pipeline_path = (
        PROJECT_ROOT
        / "app"
        / "monitoring"
        / "pipeline.py"
    )

    source = pipeline_path.read_text(
        encoding="utf-8",
    )

    calls = _collect_function_calls(
        source
    )

    assert "generate_html_report" not in calls


def test_pipeline_module_does_not_import_html_function() -> None:
    """pipeline.py must not import generate_html_report."""

    pipeline_path = (
        PROJECT_ROOT
        / "app"
        / "monitoring"
        / "pipeline.py"
    )

    source = pipeline_path.read_text(
        encoding="utf-8",
    )

    tree = ast.parse(source)

    imported_names = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.append(alias.name)

    assert "generate_html_report" not in imported_names, (
        "pipeline.py imports generate_html_report()."
    )


def test_pipeline_module_does_not_reference_html_uploads() -> None:
    """pipeline.py must not reference HTML file uploads."""

    pipeline_path = (
        PROJECT_ROOT
        / "app"
        / "monitoring"
        / "pipeline.py"
    )

    source = pipeline_path.read_text(
        encoding="utf-8",
    )

    assert "monitoring_report.html" not in source
    assert "text/html" not in source


# ---------------------------------------------------------------------------
# app/monitoring/bi_dataset_writer.py
# ---------------------------------------------------------------------------

def test_bi_dataset_writer_does_not_write_html() -> None:
    """bi_dataset_writer.py must not write HTML files."""

    writer_path = (
        PROJECT_ROOT
        / "app"
        / "monitoring"
        / "bi_dataset_writer.py"
    )

    source = writer_path.read_text(
        encoding="utf-8",
    )

    assert ".html" not in source
    assert "text/html" not in source
    assert "generate_html" not in source


# ---------------------------------------------------------------------------
# End-to-end: pipeline does not call generate_html_report
# ---------------------------------------------------------------------------

def test_pipeline_runtime_does_not_call_generate_html_report() -> None:
    """Running the pipeline must not trigger generate_html_report."""

    predictions = pd.DataFrame(
        {
            "application_id": [
                f"APP-{i}"
                for i in range(10)
            ],
            "predicted_probability": [
                0.05 + i * 0.09
                for i in range(10)
            ],
        }
    )

    outcomes = pd.DataFrame(
        {
            "application_id": [
                f"APP-{i}"
                for i in range(10)
            ],
            "actual_default": [
                0, 0, 0, 0, 0, 1, 0, 1, 1, 1,
            ],
        }
    )

    with patch(
        "app.monitoring.report.generate_html_report"
    ) as mock_html, patch(
        "app.monitoring.pipeline.MIN_PERFORMANCE_SAMPLES",
        1,
    ):
        from app.monitoring.pipeline import run_monitoring_pipeline

        run_monitoring_pipeline(
            predictions=predictions,
            outcomes=outcomes,
            minimum_samples=1,
            write_to_s3=False,
            publish_to_cloudwatch=False,
        )

        mock_html.assert_not_called()
