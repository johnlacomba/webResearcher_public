"""Tests for benchmark results serialization and comparison."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from research_tool.benchmarks.results import (
    BenchmarkReport,
    ConfigDiff,
    ConfigReport,
    IngestionMetrics,
    QueryDetail,
    RetrievalMetrics,
    build_report,
    compare_reports,
    format_comparison,
    load_report,
    save_report,
)


def _make_report(**kwargs) -> BenchmarkReport:
    defaults = dict(
        timestamp="2026-05-12T10:00:00Z",
        git_sha="abc1234",
        python_version="3.13.7",
        platform_info="Darwin arm64",
        total_wall_time_s=120.0,
        config_count=1,
        query_count=2,
        configs=[
            ConfigReport(
                config_name="test_config",
                config_overrides={"PARENT_CHILD_ENABLED": True},
                ingestion=IngestionMetrics(total_time_s=1.5, chunk_count=30, time_per_chunk_ms=50.0),
                retrieval=RetrievalMetrics(
                    mean_text_overlap=0.75,
                    p50_latency_ms=12.5,
                    p95_latency_ms=45.0,
                    p99_latency_ms=82.0,
                    query_count=2,
                ),
                queries=[
                    QueryDetail(query="test query 1", difficulty="easy", text_overlap=0.8, latency_ms=10.0, retrieved_count=5, retrieved_ids=["a", "b"]),
                    QueryDetail(query="test query 2", difficulty="hard", text_overlap=0.7, latency_ms=15.0, retrieved_count=5, retrieved_ids=["c", "d"]),
                ],
            ),
        ],
    )
    defaults.update(kwargs)
    return BenchmarkReport(**defaults)


class TestBenchmarkReportRoundTrip:
    def test_to_dict_and_from_dict(self):
        report = _make_report()
        d = report.to_dict()
        restored = BenchmarkReport.from_dict(d)
        assert restored.timestamp == report.timestamp
        assert restored.git_sha == report.git_sha
        assert restored.config_count == report.config_count
        assert len(restored.configs) == 1
        assert restored.configs[0].config_name == "test_config"
        assert restored.configs[0].ingestion.chunk_count == 30
        assert restored.configs[0].retrieval.mean_text_overlap == 0.75
        assert len(restored.configs[0].queries) == 2

    def test_save_and_load(self):
        report = _make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_report(report, output_dir=Path(tmpdir))
            assert path.exists()
            assert path.name.startswith("benchmark-results-")
            loaded = load_report(path)
            assert loaded.git_sha == "abc1234"
            assert loaded.configs[0].retrieval.p50_latency_ms == 12.5

    def test_json_is_valid(self):
        report = _make_report()
        d = report.to_dict()
        text = json.dumps(d)
        parsed = json.loads(text)
        assert parsed["git_sha"] == "abc1234"

    def test_nan_serializes_as_null(self):
        report = _make_report()
        report.configs[0].retrieval.p99_latency_ms = float("nan")
        d = report.to_dict()
        assert d["configs"][0]["retrieval"]["p99_latency_ms"] is None
        text = json.dumps(d)
        assert "NaN" not in text

    def test_empty_report_round_trips(self):
        report = BenchmarkReport(
            timestamp="2026-01-01T00:00:00Z",
            git_sha="0000000",
            python_version="3.13.0",
            platform_info="Linux x86_64",
            total_wall_time_s=0.0,
            config_count=0,
            query_count=0,
        )
        d = report.to_dict()
        restored = BenchmarkReport.from_dict(d)
        assert restored.config_count == 0
        assert restored.configs == []


class TestCompareReports:
    def test_same_config_shows_deltas(self):
        a = _make_report()
        b = _make_report()
        b.configs[0].retrieval.mean_text_overlap = 0.85
        b.configs[0].ingestion.total_time_s = 2.0
        diffs = compare_reports(a, b)
        assert len(diffs) == 1
        assert diffs[0].text_overlap_delta == pytest.approx(0.10)
        assert diffs[0].ingestion_time_delta_pct is not None

    def test_missing_config_in_b(self):
        a = _make_report()
        b = _make_report()
        b.configs = []
        diffs = compare_reports(a, b)
        assert any(d.only_in == "A" for d in diffs)

    def test_new_config_in_b(self):
        a = _make_report()
        a.configs = []
        b = _make_report()
        diffs = compare_reports(a, b)
        assert any(d.only_in == "B" for d in diffs)

    def test_format_comparison_runs(self):
        a = _make_report()
        b = _make_report()
        b.configs[0].retrieval.mean_text_overlap = 0.85
        diffs = compare_reports(a, b)
        text = format_comparison(diffs)
        assert "test_config" in text
        assert "Overlap" in text


class TestTimestampAndMetadata:
    def test_timestamp_format(self):
        report = _make_report()
        assert "T" in report.timestamp
        assert report.timestamp.endswith("Z")

    def test_platform_info_populated(self):
        report = _make_report()
        assert report.platform_info
        assert report.python_version
