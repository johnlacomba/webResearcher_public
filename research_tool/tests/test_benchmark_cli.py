"""Tests for the benchmark CLI subcommand."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from research_tool.benchmarks.results import BenchmarkReport, save_report


def _run_cli(*args: str, expect_error: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, "-m", "research_tool", *args],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if not expect_error:
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
    return result


class TestBenchmarkCLIHelp:
    def test_benchmark_help(self):
        result = _run_cli("benchmark", "--help")
        assert "run" in result.stdout
        assert "compare" in result.stdout

    def test_benchmark_run_help(self):
        result = _run_cli("benchmark", "run", "--help")
        assert "--config" in result.stdout
        assert "--llm" in result.stdout
        assert "--top-k" in result.stdout
        assert "--output" in result.stdout

    def test_benchmark_compare_help(self):
        result = _run_cli("benchmark", "compare", "--help")
        assert "file_a" in result.stdout
        assert "file_b" in result.stdout


class TestBenchmarkCompare:
    def _make_report_file(self, tmpdir: str, name: str, overlap: float = 0.5) -> Path:
        from research_tool.benchmarks.results import (
            ConfigReport,
            IngestionMetrics,
            RetrievalMetrics,
        )
        report = BenchmarkReport(
            timestamp="2026-05-12T00:00:00Z",
            git_sha="abc1234",
            python_version="3.13.7",
            platform_info="Darwin arm64",
            total_wall_time_s=10.0,
            config_count=1,
            query_count=0,
            configs=[
                ConfigReport(
                    config_name="test_config",
                    ingestion=IngestionMetrics(total_time_s=1.0, chunk_count=10, time_per_chunk_ms=100.0),
                    retrieval=RetrievalMetrics(mean_text_overlap=overlap, p50_latency_ms=10.0, p95_latency_ms=40.0, p99_latency_ms=80.0, query_count=5),
                ),
            ],
        )
        path = Path(tmpdir) / name
        path.write_text(json.dumps(report.to_dict(), indent=2))
        return path

    def test_compare_two_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            a = self._make_report_file(tmpdir, "a.json", overlap=0.5)
            b = self._make_report_file(tmpdir, "b.json", overlap=0.7)
            result = _run_cli("benchmark", "compare", str(a), str(b))
            assert "test_config" in result.stdout
            assert "+0.200" in result.stdout

    def test_compare_json_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            a = self._make_report_file(tmpdir, "a.json")
            b = self._make_report_file(tmpdir, "b.json")
            result = _run_cli("benchmark", "compare", str(a), str(b), "--json")
            diffs = json.loads(result.stdout)
            assert isinstance(diffs, list)
            assert diffs[0]["config_name"] == "test_config"

    def test_compare_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            a = self._make_report_file(tmpdir, "a.json")
            result = _run_cli("benchmark", "compare", str(a), "/nonexistent.json", expect_error=True)
            assert result.returncode != 0


class TestBenchmarkRunConfig:
    def test_unknown_config_name_errors(self):
        result = _run_cli("benchmark", "run", "--config", "nonexistent_config_xyz", expect_error=True)
        assert result.returncode != 0
        assert "unknown config" in result.stderr.lower() or "error" in result.stderr.lower()
