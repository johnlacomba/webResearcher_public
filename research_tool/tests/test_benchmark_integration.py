"""Integration tests for the benchmark pipeline end-to-end."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from research_tool.benchmarks.configs import BenchmarkConfig, generate_matrix
from research_tool.benchmarks.results import (
    BenchmarkReport,
    build_report,
    compare_reports,
    load_report,
    save_report,
)
from research_tool.benchmarks.runner import BenchmarkRunner, load_corpus


def _pick_configs(names: list[str]) -> list[BenchmarkConfig]:
    """Pick specific configs by name for fast integration testing."""
    matrix = generate_matrix(include_llm=False)
    selected = []
    for name in names:
        for cfg in matrix:
            if cfg.name == name:
                selected.append(cfg)
                break
    return selected


# Use a small subset: one with parent-child, one without, different retrieval
INTEGRATION_CONFIGS = [
    "paragraph_nomic-only_bm25-only",
    "parent-child_nomic-only_bm25-only",
    "paragraph_nomic-only_hybrid-no-hyde",
]


@pytest.fixture(scope="module")
def integration_results():
    """Run a small benchmark matrix once for all integration tests."""
    configs = _pick_configs(INTEGRATION_CONFIGS)
    if len(configs) < 2:
        pytest.skip("Could not find enough configs for integration test")
    runner = BenchmarkRunner(configs=configs, top_k=5)
    return runner.run()


class TestEndToEnd:
    def test_all_configs_complete(self, integration_results):
        assert len(integration_results) >= 2
        for result in integration_results:
            assert result.error is None, f"Config {result.config_name} failed: {result.error}"

    def test_chunks_produced(self, integration_results):
        for result in integration_results:
            assert result.chunk_count > 0, f"Config {result.config_name} produced no chunks"

    def test_ingestion_timing_positive(self, integration_results):
        for result in integration_results:
            assert result.ingestion_time > 0, f"Config {result.config_name} has zero ingestion time"

    def test_retrieval_latency_recorded(self, integration_results):
        for result in integration_results:
            assert result.retrieval_latency.count > 0
            assert result.retrieval_latency.mean > 0

    def test_text_overlap_scores_populated(self, integration_results):
        for result in integration_results:
            assert len(result.text_overlap_scores) > 0
            assert any(s > 0 for s in result.text_overlap_scores), (
                f"Config {result.config_name}: all text-overlap scores are 0"
            )

    def test_per_query_results_match_corpus(self, integration_results):
        _, queries = load_corpus()
        for result in integration_results:
            assert len(result.per_query_results) == len(queries)


class TestConfigDifferences:
    def test_parent_child_produces_more_chunks(self, integration_results):
        by_name = {r.config_name: r for r in integration_results}
        para = by_name.get("paragraph_nomic-only_bm25-only")
        pc = by_name.get("parent-child_nomic-only_bm25-only")
        if para and pc:
            assert pc.chunk_count > para.chunk_count, (
                f"Parent-child ({pc.chunk_count}) should produce more chunks than paragraph ({para.chunk_count})"
            )

    def test_different_configs_have_different_scores(self, integration_results):
        if len(integration_results) < 2:
            pytest.skip("Need at least 2 configs")
        overlaps = [
            sum(r.text_overlap_scores) / len(r.text_overlap_scores)
            for r in integration_results
            if r.text_overlap_scores
        ]
        assert len(set(f"{o:.6f}" for o in overlaps)) > 1 or len(overlaps) < 2, (
            "All configs have identical accuracy — likely a configuration issue"
        )


class TestReportRoundTrip:
    def test_full_pipeline_to_json_and_back(self, integration_results):
        report = build_report(integration_results, total_wall_time=10.0)
        assert report.config_count >= 2
        assert report.git_sha != "unknown"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_report(report, output_dir=Path(tmpdir))
            assert path.exists()

            loaded = load_report(path)
            assert loaded.config_count == report.config_count
            assert len(loaded.configs) == len(report.configs)

            for orig, loaded_cfg in zip(report.configs, loaded.configs):
                assert loaded_cfg.config_name == orig.config_name
                assert loaded_cfg.ingestion.chunk_count == orig.ingestion.chunk_count

    def test_compare_same_report_shows_zero_deltas(self, integration_results):
        report = build_report(integration_results, total_wall_time=10.0)
        diffs = compare_reports(report, report)
        for d in diffs:
            assert d.only_in is None
            if d.text_overlap_delta is not None:
                assert abs(d.text_overlap_delta) < 1e-9
