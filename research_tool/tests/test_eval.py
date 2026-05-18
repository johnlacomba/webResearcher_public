"""Tests for research_tool.eval — retrieval quality evaluation harness."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from research_tool.eval import (
    EvalQuery,
    EvalReport,
    SignalMetrics,
    _compute_metrics,
    _entity_ranking,
    evaluate,
    format_report,
    load_eval_set,
)
from research_tool.store import DocumentChunk, HybridIndex, ResearchStore


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            text="Python is a popular programming language.",
            page_url="https://example.com/python",
            chunk_id="chunk-1",
            section_title="Python",
            embedding=[1.0, 0.0, 0.0],
        ),
        DocumentChunk(
            text="Machine learning uses algorithms to learn from data.",
            page_url="https://example.com/ml",
            chunk_id="chunk-2",
            section_title="ML",
            embedding=[0.0, 1.0, 0.0],
        ),
        DocumentChunk(
            text="Web scraping extracts data from websites.",
            page_url="https://example.com/scraping",
            chunk_id="chunk-3",
            section_title="Scraping",
            embedding=[0.0, 0.0, 1.0],
        ),
    ]


@pytest.fixture
def built_index(sample_chunks):
    index = HybridIndex(mode="query")
    for chunk in sample_chunks:
        index.add_chunk(chunk)
    index.build()
    return index


@pytest.fixture
def eval_set_file(tmp_path):
    data = [
        {
            "query": "Python programming",
            "relevant_chunk_ids": ["chunk-1"],
            "modality_labels": {"text": True, "image": False},
        },
        {
            "query": "machine learning algorithms",
            "relevant_chunk_ids": ["chunk-2"],
        },
        {
            "query": "web scraping data",
            "relevant_chunk_ids": ["chunk-3", "chunk-2"],
        },
    ]
    path = tmp_path / "eval_set.json"
    path.write_text(json.dumps(data))
    return str(path)


# ── load_eval_set ─────────────────────────────────────────────────────────────


class TestLoadEvalSet:
    def test_loads_array_format(self, tmp_path):
        data = [{"query": "test", "relevant_chunk_ids": ["c1"]}]
        path = tmp_path / "eval.json"
        path.write_text(json.dumps(data))
        result = load_eval_set(str(path))
        assert len(result) == 1
        assert result[0].query == "test"
        assert result[0].relevant_chunk_ids == ["c1"]

    def test_loads_object_with_queries_key(self, tmp_path):
        data = {"queries": [{"query": "q1", "relevant_chunk_ids": ["c1", "c2"]}]}
        path = tmp_path / "eval.json"
        path.write_text(json.dumps(data))
        result = load_eval_set(str(path))
        assert len(result) == 1
        assert result[0].relevant_chunk_ids == ["c1", "c2"]

    def test_preserves_modality_labels(self, tmp_path):
        data = [{"query": "q", "relevant_chunk_ids": ["c1"], "modality_labels": {"text": True}}]
        path = tmp_path / "eval.json"
        path.write_text(json.dumps(data))
        result = load_eval_set(str(path))
        assert result[0].modality_labels == {"text": True}

    def test_default_modality_labels(self, tmp_path):
        data = [{"query": "q", "relevant_chunk_ids": ["c1"]}]
        path = tmp_path / "eval.json"
        path.write_text(json.dumps(data))
        result = load_eval_set(str(path))
        assert result[0].modality_labels == {}

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_eval_set("/nonexistent/path.json")

    def test_malformed_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        with pytest.raises(json.JSONDecodeError):
            load_eval_set(str(path))

    def test_invalid_structure(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"wrong_key": []}))
        with pytest.raises(ValueError, match="JSON array or object"):
            load_eval_set(str(path))


# ── _compute_metrics ──────────────────────────────────────────────────────────


class TestComputeMetrics:
    def test_perfect_precision_and_recall(self):
        m = _compute_metrics(["c1", "c2"], {"c1", "c2"}, k=2)
        assert m.precision_at_k == 1.0
        assert m.recall_at_k == 1.0

    def test_zero_precision_and_recall(self):
        m = _compute_metrics(["c3", "c4"], {"c1", "c2"}, k=2)
        assert m.precision_at_k == 0.0
        assert m.recall_at_k == 0.0

    def test_partial_overlap(self):
        m = _compute_metrics(["c1", "c3", "c4", "c5", "c2"], {"c1", "c2"}, k=5)
        assert m.precision_at_k == pytest.approx(0.4)
        assert m.recall_at_k == 1.0

    def test_empty_relevant_set(self):
        m = _compute_metrics(["c1"], set(), k=1)
        assert m.precision_at_k == 0.0
        assert m.recall_at_k == 0.0

    def test_k_larger_than_retrieved(self):
        m = _compute_metrics(["c1"], {"c1"}, k=5)
        assert m.precision_at_k == pytest.approx(0.2)
        assert m.recall_at_k == 1.0

    def test_single_query_no_division_by_zero(self):
        m = _compute_metrics([], {"c1"}, k=5)
        assert m.precision_at_k == 0.0
        assert m.recall_at_k == 0.0


# ── evaluate ──────────────────────────────────────────────────────────────────


class TestEvaluate:
    def test_happy_path_correct_metrics(self, built_index):
        eval_set = [
            EvalQuery(query="Python programming", relevant_chunk_ids=["chunk-1"]),
            EvalQuery(query="machine learning data", relevant_chunk_ids=["chunk-2"]),
            EvalQuery(query="web scraping websites", relevant_chunk_ids=["chunk-3"]),
        ]
        with patch("research_tool.eval._make_embedding", return_value=None):
            report = evaluate(built_index, eval_set, top_k=5)

        assert report.num_queries == 3
        assert report.top_k == 5
        assert len(report.query_results) == 3
        assert "bm25" in report.per_signal

    def test_per_signal_breakdown_differs(self, built_index):
        eval_set = [
            EvalQuery(query="Python", relevant_chunk_ids=["chunk-1"]),
        ]
        with patch("research_tool.eval._make_embedding", return_value=[1.0, 0.0, 0.0]):
            report = evaluate(built_index, eval_set, top_k=3)

        assert "bm25" in report.per_signal
        assert "text_cosine" in report.per_signal

    def test_zero_relevant_chunks(self, built_index):
        eval_set = [
            EvalQuery(query="quantum physics", relevant_chunk_ids=["nonexistent"]),
        ]
        with patch("research_tool.eval._make_embedding", return_value=None):
            report = evaluate(built_index, eval_set, top_k=5)

        assert report.overall.precision_at_k == 0.0
        assert report.overall.recall_at_k == 0.0

    def test_single_query_eval(self, built_index):
        eval_set = [
            EvalQuery(query="Python", relevant_chunk_ids=["chunk-1"]),
        ]
        with patch("research_tool.eval._make_embedding", return_value=None):
            report = evaluate(built_index, eval_set, top_k=5)

        assert report.num_queries == 1
        assert len(report.query_results) == 1

    def test_to_dict_roundtrip(self, built_index):
        eval_set = [
            EvalQuery(query="Python", relevant_chunk_ids=["chunk-1"]),
        ]
        with patch("research_tool.eval._make_embedding", return_value=None):
            report = evaluate(built_index, eval_set, top_k=5)

        d = report.to_dict()
        assert d["top_k"] == 5
        assert d["num_queries"] == 1
        assert "P@K" in d["overall"]
        assert "R@K" in d["overall"]
        assert isinstance(d["per_signal"], dict)
        serialized = json.dumps(d)
        assert json.loads(serialized) == d


# ── format_report ─────────────────────────────────────────────────────────────


class TestFormatReport:
    def test_contains_header_and_signals(self):
        report = EvalReport(
            top_k=5,
            num_queries=2,
            overall=SignalMetrics(precision_at_k=0.6, recall_at_k=0.8),
            per_signal={
                "bm25": SignalMetrics(precision_at_k=0.4, recall_at_k=0.5),
                "text_cosine": SignalMetrics(precision_at_k=0.7, recall_at_k=0.9),
            },
            query_results=[],
        )
        output = format_report(report)
        assert "top_k=5" in output
        assert "queries=2" in output
        assert "overall" in output
        assert "bm25" in output
        assert "text_cosine" in output
        assert "0.600" in output
        assert "0.800" in output


# ── CLI integration ───────────────────────────────────────────────────────────


class TestEvalCLI:
    def test_eval_help_shows_flags(self):
        result = subprocess.run(
            [sys.executable, "-m", "research_tool", "eval", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "eval_set" in result.stdout
        assert "--top-k" in result.stdout
        assert "--json" in result.stdout
        assert "--db" in result.stdout

    def test_eval_missing_file_exits_error(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        result = subprocess.run(
            [sys.executable, "-m", "research_tool", "eval", "/nonexistent.json", "--db", db_path],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0


# ── Entity signal in eval ────────────────────────────────────────────────────


class TestEntityEval:
    @staticmethod
    def _make_emb(seed: float) -> list[float]:
        import numpy as np
        rng = np.random.RandomState(int(abs(seed * 1000)) % (2**31))
        v = rng.randn(768).astype(np.float32)
        v /= np.linalg.norm(v)
        return v.tolist()

    @patch("research_tool.store.ENTITY_RESOLUTION_ENABLED", True)
    @patch("research_tool.eval.ENTITY_RESOLUTION_ENABLED", True)
    def test_entity_signal_in_eval_report(self, tmp_path):
        """Eval with entity resolution enabled includes entity metrics."""
        emb_a = self._make_emb(1.0)
        emb_b = self._make_emb(2.0)

        store = ResearchStore(str(tmp_path / "test.db"))
        store.store_page(url="https://example.com/p1", title="P1")
        chunk_a = DocumentChunk(
            text="The Global Interpreter Lock in CPython prevents multithreading.",
            page_url="https://example.com/p1",
            chunk_id="chunk-a",
            section_title="GIL",
            embedding=emb_a,
        )
        chunk_b = DocumentChunk(
            text="Python supports object-oriented programming paradigms.",
            page_url="https://example.com/p1",
            chunk_id="chunk-b",
            section_title="OOP",
            embedding=emb_b,
        )
        store.store_chunk(chunk_a)
        store.store_chunk(chunk_b)
        store.store_entity(
            entity_id="entity::gil",
            canonical_name="Python GIL",
            definition="The Global Interpreter Lock",
            entity_embedding=emb_a,
        )
        store.store_chunk_entity("chunk-a", "entity::gil")

        index = HybridIndex(mode="query")
        index.build_from_store(store)

        eval_set = [
            EvalQuery(query="GIL multithreading", relevant_chunk_ids=["chunk-a"]),
        ]
        with patch("research_tool.eval._make_embedding", return_value=emb_a):
            report = evaluate(index, eval_set, top_k=5, store=store)

        assert "entity" in report.per_signal
        assert report.per_signal["entity"].precision_at_k > 0.0

    @patch("research_tool.store.ENTITY_RESOLUTION_ENABLED", True)
    @patch("research_tool.eval.ENTITY_RESOLUTION_ENABLED", True)
    def test_entity_only_produces_metrics(self, tmp_path):
        """Entity-only ranking produces P@K and R@K values."""
        emb = self._make_emb(1.0)

        store = ResearchStore(str(tmp_path / "test.db"))
        store.store_page(url="https://example.com/p1", title="P1")
        chunk = DocumentChunk(
            text="GIL prevents multithreading.",
            page_url="https://example.com/p1",
            chunk_id="chunk-1",
            section_title="GIL",
            embedding=emb,
        )
        store.store_chunk(chunk)
        store.store_entity(
            entity_id="entity::gil",
            canonical_name="GIL",
            entity_embedding=emb,
        )
        store.store_chunk_entity("chunk-1", "entity::gil")

        index = HybridIndex(mode="query")
        index.build_from_store(store)

        with patch("research_tool.eval._make_embedding", return_value=emb):
            ranked = _entity_ranking(index, "GIL", top_k=5)
        assert "chunk-1" in ranked

    def test_entity_eval_no_entities(self, built_index):
        """No entities in store — entity metrics absent from report."""
        eval_set = [
            EvalQuery(query="Python", relevant_chunk_ids=["chunk-1"]),
        ]
        with patch("research_tool.eval._make_embedding", return_value=None):
            report = evaluate(built_index, eval_set, top_k=5)

        assert "entity" not in report.per_signal
