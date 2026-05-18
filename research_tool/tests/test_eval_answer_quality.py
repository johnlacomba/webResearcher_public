"""Tests for answer quality evaluation and RRF weight sweep."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from research_tool.eval import (
    AnswerQualityMetrics,
    AnswerQualityReport,
    EvalQuery,
    SweepReport,
    SweepResult,
    _score_answer,
    evaluate_answer_quality,
    format_answer_quality_report,
    format_sweep_report,
    load_eval_set,
    rrf_weight_sweep,
)
from research_tool.store import DocumentChunk, HybridIndex


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            text="BM25 is a ranking function used in information retrieval.",
            page_url="https://example.com/bm25",
            chunk_id="chunk-1",
            section_title="BM25",
            embedding=[1.0, 0.0, 0.0],
        ),
        DocumentChunk(
            text="Cosine similarity measures the angle between two vectors.",
            page_url="https://example.com/cosine",
            chunk_id="chunk-2",
            section_title="Cosine",
            embedding=[0.0, 1.0, 0.0],
        ),
        DocumentChunk(
            text="RRF fuses multiple ranked lists using reciprocal ranks.",
            page_url="https://example.com/rrf",
            chunk_id="chunk-3",
            section_title="RRF",
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
def eval_set():
    return [
        EvalQuery(
            query="What is BM25?",
            relevant_chunk_ids=["chunk-1"],
            reference_answer="BM25 is a bag-of-words retrieval function.",
        ),
        EvalQuery(
            query="How does cosine similarity work?",
            relevant_chunk_ids=["chunk-2"],
        ),
    ]


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate.return_value = '{"relevance": 0.9, "groundedness": 0.8, "completeness": 0.7}'
    return llm


# ── AnswerQualityMetrics ─────────────────────────────────────────────────────


class TestAnswerQualityMetrics:
    def test_mean_score(self):
        m = AnswerQualityMetrics(relevance=0.9, groundedness=0.6, completeness=0.3)
        assert m.mean_score() == pytest.approx(0.6)

    def test_to_dict(self):
        m = AnswerQualityMetrics(relevance=0.8, groundedness=0.7, completeness=0.6)
        d = m.to_dict()
        assert d["relevance"] == 0.8
        assert d["groundedness"] == 0.7
        assert d["completeness"] == 0.6
        assert "mean_score" in d


# ── _score_answer ────────────────────────────────────────────────────────────


class TestScoreAnswer:
    def test_parses_valid_json(self, mock_llm):
        m = _score_answer(mock_llm, "question", "answer", "context")
        assert m.relevance == pytest.approx(0.9)
        assert m.groundedness == pytest.approx(0.8)
        assert m.completeness == pytest.approx(0.7)

    def test_handles_llm_failure(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("API error")
        m = _score_answer(llm, "q", "a", "c")
        assert math.isnan(m.relevance)
        assert math.isnan(m.groundedness)
        assert math.isnan(m.completeness)

    def test_handles_malformed_json(self):
        llm = MagicMock()
        llm.generate.return_value = "not json at all"
        m = _score_answer(llm, "q", "a", "c")
        assert math.isnan(m.relevance)

    def test_handles_json_with_extra_text(self):
        llm = MagicMock()
        llm.generate.return_value = 'Here is my evaluation: {"relevance": 1.0, "groundedness": 0.5, "completeness": 0.8}'
        m = _score_answer(llm, "q", "a", "c")
        assert m.relevance == pytest.approx(1.0)
        assert m.groundedness == pytest.approx(0.5)

    def test_includes_reference_answer(self, mock_llm):
        _score_answer(mock_llm, "q", "a", "c", reference_answer="ref")
        call_args = mock_llm.generate.call_args
        assert "ref" in call_args[0][1]


# ── evaluate_answer_quality ──────────────────────────────────────────────────


class TestEvaluateAnswerQuality:
    def test_happy_path_returns_scores(self, built_index, eval_set, mock_llm):
        with patch("research_tool.eval._make_embedding", return_value=None):
            report = evaluate_answer_quality(
                built_index, eval_set, mock_llm, top_k=3,
            )
        assert report.num_queries == 2
        assert len(report.results) == 2
        assert 0.0 <= report.average.relevance <= 1.0
        assert 0.0 <= report.average.groundedness <= 1.0
        assert 0.0 <= report.average.completeness <= 1.0

    def test_no_reference_answer_still_scores(self, built_index, mock_llm):
        eq = [EvalQuery(query="What is RRF?", relevant_chunk_ids=["chunk-3"])]
        with patch("research_tool.eval._make_embedding", return_value=None):
            report = evaluate_answer_quality(built_index, eq, mock_llm, top_k=3)
        assert report.num_queries == 1
        assert report.results[0].reference_answer is None
        assert not math.isnan(report.results[0].metrics.relevance)

    def test_empty_eval_set(self, built_index, mock_llm):
        with patch("research_tool.eval._make_embedding", return_value=None):
            report = evaluate_answer_quality(built_index, [], mock_llm, top_k=3)
        assert report.num_queries == 0
        assert report.average.relevance == 0.0

    def test_llm_scoring_failure_produces_nan(self, built_index):
        llm = MagicMock()
        llm.generate.side_effect = [
            "answer text",
            RuntimeError("scoring failed"),
        ]
        eq = [EvalQuery(query="test", relevant_chunk_ids=["chunk-1"])]
        with patch("research_tool.eval._make_embedding", return_value=None):
            report = evaluate_answer_quality(built_index, eq, llm, top_k=3)
        assert math.isnan(report.results[0].metrics.relevance)

    def test_to_dict_roundtrip(self, built_index, eval_set, mock_llm):
        with patch("research_tool.eval._make_embedding", return_value=None):
            report = evaluate_answer_quality(
                built_index, eval_set, mock_llm, top_k=3,
            )
        d = report.to_dict()
        assert d["num_queries"] == 2
        serialized = json.dumps(d)
        assert json.loads(serialized) == d


# ── rrf_weight_sweep ─────────────────────────────────────────────────────────


class TestRrfWeightSweep:
    def test_returns_best_config(self, built_index):
        eval_set = [
            EvalQuery(query="BM25 ranking", relevant_chunk_ids=["chunk-1"]),
            EvalQuery(query="cosine vectors", relevant_chunk_ids=["chunk-2"]),
        ]
        with patch("research_tool.eval._make_embedding", return_value=None):
            report = rrf_weight_sweep(
                built_index, eval_set, top_k=3,
                sweep_weights=["bm25", "text_cosine"],
                sweep_values=[0.5, 1.0],
            )
        assert report.num_combinations == 4  # 2^2
        assert report.num_queries == 2
        assert report.best.precision_at_k >= 0.0
        assert len(report.all_results) == 4

    def test_results_sorted_by_precision(self, built_index):
        eval_set = [
            EvalQuery(query="BM25", relevant_chunk_ids=["chunk-1"]),
        ]
        with patch("research_tool.eval._make_embedding", return_value=None):
            report = rrf_weight_sweep(
                built_index, eval_set, top_k=3,
                sweep_weights=["bm25"],
                sweep_values=[0.5, 1.0, 1.5],
            )
        precisions = [r.precision_at_k for r in report.all_results]
        assert precisions == sorted(precisions, reverse=True)

    def test_empty_eval_set(self, built_index):
        with patch("research_tool.eval._make_embedding", return_value=None):
            report = rrf_weight_sweep(
                built_index, [], top_k=3,
                sweep_weights=["bm25"],
                sweep_values=[1.0],
            )
        assert report.num_queries == 0
        assert report.best.precision_at_k == 0.0

    def test_to_dict_roundtrip(self, built_index):
        eval_set = [
            EvalQuery(query="BM25", relevant_chunk_ids=["chunk-1"]),
        ]
        with patch("research_tool.eval._make_embedding", return_value=None):
            report = rrf_weight_sweep(
                built_index, eval_set, top_k=3,
                sweep_weights=["bm25"],
                sweep_values=[1.0],
            )
        d = report.to_dict()
        serialized = json.dumps(d)
        assert json.loads(serialized) == d

    def test_default_sweep_weights(self, built_index):
        eval_set = [
            EvalQuery(query="BM25", relevant_chunk_ids=["chunk-1"]),
        ]
        with patch("research_tool.eval._make_embedding", return_value=None):
            report = rrf_weight_sweep(built_index, eval_set, top_k=3)
        assert report.num_combinations == 81  # 3^4


# ── load_eval_set with reference_answer ──────────────────────────────────────


class TestLoadEvalSetReferenceAnswer:
    def test_loads_reference_answer(self, tmp_path):
        data = [
            {
                "query": "What is BM25?",
                "relevant_chunk_ids": ["c1"],
                "reference_answer": "A ranking function.",
            }
        ]
        path = tmp_path / "eval.json"
        path.write_text(json.dumps(data))
        result = load_eval_set(str(path))
        assert result[0].reference_answer == "A ranking function."

    def test_missing_reference_answer_is_none(self, tmp_path):
        data = [{"query": "q", "relevant_chunk_ids": ["c1"]}]
        path = tmp_path / "eval.json"
        path.write_text(json.dumps(data))
        result = load_eval_set(str(path))
        assert result[0].reference_answer is None


# ── format functions ─────────────────────────────────────────────────────────


class TestFormatFunctions:
    def test_format_answer_quality_report(self):
        report = AnswerQualityReport(
            num_queries=2,
            average=AnswerQualityMetrics(relevance=0.8, groundedness=0.7, completeness=0.6),
            results=[],
        )
        output = format_answer_quality_report(report)
        assert "queries=2" in output
        assert "relevance" in output
        assert "0.800" in output

    def test_format_sweep_report(self):
        report = SweepReport(
            top_k=10,
            num_queries=5,
            num_combinations=81,
            best=SweepResult(
                weights={"bm25": 1.5, "text_cosine": 1.0},
                precision_at_k=0.8,
                recall_at_k=0.9,
            ),
            all_results=[
                SweepResult(
                    weights={"bm25": 1.5, "text_cosine": 1.0},
                    precision_at_k=0.8,
                    recall_at_k=0.9,
                ),
            ],
        )
        output = format_sweep_report(report)
        assert "combinations=81" in output
        assert "0.800" in output
        assert "bm25" in output


# ── CLI flags ────────────────────────────────────────────────────────────────


class TestEvalCLIFlags:
    def test_eval_help_shows_new_flags(self):
        result = subprocess.run(
            [sys.executable, "-m", "research_tool", "eval", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "--answer-quality" in result.stdout
        assert "--sweep" in result.stdout
        assert "--llm" in result.stdout
