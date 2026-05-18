"""Lightweight evaluation harness for retrieval quality (P@K, R@K per signal)."""

from __future__ import annotations

import itertools
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from research_tool.store import (
    ENTITY_RESOLUTION_ENABLED,
    HybridIndex,
    ResearchStore,
    RetrievedEntry,
    _make_embedding,
    _make_token_embeddings,
    multi_signal_rrf,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalQuery:
    query: str
    relevant_chunk_ids: list[str]
    modality_labels: dict[str, bool] = field(default_factory=dict)
    reference_answer: str | None = None


@dataclass
class SignalMetrics:
    precision_at_k: float
    recall_at_k: float


@dataclass
class QueryResult:
    query: str
    overall: SignalMetrics
    per_signal: dict[str, SignalMetrics]


@dataclass
class EvalReport:
    top_k: int
    num_queries: int
    overall: SignalMetrics
    per_signal: dict[str, SignalMetrics]
    query_results: list[QueryResult]

    def to_dict(self) -> dict:
        return {
            "top_k": self.top_k,
            "num_queries": self.num_queries,
            "overall": {"P@K": self.overall.precision_at_k, "R@K": self.overall.recall_at_k},
            "per_signal": {
                name: {"P@K": m.precision_at_k, "R@K": m.recall_at_k}
                for name, m in self.per_signal.items()
            },
            "query_results": [
                {
                    "query": qr.query,
                    "overall": {"P@K": qr.overall.precision_at_k, "R@K": qr.overall.recall_at_k},
                    "per_signal": {
                        name: {"P@K": m.precision_at_k, "R@K": m.recall_at_k}
                        for name, m in qr.per_signal.items()
                    },
                }
                for qr in self.query_results
            ],
        }


@dataclass
class AnswerQualityMetrics:
    relevance: float
    groundedness: float
    completeness: float

    def mean_score(self) -> float:
        return (self.relevance + self.groundedness + self.completeness) / 3.0

    def to_dict(self) -> dict:
        def _safe(v: float) -> float | None:
            return None if math.isnan(v) else v
        return {
            "relevance": _safe(self.relevance),
            "groundedness": _safe(self.groundedness),
            "completeness": _safe(self.completeness),
            "mean_score": _safe(self.mean_score()),
        }


@dataclass
class AnswerQualityResult:
    query: str
    answer: str
    metrics: AnswerQualityMetrics
    reference_answer: str | None = None


@dataclass
class AnswerQualityReport:
    num_queries: int
    average: AnswerQualityMetrics
    results: list[AnswerQualityResult]

    def to_dict(self) -> dict:
        return {
            "num_queries": self.num_queries,
            "average": self.average.to_dict(),
            "results": [
                {
                    "query": r.query,
                    "answer": r.answer[:200],
                    "metrics": r.metrics.to_dict(),
                    "reference_answer": r.reference_answer,
                }
                for r in self.results
            ],
        }


@dataclass
class SweepResult:
    weights: dict[str, float]
    precision_at_k: float
    recall_at_k: float

    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "P@K": self.precision_at_k,
            "R@K": self.recall_at_k,
        }


@dataclass
class SweepReport:
    top_k: int
    num_queries: int
    num_combinations: int
    best: SweepResult
    all_results: list[SweepResult]

    def to_dict(self) -> dict:
        return {
            "top_k": self.top_k,
            "num_queries": self.num_queries,
            "num_combinations": self.num_combinations,
            "best": self.best.to_dict(),
            "all_results": [r.to_dict() for r in self.all_results],
        }


def load_eval_set(path: str) -> list[EvalQuery]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "queries" in data:
        entries = data["queries"]
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError(f"Eval set must be a JSON array or object with 'queries' key")
    return [
        EvalQuery(
            query=e["query"],
            relevant_chunk_ids=e["relevant_chunk_ids"],
            modality_labels=e.get("modality_labels", {}),
            reference_answer=e.get("reference_answer"),
        )
        for e in entries
    ]


def normalize_text(text: str) -> str:
    """Normalize text for overlap comparison.

    Applies: HTML entity decoding, whitespace collapsing, case folding.
    """
    import html as html_lib
    text = html_lib.unescape(text)
    import re as _re
    text = _re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def text_overlap_score(
    expected_passages: list[str],
    retrieved_texts: list[str],
    min_length: int = 20,
) -> float:
    """Fraction of expected passages found (as normalized substrings) in any retrieved chunk.

    Normalization: HTML entity decoding, whitespace collapsing, case-insensitive.
    Passages shorter than min_length (after normalization) are skipped.
    """
    if not expected_passages:
        return 0.0
    hits = 0
    norm_chunks = [normalize_text(t) for t in retrieved_texts]
    for passage in expected_passages:
        norm_p = normalize_text(passage)
        if len(norm_p) < min_length:
            continue
        if any(norm_p in nc for nc in norm_chunks):
            hits += 1
    return hits / len(expected_passages)


def _compute_metrics(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> SignalMetrics:
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(relevant_ids) if relevant_ids else 0.0
    return SignalMetrics(precision_at_k=precision, recall_at_k=recall)


def _bm25_ranking(index: HybridIndex, query: str, top_k: int) -> list[str]:
    results = index.bm25_retrieve(query, top_k=top_k)
    return [r.chunk_id for r in results]


def _text_cosine_ranking(index: HybridIndex, query: str, top_k: int) -> list[str]:
    query_emb = _make_embedding(query, mode="query")
    if query_emb is None:
        return []
    scores: list[tuple[str, float]] = []
    for chunk in index.chunks:
        if chunk.embedding is not None:
            cosine = sum(a * b for a, b in zip(query_emb, chunk.embedding))
            scores.append((chunk.chunk_id, cosine))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in scores[:top_k]]


def _image_cosine_ranking(index: HybridIndex, query: str, top_k: int) -> list[str]:
    ranked = index._image_cosine_rank(query)
    return [cid for cid, _ in ranked[:top_k]]


def _graph_ranking(index: HybridIndex, top_k: int) -> list[str]:
    ranked = index._graph_rank()
    return [cid for cid, _ in ranked[:top_k]]


def _maxsim_ranking(index: HybridIndex, query: str, store: "ResearchStore | None", top_k: int) -> list[str]:
    if store is None:
        return []
    ranked = index._maxsim_rank(query, store, top_k=top_k)
    return [cid for cid, _ in ranked[:top_k]]


def _entity_ranking(index: HybridIndex, query: str, top_k: int) -> list[str]:
    query_emb = _make_embedding(query, mode="query")
    if query_emb is None:
        return []
    ranked = index._entity_rank(query_emb)
    return [cid for cid, _ in ranked[:top_k]]


def evaluate(
    index: HybridIndex, eval_set: list[EvalQuery], top_k: int = 10,
    store: "ResearchStore | None" = None,
) -> EvalReport:
    query_results: list[QueryResult] = []
    all_overall = []
    all_per_signal: dict[str, list[SignalMetrics]] = {}

    for eq in eval_set:
        relevant = set(eq.relevant_chunk_ids)

        overall_results = index.hybrid_retrieve(eq.query, top_k=top_k, rerank=False, store=store)
        overall_ids = [r.chunk_id for r in overall_results]
        overall_metrics = _compute_metrics(overall_ids, relevant, top_k)

        per_signal: dict[str, SignalMetrics] = {}

        bm25_ids = _bm25_ranking(index, eq.query, top_k)
        per_signal["bm25"] = _compute_metrics(bm25_ids, relevant, top_k)

        cosine_ids = _text_cosine_ranking(index, eq.query, top_k)
        if cosine_ids:
            per_signal["text_cosine"] = _compute_metrics(cosine_ids, relevant, top_k)

        image_ids = _image_cosine_ranking(index, eq.query, top_k)
        if image_ids:
            per_signal["image_cosine"] = _compute_metrics(image_ids, relevant, top_k)

        graph_ids = _graph_ranking(index, top_k)
        if graph_ids:
            per_signal["graph"] = _compute_metrics(graph_ids, relevant, top_k)

        maxsim_ids = _maxsim_ranking(index, eq.query, store, top_k)
        if maxsim_ids:
            per_signal["maxsim"] = _compute_metrics(maxsim_ids, relevant, top_k)

        if ENTITY_RESOLUTION_ENABLED:
            entity_ids = _entity_ranking(index, eq.query, top_k)
            if entity_ids:
                per_signal["entity"] = _compute_metrics(entity_ids, relevant, top_k)

        qr = QueryResult(query=eq.query, overall=overall_metrics, per_signal=per_signal)
        query_results.append(qr)
        all_overall.append(overall_metrics)
        for name, m in per_signal.items():
            all_per_signal.setdefault(name, []).append(m)

    n = len(eval_set)
    avg_overall = SignalMetrics(
        precision_at_k=sum(m.precision_at_k for m in all_overall) / n if n else 0.0,
        recall_at_k=sum(m.recall_at_k for m in all_overall) / n if n else 0.0,
    )
    avg_per_signal: dict[str, SignalMetrics] = {}
    for name, metrics_list in all_per_signal.items():
        cnt = len(metrics_list)
        avg_per_signal[name] = SignalMetrics(
            precision_at_k=sum(m.precision_at_k for m in metrics_list) / cnt,
            recall_at_k=sum(m.recall_at_k for m in metrics_list) / cnt,
        )

    return EvalReport(
        top_k=top_k,
        num_queries=n,
        overall=avg_overall,
        per_signal=avg_per_signal,
        query_results=query_results,
    )


def format_report(report: EvalReport) -> str:
    lines = [
        f"Eval Report  (top_k={report.top_k}, queries={report.num_queries})",
        "=" * 55,
        f"{'Signal':<16} {'P@K':>8} {'R@K':>8}",
        "-" * 55,
        f"{'overall':<16} {report.overall.precision_at_k:>8.3f} {report.overall.recall_at_k:>8.3f}",
    ]
    for name, m in sorted(report.per_signal.items()):
        lines.append(f"{name:<16} {m.precision_at_k:>8.3f} {m.recall_at_k:>8.3f}")
    lines.append("=" * 55)
    return "\n".join(lines)


# ── Answer Quality Evaluation ────────────────────────────────────────────────

_ANSWER_QUALITY_SYSTEM_PROMPT = """\
You are an evaluation judge scoring the quality of a RAG-generated answer.

Score the answer on three dimensions, each from 0.0 to 1.0:

1. **relevance** — Does the answer address the question asked? \
(1.0 = directly answers; 0.0 = completely off-topic)
2. **groundedness** — Is the answer grounded in the provided context? \
(1.0 = every claim is supported by the context; 0.0 = fabricated)
3. **completeness** — Does the answer cover the key aspects of the question? \
(1.0 = comprehensive; 0.0 = misses all important points)

Respond with ONLY a JSON object: {"relevance": 0.0, "groundedness": 0.0, "completeness": 0.0}\
"""


def _score_answer(
    llm,
    question: str,
    answer: str,
    context: str,
    reference_answer: str | None = None,
) -> AnswerQualityMetrics:
    ref_block = ""
    if reference_answer:
        ref_block = f"\n\nReference answer (for comparison):\n{reference_answer}"

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Context provided to the system:\n{context[:3000]}\n\n"
        f"System's answer:\n{answer}{ref_block}"
    )

    try:
        raw = llm.generate(_ANSWER_QUALITY_SYSTEM_PROMPT, user_prompt)
    except Exception:
        logger.warning("Answer quality LLM scoring failed", exc_info=True)
        return AnswerQualityMetrics(relevance=float("nan"), groundedness=float("nan"), completeness=float("nan"))

    try:
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("no JSON found")
        parsed = json.loads(text[start:end + 1])
        return AnswerQualityMetrics(
            relevance=float(parsed.get("relevance", 0.0)),
            groundedness=float(parsed.get("groundedness", 0.0)),
            completeness=float(parsed.get("completeness", 0.0)),
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("Answer quality scorer returned malformed JSON: %s", raw[:200])
        return AnswerQualityMetrics(relevance=float("nan"), groundedness=float("nan"), completeness=float("nan"))


def evaluate_answer_quality(
    index: HybridIndex,
    eval_set: list[EvalQuery],
    llm,
    top_k: int = 5,
    store: "ResearchStore | None" = None,
) -> AnswerQualityReport:
    results: list[AnswerQualityResult] = []

    for eq in eval_set:
        retrieved = index.hybrid_retrieve(eq.query, top_k=top_k, rerank=False, store=store)
        context = "\n\n---\n\n".join(r.text for r in retrieved)

        answer_prompt = (
            f"Based on the following context, answer the question.\n\n"
            f"Context:\n{context[:4000]}\n\n"
            f"Question: {eq.query}"
        )
        try:
            answer = llm.generate("You are a helpful research assistant.", answer_prompt)
        except Exception:
            logger.warning("Answer generation failed for query: %s", eq.query[:80])
            answer = ""

        metrics = _score_answer(llm, eq.query, answer, context, eq.reference_answer)
        results.append(AnswerQualityResult(
            query=eq.query,
            answer=answer,
            metrics=metrics,
            reference_answer=eq.reference_answer,
        ))

    valid = [r.metrics for r in results if not math.isnan(r.metrics.relevance)]
    n = len(valid)
    if n > 0:
        avg = AnswerQualityMetrics(
            relevance=sum(m.relevance for m in valid) / n,
            groundedness=sum(m.groundedness for m in valid) / n,
            completeness=sum(m.completeness for m in valid) / n,
        )
    else:
        avg = AnswerQualityMetrics(relevance=0.0, groundedness=0.0, completeness=0.0)

    return AnswerQualityReport(num_queries=len(results), average=avg, results=results)


def format_answer_quality_report(report: AnswerQualityReport) -> str:
    lines = [
        f"Answer Quality Report  (queries={report.num_queries})",
        "=" * 55,
        f"{'Dimension':<16} {'Score':>8}",
        "-" * 55,
        f"{'relevance':<16} {report.average.relevance:>8.3f}",
        f"{'groundedness':<16} {report.average.groundedness:>8.3f}",
        f"{'completeness':<16} {report.average.completeness:>8.3f}",
        f"{'mean':<16} {report.average.mean_score():>8.3f}",
        "=" * 55,
    ]
    return "\n".join(lines)


# ── RRF Weight Sweep ─────────────────────────────────────────────────────────

_SWEEP_WEIGHTS = ["bm25", "text_cosine", "maxsim", "hyde_cosine"]
_SWEEP_VALUES = [0.5, 1.0, 1.5]
_MAX_SWEEP_COMBINATIONS = 10_000


def rrf_weight_sweep(
    index: HybridIndex,
    eval_set: list[EvalQuery],
    top_k: int = 10,
    store: "ResearchStore | None" = None,
    sweep_weights: list[str] | None = None,
    sweep_values: list[float] | None = None,
) -> SweepReport:
    weights_to_sweep = sweep_weights or _SWEEP_WEIGHTS
    values = sweep_values or _SWEEP_VALUES

    num_combinations = len(values) ** len(weights_to_sweep)
    if num_combinations > _MAX_SWEEP_COMBINATIONS:
        raise ValueError(
            f"Sweep would produce {num_combinations:,} combinations "
            f"({len(values)} values ^ {len(weights_to_sweep)} weights), "
            f"exceeding cap of {_MAX_SWEEP_COMBINATIONS:,}. "
            f"Reduce sweep_values or sweep_weights."
        )

    combinations = list(itertools.product(values, repeat=len(weights_to_sweep)))
    all_results: list[SweepResult] = []

    for combo in combinations:
        weight_overrides = dict(zip(weights_to_sweep, combo))

        all_overall: list[SignalMetrics] = []
        for eq in eval_set:
            relevant = set(eq.relevant_chunk_ids)
            results = index.hybrid_retrieve(
                eq.query, top_k=top_k, rerank=False, store=store,
                weight_overrides=weight_overrides,
            )
            retrieved_ids = [r.chunk_id for r in results]
            metrics = _compute_metrics(retrieved_ids, relevant, top_k)
            all_overall.append(metrics)

        n = len(eval_set)
        avg_p = sum(m.precision_at_k for m in all_overall) / n if n else 0.0
        avg_r = sum(m.recall_at_k for m in all_overall) / n if n else 0.0

        all_results.append(SweepResult(
            weights=weight_overrides,
            precision_at_k=avg_p,
            recall_at_k=avg_r,
        ))

    all_results.sort(key=lambda r: r.precision_at_k, reverse=True)
    best = all_results[0] if all_results else SweepResult(weights={}, precision_at_k=0.0, recall_at_k=0.0)

    return SweepReport(
        top_k=top_k,
        num_queries=len(eval_set),
        num_combinations=len(combinations),
        best=best,
        all_results=all_results,
    )


def format_sweep_report(report: SweepReport) -> str:
    lines = [
        f"RRF Weight Sweep  (top_k={report.top_k}, queries={report.num_queries}, "
        f"combinations={report.num_combinations})",
        "=" * 70,
        "Best configuration:",
        f"  P@K: {report.best.precision_at_k:.3f}  R@K: {report.best.recall_at_k:.3f}",
        f"  Weights: {json.dumps(report.best.weights)}",
        "-" * 70,
        "Top 5 configurations:",
    ]
    for i, r in enumerate(report.all_results[:5], 1):
        lines.append(f"  {i}. P@K={r.precision_at_k:.3f} R@K={r.recall_at_k:.3f} {json.dumps(r.weights)}")
    lines.append("=" * 70)
    return "\n".join(lines)
