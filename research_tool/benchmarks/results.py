"""Benchmark results schema, serialization, and comparison."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _sanitize_for_json(obj: Any) -> Any:
    """Replace NaN/Inf with None for JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


@dataclass
class IngestionMetrics:
    total_time_s: float = 0.0
    chunk_count: int = 0
    time_per_chunk_ms: float = 0.0


@dataclass
class RetrievalMetrics:
    mean_text_overlap: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    query_count: int = 0


@dataclass
class QueryDetail:
    query: str = ""
    source_doc: str = ""
    difficulty: str = ""
    text_overlap: float = 0.0
    latency_ms: float = 0.0
    retrieved_count: int = 0
    retrieved_ids: list[str] = field(default_factory=list)


@dataclass
class ConfigReport:
    config_name: str = ""
    config_overrides: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    ingestion: IngestionMetrics = field(default_factory=IngestionMetrics)
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    queries: list[QueryDetail] = field(default_factory=list)
    error: str | None = None


@dataclass
class BenchmarkReport:
    timestamp: str = ""
    git_sha: str = ""
    python_version: str = ""
    platform_info: str = ""
    total_wall_time_s: float = 0.0
    config_count: int = 0
    query_count: int = 0
    configs: list[ConfigReport] = field(default_factory=list)

    def to_dict(self) -> dict:
        return _sanitize_for_json(asdict(self))

    @classmethod
    def from_dict(cls, data: dict) -> BenchmarkReport:
        report = cls(
            timestamp=data.get("timestamp", ""),
            git_sha=data.get("git_sha", ""),
            python_version=data.get("python_version", ""),
            platform_info=data.get("platform_info", ""),
            total_wall_time_s=data.get("total_wall_time_s", 0.0),
            config_count=data.get("config_count", 0),
            query_count=data.get("query_count", 0),
        )
        for cfg_data in data.get("configs", []):
            ing = cfg_data.get("ingestion", {})
            ret = cfg_data.get("retrieval", {})
            queries = []
            for qd in cfg_data.get("queries", []):
                queries.append(QueryDetail(
                    query=qd.get("query", ""),
                    source_doc=qd.get("source_doc", ""),
                    difficulty=qd.get("difficulty", ""),
                    text_overlap=qd.get("text_overlap", 0.0) or 0.0,
                    latency_ms=qd.get("latency_ms", 0.0) or 0.0,
                    retrieved_count=qd.get("retrieved_count", 0),
                    retrieved_ids=qd.get("retrieved_ids", []),
                ))
            report.configs.append(ConfigReport(
                config_name=cfg_data.get("config_name", ""),
                config_overrides=cfg_data.get("config_overrides", {}),
                description=cfg_data.get("description", ""),
                ingestion=IngestionMetrics(
                    total_time_s=ing.get("total_time_s", 0.0) or 0.0,
                    chunk_count=ing.get("chunk_count", 0),
                    time_per_chunk_ms=ing.get("time_per_chunk_ms", 0.0) or 0.0,
                ),
                retrieval=RetrievalMetrics(
                    mean_text_overlap=ret.get("mean_text_overlap", 0.0) or 0.0,
                    p50_latency_ms=ret.get("p50_latency_ms", 0.0) or 0.0,
                    p95_latency_ms=ret.get("p95_latency_ms", 0.0) or 0.0,
                    p99_latency_ms=ret.get("p99_latency_ms", 0.0) or 0.0,
                    query_count=ret.get("query_count", 0),
                ),
                queries=queries,
                error=cfg_data.get("error"),
            ))
        return report


def build_report(config_results: list, total_wall_time: float) -> BenchmarkReport:
    """Build a BenchmarkReport from runner ConfigResult objects."""
    from research_tool.benchmarks.runner import ConfigResult

    report = BenchmarkReport(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        git_sha=_get_git_sha(),
        python_version=sys.version.split()[0],
        platform_info=f"{platform.system()} {platform.machine()}",
        total_wall_time_s=total_wall_time,
        config_count=len(config_results),
        query_count=len(config_results[0].per_query_results) if config_results else 0,
    )

    for cr in config_results:
        tpc = (cr.ingestion_time * 1000 / cr.chunk_count) if cr.chunk_count > 0 else 0.0
        cfg_report = ConfigReport(
            config_name=cr.config_name,
            config_overrides={k: _coerce_override(v) for k, v in cr.config_overrides.items()},
            ingestion=IngestionMetrics(
                total_time_s=cr.ingestion_time,
                chunk_count=cr.chunk_count,
                time_per_chunk_ms=tpc,
            ),
            retrieval=RetrievalMetrics(
                mean_text_overlap=cr.retrieval_metrics.get("mean_text_overlap", 0.0),
                p50_latency_ms=cr.retrieval_metrics.get("retrieval_p50_ms", 0.0),
                p95_latency_ms=cr.retrieval_metrics.get("retrieval_p95_ms", 0.0),
                p99_latency_ms=cr.retrieval_metrics.get("retrieval_p99_ms", 0.0),
                query_count=len(cr.per_query_results),
            ),
            error=cr.error,
        )
        for pqr in cr.per_query_results:
            cfg_report.queries.append(QueryDetail(
                query=pqr["query"],
                source_doc=pqr.get("source_doc", ""),
                difficulty=pqr["difficulty"],
                text_overlap=pqr["text_overlap"],
                latency_ms=pqr["latency"] * 1000,
                retrieved_count=pqr["retrieved_count"],
                retrieved_ids=pqr["retrieved_ids"],
            ))
        report.configs.append(cfg_report)

    return report


def _coerce_override(v: Any) -> Any:
    """Ensure override values are JSON-serializable."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float, str)):
        return v
    return str(v)


def save_report(report: BenchmarkReport, output_dir: Path | None = None) -> Path:
    """Save report to a timestamped JSON file. Returns the path."""
    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    path = out_dir / f"benchmark-results-{ts}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    return path


def load_report(path: Path | str) -> BenchmarkReport:
    """Load a BenchmarkReport from a JSON file."""
    data = json.loads(Path(path).read_text())
    return BenchmarkReport.from_dict(data)


# ── Comparison ────────────────────────────────────────────────────────────────


@dataclass
class ConfigDiff:
    config_name: str
    ingestion_time_delta_pct: float | None = None
    text_overlap_delta: float | None = None
    p50_latency_delta_pct: float | None = None
    p95_latency_delta_pct: float | None = None
    chunk_count_a: int = 0
    chunk_count_b: int = 0
    only_in: str | None = None


def compare_reports(report_a: BenchmarkReport, report_b: BenchmarkReport) -> list[ConfigDiff]:
    """Compare two benchmark reports, returning per-config diffs.

    report_a is the baseline, report_b is the new run.
    Positive delta = improvement (higher overlap or lower latency).
    """
    a_map = {c.config_name: c for c in report_a.configs}
    b_map = {c.config_name: c for c in report_b.configs}

    all_names = sorted(set(a_map) | set(b_map))
    diffs: list[ConfigDiff] = []

    for name in all_names:
        ca = a_map.get(name)
        cb = b_map.get(name)

        if ca is None:
            diffs.append(ConfigDiff(config_name=name, only_in="B"))
            continue
        if cb is None:
            diffs.append(ConfigDiff(config_name=name, only_in="A"))
            continue

        def _pct_delta(old: float, new: float) -> float | None:
            if old == 0:
                return None
            return ((new - old) / old) * 100

        diffs.append(ConfigDiff(
            config_name=name,
            ingestion_time_delta_pct=_pct_delta(ca.ingestion.total_time_s, cb.ingestion.total_time_s),
            text_overlap_delta=cb.retrieval.mean_text_overlap - ca.retrieval.mean_text_overlap,
            p50_latency_delta_pct=_pct_delta(ca.retrieval.p50_latency_ms, cb.retrieval.p50_latency_ms),
            p95_latency_delta_pct=_pct_delta(ca.retrieval.p95_latency_ms, cb.retrieval.p95_latency_ms),
            chunk_count_a=ca.ingestion.chunk_count,
            chunk_count_b=cb.ingestion.chunk_count,
        ))

    return diffs


def format_comparison(diffs: list[ConfigDiff]) -> str:
    """Format comparison as a table."""
    lines = [
        f"{'Config':<45} {'Overlap Δ':>10} {'Ingest Δ%':>10} {'p50 Δ%':>8} {'p95 Δ%':>8} {'Chunks A→B':>12}",
        "=" * 95,
    ]
    for d in diffs:
        if d.only_in:
            lines.append(f"{d.config_name:<45} {'N/A (only in ' + d.only_in + ')':>50}")
            continue

        def _fmt_pct(v: float | None) -> str:
            if v is None:
                return "N/A"
            sign = "+" if v > 0 else ""
            return f"{sign}{v:.1f}%"

        def _fmt_delta(v: float | None) -> str:
            if v is None:
                return "N/A"
            sign = "+" if v > 0 else ""
            return f"{sign}{v:.3f}"

        lines.append(
            f"{d.config_name:<45} "
            f"{_fmt_delta(d.text_overlap_delta):>10} "
            f"{_fmt_pct(d.ingestion_time_delta_pct):>10} "
            f"{_fmt_pct(d.p50_latency_delta_pct):>8} "
            f"{_fmt_pct(d.p95_latency_delta_pct):>8} "
            f"{d.chunk_count_a}→{d.chunk_count_b:>5}"
        )
    return "\n".join(lines)
