"""Benchmark runner: iterates configs, ingests corpus, evaluates retrieval."""

from __future__ import annotations

import hashlib
import html as html_lib
import json
import logging
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research_tool.benchmarks.configs import (
    DUAL_PATCH_ATTRS,
    STORE_ONLY_ATTRS,
    BenchmarkConfig,
)

logger = logging.getLogger(__name__)

BENCHMARKS_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BENCHMARKS_DIR / "corpus"
GROUND_TRUTH_PATH = BENCHMARKS_DIR / "ground_truth.json"
HYDE_CACHE_PATH = BENCHMARKS_DIR / "hyde_cache.json"


# ── Text-overlap metric ──────────────────────────────────────────────────────


def normalize_text(text: str) -> str:
    """Normalize text for overlap comparison.

    Applies: HTML entity decoding, whitespace collapsing, case folding.
    """
    text = html_lib.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


_BOUNDARY_CHARS = set(' \t\n\r.,;:!?"\'-()[]{}/<>=@#$%^&*~`|\\')


def _is_boundary(ch: str) -> bool:
    """Return True if character is a word boundary (whitespace or punctuation)."""
    return ch in _BOUNDARY_CHARS


def text_overlap(passage: str, chunk_text: str, min_length: int = 4) -> bool:
    """Check if normalized passage appears as substring in normalized chunk.

    For passages shorter than 20 characters, a word-boundary check is applied:
    the match must not be embedded inside a longer word. The character immediately
    before and after the match (if any) must be whitespace or punctuation.

    Passages shorter than ``min_length`` (default 4) are rejected outright because
    they are too ambiguous for reliable substring matching.
    """
    norm_passage = normalize_text(passage)
    if len(norm_passage) < min_length:
        return False
    norm_chunk = normalize_text(chunk_text)

    # Long passages: simple substring match (no boundary check needed)
    if len(norm_passage) >= 20:
        return norm_passage in norm_chunk

    # Short passages (4-19 chars): require word-boundary check
    start = 0
    while True:
        idx = norm_chunk.find(norm_passage, start)
        if idx == -1:
            return False
        # Check boundary before match
        if idx > 0 and not _is_boundary(norm_chunk[idx - 1]):
            start = idx + 1
            continue
        # Check boundary after match
        end = idx + len(norm_passage)
        if end < len(norm_chunk) and not _is_boundary(norm_chunk[end]):
            start = idx + 1
            continue
        return True


def text_overlap_score(
    expected_passages: list[str],
    retrieved_texts: list[str],
) -> float:
    """Fraction of expected passages found in any retrieved chunk."""
    if not expected_passages:
        return 0.0
    hits = 0
    for passage in expected_passages:
        for chunk_text in retrieved_texts:
            if text_overlap(passage, chunk_text):
                hits += 1
                break
    return hits / len(expected_passages)


# ── Module attribute patching ─────────────────────────────────────────────────


def _save_originals() -> dict[str, Any]:
    """Snapshot current module attribute values for restoration."""
    import research_tool.brain as brain
    import research_tool.store as store

    originals: dict[str, Any] = {}
    for attr in DUAL_PATCH_ATTRS:
        originals[f"store.{attr}"] = getattr(store, attr)
        if hasattr(brain, attr):
            originals[f"brain.{attr}"] = getattr(brain, attr)
    for attr in STORE_ONLY_ATTRS:
        originals[f"store.{attr}"] = getattr(store, attr)
    return originals


def _apply_overrides(overrides: dict[str, object]) -> None:
    """Patch module attributes for a benchmark configuration."""
    import research_tool.brain as brain
    import research_tool.store as store

    for attr, value in overrides.items():
        if attr in DUAL_PATCH_ATTRS:
            setattr(store, attr, value)
            if hasattr(brain, attr):
                setattr(brain, attr, value)
        elif attr in STORE_ONLY_ATTRS:
            setattr(store, attr, value)
        else:
            logger.warning("Unknown config attribute: %s", attr)


def _restore_originals(originals: dict[str, Any]) -> None:
    """Restore module attributes from snapshot."""
    import research_tool.brain as brain
    import research_tool.store as store

    for key, value in originals.items():
        module_name, attr = key.split(".", 1)
        mod = store if module_name == "store" else brain
        setattr(mod, attr, value)


# ── Corpus loading ────────────────────────────────────────────────────────────


@dataclass
class CorpusDocument:
    path: str
    url: str
    doc_type: str
    language: str | None = None
    text: str = ""


@dataclass
class GroundTruthQuery:
    query: str
    expected_passages: list[str]
    source_doc: str
    expected_content_types: list[str]
    difficulty: str


def load_corpus() -> tuple[list[CorpusDocument], list[GroundTruthQuery]]:
    """Load benchmark corpus documents and ground-truth queries."""
    data = json.loads(GROUND_TRUTH_PATH.read_text())

    docs = []
    for d in data["documents"]:
        path = BENCHMARKS_DIR / d["path"]
        docs.append(CorpusDocument(
            path=d["path"],
            url=d["url"],
            doc_type=d["type"],
            language=d.get("language"),
            text=path.read_text(),
        ))

    queries = []
    for q in data["queries"]:
        queries.append(GroundTruthQuery(
            query=q["query"],
            expected_passages=q["expected_passages"],
            source_doc=q["source_doc"],
            expected_content_types=q["expected_content_types"],
            difficulty=q["difficulty"],
        ))

    return docs, queries


# ── Ingestion pipeline ────────────────────────────────────────────────────────


def _make_chunk_id(url: str, index: int) -> str:
    key = f"{url}::bench::{index}"
    return hashlib.sha256(key.encode()).hexdigest()[:16] + f"::chunk-{index}"


def _ingest_document(doc: CorpusDocument, store_inst: "ResearchStore", config: BenchmarkConfig) -> int:
    """Ingest a single document using current module config. Returns chunk count."""
    from research_tool.code_chunker import chunk_code_file
    from research_tool.store import (
        DocumentChunk,
        _make_code_embedding,
        _make_embedding,
        _make_token_embeddings,
        classify_chunk_content_type,
        split_into_children,
    )
    from research_tool.wiki_chunker import chunk_wiki_content

    import research_tool.store as store_mod

    # Store page first — chunks FK references pages.url
    store_inst.store_page(url=doc.url, title=doc.path, extracted_text=doc.text[:500])

    chunks: list[DocumentChunk] = []

    chunk_mode = getattr(store_mod, "CHUNK_MODE", "auto")

    def _do_wiki() -> list[DocumentChunk]:
        return chunk_wiki_content(doc.text, doc.url)

    def _do_code() -> list[DocumentChunk]:
        code_chunks = chunk_code_file(
            doc.path, doc.text, language=doc.language,
        )
        result: list[DocumentChunk] = []
        for i, cc in enumerate(code_chunks):
            cid = _make_chunk_id(doc.url, i)
            result.append(DocumentChunk(
                text=cc.text,
                page_url=doc.url,
                chunk_id=cid,
                section_title=cc.metadata.get("signature", doc.path),
                content_type="code",
            ))
        return result

    def _do_paragraph() -> list[DocumentChunk]:
        from research_tool.store import chunk_web_content
        return chunk_web_content(doc.text, doc.url)

    if chunk_mode == "wiki":
        chunks = _do_wiki()
    elif chunk_mode == "code":
        if doc.doc_type == "code":
            chunks = _do_code()
        else:
            chunks = _do_paragraph()
    elif chunk_mode == "paragraph":
        chunks = _do_paragraph()
    else:
        # auto: route by doc_type (original production behavior)
        if doc.doc_type == "wiki":
            chunks = _do_wiki()
        elif doc.doc_type == "code":
            chunks = _do_code()
        else:
            chunks = _do_paragraph()

    if not chunks:
        return 0

    # Classify content type
    for chunk in chunks:
        chunk.content_type = classify_chunk_content_type(chunk.text)

    # Embed
    jina_enabled = getattr(store_mod, "JINA_CODE_EMBEDDING_ENABLED", True)
    token_enabled = getattr(store_mod, "TOKEN_EMBEDDING_ENABLED", True)

    for chunk in chunks:
        chunk.embedding = _make_embedding(chunk.text, mode="document")
        if jina_enabled and chunk.content_type == "code":
            chunk.code_embedding = _make_code_embedding(chunk.text)

    # Parent-child splitting
    if store_mod.PARENT_CHILD_ENABLED:
        all_chunks = []
        for chunk in chunks:
            children = split_into_children(chunk, max_tokens=store_mod.CHILD_CHUNK_MAX_TOKENS)
            if children:
                for child in children:
                    child.embedding = _make_embedding(child.text, mode="document")
                    if jina_enabled and child.content_type == "code":
                        child.code_embedding = _make_code_embedding(child.text)
                all_chunks.append(chunk)
                all_chunks.extend(children)
            else:
                all_chunks.append(chunk)
        chunks = all_chunks

    # Store chunks
    store_inst.store_chunks(chunks)

    # Token embeddings (ColBERT/FDE)
    if token_enabled:
        for chunk in chunks:
            tok_emb = _make_token_embeddings(chunk.text)
            if tok_emb is not None:
                store_inst.store_token_embeddings(chunk.chunk_id, tok_emb)

    return len(chunks)


# ── Timing helpers ────────────────────────────────────────────────────────────


@dataclass
class TimingStats:
    values: list[float] = field(default_factory=list)

    def add(self, v: float) -> None:
        self.values.append(v)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def total(self) -> float:
        return sum(self.values)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def percentile(self, p: float) -> float:
        if not self.values:
            return 0.0
        s = sorted(self.values)
        idx = int(len(s) * p / 100.0)
        idx = min(idx, len(s) - 1)
        return s[idx]


# ── Per-config result ─────────────────────────────────────────────────────────


@dataclass
class ConfigResult:
    config_name: str
    config_overrides: dict[str, object]
    ingestion_time: float = 0.0
    chunk_count: int = 0
    retrieval_metrics: dict[str, Any] = field(default_factory=dict)
    per_query_results: list[dict[str, Any]] = field(default_factory=list)
    retrieval_latency: TimingStats = field(default_factory=TimingStats)
    text_overlap_scores: list[float] = field(default_factory=list)
    error: str | None = None


# ── Parent expansion (matches production brain.py behavior) ──────────────────


def _expand_to_parents(
    results: list, store_inst: "ResearchStore", top_k: int = 10,
) -> list:
    """Replace child chunk text with parent text, deduplicating shared parents.

    After deduplication, backfills from remaining ranked results to maintain top_k.
    """
    from dataclasses import replace

    expanded = []
    seen_parents: set[str] = set()
    skipped = []
    for r in results:
        child_chunk = store_inst.get_chunk(r.chunk_id)
        if child_chunk and child_chunk.is_child and child_chunk.parent_chunk_id:
            if child_chunk.parent_chunk_id in seen_parents:
                skipped.append(r)
                continue
            seen_parents.add(child_chunk.parent_chunk_id)
            parent = store_inst.get_chunk(child_chunk.parent_chunk_id)
            if parent:
                r = replace(r, text=parent.text, chunk_id=parent.chunk_id)
        expanded.append(r)

    if len(expanded) < top_k and skipped:
        seen_ids = {r.chunk_id for r in expanded}
        for r in skipped:
            if r.chunk_id not in seen_ids:
                expanded.append(r)
                seen_ids.add(r.chunk_id)
            if len(expanded) >= top_k:
                break

    return expanded


# ── Main runner ───────────────────────────────────────────────────────────────


@dataclass
class ProgressCallback:
    """Override to receive progress updates."""
    def on_config_start(self, index: int, total: int, config_name: str) -> None:
        pass

    def on_config_done(self, index: int, total: int, result: "ConfigResult") -> None:
        pass


_INGESTION_KEY_ATTRS = ("CHUNK_MODE", "PARENT_CHILD_ENABLED",
                        "JINA_CODE_EMBEDDING_ENABLED", "TOKEN_EMBEDDING_ENABLED")


def _ingestion_cache_key(config: BenchmarkConfig) -> str:
    parts = {attr: config.module_overrides.get(attr) for attr in _INGESTION_KEY_ATTRS}
    return json.dumps(parts, sort_keys=True)


def _install_embedding_cache() -> dict:
    import research_tool.store as store_mod

    cache: dict[tuple, Any] = {}
    orig_make_embedding = store_mod._make_embedding
    orig_make_code_embedding = store_mod._make_code_embedding
    orig_make_token_embeddings = store_mod._make_token_embeddings

    def cached_make_embedding(text, mode="document"):
        key = ("emb", text, mode)
        if key not in cache:
            cache[key] = orig_make_embedding(text, mode)
        return cache[key]

    def cached_make_code_embedding(text):
        key = ("code", text)
        if key not in cache:
            cache[key] = orig_make_code_embedding(text)
        return cache[key]

    def cached_make_token_embeddings(text):
        key = ("tok", text)
        if key not in cache:
            cache[key] = orig_make_token_embeddings(text)
        return cache[key]

    store_mod._make_embedding = cached_make_embedding
    store_mod._make_code_embedding = cached_make_code_embedding
    store_mod._make_token_embeddings = cached_make_token_embeddings

    return {
        "_make_embedding": orig_make_embedding,
        "_make_code_embedding": orig_make_code_embedding,
        "_make_token_embeddings": orig_make_token_embeddings,
    }


def _uninstall_embedding_cache(originals: dict) -> None:
    import research_tool.store as store_mod
    for name, func in originals.items():
        setattr(store_mod, name, func)


class BenchmarkRunner:
    def __init__(
        self,
        configs: list[BenchmarkConfig],
        top_k: int = 10,
        progress: ProgressCallback | None = None,
    ):
        self.configs = configs
        self.top_k = top_k
        self.progress = progress or ProgressCallback()

    def run(self) -> list[ConfigResult]:
        docs, queries = load_corpus()
        results: list[ConfigResult] = []
        originals = _save_originals()

        cache_dir = tempfile.mkdtemp(prefix="bench_cache_")
        ingestion_cache: dict[str, tuple[str, int]] = {}
        embed_originals = _install_embedding_cache()

        try:
            total = len(self.configs)
            for i, config in enumerate(self.configs):
                self.progress.on_config_start(i, total, config.name)
                result = self._run_single(
                    config, docs, queries, originals,
                    cache_dir, ingestion_cache,
                )
                results.append(result)

                self.progress.on_config_done(i, total, result)
        finally:
            _uninstall_embedding_cache(embed_originals)
            shutil.rmtree(cache_dir, ignore_errors=True)

        return results

    def _run_single(
        self,
        config: BenchmarkConfig,
        docs: list[CorpusDocument],
        queries: list[GroundTruthQuery],
        originals: dict[str, Any],
        cache_dir: str = "",
        ingestion_cache: dict[str, tuple[str, int]] | None = None,
    ) -> ConfigResult:
        from research_tool.eval import EvalQuery, evaluate
        from research_tool.store import HybridIndex, ResearchStore

        result = ConfigResult(
            config_name=config.name,
            config_overrides=config.module_overrides,
        )

        try:
            _apply_overrides(config.module_overrides)

            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = str(Path(tmpdir) / "bench.db")

                ikey = _ingestion_cache_key(config) if ingestion_cache is not None else None
                cached = ingestion_cache.get(ikey) if ikey else None

                if cached:
                    cached_db_path, total_chunks = cached
                    t0 = time.perf_counter()
                    shutil.copy2(cached_db_path, db_path)
                    result.ingestion_time = time.perf_counter() - t0
                    result.chunk_count = total_chunks
                    store_inst = ResearchStore(db_path=db_path)
                else:
                    store_inst = ResearchStore(db_path=db_path)
                    t0 = time.perf_counter()
                    total_chunks = 0
                    for doc in docs:
                        total_chunks += _ingest_document(doc, store_inst, config)
                    result.ingestion_time = time.perf_counter() - t0
                    result.chunk_count = total_chunks

                    if ikey is not None and ingestion_cache is not None:
                        store_inst.close()
                        cached_db = str(Path(cache_dir) / f"{hashlib.md5(ikey.encode()).hexdigest()}.db")
                        shutil.copy2(db_path, cached_db)
                        ingestion_cache[ikey] = (cached_db, total_chunks)
                        store_inst = ResearchStore(db_path=db_path)

                # Build index
                index = HybridIndex(mode="query")
                index.build_from_store(store_inst)

                # Retrieval phase — per-query timing and text-overlap
                eval_queries: list[EvalQuery] = []
                for gq in queries:
                    eq = EvalQuery(
                        query=gq.query,
                        relevant_chunk_ids=[],
                        reference_answer=None,
                    )
                    eval_queries.append(eq)

                parent_child = config.module_overrides.get("PARENT_CHILD_ENABLED", False)

                retrieval_k = self.top_k * 2 if parent_child else self.top_k

                for j, gq in enumerate(queries):
                    t_start = time.perf_counter()
                    retrieved = index.hybrid_retrieve(
                        gq.query, top_k=retrieval_k, rerank=False, store=store_inst,
                    )
                    if parent_child:
                        retrieved = _expand_to_parents(retrieved, store_inst, self.top_k)
                        retrieved = retrieved[:self.top_k]
                    latency = time.perf_counter() - t_start
                    result.retrieval_latency.add(latency)

                    retrieved_texts = [r.text for r in retrieved]
                    overlap = text_overlap_score(gq.expected_passages, retrieved_texts)
                    result.text_overlap_scores.append(overlap)

                    result.per_query_results.append({
                        "query": gq.query,
                        "source_doc": gq.source_doc,
                        "difficulty": gq.difficulty,
                        "latency": latency,
                        "text_overlap": overlap,
                        "retrieved_count": len(retrieved),
                        "retrieved_ids": [r.chunk_id for r in retrieved],
                    })

                # Aggregate retrieval metrics
                if result.text_overlap_scores:
                    scores = result.text_overlap_scores
                    result.retrieval_metrics["mean_text_overlap"] = sum(scores) / len(scores)
                    result.retrieval_metrics["retrieval_p50_ms"] = result.retrieval_latency.percentile(50) * 1000
                    result.retrieval_metrics["retrieval_p95_ms"] = result.retrieval_latency.percentile(95) * 1000
                    result.retrieval_metrics["retrieval_p99_ms"] = result.retrieval_latency.percentile(99) * 1000

                store_inst.close()

        except Exception as e:
            logger.exception("Config %s failed", config.name)
            result.error = str(e)
        finally:
            _restore_originals(originals)

        return result
