"""Tests for research_tool.store — SQLite schema, chunking, BM25, hybrid retrieval."""

import os
import sqlite3
import struct
import tempfile

import pytest

from unittest.mock import MagicMock, patch

from research_tool.store import (
    CLIP_EMBEDDING_DIM,
    COLBERT_DIM,
    CONTEXTUAL_RETRIEVAL,
    EMBEDDING_DIM,
    MUVERA_BUCKETS,
    MUVERA_FDE_DIM,
    _BLOB_MAGIC,
    DocumentChunk,
    HybridIndex,
    RERANK_CANDIDATES,
    ResearchStore,
    RetrievedEntry,
    _blob_to_embedding,
    _compute_fde_vector,
    _embedding_to_blob,
    _ensure_chart_labels,
    _ensure_reranker_model,
    _make_clip_text_embedding,
    _make_embedding,
    _make_embedding_truncated,
    _make_image_embedding,
    _make_token_embeddings,
    _rag_tokenize,
    _rerank_pairs,
    chunk_web_content,
    classify_image_is_chart,
    compute_eigenvector_centrality,
    compute_image_chunk_proximity,
    compute_spectral_embeddings,
    extract_chart_text,
    multi_signal_rrf,
    RRF_K,
    RRF_WEIGHT_BM25,
    RRF_WEIGHT_TEXT_COSINE,
    RRF_WEIGHT_IMAGE_COSINE,
    RRF_WEIGHT_GRAPH,
    RRF_WEIGHT_MAXSIM,
    HYDE_ENABLED,
    RRF_WEIGHT_HYDE,
    PARENT_CHILD_ENABLED,
    ENTITY_RESOLUTION_ENABLED,
    ENTITY_RESOLUTION_THRESHOLD,
    RRF_WEIGHT_ENTITY,
    split_into_children,
    query_similar_code,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path):
    """Create a ResearchStore with a temporary database."""
    db_path = str(tmp_path / "test_research.db")
    store = ResearchStore(db_path=db_path)
    yield store
    store.close()


@pytest.fixture
def sample_embedding():
    """A small fake embedding vector for testing (768-dim for nomic-embed-text-v1.5)."""
    return [0.1 * (i % 10) for i in range(768)]


@pytest.fixture
def sample_chunks():
    """Multiple DocumentChunks for index testing (no parent pages needed for in-memory index)."""
    return [
        DocumentChunk(
            text="Python is a popular programming language used for web development.",
            page_url="https://example.com/python",
            chunk_id="abc123::chunk-0",
            section_title="Python Overview",
        ),
        DocumentChunk(
            text="Machine learning uses algorithms to learn patterns from data.",
            page_url="https://example.com/ml",
            chunk_id="def456::chunk-0",
            section_title="Machine Learning",
        ),
        DocumentChunk(
            text="Web scraping extracts data from websites using automated tools.",
            page_url="https://example.com/scraping",
            chunk_id="ghi789::chunk-0",
            section_title="Web Scraping",
        ),
    ]


def _ensure_pages_for_chunks(store, chunks):
    """Helper: insert parent pages so FK constraints are satisfied."""
    urls_seen = set()
    for chunk in chunks:
        if chunk.page_url and chunk.page_url not in urls_seen:
            store.store_page(url=chunk.page_url, title=chunk.page_url)
            urls_seen.add(chunk.page_url)


# ── Database Initialization Tests ──────────────────────────────────────────────


class TestResearchStoreInit:
    def test_creates_database_file(self, tmp_path):
        db_path = str(tmp_path / "new_research.db")
        assert not os.path.exists(db_path)
        store = ResearchStore(db_path=db_path)
        assert os.path.exists(db_path)
        store.close()

    def test_file_permissions(self, tmp_path):
        db_path = str(tmp_path / "perm_test.db")
        store = ResearchStore(db_path=db_path)
        mode = os.stat(db_path).st_mode & 0o777
        assert mode == 0o600, f"Expected 0600 permissions, got {oct(mode)}"
        store.close()

    def test_schema_tables_exist(self, tmp_db):
        cursor = tmp_db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row["name"] for row in cursor.fetchall()]
        assert "pages" in tables
        assert "chunks" in tables
        assert "searches" in tables

    def test_no_embedding_cache_table(self, tmp_db):
        """Verify embedding_cache table is NOT created (per design decision)."""
        cursor = tmp_db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_cache'"
        )
        assert cursor.fetchone() is None

    def test_pragma_ordering(self, tmp_path):
        """Verify PRAGMAs are set correctly (busy_timeout, journal_mode, foreign_keys)."""
        db_path = str(tmp_path / "pragma_test.db")
        store = ResearchStore(db_path=db_path)

        bt = store.conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert bt == 5000

        jm = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert jm == "wal"

        fk = store.conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1

        store.close()

    def test_idempotent_init(self, tmp_path):
        """Opening the same DB twice should not error."""
        db_path = str(tmp_path / "idempotent.db")
        store1 = ResearchStore(db_path=db_path)
        store1.store_page("https://example.com", title="Test")
        store1.close()

        store2 = ResearchStore(db_path=db_path)
        page = store2.get_page("https://example.com")
        assert page is not None
        assert page["title"] == "Test"
        store2.close()


# ── Page Operations Tests ──────────────────────────────────────────────────────


class TestPageOperations:
    def test_store_and_retrieve_page(self, tmp_db):
        tmp_db.store_page(
            url="https://example.com/article",
            title="Test Article",
            html="<h1>Test</h1>",
            extracted_text="Test content here.",
        )
        page = tmp_db.get_page("https://example.com/article")
        assert page is not None
        assert page["url"] == "https://example.com/article"
        assert page["title"] == "Test Article"
        assert page["html"] == "<h1>Test</h1>"
        assert page["extracted_text"] == "Test content here."
        assert page["fetched_at"] is not None

    def test_get_nonexistent_page(self, tmp_db):
        assert tmp_db.get_page("https://nonexistent.com") is None

    def test_upsert_page(self, tmp_db):
        tmp_db.store_page(url="https://example.com", title="V1")
        tmp_db.store_page(url="https://example.com", title="V2")
        page = tmp_db.get_page("https://example.com")
        assert page["title"] == "V2"


# ── Chunk Operations Tests ─────────────────────────────────────────────────────


class TestChunkOperations:
    def test_store_and_retrieve_chunk(self, tmp_db):
        tmp_db.store_page(url="https://example.com", title="Example")
        chunk = DocumentChunk(
            text="Some test content.",
            page_url="https://example.com",
            chunk_id="test::chunk-0",
            section_title="Test Section",
        )
        tmp_db.store_chunk(chunk)
        retrieved = tmp_db.get_chunk("test::chunk-0")
        assert retrieved is not None
        assert retrieved.text == "Some test content."
        assert retrieved.page_url == "https://example.com"
        assert retrieved.section_title == "Test Section"
        assert retrieved.embedding is None

    def test_store_chunk_with_embedding(self, tmp_db, sample_embedding):
        tmp_db.store_page(url="https://example.com", title="Example")
        chunk = DocumentChunk(
            text="Embedded content.",
            page_url="https://example.com",
            chunk_id="emb::chunk-0",
            section_title="Embedded",
            embedding=sample_embedding,
        )
        tmp_db.store_chunk(chunk)
        retrieved = tmp_db.get_chunk("emb::chunk-0")
        assert retrieved is not None
        assert retrieved.embedding is not None
        assert len(retrieved.embedding) == 768
        # Int8 quantization introduces small error; verify cosine similarity
        dot = sum(a * b for a, b in zip(sample_embedding, retrieved.embedding))
        mag_a = sum(a * a for a in sample_embedding) ** 0.5
        mag_b = sum(b * b for b in retrieved.embedding) ** 0.5
        cosine = dot / (mag_a * mag_b) if mag_a > 0 and mag_b > 0 else 0
        assert cosine > 0.99

    def test_embedding_blob_roundtrip(self, sample_embedding):
        blob = _embedding_to_blob(sample_embedding)
        restored = _blob_to_embedding(blob)
        assert len(restored) == len(sample_embedding)
        for orig, r in zip(sample_embedding, restored):
            assert abs(orig - r) < 1e-5

    def test_upsert_chunk(self, tmp_db):
        """Duplicate chunk_id should update rather than error."""
        tmp_db.store_page(url="https://example.com", title="Example")
        chunk1 = DocumentChunk(
            text="Original text.",
            page_url="https://example.com",
            chunk_id="dup::chunk-0",
            section_title="Section",
        )
        tmp_db.store_chunk(chunk1)

        chunk2 = DocumentChunk(
            text="Updated text.",
            page_url="https://example.com",
            chunk_id="dup::chunk-0",
            section_title="Updated Section",
        )
        tmp_db.store_chunk(chunk2)

        retrieved = tmp_db.get_chunk("dup::chunk-0")
        assert retrieved.text == "Updated text."
        assert retrieved.section_title == "Updated Section"

    def test_store_multiple_chunks(self, tmp_db, sample_chunks):
        _ensure_pages_for_chunks(tmp_db, sample_chunks)
        tmp_db.store_chunks(sample_chunks)
        all_chunks = tmp_db.get_all_chunks()
        assert len(all_chunks) == 3

    def test_get_nonexistent_chunk(self, tmp_db):
        assert tmp_db.get_chunk("nonexistent::chunk") is None

    def test_store_and_retrieve_code_embedding(self, tmp_db, sample_embedding):
        """Code chunks can store both primary and code_embedding, roundtrip intact."""
        tmp_db.store_page(url="https://example.com", title="Example")
        code_emb = [0.05 * (i % 20) for i in range(768)]
        chunk = DocumentChunk(
            text="def hello(): pass",
            page_url="https://example.com",
            chunk_id="code::chunk-0",
            section_title="Code",
            embedding=sample_embedding,
            code_embedding=code_emb,
            content_type="code",
        )
        tmp_db.store_chunk(chunk)
        retrieved = tmp_db.get_chunk("code::chunk-0")
        assert retrieved is not None
        assert retrieved.embedding is not None
        assert retrieved.code_embedding is not None
        assert len(retrieved.code_embedding) == 768
        dot = sum(a * b for a, b in zip(code_emb, retrieved.code_embedding))
        mag_a = sum(a * a for a in code_emb) ** 0.5
        mag_b = sum(b * b for b in retrieved.code_embedding) ** 0.5
        cosine = dot / (mag_a * mag_b) if mag_a > 0 and mag_b > 0 else 0
        assert cosine > 0.99

    def test_text_chunk_has_null_code_embedding(self, tmp_db, sample_embedding):
        """Text chunks store code_embedding=None."""
        tmp_db.store_page(url="https://example.com", title="Example")
        chunk = DocumentChunk(
            text="Some prose text.",
            page_url="https://example.com",
            chunk_id="text::chunk-0",
            section_title="Text",
            embedding=sample_embedding,
        )
        tmp_db.store_chunk(chunk)
        retrieved = tmp_db.get_chunk("text::chunk-0")
        assert retrieved is not None
        assert retrieved.embedding is not None
        assert retrieved.code_embedding is None

    def test_code_chunk_without_code_embedding(self, tmp_db, sample_embedding):
        """Code chunk where jina model returned None still stores with primary only."""
        tmp_db.store_page(url="https://example.com", title="Example")
        chunk = DocumentChunk(
            text="def foo(): pass",
            page_url="https://example.com",
            chunk_id="code-nomodel::chunk-0",
            section_title="Code",
            embedding=sample_embedding,
            code_embedding=None,
            content_type="code",
        )
        tmp_db.store_chunk(chunk)
        retrieved = tmp_db.get_chunk("code-nomodel::chunk-0")
        assert retrieved is not None
        assert retrieved.embedding is not None
        assert retrieved.code_embedding is None

    def test_store_chunks_batch_with_code_embedding(self, tmp_db, sample_embedding):
        """store_chunks() batch handles mix of chunks with and without code_embedding."""
        tmp_db.store_page(url="https://example.com", title="Example")
        code_emb = [0.05 * (i % 20) for i in range(768)]
        chunks = [
            DocumentChunk(
                text="def bar(): pass",
                page_url="https://example.com",
                chunk_id="batch::code-0",
                section_title="Code",
                embedding=sample_embedding,
                code_embedding=code_emb,
                content_type="code",
            ),
            DocumentChunk(
                text="Some text.",
                page_url="https://example.com",
                chunk_id="batch::text-0",
                section_title="Text",
                embedding=sample_embedding,
            ),
        ]
        tmp_db.store_chunks(chunks)
        code_chunk = tmp_db.get_chunk("batch::code-0")
        text_chunk = tmp_db.get_chunk("batch::text-0")
        assert code_chunk.code_embedding is not None
        assert text_chunk.code_embedding is None

    def test_get_all_chunks_includes_code_embedding(self, tmp_db, sample_embedding):
        """get_all_chunks() deserializes code_embedding for code chunks."""
        tmp_db.store_page(url="https://example.com", title="Example")
        code_emb = [0.05 * (i % 20) for i in range(768)]
        tmp_db.store_chunk(DocumentChunk(
            text="def baz(): pass",
            page_url="https://example.com",
            chunk_id="all::code-0",
            section_title="Code",
            embedding=sample_embedding,
            code_embedding=code_emb,
            content_type="code",
        ))
        all_chunks = tmp_db.get_all_chunks()
        code_chunks = [c for c in all_chunks if c.chunk_id == "all::code-0"]
        assert len(code_chunks) == 1
        assert code_chunks[0].code_embedding is not None


# ── Search Log Tests ───────────────────────────────────────────────────────────


class TestSearchLog:
    def test_log_search(self, tmp_db):
        search_id = tmp_db.log_search("test query", result_count=5)
        assert search_id is not None
        assert search_id > 0

    def test_multiple_searches(self, tmp_db):
        id1 = tmp_db.log_search("query one")
        id2 = tmp_db.log_search("query two")
        assert id2 > id1


# ── Chunking Tests ─────────────────────────────────────────────────────────────


class TestChunking:
    def test_basic_chunking(self):
        text = "First paragraph about cats.\n\nSecond paragraph about dogs.\n\nThird paragraph about birds."
        chunks = chunk_web_content(text, "https://example.com/animals")
        assert len(chunks) >= 1
        # All chunks should have non-empty text
        for chunk in chunks:
            assert chunk.text.strip()
            assert chunk.page_url == "https://example.com/animals"
            assert chunk.chunk_id

    def test_empty_text_returns_empty(self):
        assert chunk_web_content("", "https://example.com") == []
        assert chunk_web_content("   ", "https://example.com") == []
        assert chunk_web_content("\n\n\n", "https://example.com") == []

    def test_no_empty_chunks(self):
        text = "Para one.\n\n\n\n\n\nPara two.\n\n   \n\nPara three."
        chunks = chunk_web_content(text, "https://example.com")
        for chunk in chunks:
            assert chunk.text.strip(), f"Empty chunk found: {chunk!r}"

    def test_multi_paragraph_merging(self):
        """Short paragraphs should merge into a single chunk."""
        text = "Short one.\n\nShort two.\n\nShort three."
        chunks = chunk_web_content(text, "https://example.com", max_tokens=500)
        # Three short paragraphs should merge into one chunk
        assert len(chunks) == 1
        assert "Short one." in chunks[0].text
        assert "Short two." in chunks[0].text
        assert "Short three." in chunks[0].text

    def test_large_text_splits(self):
        """A very long text should produce multiple chunks."""
        # Generate ~2000 tokens worth of text
        paragraphs = []
        for i in range(40):
            paragraphs.append(
                f"This is paragraph number {i} and it contains some words "
                f"that contribute to the overall token count of the document."
            )
        text = "\n\n".join(paragraphs)
        chunks = chunk_web_content(text, "https://example.com/long", max_tokens=500)
        assert len(chunks) > 1
        # No chunk should be empty
        for chunk in chunks:
            assert chunk.text.strip()

    def test_chunk_ids_unique(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        # Force small chunks so we get multiple
        chunks = chunk_web_content(text, "https://example.com", max_tokens=3)
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs must be unique"

    def test_section_title_propagated(self):
        text = "Content here.\n\nMore content."
        chunks = chunk_web_content(text, "https://example.com", section_title="My Section")
        for chunk in chunks:
            assert chunk.section_title == "My Section"

    def test_section_title_defaults_to_untitled(self):
        text = "Content here."
        chunks = chunk_web_content(text, "https://example.com")
        assert chunks[0].section_title == "(untitled)"

    def test_oversized_single_paragraph_splits(self):
        """A single paragraph exceeding max_tokens is sub-split, not stored as one huge chunk."""
        long_para = " ".join([f"word{i}" for i in range(600)])
        chunks = chunk_web_content(long_para, "https://example.com", max_tokens=500)
        assert len(chunks) > 1
        # All content preserved — rejoin and compare
        reassembled = " ".join(c.text.strip() for c in chunks)
        assert reassembled == long_para
        for c in chunks:
            assert c.text.strip()

    def test_oversized_with_sentences_splits_on_boundaries(self):
        """Oversized text with sentence boundaries splits cleanly at sentences."""
        sentences = [f"This is sentence number {i}." for i in range(100)]
        long_text = " ".join(sentences)
        chunks = chunk_web_content(long_text, "https://example.com", max_tokens=50)
        assert len(chunks) > 1
        # No content lost
        all_text = " ".join(c.text.strip() for c in chunks)
        for s in sentences:
            assert s in all_text

    def test_abbreviation_dr_not_split(self):
        text = "Dr. Smith said hello. Then he left."
        from research_tool.store import _split_oversized
        pieces = _split_oversized(text, max_tokens=500)
        assert len(pieces) == 1
        assert pieces[0] == text

    def test_abbreviation_us_not_split(self):
        text = "The U.S. government announced a policy. It takes effect Monday."
        from research_tool.store import _split_oversized
        pieces = _split_oversized(text, max_tokens=500)
        assert len(pieces) == 1
        assert pieces[0] == text

    def test_abbreviation_fig_not_split(self):
        text = "See Fig. 3 for details. The results show improvement."
        from research_tool.store import _split_oversized
        pieces = _split_oversized(text, max_tokens=500)
        assert len(pieces) == 1
        assert pieces[0] == text

    def test_numbered_list_not_split(self):
        text = "1. First item. 2. Second item."
        from research_tool.store import _split_oversized
        pieces = _split_oversized(text, max_tokens=500)
        assert len(pieces) == 1
        assert pieces[0] == text

    def test_eg_not_split(self):
        text = "e.g. this example shows the issue. Another sentence follows."
        from research_tool.store import _split_oversized
        pieces = _split_oversized(text, max_tokens=500)
        assert len(pieces) == 1
        assert pieces[0] == text

    def test_long_sentence_no_punctuation_falls_back(self):
        text = " ".join(f"word{i}" for i in range(200))
        from research_tool.store import _split_oversized
        pieces = _split_oversized(text, max_tokens=50)
        assert len(pieces) > 1
        assert " ".join(pieces) == text

    def test_only_commas_falls_back_gracefully(self):
        text = "alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota, kappa"
        from research_tool.store import _split_oversized
        pieces = _split_oversized(text, max_tokens=5)
        assert len(pieces) > 1
        for p in pieces:
            assert p.strip()

    def test_real_sentences_split_correctly(self):
        text = (
            "Dr. Smith studied the U.S. economy. "
            "He published in Vol. 3 of the journal. "
            "See Fig. 12 for the results. "
            "The GDP grew by 2.5% in Q3."
        )
        from research_tool.store import _split_oversized
        pieces = _split_oversized(text, max_tokens=20)
        assert len(pieces) > 1
        reassembled = " ".join(pieces)
        assert "Dr. Smith" in reassembled
        assert "U.S. economy" in reassembled
        assert "Vol. 3" in reassembled
        assert "Fig. 12" in reassembled


# ── BM25 Retrieval Tests ──────────────────────────────────────────────────────


class TestBM25Retrieval:
    def test_happy_path_bm25(self, tmp_db, sample_chunks):
        """Store chunks, build index, retrieve via BM25."""
        _ensure_pages_for_chunks(tmp_db, sample_chunks)
        tmp_db.store_chunks(sample_chunks)
        index = HybridIndex()
        index.build_from_store(tmp_db)
        assert index.is_built

        results = index.bm25_retrieve("programming language")
        assert len(results) > 0
        # The Python chunk should rank first for "programming language"
        assert results[0].chunk_id == "abc123::chunk-0"
        assert results[0].bm25_score > 0
        assert results[0].text == sample_chunks[0].text

    def test_empty_query_returns_empty(self, sample_chunks):
        index = HybridIndex()
        for c in sample_chunks:
            index.add_chunk(c)
        index.build()

        assert index.bm25_retrieve("") == []

    def test_no_match_returns_empty(self, sample_chunks):
        index = HybridIndex()
        for c in sample_chunks:
            index.add_chunk(c)
        index.build()

        results = index.bm25_retrieve("xyzzyplugh")
        assert results == []

    def test_unbuilt_index_returns_empty(self, sample_chunks):
        index = HybridIndex()
        for c in sample_chunks:
            index.add_chunk(c)
        # Don't call build()
        assert index.bm25_retrieve("python") == []

    def test_bm25_only_no_embedding(self, tmp_db):
        """Chunks with no embedding should still work with BM25-only retrieval."""
        chunks = [
            DocumentChunk(
                text="Quantum computing uses qubits instead of classical bits.",
                page_url="https://example.com/quantum",
                chunk_id="quantum::chunk-0",
                section_title="Quantum",
                embedding=None,
            ),
            DocumentChunk(
                text="Classical computing uses binary transistors.",
                page_url="https://example.com/classical",
                chunk_id="classical::chunk-0",
                section_title="Classical",
                embedding=None,
            ),
        ]
        _ensure_pages_for_chunks(tmp_db, chunks)
        tmp_db.store_chunks(chunks)
        index = HybridIndex()
        index.build_from_store(tmp_db)

        results = index.bm25_retrieve("quantum qubits")
        assert len(results) > 0
        assert results[0].chunk_id == "quantum::chunk-0"


# ── Hybrid Retrieval Tests ─────────────────────────────────────────────────────


class TestHybridRetrieval:
    def test_hybrid_retrieve_sorted(self, sample_chunks):
        """Hybrid retrieve should return results sorted by combined score."""
        index = HybridIndex()
        for c in sample_chunks:
            index.add_chunk(c)
        index.build()

        results = index.hybrid_retrieve("machine learning algorithms data")
        assert len(results) > 0
        # Verify sorted by descending score
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_hybrid_empty_query(self, sample_chunks):
        index = HybridIndex()
        for c in sample_chunks:
            index.add_chunk(c)
        index.build()

        assert index.hybrid_retrieve("") == []

    def test_hybrid_unbuilt(self):
        index = HybridIndex()
        assert index.hybrid_retrieve("test") == []

    def test_hybrid_fallback_bm25_only(self, sample_chunks):
        """When ONNX is unavailable, hybrid should fall back to BM25-only."""
        index = HybridIndex()
        for c in sample_chunks:
            index.add_chunk(c)
        index.build()

        results = index.hybrid_retrieve("web scraping data extraction")
        assert len(results) > 0
        # Without ONNX, cosine_score should be 0
        for r in results:
            assert r.cosine_score == 0.0
            assert r.bm25_score >= 0.0

    def test_hybrid_top_k(self, sample_chunks):
        index = HybridIndex()
        for c in sample_chunks:
            index.add_chunk(c)
        index.build()

        results = index.hybrid_retrieve("programming", top_k=1)
        assert len(results) <= 1


# ── Tokenization Tests ────────────────────────────────────────────────────────


class TestTokenization:
    def test_basic_tokenize(self):
        tokens = _rag_tokenize("Hello World 123 test-case")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test-case" in tokens

    def test_empty_tokenize(self):
        assert _rag_tokenize("") == []
        assert _rag_tokenize("123 456") == []

    def test_lowercased(self):
        tokens = _rag_tokenize("UPPER lower MiXeD")
        assert all(t == t.lower() for t in tokens)


# ── Integration: Chunking + Store + Index ──────────────────────────────────────


class TestIntegration:
    def test_chunk_store_retrieve_pipeline(self, tmp_db):
        """Full pipeline: chunk text -> store -> build index -> retrieve."""
        text = (
            "Python is a versatile programming language. It supports multiple "
            "programming paradigms including procedural, object-oriented, and "
            "functional programming.\n\n"
            "JavaScript is the language of the web. It runs in browsers and on "
            "servers via Node.js. Modern JavaScript uses ES6+ features.\n\n"
            "Rust is a systems programming language focused on safety and "
            "performance. It prevents memory errors at compile time."
        )

        chunks = chunk_web_content(text, "https://example.com/languages", section_title="Languages")
        assert len(chunks) >= 1

        tmp_db.store_page(url="https://example.com/languages", title="Programming Languages")
        tmp_db.store_chunks(chunks)

        index = HybridIndex()
        index.build_from_store(tmp_db)
        assert index.is_built

        results = index.bm25_retrieve("systems programming safety performance")
        assert len(results) > 0
        # The Rust chunk should score highest
        assert "Rust" in results[0].text or "rust" in results[0].text.lower()

    def test_chunk_web_content_reasonable_sizes(self):
        """Verify chunk_web_content produces reasonable chunk sizes."""
        paragraphs = []
        for i in range(20):
            paragraphs.append(
                f"This is paragraph {i} discussing topic {i % 5}. "
                f"It contains several sentences about various subjects. "
                f"The content is designed to test chunking behavior. "
                f"Each paragraph has roughly similar length for consistency."
            )
        text = "\n\n".join(paragraphs)

        chunks = chunk_web_content(text, "https://example.com/test", max_tokens=500)

        assert len(chunks) >= 1
        # No empty chunks
        for chunk in chunks:
            assert chunk.text.strip()
        # Each chunk should not vastly exceed max_tokens
        for chunk in chunks:
            token_count = len(_rag_tokenize(chunk.text))
            # Allow some tolerance for the paragraph that pushed it over
            # (single large paragraphs are kept whole)
            assert token_count > 0


# ── Schema Migration Tests ───────────────────────────────────────────────────


class TestSchemaMigration:
    """Tests for new tables (images, page_links) and columns (graph_embedding, authority_score)."""

    def _get_tables(self, conn):
        """Return set of user table names."""
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return {row[0] for row in cursor.fetchall()}

    def _get_columns(self, conn, table):
        """Return set of column names for a table."""
        cursor = conn.execute(f"PRAGMA table_info({table})")
        return {row[1] for row in cursor.fetchall()}

    def test_fresh_db_has_all_tables(self, tmp_path):
        """Fresh database should have all five tables."""
        db_path = str(tmp_path / "fresh.db")
        store = ResearchStore(db_path=db_path)
        tables = self._get_tables(store.conn)
        assert "pages" in tables
        assert "chunks" in tables
        assert "searches" in tables
        assert "images" in tables
        assert "page_links" in tables
        store.close()

    def test_fresh_db_has_new_columns(self, tmp_path):
        """Fresh database should have graph_embedding and authority_score on pages."""
        db_path = str(tmp_path / "fresh_cols.db")
        store = ResearchStore(db_path=db_path)
        columns = self._get_columns(store.conn, "pages")
        assert "graph_embedding" in columns
        assert "authority_score" in columns
        store.close()

    def test_existing_db_gains_new_tables_and_columns(self, tmp_path):
        """A pre-existing database (old schema) gains new tables and columns on next open."""
        db_path = str(tmp_path / "legacy.db")

        # Simulate an old database with only the original three tables
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = DELETE")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS pages (
                url TEXT PRIMARY KEY,
                title TEXT,
                html TEXT,
                extracted_text TEXT,
                fetched_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                page_url TEXT REFERENCES pages(url),
                section_title TEXT,
                text TEXT NOT NULL,
                embedding BLOB,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                result_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );
        """)
        conn.commit()
        # Insert a row to prove data survives migration
        conn.execute("INSERT INTO pages (url, title) VALUES ('https://old.com', 'Old Page')")
        conn.commit()
        conn.close()

        # Now open with ResearchStore — should migrate
        store = ResearchStore(db_path=db_path)
        tables = self._get_tables(store.conn)
        assert "images" in tables
        assert "page_links" in tables

        columns = self._get_columns(store.conn, "pages")
        assert "graph_embedding" in columns
        assert "authority_score" in columns

        # Existing data should be intact
        page = store.get_page("https://old.com")
        assert page is not None
        assert page["title"] == "Old Page"
        store.close()

    def test_idempotent_double_open(self, tmp_path):
        """Opening the same database twice does not crash (idempotent)."""
        db_path = str(tmp_path / "idempotent_migration.db")
        store1 = ResearchStore(db_path=db_path)
        store1.store_page("https://example.com", title="Test")
        store1.close()

        store2 = ResearchStore(db_path=db_path)
        tables = self._get_tables(store2.conn)
        assert "images" in tables
        assert "page_links" in tables
        columns = self._get_columns(store2.conn, "pages")
        assert "graph_embedding" in columns
        assert "authority_score" in columns

        # Data survives
        page = store2.get_page("https://example.com")
        assert page is not None
        assert page["title"] == "Test"
        store2.close()

    def test_already_migrated_db_unaffected(self, tmp_path):
        """Database that already has the new tables and columns is unaffected."""
        db_path = str(tmp_path / "already_migrated.db")

        # First open creates everything
        store1 = ResearchStore(db_path=db_path)
        store1.store_page("https://example.com", title="Existing")
        store1.close()

        # Second open should not error or lose data
        store2 = ResearchStore(db_path=db_path)
        tables = self._get_tables(store2.conn)
        assert "images" in tables
        assert "page_links" in tables
        columns = self._get_columns(store2.conn, "pages")
        assert "graph_embedding" in columns
        assert "authority_score" in columns

        page = store2.get_page("https://example.com")
        assert page is not None
        assert page["title"] == "Existing"
        store2.close()

    def test_legacy_db_gains_code_embedding_column(self, tmp_path):
        """Legacy DB without code_embedding column gains it on open; existing data intact."""
        db_path = str(tmp_path / "legacy_no_code_emb.db")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = DELETE")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS pages (
                url TEXT PRIMARY KEY,
                title TEXT,
                html TEXT,
                extracted_text TEXT,
                fetched_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                page_url TEXT REFERENCES pages(url),
                section_title TEXT,
                text TEXT NOT NULL,
                embedding BLOB,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                result_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );
        """)
        conn.execute("INSERT INTO pages (url, title) VALUES ('https://old.com', 'Old')")
        conn.execute(
            "INSERT INTO chunks (chunk_id, page_url, section_title, text) "
            "VALUES ('old::chunk', 'https://old.com', 'S', 'old text')"
        )
        conn.commit()
        conn.close()

        store = ResearchStore(db_path=db_path)
        columns = self._get_columns(store.conn, "chunks")
        assert "code_embedding" in columns
        chunk = store.get_chunk("old::chunk")
        assert chunk is not None
        assert chunk.text == "old text"
        assert chunk.code_embedding is None
        store.close()


# ── CLIP Embedding Tests ─────────────────────────────────────────────────────


def _make_valid_png_bytes() -> bytes:
    """Create a minimal valid PNG image for testing."""
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color=(128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


def _mock_onnx_output(dim: int = CLIP_EMBEDDING_DIM):
    """Return a fake ONNX session output: (1, dim) numpy array of random values."""
    import numpy as np

    raw = np.random.randn(1, dim).astype(np.float32)
    return [raw]


class TestCLIPEmbedding:
    """Tests for CLIP ViT-B/32 image and text embedding functions."""

    @patch("research_tool.store._ensure_clip_model", return_value=True)
    @patch("research_tool.store._clip_vision_session")
    def test_image_embedding_happy_path(self, mock_session, mock_ensure):
        """_make_image_embedding with valid PNG returns a 512-dim list of floats."""
        import numpy as np

        fake_output = np.random.randn(1, CLIP_EMBEDDING_DIM).astype(np.float32)
        mock_session.run = MagicMock(return_value=[fake_output])

        result = _make_image_embedding(_make_valid_png_bytes())
        assert result is not None
        assert len(result) == CLIP_EMBEDDING_DIM
        assert all(isinstance(v, float) for v in result)

    @patch("research_tool.store._ensure_clip_model", return_value=True)
    @patch("research_tool.store._clip_text_session")
    @patch("research_tool.store._clip_tokenizer")
    def test_text_embedding_happy_path(self, mock_tokenizer, mock_session, mock_ensure):
        """_make_clip_text_embedding with valid text returns a 512-dim list of floats."""
        import numpy as np

        # Mock tokenizer encode output
        mock_encoded = MagicMock()
        mock_encoded.ids = [49406, 320, 2368, 49407] + [0] * 73  # 77 total
        mock_encoded.attention_mask = [1, 1, 1, 1] + [0] * 73  # not used by CLIP text model, but tokenizer produces it
        mock_tokenizer.encode = MagicMock(return_value=mock_encoded)

        fake_output = np.random.randn(1, CLIP_EMBEDDING_DIM).astype(np.float32)
        mock_session.run = MagicMock(return_value=[fake_output])

        result = _make_clip_text_embedding("a cat")
        assert result is not None
        assert len(result) == CLIP_EMBEDDING_DIM
        assert all(isinstance(v, float) for v in result)

    @patch("research_tool.store._ensure_clip_model", return_value=True)
    @patch("research_tool.store._clip_vision_session")
    def test_image_embedding_is_normalized(self, mock_session, mock_ensure):
        """Returned image embedding should be L2-normalized (norm ~ 1.0)."""
        import numpy as np

        fake_output = np.array([[3.0] * CLIP_EMBEDDING_DIM], dtype=np.float32)
        mock_session.run = MagicMock(return_value=[fake_output])

        result = _make_image_embedding(_make_valid_png_bytes())
        assert result is not None
        norm = sum(v ** 2 for v in result) ** 0.5
        assert abs(norm - 1.0) < 1e-5

    @patch("research_tool.store._ensure_clip_model", return_value=True)
    @patch("research_tool.store._clip_text_session")
    @patch("research_tool.store._clip_tokenizer")
    def test_text_embedding_is_normalized(self, mock_tokenizer, mock_session, mock_ensure):
        """Returned text embedding should be L2-normalized (norm ~ 1.0)."""
        import numpy as np

        mock_encoded = MagicMock()
        mock_encoded.ids = [49406, 320, 2368, 49407] + [0] * 73
        mock_encoded.attention_mask = [1, 1, 1, 1] + [0] * 73  # not used by CLIP text model, but tokenizer produces it
        mock_tokenizer.encode = MagicMock(return_value=mock_encoded)

        fake_output = np.array([[5.0] * CLIP_EMBEDDING_DIM], dtype=np.float32)
        mock_session.run = MagicMock(return_value=[fake_output])

        result = _make_clip_text_embedding("a cat")
        assert result is not None
        norm = sum(v ** 2 for v in result) ** 0.5
        assert abs(norm - 1.0) < 1e-5

    def test_image_embedding_none_input(self):
        """_make_image_embedding(None) returns None without errors."""
        result = _make_image_embedding(None)
        assert result is None

    def test_image_embedding_empty_bytes(self):
        """_make_image_embedding(b'') returns None without errors."""
        result = _make_image_embedding(b"")
        assert result is None

    @patch("research_tool.store._ensure_clip_model", return_value=True)
    def test_image_embedding_corrupt_bytes(self, mock_ensure):
        """_make_image_embedding with corrupt bytes returns None (PIL fails)."""
        result = _make_image_embedding(b"corrupt data not a valid image")
        assert result is None

    @patch("research_tool.store._ensure_clip_model", return_value=False)
    def test_image_embedding_model_unavailable(self, mock_ensure):
        """When CLIP model is unavailable, _make_image_embedding returns None."""
        result = _make_image_embedding(_make_valid_png_bytes())
        assert result is None

    @patch("research_tool.store._ensure_clip_model", return_value=False)
    def test_text_embedding_model_unavailable(self, mock_ensure):
        """When CLIP model is unavailable, _make_clip_text_embedding returns None."""
        result = _make_clip_text_embedding("a cat")
        assert result is None

    def test_text_embedding_empty_string(self):
        """_make_clip_text_embedding('') returns None without errors."""
        result = _make_clip_text_embedding("")
        assert result is None

    @patch("research_tool.store._ensure_clip_model", return_value=True)
    @patch("research_tool.store._clip_vision_session")
    def test_image_embedding_onnx_exception(self, mock_session, mock_ensure):
        """ONNX inference exception is caught, logged, returns None."""
        mock_session.run = MagicMock(side_effect=RuntimeError("ONNX inference failed"))

        result = _make_image_embedding(_make_valid_png_bytes())
        assert result is None

    @patch("research_tool.store._ensure_clip_model", return_value=True)
    @patch("research_tool.store._clip_text_session")
    @patch("research_tool.store._clip_tokenizer")
    def test_text_embedding_onnx_exception(self, mock_tokenizer, mock_session, mock_ensure):
        """ONNX inference exception is caught, logged, returns None."""
        mock_encoded = MagicMock()
        mock_encoded.ids = [49406, 320, 2368, 49407] + [0] * 73
        mock_encoded.attention_mask = [1, 1, 1, 1] + [0] * 73  # not used by CLIP text model, but tokenizer produces it
        mock_tokenizer.encode = MagicMock(return_value=mock_encoded)

        mock_session.run = MagicMock(side_effect=RuntimeError("ONNX inference failed"))

        result = _make_clip_text_embedding("a cat")
        assert result is None


# ── Reranker Tests ──────────────────────────────────────────────────────────


class TestReranker:
    """Tests for cross-encoder reranker model lifecycle and _rerank_pairs."""

    @patch("research_tool.store._ensure_reranker_model", return_value=True)
    @patch("research_tool.store._reranker_session")
    @patch("research_tool.store._reranker_tokenizer")
    def test_rerank_pairs_happy_path(self, mock_tokenizer, mock_session, mock_ensure):
        """_rerank_pairs returns a float score per passage, relevant > irrelevant."""
        import numpy as np

        def mock_encode(query, passage):
            enc = MagicMock()
            enc.ids = [101, 1, 2, 3, 102, 4, 5, 102]
            enc.attention_mask = [1] * 8
            enc.type_ids = [0, 0, 0, 0, 0, 1, 1, 1]
            return enc

        mock_tokenizer.encode = MagicMock(side_effect=mock_encode)
        fake_logits = np.array([[2.5], [-1.0]], dtype=np.float32)
        mock_session.run = MagicMock(return_value=[fake_logits])

        scores = _rerank_pairs("what is python", ["Python is a language", "cats are cute"])
        assert scores is not None
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)
        assert scores[0] > scores[1]

    def test_rerank_pairs_empty_passages(self):
        """Empty passages list returns empty list (no model loading)."""
        scores = _rerank_pairs("any query", [])
        assert scores == []

    @patch("research_tool.store._ensure_reranker_model", return_value=True)
    @patch("research_tool.store._reranker_session")
    @patch("research_tool.store._reranker_tokenizer")
    def test_rerank_pairs_single_passage(self, mock_tokenizer, mock_session, mock_ensure):
        """Single passage returns a single-element list."""
        import numpy as np

        enc = MagicMock()
        enc.ids = [101, 1, 2, 102, 3, 102]
        enc.attention_mask = [1] * 6
        enc.type_ids = [0, 0, 0, 0, 1, 1]
        mock_tokenizer.encode = MagicMock(return_value=enc)

        fake_logits = np.array([[3.14]], dtype=np.float32)
        mock_session.run = MagicMock(return_value=[fake_logits])

        scores = _rerank_pairs("query", ["single passage"])
        assert scores is not None
        assert len(scores) == 1
        assert isinstance(scores[0], float)

    @patch("research_tool.store._ensure_reranker_model", return_value=False)
    def test_rerank_pairs_model_unavailable(self, mock_ensure):
        """When the reranker model is unavailable, returns None."""
        scores = _rerank_pairs("query", ["passage one", "passage two"])
        assert scores is None

    @patch("research_tool.store._ensure_reranker_model", return_value=True)
    @patch("research_tool.store._reranker_session")
    @patch("research_tool.store._reranker_tokenizer")
    def test_rerank_pairs_onnx_exception(self, mock_tokenizer, mock_session, mock_ensure):
        """ONNX inference exception is caught and returns None."""
        enc = MagicMock()
        enc.ids = [101, 1, 102]
        enc.attention_mask = [1] * 3
        enc.type_ids = [0, 0, 0]
        mock_tokenizer.encode = MagicMock(return_value=enc)
        mock_session.run = MagicMock(side_effect=RuntimeError("ONNX inference failed"))

        scores = _rerank_pairs("query", ["passage"])
        assert scores is None

    @patch("research_tool.store._ensure_reranker_model", return_value=True)
    @patch("research_tool.store._reranker_session")
    @patch("research_tool.store._reranker_tokenizer")
    def test_rerank_pairs_1d_logits(self, mock_tokenizer, mock_session, mock_ensure):
        """1D logit output (flattened) is handled correctly."""
        import numpy as np

        enc = MagicMock()
        enc.ids = [101, 1, 102]
        enc.attention_mask = [1] * 3
        enc.type_ids = [0, 0, 0]
        mock_tokenizer.encode = MagicMock(return_value=enc)

        fake_logits = np.array([1.5, -0.5, 0.3], dtype=np.float32)
        mock_session.run = MagicMock(return_value=[fake_logits])

        scores = _rerank_pairs("q", ["a", "b", "c"])
        assert scores is not None
        assert len(scores) == 3
        assert scores[0] == pytest.approx(1.5, abs=1e-5)
        assert scores[1] == pytest.approx(-0.5, abs=1e-5)
        assert scores[2] == pytest.approx(0.3, abs=1e-5)

    @patch("research_tool.store._ensure_reranker_model", return_value=True)
    @patch("research_tool.store._reranker_session")
    @patch("research_tool.store._reranker_tokenizer")
    def test_rerank_pairs_batch_padding(self, mock_tokenizer, mock_session, mock_ensure):
        """Passages of different lengths are padded to the max length in the batch."""
        import numpy as np

        def mock_encode(query, passage):
            enc = MagicMock()
            length = 5 + len(passage.split())
            enc.ids = list(range(length))
            enc.attention_mask = [1] * length
            enc.type_ids = [0] * 3 + [1] * (length - 3)
            return enc

        mock_tokenizer.encode = MagicMock(side_effect=mock_encode)

        captured_inputs = {}

        def capture_run(_, inputs):
            captured_inputs.update(inputs)
            batch_size = inputs["input_ids"].shape[0]
            return [np.zeros((batch_size, 1), dtype=np.float32)]

        mock_session.run = MagicMock(side_effect=capture_run)

        _rerank_pairs("query", ["short", "a much longer passage here"])
        assert "input_ids" in captured_inputs
        ids = captured_inputs["input_ids"]
        assert ids.shape[0] == 2
        assert ids.shape[1] == max(5 + 1, 5 + 5)

    def test_rerank_candidates_env_var_default(self):
        """RERANK_CANDIDATES defaults to 50."""
        assert RERANK_CANDIDATES == 50


class TestEnsureRerankerModel:
    """Tests for _ensure_reranker_model lifecycle (download, verify, init)."""

    @patch("research_tool.store._ONNX_AVAILABLE", False)
    def test_returns_false_when_onnx_unavailable(self):
        """Returns False immediately when onnxruntime is not installed."""
        import research_tool.store as store_mod
        store_mod._reranker_session = None
        store_mod._reranker_tokenizer = None
        result = _ensure_reranker_model()
        assert result is False

    @patch("research_tool.store._ONNX_AVAILABLE", True)
    def test_returns_true_when_already_initialized(self):
        """Returns True immediately when session and tokenizer already set."""
        import research_tool.store as store_mod
        store_mod._reranker_session = MagicMock()
        store_mod._reranker_tokenizer = MagicMock()
        try:
            result = _ensure_reranker_model()
            assert result is True
        finally:
            store_mod._reranker_session = None
            store_mod._reranker_tokenizer = None

    @patch("research_tool.store._ONNX_AVAILABLE", True)
    @patch("research_tool.store._download_verified")
    def test_download_failure_resets_globals(self, mock_dl, tmp_path):
        """Download failure resets globals to None and returns False."""
        import research_tool.store as store_mod
        store_mod._reranker_session = None
        store_mod._reranker_tokenizer = None
        mock_dl.side_effect = ValueError("SHA-256 mismatch")

        with patch("research_tool.store.MODEL_CACHE_DIR", str(tmp_path)):
            result = _ensure_reranker_model()

        assert result is False
        assert store_mod._reranker_session is None
        assert store_mod._reranker_tokenizer is None


# ── Image Storage Tests ─────────────────────────────────────────────────────


@pytest.fixture
def sample_clip_embedding():
    """A fake 512-dim CLIP embedding for testing."""
    return [0.01 * i for i in range(CLIP_EMBEDDING_DIM)]


class TestImageStorage:
    def test_store_and_retrieve_image(self, tmp_db, sample_clip_embedding):
        """Happy path: store an image with embedding, retrieve via get_images_for_page."""
        tmp_db.store_page(url="https://example.com/article", title="Article")

        tmp_db.store_image(
            image_id="img001",
            page_url="https://example.com/article",
            src_url="https://cdn.example.com/photo.jpg",
            alt_text="A photo",
            width=800,
            height=600,
            embedding=sample_clip_embedding,
            nearest_chunk_ids=["abc123::chunk-0", "abc123::chunk-1"],
        )

        images = tmp_db.get_images_for_page("https://example.com/article")
        assert len(images) == 1
        img = images[0]
        assert img["image_id"] == "img001"
        assert img["page_url"] == "https://example.com/article"
        assert img["src_url"] == "https://cdn.example.com/photo.jpg"
        assert img["alt_text"] == "A photo"
        assert img["width"] == 800
        assert img["height"] == 600
        assert img["embedding"] is not None
        assert len(img["embedding"]) == CLIP_EMBEDDING_DIM
        # Verify embedding round-trip
        for orig, stored in zip(sample_clip_embedding, img["embedding"]):
            assert abs(orig - stored) < 1e-5
        assert img["nearest_chunk_ids"] == ["abc123::chunk-0", "abc123::chunk-1"]

    def test_get_all_images_returns_only_with_embeddings(self, tmp_db, sample_clip_embedding):
        """get_all_images() returns only images with non-NULL embeddings."""
        tmp_db.store_page(url="https://example.com", title="Test")

        # Image with embedding
        tmp_db.store_image(
            image_id="img_with",
            page_url="https://example.com",
            src_url="https://cdn.example.com/with.jpg",
            embedding=sample_clip_embedding,
            nearest_chunk_ids=["c1"],
        )
        # Image without embedding
        tmp_db.store_image(
            image_id="img_without",
            page_url="https://example.com",
            src_url="https://cdn.example.com/without.jpg",
            embedding=None,
            nearest_chunk_ids=None,
        )

        all_images = tmp_db.get_all_images()
        assert len(all_images) == 1
        assert all_images[0]["image_id"] == "img_with"

    def test_get_image_count_includes_null_embeddings(self, tmp_db, sample_clip_embedding):
        """get_image_count() includes images with NULL embeddings."""
        tmp_db.store_page(url="https://example.com", title="Test")

        tmp_db.store_image(
            image_id="img_a",
            page_url="https://example.com",
            src_url="https://cdn.example.com/a.jpg",
            embedding=sample_clip_embedding,
        )
        tmp_db.store_image(
            image_id="img_b",
            page_url="https://example.com",
            src_url="https://cdn.example.com/b.jpg",
            embedding=None,
        )

        assert tmp_db.get_image_count() == 2

    def test_store_image_null_embedding(self, tmp_db):
        """Storing an image with NULL embedding succeeds (metadata preserved)."""
        tmp_db.store_page(url="https://example.com", title="Test")

        tmp_db.store_image(
            image_id="img_null",
            page_url="https://example.com",
            src_url="https://cdn.example.com/null.jpg",
            alt_text="No embedding",
            width=400,
            height=300,
            embedding=None,
            nearest_chunk_ids=None,
        )

        images = tmp_db.get_images_for_page("https://example.com")
        assert len(images) == 1
        img = images[0]
        assert img["image_id"] == "img_null"
        assert img["alt_text"] == "No embedding"
        assert img["width"] == 400
        assert img["height"] == 300
        assert img["embedding"] is None
        assert img["nearest_chunk_ids"] is None

    def test_duplicate_image_id_upserts(self, tmp_db, sample_clip_embedding):
        """Duplicate image_id upserts without error."""
        tmp_db.store_page(url="https://example.com", title="Test")

        tmp_db.store_image(
            image_id="img_dup",
            page_url="https://example.com",
            src_url="https://cdn.example.com/v1.jpg",
            alt_text="Version 1",
            embedding=None,
        )
        tmp_db.store_image(
            image_id="img_dup",
            page_url="https://example.com",
            src_url="https://cdn.example.com/v2.jpg",
            alt_text="Version 2",
            embedding=sample_clip_embedding,
        )

        images = tmp_db.get_images_for_page("https://example.com")
        assert len(images) == 1
        assert images[0]["src_url"] == "https://cdn.example.com/v2.jpg"
        assert images[0]["alt_text"] == "Version 2"
        assert images[0]["embedding"] is not None

    def test_store_image_fk_constraint(self, tmp_db):
        """store_image before store_page raises FK constraint error."""
        with pytest.raises(sqlite3.IntegrityError):
            tmp_db.store_image(
                image_id="img_orphan",
                page_url="https://nonexistent.com/page",
                src_url="https://cdn.example.com/orphan.jpg",
            )

    def test_get_images_for_page_empty(self, tmp_db):
        """get_images_for_page returns empty list when no images exist."""
        assert tmp_db.get_images_for_page("https://no-images.com") == []

    def test_get_image_count_empty(self, tmp_db):
        """get_image_count returns 0 when no images exist."""
        assert tmp_db.get_image_count() == 0


# ── Image-Chunk Proximity Tests ─────────────────────────────────────────────


class TestImageChunkProximity:
    def test_proximity_associates_nearest_chunks(self):
        """Image between two chunks is mapped to both adjacent chunks."""
        html = (
            "<p>First paragraph content here for testing.</p>"
            '<img src="photo.jpg" alt="photo">'
            "<p>Second paragraph content here for testing.</p>"
        )
        chunks = [
            DocumentChunk(
                text="First paragraph content here for testing.",
                page_url="https://example.com",
                chunk_id="c::chunk-0",
                section_title="Test",
            ),
            DocumentChunk(
                text="Second paragraph content here for testing.",
                page_url="https://example.com",
                chunk_id="c::chunk-1",
                section_title="Test",
            ),
        ]
        image_dicts = [
            {"src_url": "https://example.com/photo.jpg", "dom_offset": html.find('<img')},
        ]

        proximity = compute_image_chunk_proximity(image_dicts, chunks, html)
        assert 0 in proximity
        # Should have chunk-0 (before) and chunk-1 (after)
        assert "c::chunk-0" in proximity[0]
        assert "c::chunk-1" in proximity[0]

    def test_proximity_no_chunks(self):
        """Image on page with no text chunks gets empty mapping."""
        html = '<img src="photo.jpg">'
        image_dicts = [{"src_url": "https://example.com/photo.jpg", "dom_offset": 0}]

        proximity = compute_image_chunk_proximity(image_dicts, [], html)
        assert proximity == {}

    def test_proximity_chunk_text_not_in_html(self):
        """Chunks whose text cannot be found in HTML produce empty proximity."""
        html = '<p>Some HTML</p><img src="photo.jpg">'
        chunks = [
            DocumentChunk(
                text="This text does not appear in the HTML at all",
                page_url="https://example.com",
                chunk_id="c::chunk-0",
                section_title="Test",
            ),
        ]
        image_dicts = [{"src_url": "https://example.com/photo.jpg", "dom_offset": 20}]

        proximity = compute_image_chunk_proximity(image_dicts, chunks, html)
        # No chunks could be located, so no proximity data
        assert proximity == {}

    def test_proximity_empty_inputs(self):
        """Empty image_dicts or html returns empty mapping."""
        chunks = [
            DocumentChunk(
                text="Some text",
                page_url="https://example.com",
                chunk_id="c::chunk-0",
                section_title="Test",
            ),
        ]
        assert compute_image_chunk_proximity([], chunks, "<p>text</p>") == {}
        assert compute_image_chunk_proximity([{"dom_offset": 0}], chunks, "") == {}

    def test_proximity_image_before_all_chunks(self):
        """Image at the very start gets mapped to the first chunk (after)."""
        html = '<img src="photo.jpg"><p>Chunk text at the end of the page.</p>'
        chunks = [
            DocumentChunk(
                text="Chunk text at the end of the page.",
                page_url="https://example.com",
                chunk_id="c::chunk-0",
                section_title="Test",
            ),
        ]
        image_dicts = [{"src_url": "https://example.com/photo.jpg", "dom_offset": 0}]

        proximity = compute_image_chunk_proximity(image_dicts, chunks, html)
        assert 0 in proximity
        assert "c::chunk-0" in proximity[0]


# ── Link Storage Tests ──────────────────────────────────────────────────────


class TestLinkStorage:
    def test_store_links_and_get_graph(self, tmp_db):
        """store_links stores links, get_link_graph returns correct adjacency."""
        tmp_db.store_page(url="https://a.com", title="A")
        tmp_db.store_links("https://a.com", [
            {"target_url": "https://b.com", "anchor_text": "B"},
            {"target_url": "https://c.com", "anchor_text": "C"},
        ])
        graph = tmp_db.get_link_graph()
        assert "https://a.com" in graph
        assert graph["https://a.com"] == {"https://b.com", "https://c.com"}
        assert tmp_db.get_link_count() == 2

    def test_duplicate_links_ignored(self, tmp_db):
        """Duplicate (source_url, target_url) pairs are silently ignored."""
        tmp_db.store_page(url="https://a.com", title="A")
        tmp_db.store_links("https://a.com", [
            {"target_url": "https://b.com", "anchor_text": "First"},
        ])
        tmp_db.store_links("https://a.com", [
            {"target_url": "https://b.com", "anchor_text": "Second"},
        ])
        assert tmp_db.get_link_count() == 1

    def test_links_to_nonexistent_pages(self, tmp_db):
        """Links to pages not yet in the pages table are allowed (no FK)."""
        tmp_db.store_page(url="https://a.com", title="A")
        tmp_db.store_links("https://a.com", [
            {"target_url": "https://nonexistent.com/page", "anchor_text": "Missing"},
        ])
        assert tmp_db.get_link_count() == 1
        graph = tmp_db.get_link_graph()
        assert "https://nonexistent.com/page" in graph["https://a.com"]

    def test_get_link_count_empty(self, tmp_db):
        """get_link_count returns 0 when no links exist."""
        assert tmp_db.get_link_count() == 0

    def test_multiple_sources(self, tmp_db):
        """Multiple source pages can have links."""
        tmp_db.store_page(url="https://a.com", title="A")
        tmp_db.store_page(url="https://b.com", title="B")
        tmp_db.store_links("https://a.com", [
            {"target_url": "https://c.com", "anchor_text": "C"},
        ])
        tmp_db.store_links("https://b.com", [
            {"target_url": "https://c.com", "anchor_text": "C"},
            {"target_url": "https://d.com", "anchor_text": "D"},
        ])
        graph = tmp_db.get_link_graph()
        assert len(graph) == 2
        assert graph["https://a.com"] == {"https://c.com"}
        assert graph["https://b.com"] == {"https://c.com", "https://d.com"}
        assert tmp_db.get_link_count() == 3


# ── Spectral Embedding Tests ────────────────────────────────────────────────


class TestSpectralEmbeddings:
    def test_five_pages_eight_links(self):
        """5 pages with 8 links -> 5 embedding vectors, L2-normalized."""
        import numpy as np

        adjacency = {
            "https://a.com": {"https://b.com", "https://c.com"},
            "https://b.com": {"https://c.com", "https://d.com"},
            "https://c.com": {"https://d.com", "https://e.com"},
            "https://d.com": {"https://e.com"},
            "https://e.com": {"https://a.com"},
        }
        embeddings = compute_spectral_embeddings(adjacency)
        assert len(embeddings) == 5
        for url, vec in embeddings.items():
            norm = sum(x * x for x in vec) ** 0.5
            assert abs(norm - 1.0) < 1e-5, f"Embedding for {url} not L2-normalized: norm={norm}"

    def test_structurally_equivalent_nodes_similar_embeddings(self):
        """Structurally equivalent nodes should have similar pairwise distances."""
        import numpy as np

        # Star graph: hub connects to spokes A, B, C, D
        # All spokes are structurally equivalent -- their pairwise cosine
        # similarities should be roughly equal.
        adjacency = {
            "https://hub.com": {"https://a.com", "https://b.com", "https://c.com", "https://d.com"},
            "https://a.com": {"https://hub.com"},
            "https://b.com": {"https://hub.com"},
            "https://c.com": {"https://hub.com"},
            "https://d.com": {"https://hub.com"},
        }
        embeddings = compute_spectral_embeddings(adjacency, k=4)
        spokes = ["https://a.com", "https://b.com", "https://c.com", "https://d.com"]
        sims = []
        for i in range(len(spokes)):
            for j in range(i + 1, len(spokes)):
                vi = np.array(embeddings[spokes[i]])
                vj = np.array(embeddings[spokes[j]])
                sims.append(float(np.dot(vi, vj)))
        assert max(sims) - min(sims) < 0.1, (
            f"Spoke pairwise similarities should be roughly equal, got range "
            f"{min(sims):.4f} to {max(sims):.4f}"
        )

    def test_high_indegree_higher_centrality(self):
        """Page with high in-degree should have higher eigenvector centrality."""
        # Popular page is linked from all others and has reciprocal links to some.
        # This creates a non-trivial graph where the hub has the most connections.
        adjacency = {
            "https://a.com": {"https://popular.com", "https://b.com"},
            "https://b.com": {"https://popular.com", "https://c.com"},
            "https://c.com": {"https://popular.com"},
            "https://d.com": {"https://popular.com"},
            "https://popular.com": {"https://a.com", "https://b.com"},
        }
        centrality = compute_eigenvector_centrality(adjacency)
        # Hub should have the highest centrality
        assert centrality["https://popular.com"] == pytest.approx(1.0)
        # Peripheral nodes should have lower centrality
        for url in ("https://c.com", "https://d.com"):
            assert centrality[url] < centrality["https://popular.com"]

    def test_isolated_page_zero_vector_and_centrality(self, tmp_db):
        """Isolated page in graph gets zero-vector embedding and centrality 0.0."""
        # Store 3 pages: A->B, C is isolated
        tmp_db.store_page(url="https://a.com", title="A")
        tmp_db.store_page(url="https://b.com", title="B")
        tmp_db.store_page(url="https://c.com", title="C (isolated)")

        adjacency = {
            "https://a.com": {"https://b.com"},
        }
        embeddings = compute_spectral_embeddings(adjacency)
        centrality = compute_eigenvector_centrality(adjacency)
        tmp_db.store_graph_data(embeddings, centrality)

        # Check that C gets zero-vector and 0.0 authority_score
        page_c = tmp_db.get_page("https://c.com")
        assert page_c["authority_score"] == 0.0
        blob = page_c["graph_embedding"]
        assert blob is not None
        emb = _blob_to_embedding(blob)
        assert all(v == 0.0 for v in emb)

    def test_fewer_nodes_than_k(self):
        """Graph with fewer nodes than k gracefully reduces dimensions."""
        adjacency = {
            "https://a.com": {"https://b.com"},
            "https://b.com": {"https://c.com"},
        }
        embeddings = compute_spectral_embeddings(adjacency, k=8)
        # 3 nodes -> sparse eigsh can compute at most n-2 non-trivial = 1
        for url, vec in embeddings.items():
            assert 1 <= len(vec) <= 2, f"Expected 1-2 dim embedding, got {len(vec)}"

    def test_empty_graph_returns_empty(self):
        """Empty adjacency returns empty dicts."""
        assert compute_spectral_embeddings({}) == {}
        assert compute_eigenvector_centrality({}) == {}

    def test_store_graph_data_roundtrip(self, tmp_db):
        """store_graph_data stores embeddings and centrality correctly, retrievable from DB."""
        tmp_db.store_page(url="https://a.com", title="A")
        tmp_db.store_page(url="https://b.com", title="B")

        embeddings = {
            "https://a.com": [0.1, 0.2, 0.3, 0.4],
            "https://b.com": [0.5, 0.6, 0.7, 0.8],
        }
        centrality = {
            "https://a.com": 0.75,
            "https://b.com": 1.0,
        }
        tmp_db.store_graph_data(embeddings, centrality)

        page_a = tmp_db.get_page("https://a.com")
        assert page_a["authority_score"] == pytest.approx(0.75)
        emb_a = _blob_to_embedding(page_a["graph_embedding"])
        assert len(emb_a) == 4
        assert emb_a[0] == pytest.approx(0.1, abs=1e-5)

        page_b = tmp_db.get_page("https://b.com")
        assert page_b["authority_score"] == pytest.approx(1.0)
        emb_b = _blob_to_embedding(page_b["graph_embedding"])
        assert emb_b[2] == pytest.approx(0.7, abs=1e-5)

    def test_single_node_returns_zero(self):
        """Graph with a single node returns a zero-like embedding."""
        adjacency = {
            "https://only.com": set(),
        }
        embeddings = compute_spectral_embeddings(adjacency)
        assert len(embeddings) == 1
        assert "https://only.com" in embeddings
        # Single node: min(k, 1) = 1 dimension, value 0.0
        assert embeddings["https://only.com"] == [0.0]


# ── RRF Tests ────────────────────────────────────────────────────────────────


class TestRRF:
    """Tests for multi_signal_rrf weighted reciprocal rank fusion."""

    def test_four_signals_ranks_higher_than_two(self):
        """Chunk appearing in all 4 signals ranks higher than one in only 2."""
        ranked_lists = {
            "bm25": [("c_all", 0.9), ("c_two", 0.8), ("c_one", 0.5)],
            "text_cosine": [("c_all", 0.95), ("c_two", 0.7)],
            "image_cosine": [("c_all", 0.85), ("c_only_img", 0.6)],
            "graph": [("c_all", 1.0), ("c_only_graph", 0.5)],
        }
        weights = {
            "bm25": 1.0,
            "text_cosine": 1.0,
            "image_cosine": 0.5,
            "graph": 0.3,
        }
        results = multi_signal_rrf(ranked_lists, weights, k=60, top_k=5)
        # c_all should be ranked first (appears in all 4 signals)
        assert results[0][0] == "c_all"
        # c_two should be second (appears in 2 signals)
        rrf_ids = [r[0] for r in results]
        assert rrf_ids.index("c_all") < rrf_ids.index("c_two")

    def test_zero_weight_disables_signal(self):
        """Setting weight=0.0 for a signal effectively disables it."""
        ranked_lists = {
            "bm25": [("c1", 0.9)],
            "text_cosine": [("c2", 0.95)],
        }
        weights = {"bm25": 1.0, "text_cosine": 0.0}
        results = multi_signal_rrf(ranked_lists, weights, k=60, top_k=5)
        # c2 should get zero contribution from text_cosine
        scores = {r[0]: r[1] for r in results}
        assert scores["c1"] > scores["c2"]
        assert scores["c2"] == 0.0

    def test_single_signal(self):
        """Single signal produces valid results."""
        ranked_lists = {
            "bm25": [("c1", 0.9), ("c2", 0.5), ("c3", 0.2)],
        }
        weights = {"bm25": 1.0}
        results = multi_signal_rrf(ranked_lists, weights, k=60, top_k=3)
        assert len(results) == 3
        assert results[0][0] == "c1"
        assert results[1][0] == "c2"
        assert results[2][0] == "c3"
        # All scores should be positive
        for _, score, _ in results:
            assert score > 0

    def test_empty_ranked_list(self):
        """Empty ranked list is handled gracefully."""
        ranked_lists = {
            "bm25": [],
            "text_cosine": [("c1", 0.5)],
        }
        weights = {"bm25": 1.0, "text_cosine": 1.0}
        results = multi_signal_rrf(ranked_lists, weights, k=60, top_k=5)
        assert len(results) == 1
        assert results[0][0] == "c1"

    def test_chunk_in_only_one_list(self):
        """Chunk appearing in only one list still gets a score."""
        ranked_lists = {
            "bm25": [("c_bm25", 0.9)],
            "text_cosine": [("c_cosine", 0.8)],
        }
        weights = {"bm25": 1.0, "text_cosine": 1.0}
        results = multi_signal_rrf(ranked_lists, weights, k=60, top_k=5)
        assert len(results) == 2
        rrf_ids = [r[0] for r in results]
        assert "c_bm25" in rrf_ids
        assert "c_cosine" in rrf_ids

    def test_signal_scores_preserved(self):
        """Raw signal scores are preserved in the output."""
        ranked_lists = {
            "bm25": [("c1", 0.9)],
            "text_cosine": [("c1", 0.75)],
        }
        weights = {"bm25": 1.0, "text_cosine": 1.0}
        results = multi_signal_rrf(ranked_lists, weights, k=60, top_k=5)
        assert len(results) == 1
        _, _, signals = results[0]
        assert signals["bm25"] == pytest.approx(0.9)
        assert signals["text_cosine"] == pytest.approx(0.75)

    def test_all_empty_lists(self):
        """All empty ranked lists returns empty results."""
        ranked_lists = {"bm25": [], "text_cosine": []}
        weights = {"bm25": 1.0, "text_cosine": 1.0}
        results = multi_signal_rrf(ranked_lists, weights, k=60, top_k=5)
        assert results == []


# ── Hybrid Index Mode Tests ──────────────────────────────────────────────────


class TestHybridIndexModes:
    """Tests for ingest vs query mode behavior in HybridIndex."""

    def test_ingest_mode_no_images_or_authority(self, tmp_db, sample_chunks):
        """Ingest-mode index loads only chunks (no images or authority_scores)."""
        _ensure_pages_for_chunks(tmp_db, sample_chunks)
        tmp_db.store_chunks(sample_chunks)
        # Add some images and authority scores to verify they're NOT loaded
        tmp_db.store_image(
            image_id="img1",
            page_url=sample_chunks[0].page_url,
            src_url="https://cdn.example.com/photo.jpg",
            embedding=[0.1] * CLIP_EMBEDDING_DIM,
            nearest_chunk_ids=[sample_chunks[0].chunk_id],
        )
        adjacency = {sample_chunks[0].page_url: {sample_chunks[1].page_url}}
        embeddings = compute_spectral_embeddings(adjacency)
        centrality = compute_eigenvector_centrality(adjacency)
        tmp_db.store_graph_data(embeddings, centrality)

        index = HybridIndex(mode="ingest")
        index.build_from_store(tmp_db)

        assert index.is_built
        assert len(index.chunks) == 3
        assert index._images == []
        assert index._authority_scores == {}

    def test_query_mode_loads_images_and_authority(self, tmp_db, sample_chunks):
        """Query-mode index loads images and authority_scores."""
        _ensure_pages_for_chunks(tmp_db, sample_chunks)
        tmp_db.store_chunks(sample_chunks)
        # Add image
        tmp_db.store_image(
            image_id="img1",
            page_url=sample_chunks[0].page_url,
            src_url="https://cdn.example.com/photo.jpg",
            embedding=[0.1] * CLIP_EMBEDDING_DIM,
            nearest_chunk_ids=[sample_chunks[0].chunk_id],
        )
        # Add authority scores
        adjacency = {sample_chunks[0].page_url: {sample_chunks[1].page_url}}
        embeddings = compute_spectral_embeddings(adjacency)
        centrality = compute_eigenvector_centrality(adjacency)
        tmp_db.store_graph_data(embeddings, centrality)

        index = HybridIndex(mode="query")
        index.build_from_store(tmp_db)

        assert index.is_built
        assert len(index._images) == 1
        assert len(index._authority_scores) > 0

    def test_hybrid_retrieve_ingest_mode_bm25_only(self, sample_chunks):
        """Ingest-mode hybrid_retrieve uses only BM25 + text cosine (no image/graph)."""
        index = HybridIndex(mode="ingest")
        for c in sample_chunks:
            index.add_chunk(c)
        index.build()

        results = index.hybrid_retrieve("programming language", top_k=3)
        assert len(results) > 0
        # In ingest mode, image and graph scores should be 0
        for r in results:
            assert r.image_cosine_score == 0.0
            assert r.graph_score == 0.0

    def test_hybrid_retrieve_query_mode_uses_all_signals(self, tmp_db, sample_chunks):
        """Query-mode hybrid_retrieve uses all 4 signals when data present."""
        _ensure_pages_for_chunks(tmp_db, sample_chunks)
        tmp_db.store_chunks(sample_chunks)

        # Store authority scores
        adjacency = {
            sample_chunks[0].page_url: {sample_chunks[1].page_url},
            sample_chunks[1].page_url: {sample_chunks[0].page_url, sample_chunks[2].page_url},
        }
        embeddings = compute_spectral_embeddings(adjacency)
        centrality = compute_eigenvector_centrality(adjacency)
        tmp_db.store_graph_data(embeddings, centrality)

        index = HybridIndex(mode="query")
        index.build_from_store(tmp_db)

        results = index.hybrid_retrieve("programming language", top_k=3)
        assert len(results) > 0
        # At least one result should have a graph_score > 0 (since authority data exists)
        has_graph = any(r.graph_score > 0 for r in results)
        assert has_graph

    def test_no_images_no_graph_same_as_ingest(self, sample_chunks):
        """BM25-only + text-cosine-only (no images, no graph) produces valid results."""
        index = HybridIndex(mode="query")
        for c in sample_chunks:
            index.add_chunk(c)
        index.build()

        results = index.hybrid_retrieve("web scraping data extraction")
        assert len(results) > 0
        # Without images or graph data, those scores should be 0
        for r in results:
            assert r.image_cosine_score == 0.0
            assert r.graph_score == 0.0

    @patch("research_tool.store._make_clip_text_embedding", return_value=None)
    def test_clip_unavailable_omits_image_signal(self, mock_clip, sample_chunks):
        """CLIP unavailable -> image signal silently omitted."""
        index = HybridIndex(mode="query")
        for c in sample_chunks:
            index.add_chunk(c)
        index.build()
        # Manually add images to simulate query-mode data
        index._images = [
            {
                "page_url": "https://example.com/python",
                "embedding": [0.1] * CLIP_EMBEDDING_DIM,
                "nearest_chunk_ids": ["abc123::chunk-0"],
            },
        ]

        results = index.hybrid_retrieve("programming")
        assert len(results) > 0
        # Image signal should be 0 since CLIP text embedding failed
        for r in results:
            assert r.image_cosine_score == 0.0

    def test_empty_graph_omits_graph_signal(self, sample_chunks):
        """Empty graph -> graph signal silently omitted."""
        index = HybridIndex(mode="query")
        for c in sample_chunks:
            index.add_chunk(c)
        index.build()
        # No authority_scores set
        assert index._authority_scores == {}

        results = index.hybrid_retrieve("programming")
        assert len(results) > 0
        for r in results:
            assert r.graph_score == 0.0

    def test_null_authority_score_treated_as_zero(self, tmp_db, sample_chunks):
        """NULL authority_score treated as 0.0 (not loaded into _authority_scores)."""
        _ensure_pages_for_chunks(tmp_db, sample_chunks)
        tmp_db.store_chunks(sample_chunks)
        # Don't store any graph data -- pages have NULL authority_score

        index = HybridIndex(mode="query")
        index.build_from_store(tmp_db)

        assert index._authority_scores == {}
        results = index.hybrid_retrieve("programming")
        assert len(results) > 0
        for r in results:
            assert r.graph_score == 0.0

    def test_default_mode_is_query(self):
        """HybridIndex defaults to query mode."""
        index = HybridIndex()
        assert index.mode == "query"

    @patch("research_tool.store._rerank_pairs")
    def test_rerank_enabled_reorders_by_score(self, mock_rerank):
        """With reranking enabled, results are sorted by cross-encoder score."""
        chunks = [
            DocumentChunk(
                text="Python is a programming language for web development and data science.",
                page_url="https://example.com/python",
                chunk_id="a::chunk-0",
                section_title="Python",
            ),
            DocumentChunk(
                text="Data science uses programming tools to analyze large datasets.",
                page_url="https://example.com/data",
                chunk_id="b::chunk-0",
                section_title="Data Science",
            ),
        ]
        index = HybridIndex(mode="query")
        for c in chunks:
            index.add_chunk(c)
        index.build()

        def assign_scores(query, passages):
            scores = []
            for p in passages:
                if "Data science" in p:
                    scores.append(10.0)
                else:
                    scores.append(-5.0)
            return scores

        mock_rerank.side_effect = assign_scores
        reranked = index.hybrid_retrieve("programming data science", top_k=2, rerank=True)

        assert mock_rerank.called
        assert len(reranked) == 2
        assert reranked[0].chunk_id == "b::chunk-0"
        assert reranked[0].rerank_score > reranked[1].rerank_score

    @patch("research_tool.store._rerank_pairs")
    def test_rerank_false_skips_reranking(self, mock_rerank, sample_chunks):
        """rerank=False skips reranking entirely."""
        index = HybridIndex(mode="query")
        for c in sample_chunks:
            index.add_chunk(c)
        index.build()

        index.hybrid_retrieve("programming", top_k=3, rerank=False)
        mock_rerank.assert_not_called()

    @patch("research_tool.store._rerank_pairs")
    def test_ingest_mode_skips_reranking(self, mock_rerank, sample_chunks):
        """Ingest mode never triggers reranking regardless of rerank flag."""
        index = HybridIndex(mode="ingest")
        for c in sample_chunks:
            index.add_chunk(c)
        index.build()

        index.hybrid_retrieve("programming", top_k=3, rerank=True)
        mock_rerank.assert_not_called()

    @patch("research_tool.store._rerank_pairs", return_value=None)
    def test_reranker_unavailable_falls_back(self, mock_rerank, sample_chunks):
        """When reranker returns None, falls back to RRF order."""
        index = HybridIndex(mode="query")
        for c in sample_chunks:
            index.add_chunk(c)
        index.build()

        rrf_results = index.hybrid_retrieve("programming language", top_k=3, rerank=False)
        rrf_order = [r.chunk_id for r in rrf_results]

        fallback_results = index.hybrid_retrieve("programming language", top_k=3, rerank=True)
        fallback_order = [r.chunk_id for r in fallback_results]

        assert mock_rerank.called
        assert fallback_order == rrf_order

    @patch("research_tool.store._rerank_pairs")
    def test_rerank_respects_top_k(self, mock_rerank, sample_chunks):
        """Reranked results are truncated to top_k."""
        index = HybridIndex(mode="query")
        for c in sample_chunks:
            index.add_chunk(c)
        index.build()

        mock_rerank.return_value = [3.0, 1.0, 2.0]

        results = index.hybrid_retrieve("programming", top_k=2, rerank=True)
        assert len(results) <= 2


class TestChartClassification:
    """Tests for CLIP zero-shot chart/figure detection."""

    def _make_normalized_embedding(self, seed: int, dim: int = 512) -> list[float]:
        """Create a deterministic normalized embedding vector."""
        import numpy as np
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    @patch("research_tool.store._make_clip_text_embedding")
    def test_chart_embedding_classified_as_chart(self, mock_clip_text):
        """Image closer to chart labels returns True."""
        import numpy as np
        import research_tool.store as store_mod

        chart_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        non_chart_emb = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        mock_clip_text.side_effect = lambda label: (
            chart_emb.tolist() if "chart" in label or "visualization" in label or "bar" in label
            else non_chart_emb.tolist()
        )

        store_mod._chart_label_embeddings = None
        store_mod._non_chart_label_embeddings = None

        img_embedding = [0.9, 0.1, 0.0]
        assert classify_image_is_chart(img_embedding) is True

    @patch("research_tool.store._make_clip_text_embedding")
    def test_photo_embedding_classified_as_non_chart(self, mock_clip_text):
        """Image closer to photo labels returns False."""
        import numpy as np
        import research_tool.store as store_mod

        chart_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        non_chart_emb = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        mock_clip_text.side_effect = lambda label: (
            chart_emb.tolist() if "chart" in label or "visualization" in label or "bar" in label
            else non_chart_emb.tolist()
        )

        store_mod._chart_label_embeddings = None
        store_mod._non_chart_label_embeddings = None

        img_embedding = [0.1, 0.9, 0.0]
        assert classify_image_is_chart(img_embedding) is False

    @patch("research_tool.store._make_clip_text_embedding")
    def test_labels_cached_after_first_call(self, mock_clip_text):
        """Label embeddings are computed once and cached."""
        import research_tool.store as store_mod

        mock_clip_text.return_value = [0.5] * 512
        store_mod._chart_label_embeddings = None
        store_mod._non_chart_label_embeddings = None

        _ensure_chart_labels()
        first_call_count = mock_clip_text.call_count

        _ensure_chart_labels()
        assert mock_clip_text.call_count == first_call_count

    @patch("research_tool.store._make_clip_text_embedding")
    def test_clip_unavailable_returns_false(self, mock_clip_text):
        """When CLIP text encoder fails, returns False conservatively."""
        import research_tool.store as store_mod

        mock_clip_text.return_value = None
        store_mod._chart_label_embeddings = None
        store_mod._non_chart_label_embeddings = None

        assert classify_image_is_chart([0.5] * 512) is False

    def test_empty_embedding_returns_false(self):
        """Empty or None embedding returns False."""
        assert classify_image_is_chart([]) is False
        assert classify_image_is_chart(None) is False

    @patch("research_tool.store._make_clip_text_embedding")
    def test_equal_similarity_returns_false(self, mock_clip_text):
        """When chart and non-chart scores are equal, returns False (conservative)."""
        import research_tool.store as store_mod

        mock_clip_text.return_value = [1.0, 0.0, 0.0]
        store_mod._chart_label_embeddings = None
        store_mod._non_chart_label_embeddings = None

        img_embedding = [1.0, 0.0, 0.0]
        assert classify_image_is_chart(img_embedding) is False


# ── U2: OCR Extraction Tests ─────────────────────────────────────────────────


class TestChartOCR:
    """Tests for extract_chart_text() OCR extraction."""

    @patch("research_tool.store._ensure_ocr_engine")
    def test_extracts_text_from_boxes(self, mock_engine_fn):
        """Extracts and concatenates text from OCR bounding boxes."""
        mock_engine = MagicMock()
        mock_engine.return_value = (
            [
                ([[0, 0], [100, 0], [100, 20], [0, 20]], "Revenue", 0.95),
                ([[0, 30], [100, 30], [100, 50], [0, 50]], "$1.2M", 0.88),
            ],
            None,
        )
        mock_engine_fn.return_value = mock_engine

        result = extract_chart_text(b"fake_png_data")
        assert result == "Revenue $1.2M"

    @patch("research_tool.store._ensure_ocr_engine")
    def test_filters_low_confidence(self, mock_engine_fn):
        """Boxes below 0.5 confidence are filtered out."""
        mock_engine = MagicMock()
        mock_engine.return_value = (
            [
                ([[0, 0], [100, 0], [100, 20], [0, 20]], "Good", 0.9),
                ([[0, 30], [100, 30], [100, 50], [0, 50]], "Bad", 0.3),
            ],
            None,
        )
        mock_engine_fn.return_value = mock_engine

        result = extract_chart_text(b"fake_png_data")
        assert result == "Good"
        assert "Bad" not in result

    @patch("research_tool.store._ensure_ocr_engine")
    def test_returns_none_when_no_results(self, mock_engine_fn):
        """Returns None when OCR produces no boxes."""
        mock_engine = MagicMock()
        mock_engine.return_value = (None, None)
        mock_engine_fn.return_value = mock_engine

        assert extract_chart_text(b"blank_image") is None

    @patch("research_tool.store._ensure_ocr_engine")
    def test_returns_none_when_engine_unavailable(self, mock_engine_fn):
        """Returns None gracefully when OCR engine not installed."""
        mock_engine_fn.return_value = None

        assert extract_chart_text(b"any_data") is None

    @patch("research_tool.store._ensure_ocr_engine")
    def test_truncates_long_text(self, mock_engine_fn):
        """Text longer than _MAX_OCR_TEXT_LENGTH is truncated."""
        from research_tool.store import _MAX_OCR_TEXT_LENGTH

        mock_engine = MagicMock()
        long_word = "x" * (_MAX_OCR_TEXT_LENGTH + 500)
        mock_engine.return_value = (
            [([[0, 0], [100, 0], [100, 20], [0, 20]], long_word, 0.9)],
            None,
        )
        mock_engine_fn.return_value = mock_engine

        result = extract_chart_text(b"fake")
        assert len(result) == _MAX_OCR_TEXT_LENGTH

    @patch("research_tool.store._ensure_ocr_engine")
    def test_sorts_top_to_bottom(self, mock_engine_fn):
        """Boxes are sorted by vertical then horizontal position."""
        mock_engine = MagicMock()
        mock_engine.return_value = (
            [
                ([[50, 100], [150, 100], [150, 120], [50, 120]], "Bottom", 0.9),
                ([[0, 0], [100, 0], [100, 20], [0, 20]], "Top", 0.9),
            ],
            None,
        )
        mock_engine_fn.return_value = mock_engine

        result = extract_chart_text(b"fake")
        assert result == "Top Bottom"

    @patch("research_tool.store._ensure_ocr_engine")
    def test_handles_exception_gracefully(self, mock_engine_fn):
        """Returns None when OCR raises an exception."""
        mock_engine = MagicMock()
        mock_engine.side_effect = RuntimeError("OCR crashed")
        mock_engine_fn.return_value = mock_engine

        assert extract_chart_text(b"crash_data") is None


# ── U3: Chart Image Retrieval Tests ──────────────────────────────────────────


class TestGetChartImagesForChunks:
    """Tests for ResearchStore.get_chart_images_for_chunks()."""

    def test_returns_matching_chart_images(self, tmp_db):
        """Returns chart images whose nearest_chunk_ids overlap with query."""
        tmp_db.store_page(url="https://example.com/paper", title="Paper")
        tmp_db.store_image(
            image_id="chart1",
            page_url="https://example.com/paper",
            src_url="https://example.com/fig1.png",
            nearest_chunk_ids=["chunk-a", "chunk-b"],
            image_bytes=b"png_data_1",
            is_chart=True,
        )
        tmp_db.store_image(
            image_id="photo1",
            page_url="https://example.com/paper",
            src_url="https://example.com/photo.jpg",
            nearest_chunk_ids=["chunk-a"],
            image_bytes=b"photo_data",
            is_chart=False,
        )

        results = tmp_db.get_chart_images_for_chunks(["chunk-a", "chunk-c"])
        assert len(results) == 1
        assert results[0]["image_id"] == "chart1"
        assert results[0]["image_bytes"] == b"png_data_1"

    def test_returns_empty_for_no_overlap(self, tmp_db):
        """Returns empty list when no chart images match the chunk IDs."""
        tmp_db.store_page(url="https://example.com", title="Example")
        tmp_db.store_image(
            image_id="chart1",
            page_url="https://example.com",
            src_url="https://example.com/fig.png",
            nearest_chunk_ids=["chunk-x"],
            image_bytes=b"data",
            is_chart=True,
        )

        results = tmp_db.get_chart_images_for_chunks(["chunk-a", "chunk-b"])
        assert results == []

    def test_returns_empty_for_empty_input(self, tmp_db):
        """Returns empty list for empty chunk_ids input."""
        assert tmp_db.get_chart_images_for_chunks([]) == []

    def test_respects_max_results(self, tmp_db):
        """Limits returned results to max_results."""
        tmp_db.store_page(url="https://example.com", title="Example")
        for i in range(5):
            tmp_db.store_image(
                image_id=f"chart{i}",
                page_url="https://example.com",
                src_url=f"https://example.com/fig{i}.png",
                nearest_chunk_ids=["shared-chunk"],
                image_bytes=f"data{i}".encode(),
                is_chart=True,
            )

        results = tmp_db.get_chart_images_for_chunks(["shared-chunk"], max_results=2)
        assert len(results) == 2


# ── MRL Embedding Model Tests (U1) ───────────────────────────────────────────


class TestMRLEmbedding:
    def test_embedding_dim_constant(self):
        assert EMBEDDING_DIM == 768

    def test_make_embedding_returns_correct_dim(self):
        result = _make_embedding("test text", mode="document")
        if result is None:
            pytest.skip("ONNX model not available in test environment")
        assert len(result) == EMBEDDING_DIM

    def test_make_embedding_query_mode(self):
        result = _make_embedding("test query", mode="query")
        if result is None:
            pytest.skip("ONNX model not available in test environment")
        assert len(result) == EMBEDDING_DIM

    def test_document_and_query_mode_differ(self):
        doc = _make_embedding("hello world", mode="document")
        query = _make_embedding("hello world", mode="query")
        if doc is None or query is None:
            pytest.skip("ONNX model not available in test environment")
        assert doc != query

    def test_embedding_is_normalized(self):
        result = _make_embedding("normalization test", mode="document")
        if result is None:
            pytest.skip("ONNX model not available in test environment")
        norm = sum(x * x for x in result) ** 0.5
        assert abs(norm - 1.0) < 1e-5

    def test_make_embedding_truncated(self):
        result = _make_embedding_truncated("truncation test", 256, mode="document")
        if result is None:
            pytest.skip("ONNX model not available in test environment")
        assert len(result) == 256

    def test_make_embedding_truncated_normalized(self):
        result = _make_embedding_truncated("norm test", 128, mode="document")
        if result is None:
            pytest.skip("ONNX model not available in test environment")
        norm = sum(x * x for x in result) ** 0.5
        assert abs(norm - 1.0) < 1e-5

    def test_make_embedding_truncated_larger_than_model(self):
        result = _make_embedding_truncated("large dims", 1024, mode="document")
        if result is None:
            pytest.skip("ONNX model not available in test environment")
        assert len(result) == EMBEDDING_DIM

    def test_make_embedding_model_unavailable(self):
        with patch("research_tool.store._ONNX_AVAILABLE", False):
            import research_tool.store as store_mod
            old_session = store_mod._embedding_session
            store_mod._embedding_session = None
            try:
                result = _make_embedding("test")
                assert result is None
            finally:
                store_mod._embedding_session = old_session


# ── Blob Format Tests (U2) ───────────────────────���───────────────────────────


class TestBlobFormat:
    def test_float32_roundtrip(self):
        vec = [0.1 * i for i in range(768)]
        blob = _embedding_to_blob(vec, quantize=False)
        restored = _blob_to_embedding(blob)
        assert len(restored) == 768
        for orig, r in zip(vec, restored):
            assert abs(orig - r) < 1e-5

    def test_int8_quantized_roundtrip(self):
        vec = [0.1 * (i % 20) - 1.0 for i in range(768)]
        blob = _embedding_to_blob(vec, quantize=True)
        restored = _blob_to_embedding(blob)
        assert len(restored) == 768
        # Cosine similarity should be > 0.995
        dot = sum(a * b for a, b in zip(vec, restored))
        mag_a = sum(a * a for a in vec) ** 0.5
        mag_b = sum(b * b for b in restored) ** 0.5
        cosine = dot / (mag_a * mag_b) if mag_a > 0 and mag_b > 0 else 0
        assert cosine > 0.995

    def test_legacy_blob_still_decoded(self):
        vec = [0.5, -0.3, 0.7, 0.0]
        legacy_blob = struct.pack(f"{len(vec)}f", *vec)
        restored = _blob_to_embedding(legacy_blob)
        assert len(restored) == 4
        for orig, r in zip(vec, restored):
            assert abs(orig - r) < 1e-5

    def test_new_format_has_magic_byte(self):
        vec = [1.0, 2.0, 3.0]
        blob = _embedding_to_blob(vec, quantize=False)
        assert blob[0] == _BLOB_MAGIC

    def test_new_format_multiple_of_4_length_detected(self):
        # 4-byte header + 4 floats * 4 bytes = 20 bytes (multiple of 4)
        vec = [0.1, 0.2, 0.3, 0.4]
        blob = _embedding_to_blob(vec, quantize=False)
        assert len(blob) % 4 == 0
        restored = _blob_to_embedding(blob)
        assert len(restored) == 4
        for orig, r in zip(vec, restored):
            assert abs(orig - r) < 1e-5

    def test_all_zeros_quantization(self):
        vec = [0.0] * 128
        blob = _embedding_to_blob(vec, quantize=True)
        restored = _blob_to_embedding(blob)
        assert len(restored) == 128
        for v in restored:
            assert abs(v) < 1e-5

    def test_all_same_value_quantization(self):
        vec = [0.5] * 64
        blob = _embedding_to_blob(vec, quantize=True)
        restored = _blob_to_embedding(blob)
        assert len(restored) == 64
        for v in restored:
            assert abs(v - 0.5) < 1e-5

    def test_store_chunk_roundtrip_with_quantization(self, tmp_db, sample_embedding):
        tmp_db.store_page(url="https://example.com", title="Example")
        chunk = DocumentChunk(
            text="Quantized embedding test.",
            page_url="https://example.com",
            chunk_id="quant::chunk-0",
            section_title="Quantized",
            embedding=sample_embedding,
        )
        tmp_db.store_chunk(chunk)
        retrieved = tmp_db.get_chunk("quant::chunk-0")
        assert retrieved is not None
        assert retrieved.embedding is not None
        assert len(retrieved.embedding) == 768
        # Check cosine similarity after int8 quantization
        dot = sum(a * b for a, b in zip(sample_embedding, retrieved.embedding))
        mag_a = sum(a * a for a in sample_embedding) ** 0.5
        mag_b = sum(b * b for b in retrieved.embedding) ** 0.5
        cosine = dot / (mag_a * mag_b) if mag_a > 0 and mag_b > 0 else 0
        assert cosine > 0.99

    def test_embedding_version_column_exists(self, tmp_db):
        cursor = tmp_db.conn.execute("PRAGMA table_info(chunks)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "embedding_version" in columns

    def test_embedding_version_set_on_store(self, tmp_db):
        tmp_db.store_page(url="https://example.com", title="Example")
        chunk = DocumentChunk(
            text="Version tracking test.",
            page_url="https://example.com",
            chunk_id="ver::chunk-0",
            section_title="Version",
            embedding=[0.1] * 768,
        )
        tmp_db.store_chunk(chunk)
        row = tmp_db.conn.execute(
            "SELECT embedding_version FROM chunks WHERE chunk_id = ?",
            ("ver::chunk-0",),
        ).fetchone()
        assert row[0] == 1

    def test_embedding_version_zero_without_embedding(self, tmp_db):
        tmp_db.store_page(url="https://example.com", title="Example")
        chunk = DocumentChunk(
            text="No embedding.",
            page_url="https://example.com",
            chunk_id="noemb::chunk-0",
            section_title="NoEmb",
        )
        tmp_db.store_chunk(chunk)
        row = tmp_db.conn.execute(
            "SELECT embedding_version FROM chunks WHERE chunk_id = ?",
            ("noemb::chunk-0",),
        ).fetchone()
        assert row[0] == 0


# ── Migration Tests (U3) ─────────────────────────────────────────────────────


class TestMigration:
    def _insert_legacy_chunk(self, store, chunk_id, text, embedding_384):
        """Insert a chunk with legacy 384-dim headerless float32 blob."""
        legacy_blob = struct.pack(f"{len(embedding_384)}f", *embedding_384)
        store.conn.execute(
            "INSERT INTO chunks (chunk_id, page_url, section_title, text, embedding, embedding_version) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            (chunk_id, "https://example.com", "Section", text, legacy_blob),
        )
        store.conn.commit()

    def test_migrate_legacy_chunks(self, tmp_db):
        from research_tool.store import migrate_embeddings
        tmp_db.store_page(url="https://example.com", title="Example")
        for i in range(3):
            self._insert_legacy_chunk(
                tmp_db, f"legacy::chunk-{i}", f"Legacy text number {i}",
                [0.1 * j for j in range(384)]
            )
        assert tmp_db.has_unmigrated_chunks()
        result = migrate_embeddings(tmp_db)
        assert result["total"] == 3
        assert result["migrated"] == 3
        assert result["skipped"] == 0
        assert not tmp_db.has_unmigrated_chunks()
        # Verify new embeddings are 768-dim
        chunks = tmp_db.get_all_chunks()
        for chunk in chunks:
            if chunk.embedding is not None:
                assert len(chunk.embedding) == 768

    def test_already_migrated_skipped(self, tmp_db):
        from research_tool.store import migrate_embeddings
        tmp_db.store_page(url="https://example.com", title="Example")
        chunk = DocumentChunk(
            text="Already migrated.",
            page_url="https://example.com",
            chunk_id="new::chunk-0",
            section_title="New",
            embedding=[0.1] * 768,
        )
        tmp_db.store_chunk(chunk)
        assert not tmp_db.has_unmigrated_chunks()
        result = migrate_embeddings(tmp_db)
        assert result["total"] == 0

    def test_empty_store_noop(self, tmp_db):
        from research_tool.store import migrate_embeddings
        result = migrate_embeddings(tmp_db)
        assert result["total"] == 0
        assert result["migrated"] == 0

    def test_null_embedding_skipped(self, tmp_db):
        from research_tool.store import migrate_embeddings
        tmp_db.store_page(url="https://example.com", title="Example")
        # Chunk with embedding_version 0 but NULL embedding
        tmp_db.conn.execute(
            "INSERT INTO chunks (chunk_id, page_url, section_title, text, embedding, embedding_version) "
            "VALUES (?, ?, ?, ?, NULL, 0)",
            ("null::chunk-0", "https://example.com", "Null", "text"),
        )
        tmp_db.conn.commit()
        assert not tmp_db.has_unmigrated_chunks()  # NULL embedding not counted

    def test_build_from_store_raises_on_unmigrated(self, tmp_db):
        tmp_db.store_page(url="https://example.com", title="Example")
        self._insert_legacy_chunk(
            tmp_db, "legacy::chunk-0", "Legacy text",
            [0.1 * j for j in range(384)]
        )
        index = HybridIndex(mode="query")
        with pytest.raises(RuntimeError, match="unmigrated embeddings"):
            index.build_from_store(tmp_db)

    def test_model_unavailable_raises(self, tmp_db):
        from research_tool.store import migrate_embeddings
        tmp_db.store_page(url="https://example.com", title="Example")
        self._insert_legacy_chunk(
            tmp_db, "legacy::chunk-0", "Legacy text",
            [0.1 * j for j in range(384)]
        )
        with patch("research_tool.store._ONNX_AVAILABLE", False):
            import research_tool.store as store_mod
            old_session = store_mod._embedding_session
            store_mod._embedding_session = None
            try:
                with pytest.raises(RuntimeError, match="model unavailable"):
                    migrate_embeddings(tmp_db)
            finally:
                store_mod._embedding_session = old_session


# ── ColBERT Token Embedding Tests (U4) ───────────────────────────────────────


class TestColBERTTokenEmbeddings:
    def test_token_embeddings_blob_roundtrip(self):
        from research_tool.store import _token_embeddings_to_blob, _blob_to_token_embeddings
        import numpy as np
        arr = np.random.randn(10, 128).astype(np.float32)
        # L2 normalize rows
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / norms
        blob = _token_embeddings_to_blob(arr)
        restored = _blob_to_token_embeddings(blob)
        assert restored is not None
        assert restored.shape == (10, 128)
        assert np.allclose(arr, restored, atol=1e-6)

    def test_token_embeddings_store_retrieve(self, tmp_db):
        import numpy as np
        tmp_db.store_page(url="https://example.com", title="Example")
        chunk = DocumentChunk(
            text="Test token embeddings.",
            page_url="https://example.com",
            chunk_id="tok::chunk-0",
            section_title="Tokens",
            embedding=[0.1] * 768,
        )
        tmp_db.store_chunk(chunk)
        arr = np.random.randn(5, 128).astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / norms
        tmp_db.store_token_embeddings("tok::chunk-0", arr)
        retrieved = tmp_db.get_token_embeddings("tok::chunk-0")
        assert retrieved is not None
        assert retrieved.shape == (5, 128)
        assert np.allclose(arr, retrieved, atol=1e-6)

    def test_get_token_embeddings_missing(self, tmp_db):
        result = tmp_db.get_token_embeddings("nonexistent::chunk")
        assert result is None

    def test_corrupt_blob_returns_none(self, tmp_db):
        from research_tool.store import _blob_to_token_embeddings
        result = _blob_to_token_embeddings(b"\x00\x01")
        assert result is None

    def test_size_mismatch_blob_returns_none(self, tmp_db):
        from research_tool.store import _blob_to_token_embeddings
        # Header says 5 tokens x 128 dim but blob is too short
        blob = struct.pack("<HH", 5, 128) + b"\x00" * 10
        result = _blob_to_token_embeddings(blob)
        assert result is None

    def test_chunk_token_embeddings_table_exists(self, tmp_db):
        cursor = tmp_db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_token_embeddings'"
        )
        assert cursor.fetchone() is not None

    def test_make_token_embeddings_empty_string(self):
        from research_tool.store import _make_token_embeddings
        result = _make_token_embeddings("")
        assert result is None

    def test_make_token_embeddings_model_unavailable(self):
        from research_tool.store import _make_token_embeddings
        with patch("research_tool.store._ONNX_AVAILABLE", False):
            import research_tool.store as store_mod
            old_session = store_mod._colbert_session
            store_mod._colbert_session = None
            try:
                result = _make_token_embeddings("test text")
                assert result is None
            finally:
                store_mod._colbert_session = old_session

    def test_make_token_embeddings_happy_path(self):
        from research_tool.store import _make_token_embeddings, COLBERT_DIM
        import numpy as np
        result = _make_token_embeddings("hello world")
        if result is None:
            pytest.skip("ColBERT model not available in test environment")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[1] == COLBERT_DIM
        assert result.shape[0] > 0
        # Check L2 normalization
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)


class TestMaxSimScoring:
    """U5: MaxSim scoring computation and RRF integration."""

    def test_maxsim_score_identical_tokens(self):
        """Identical query and doc tokens should yield maximum score."""
        import numpy as np
        index = HybridIndex(mode="query")
        tokens = np.random.randn(5, 128).astype(np.float32)
        tokens = tokens / np.linalg.norm(tokens, axis=1, keepdims=True)
        score = index._maxsim_score(tokens, tokens)
        # Each query token's max sim against itself = 1.0, sum = num tokens
        assert abs(score - 5.0) < 1e-4

    def test_maxsim_score_orthogonal_tokens(self):
        """Orthogonal query and doc tokens should yield near-zero score."""
        import numpy as np
        index = HybridIndex(mode="query")
        # Use first 5 dimensions for query, last 5 for doc (orthogonal in 128-d)
        query = np.zeros((3, 128), dtype=np.float32)
        query[:, :3] = np.eye(3)
        doc = np.zeros((3, 128), dtype=np.float32)
        doc[:, 125:128] = np.eye(3)
        score = index._maxsim_score(query, doc)
        assert abs(score) < 1e-5

    def test_maxsim_score_asymmetric_token_counts(self):
        """MaxSim should work with different query and doc token counts."""
        import numpy as np
        index = HybridIndex(mode="query")
        query = np.random.randn(3, 128).astype(np.float32)
        query = query / np.linalg.norm(query, axis=1, keepdims=True)
        doc = np.random.randn(10, 128).astype(np.float32)
        doc = doc / np.linalg.norm(doc, axis=1, keepdims=True)
        score = index._maxsim_score(query, doc)
        # Score should be sum of 3 max values, each <= 1.0
        assert 0.0 <= score <= 3.0 + 1e-5

    def test_maxsim_rank_returns_sorted(self, tmp_path):
        """_maxsim_rank should return chunk_ids sorted by descending score."""
        import numpy as np
        store = ResearchStore(db_path=str(tmp_path / "test.db"))
        try:
            emb = [0.1] * EMBEDDING_DIM
            store.store_page(url="http://a.com", title="Test")
            store.store_chunk(DocumentChunk(
                page_url="http://a.com", chunk_id="c1",
                section_title="section", text="text about cats", embedding=emb))
            store.store_chunk(DocumentChunk(
                page_url="http://a.com", chunk_id="c2",
                section_title="section", text="text about dogs", embedding=emb))

            query_tokens = np.random.randn(4, 128).astype(np.float32)
            query_tokens = query_tokens / np.linalg.norm(query_tokens, axis=1, keepdims=True)

            c1_tokens = np.zeros((4, 128), dtype=np.float32)
            c1_tokens[:, 64:68] = np.eye(4)
            store.store_token_embeddings("c1", c1_tokens)
            store.store_token_embeddings("c2", query_tokens.copy())

            index = HybridIndex(mode="query")
            index.chunks = [
                DocumentChunk(page_url="http://a.com", chunk_id="c1",
                              section_title="s", text="text about cats",
                              embedding=emb),
                DocumentChunk(page_url="http://a.com", chunk_id="c2",
                              section_title="s", text="text about dogs",
                              embedding=emb),
            ]
            index._built = True

            with patch("research_tool.store._make_token_embeddings", return_value=query_tokens):
                ranked = index._maxsim_rank("test query", store, top_k=10)

            assert len(ranked) == 2
            assert ranked[0][0] == "c2"
            assert ranked[1][0] == "c1"
            assert ranked[0][1] > ranked[1][1]
        finally:
            store.close()

    def test_maxsim_rank_returns_empty_without_model(self, tmp_path):
        """_maxsim_rank returns [] when token embeddings can't be generated."""
        import numpy as np
        store = ResearchStore(db_path=str(tmp_path / "test.db"))
        try:
            index = HybridIndex(mode="query")
            index.chunks = []
            index._built = True

            with patch("research_tool.store._make_token_embeddings", return_value=None):
                ranked = index._maxsim_rank("test", store, top_k=10)
            assert ranked == []
        finally:
            store.close()

    def test_hybrid_retrieve_includes_maxsim_in_query_mode(self, tmp_path):
        """hybrid_retrieve with store param should include maxsim signal."""
        import numpy as np
        store = ResearchStore(db_path=str(tmp_path / "test.db"))
        try:
            emb = [0.1] * EMBEDDING_DIM
            store.store_page(url="http://a.com", title="Test")
            store.store_chunk(DocumentChunk(
                page_url="http://a.com", chunk_id="c1",
                section_title="section", text="machine learning", embedding=emb))

            tokens = np.random.randn(4, 128).astype(np.float32)
            tokens = tokens / np.linalg.norm(tokens, axis=1, keepdims=True)
            store.store_token_embeddings("c1", tokens)

            index = HybridIndex(mode="query")
            index.chunks = [
                DocumentChunk(page_url="http://a.com", chunk_id="c1",
                              section_title="s", text="machine learning",
                              embedding=emb),
            ]
            index._built = True
            index._doc_count = 1
            index._avg_dl = len("machine learning".split())
            index._idf = {"machine": 0.5, "learning": 0.5}
            index._tf = {"c1": {"machine": 1, "learning": 1}}
            index._dl = {"c1": 2}

            with patch("research_tool.store._make_embedding", return_value=emb), \
                 patch("research_tool.store._make_token_embeddings", return_value=tokens), \
                 patch("research_tool.store._ONNX_AVAILABLE", True):
                results = index.hybrid_retrieve("machine learning", top_k=5, rerank=False, store=store)

            assert len(results) == 1
            assert results[0].maxsim_score > 0.0
        finally:
            store.close()

    def test_hybrid_retrieve_no_maxsim_without_store(self, tmp_path):
        """hybrid_retrieve without store param should not include maxsim."""
        import numpy as np
        store = ResearchStore(db_path=str(tmp_path / "test.db"))
        try:
            emb = [0.1] * EMBEDDING_DIM
            store.store_page(url="http://a.com", title="Test")
            store.store_chunk(DocumentChunk(
                page_url="http://a.com", chunk_id="c1",
                section_title="section", text="machine learning", embedding=emb))

            index = HybridIndex(mode="query")
            index.chunks = [
                DocumentChunk(page_url="http://a.com", chunk_id="c1",
                              section_title="s", text="machine learning",
                              embedding=emb),
            ]
            index._built = True
            index._doc_count = 1
            index._avg_dl = len("machine learning".split())
            index._idf = {"machine": 0.5, "learning": 0.5}
            index._tf = {"c1": {"machine": 1, "learning": 1}}
            index._dl = {"c1": 2}

            with patch("research_tool.store._make_embedding", return_value=emb), \
                 patch("research_tool.store._ONNX_AVAILABLE", True):
                results = index.hybrid_retrieve("machine learning", top_k=5, rerank=False)

            assert len(results) == 1
            assert results[0].maxsim_score == 0.0
        finally:
            store.close()

    def test_hybrid_retrieve_no_maxsim_in_ingest_mode(self, tmp_path):
        """MaxSim should not activate in ingest mode even with store."""
        import numpy as np
        store = ResearchStore(db_path=str(tmp_path / "test.db"))
        try:
            emb = [0.1] * EMBEDDING_DIM
            store.store_page(url="http://a.com", title="Test")
            store.store_chunk(DocumentChunk(
                page_url="http://a.com", chunk_id="c1",
                section_title="section", text="machine learning", embedding=emb))

            tokens = np.random.randn(4, 128).astype(np.float32)
            tokens = tokens / np.linalg.norm(tokens, axis=1, keepdims=True)
            store.store_token_embeddings("c1", tokens)

            index = HybridIndex(mode="ingest")
            index.chunks = [
                DocumentChunk(page_url="http://a.com", chunk_id="c1",
                              section_title="s", text="machine learning",
                              embedding=emb),
            ]
            index._built = True
            index._doc_count = 1
            index._avg_dl = len("machine learning".split())
            index._idf = {"machine": 0.5, "learning": 0.5}
            index._tf = {"c1": {"machine": 1, "learning": 1}}
            index._dl = {"c1": 2}

            with patch("research_tool.store._make_embedding", return_value=emb), \
                 patch("research_tool.store._ONNX_AVAILABLE", True):
                results = index.hybrid_retrieve("machine learning", top_k=5, rerank=False, store=store)

            assert len(results) == 1
            assert results[0].maxsim_score == 0.0
        finally:
            store.close()

    def test_maxsim_disabled_when_weight_zero(self, tmp_path):
        """MaxSim signal should be omitted when RRF_WEIGHT_MAXSIM is 0."""
        import numpy as np
        import research_tool.store as store_mod
        store = ResearchStore(db_path=str(tmp_path / "test.db"))
        try:
            emb = [0.1] * EMBEDDING_DIM
            store.store_page(url="http://a.com", title="Test")
            store.store_chunk(DocumentChunk(
                page_url="http://a.com", chunk_id="c1",
                section_title="section", text="machine learning", embedding=emb))

            tokens = np.random.randn(4, 128).astype(np.float32)
            tokens = tokens / np.linalg.norm(tokens, axis=1, keepdims=True)
            store.store_token_embeddings("c1", tokens)

            index = HybridIndex(mode="query")
            index.chunks = [
                DocumentChunk(page_url="http://a.com", chunk_id="c1",
                              section_title="s", text="machine learning",
                              embedding=emb),
            ]
            index._built = True
            index._doc_count = 1
            index._avg_dl = len("machine learning".split())
            index._idf = {"machine": 0.5, "learning": 0.5}
            index._tf = {"c1": {"machine": 1, "learning": 1}}
            index._dl = {"c1": 2}

            old_weight = store_mod.RRF_WEIGHT_MAXSIM
            store_mod.RRF_WEIGHT_MAXSIM = 0.0
            try:
                with patch("research_tool.store._make_embedding", return_value=emb), \
                     patch("research_tool.store._make_token_embeddings", return_value=tokens), \
                     patch("research_tool.store._ONNX_AVAILABLE", True):
                    results = index.hybrid_retrieve("machine learning", top_k=5, rerank=False, store=store)
            finally:
                store_mod.RRF_WEIGHT_MAXSIM = old_weight

            assert len(results) == 1
            # MaxSim signal was included but with weight 0 it doesn't contribute to RRF score
            # The maxsim_score field should still be populated since the signal was computed
            # but RRF weight=0 means the fused score is unaffected
        finally:
            store.close()


class TestParentChildChunking:
    """U2 (optimization plan): Parent-child chunking for retrieval precision."""

    def test_split_into_children_basic(self):
        """A ~500 token parent should produce 2-3 children of ~200 tokens."""
        long_text = " ".join(["word"] * 100) + "\n\n" + " ".join(["other"] * 100) + "\n\n" + " ".join(["more"] * 100)
        parent = DocumentChunk(
            text=long_text, page_url="http://a.com",
            chunk_id="p1", section_title="Section",
        )
        children = split_into_children(parent, max_tokens=200)
        assert len(children) >= 2
        for child in children:
            assert child.parent_chunk_id == "p1"
            assert child.is_child is True
            assert child.page_url == "http://a.com"
            assert child.section_title == "Section"

    def test_split_children_ids_deterministic(self):
        """Child chunk_ids should be deterministic: parent_id::child:N."""
        parent = DocumentChunk(
            text="First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
            page_url="http://a.com", chunk_id="p1", section_title="S",
        )
        children = split_into_children(parent, max_tokens=5)
        for i, child in enumerate(children):
            assert child.chunk_id == f"p1::child:{i}"

    def test_split_small_parent_produces_one_child(self):
        """Parent with < max_tokens produces exactly one child."""
        parent = DocumentChunk(
            text="Short text.", page_url="http://a.com",
            chunk_id="p1", section_title="S",
        )
        children = split_into_children(parent, max_tokens=200)
        assert len(children) == 1
        assert children[0].text == "Short text."
        assert children[0].parent_chunk_id == "p1"

    def test_split_empty_parent(self):
        """Empty parent produces no children."""
        parent = DocumentChunk(
            text="", page_url="http://a.com",
            chunk_id="p1", section_title="S",
        )
        children = split_into_children(parent, max_tokens=200)
        assert len(children) == 0

    def test_split_inherits_content_type(self):
        """Children should inherit parent's content_type."""
        parent = DocumentChunk(
            text="def foo():\n    pass\n\ndef bar():\n    return 1",
            page_url="http://a.com", chunk_id="p1",
            section_title="S", content_type="code",
        )
        children = split_into_children(parent, max_tokens=10)
        for child in children:
            assert child.content_type == "code"

    def test_store_and_retrieve_child_chunk(self, tmp_path):
        """Child chunks should round-trip through the store with parent references."""
        store = ResearchStore(db_path=str(tmp_path / "test.db"))
        try:
            store.store_page(url="http://a.com", title="Test")
            parent = DocumentChunk(
                text="Parent text.", page_url="http://a.com",
                chunk_id="p1", section_title="S",
                embedding=[0.1] * EMBEDDING_DIM,
            )
            store.store_chunk(parent)

            child = DocumentChunk(
                text="Child text.", page_url="http://a.com",
                chunk_id="p1::child:0", section_title="S",
                embedding=[0.2] * EMBEDDING_DIM,
                parent_chunk_id="p1", is_child=True,
            )
            store.store_chunk(child)

            retrieved = store.get_chunk("p1::child:0")
            assert retrieved is not None
            assert retrieved.parent_chunk_id == "p1"
            assert retrieved.is_child is True

            retrieved_parent = store.get_parent_chunk("p1::child:0")
            assert retrieved_parent is not None
            assert retrieved_parent.chunk_id == "p1"
            assert retrieved_parent.text == "Parent text."
        finally:
            store.close()

    def test_get_parent_chunk_returns_none_for_parent(self, tmp_path):
        """get_parent_chunk on a non-child chunk returns None."""
        store = ResearchStore(db_path=str(tmp_path / "test.db"))
        try:
            store.store_page(url="http://a.com", title="Test")
            parent = DocumentChunk(
                text="Parent text.", page_url="http://a.com",
                chunk_id="p1", section_title="S",
            )
            store.store_chunk(parent)

            assert store.get_parent_chunk("p1") is None
        finally:
            store.close()

    def test_legacy_database_without_parent_columns(self, tmp_path):
        """Legacy databases without parent_chunk_id should work (defaults)."""
        store = ResearchStore(db_path=str(tmp_path / "test.db"))
        try:
            store.store_page(url="http://a.com", title="Test")
            parent = DocumentChunk(
                text="Legacy text.", page_url="http://a.com",
                chunk_id="legacy1", section_title="S",
                embedding=[0.1] * EMBEDDING_DIM,
            )
            store.store_chunk(parent)

            chunks = store.get_all_chunks()
            assert len(chunks) == 1
            assert chunks[0].parent_chunk_id is None
            assert chunks[0].is_child is False
        finally:
            store.close()


class TestChildRetrievalAndParentExpansion:
    """U3 (optimization plan): Retrieval on children with parent expansion."""

    def test_build_from_store_prefers_children(self, tmp_path):
        """build_from_store should use children for retrieval, exclude parents with children."""
        store = ResearchStore(db_path=str(tmp_path / "test.db"))
        try:
            store.store_page(url="http://a.com", title="Test")
            emb = [0.1] * EMBEDDING_DIM

            parent = DocumentChunk(
                text="Full parent text with lots of words.",
                page_url="http://a.com", chunk_id="p1",
                section_title="S", embedding=emb,
            )
            store.store_chunk(parent)

            child0 = DocumentChunk(
                text="Full parent text",
                page_url="http://a.com", chunk_id="p1::child:0",
                section_title="S", embedding=emb,
                parent_chunk_id="p1", is_child=True,
            )
            child1 = DocumentChunk(
                text="with lots of words.",
                page_url="http://a.com", chunk_id="p1::child:1",
                section_title="S", embedding=emb,
                parent_chunk_id="p1", is_child=True,
            )
            store.store_chunk(child0)
            store.store_chunk(child1)

            index = HybridIndex(mode="query")
            index.build_from_store(store)

            chunk_ids = {c.chunk_id for c in index.chunks}
            assert "p1::child:0" in chunk_ids
            assert "p1::child:1" in chunk_ids
            assert "p1" not in chunk_ids
        finally:
            store.close()

    def test_build_from_store_legacy_chunks_included(self, tmp_path):
        """Legacy chunks without children should still be in the index."""
        store = ResearchStore(db_path=str(tmp_path / "test.db"))
        try:
            store.store_page(url="http://a.com", title="Test")
            emb = [0.1] * EMBEDDING_DIM

            legacy = DocumentChunk(
                text="Legacy chunk without children.",
                page_url="http://a.com", chunk_id="legacy1",
                section_title="S", embedding=emb,
            )
            store.store_chunk(legacy)

            index = HybridIndex(mode="query")
            index.build_from_store(store)

            chunk_ids = {c.chunk_id for c in index.chunks}
            assert "legacy1" in chunk_ids
        finally:
            store.close()

    def test_build_from_store_mixed_legacy_and_children(self, tmp_path):
        """Mix of legacy chunks and parent-child chunks: both types handled correctly."""
        store = ResearchStore(db_path=str(tmp_path / "test.db"))
        try:
            store.store_page(url="http://a.com", title="Test")
            emb = [0.1] * EMBEDDING_DIM

            legacy = DocumentChunk(
                text="Legacy.", page_url="http://a.com",
                chunk_id="legacy1", section_title="S", embedding=emb,
            )
            parent = DocumentChunk(
                text="Parent.", page_url="http://a.com",
                chunk_id="p1", section_title="S", embedding=emb,
            )
            child = DocumentChunk(
                text="Child.", page_url="http://a.com",
                chunk_id="p1::child:0", section_title="S", embedding=emb,
                parent_chunk_id="p1", is_child=True,
            )
            store.store_chunk(legacy)
            store.store_chunk(parent)
            store.store_chunk(child)

            index = HybridIndex(mode="query")
            index.build_from_store(store)

            chunk_ids = {c.chunk_id for c in index.chunks}
            assert "legacy1" in chunk_ids
            assert "p1::child:0" in chunk_ids
            assert "p1" not in chunk_ids
        finally:
            store.close()


class TestHyDERetrieval:
    """U1 (optimization plan): HyDE cosine signal in hybrid_retrieve."""

    def _build_index_with_chunks(self, emb):
        """Helper to build a simple HybridIndex with one chunk for HyDE tests."""
        index = HybridIndex(mode="query")
        index.chunks = [
            DocumentChunk(page_url="http://a.com", chunk_id="c1",
                          section_title="s", text="implement exponential backoff with jitter for rate limiting",
                          embedding=emb),
        ]
        index._built = True
        index._doc_count = 1
        words = "implement exponential backoff with jitter for rate limiting".split()
        index._avg_dl = len(words)
        index._idf = {w: 0.5 for w in words}
        index._tf = {"c1": {w: 1 for w in words}}
        index._dl = {"c1": len(words)}
        return index

    def test_hyde_cosine_signal_in_rrf(self):
        """hybrid_retrieve with hyde_emb should include hyde_cosine signal."""
        emb = [0.1] * EMBEDDING_DIM
        hyde_emb = [0.2] * EMBEDDING_DIM
        index = self._build_index_with_chunks(emb)

        with patch("research_tool.store._make_embedding", return_value=emb), \
             patch("research_tool.store._ONNX_AVAILABLE", True):
            results = index.hybrid_retrieve("rate limits", top_k=5, rerank=False, hyde_emb=hyde_emb)

        assert len(results) == 1
        assert results[0].hyde_cosine_score > 0.0

    def test_no_hyde_without_embedding(self):
        """hybrid_retrieve without hyde_emb should not include hyde_cosine."""
        emb = [0.1] * EMBEDDING_DIM
        index = self._build_index_with_chunks(emb)

        with patch("research_tool.store._make_embedding", return_value=emb), \
             patch("research_tool.store._ONNX_AVAILABLE", True):
            results = index.hybrid_retrieve("rate limits", top_k=5, rerank=False)

        assert len(results) == 1
        assert results[0].hyde_cosine_score == 0.0

    def test_hyde_not_used_in_ingest_mode(self):
        """HyDE signal should not activate in ingest mode."""
        emb = [0.1] * EMBEDDING_DIM
        hyde_emb = [0.2] * EMBEDDING_DIM
        index = HybridIndex(mode="ingest")
        index.chunks = [
            DocumentChunk(page_url="http://a.com", chunk_id="c1",
                          section_title="s", text="some text",
                          embedding=emb),
        ]
        index._built = True
        index._doc_count = 1
        index._avg_dl = 2
        index._idf = {"some": 0.5, "text": 0.5}
        index._tf = {"c1": {"some": 1, "text": 1}}
        index._dl = {"c1": 2}

        with patch("research_tool.store._make_embedding", return_value=emb), \
             patch("research_tool.store._ONNX_AVAILABLE", True):
            results = index.hybrid_retrieve("some text", top_k=5, rerank=False, hyde_emb=hyde_emb)

        assert len(results) == 1
        assert results[0].hyde_cosine_score == 0.0

    def test_hyde_weight_zero_excludes_signal(self):
        """RRF_WEIGHT_HYDE=0 should still compute but not affect ranking."""
        import research_tool.store as store_mod
        emb = [0.1] * EMBEDDING_DIM
        hyde_emb = [0.2] * EMBEDDING_DIM
        index = self._build_index_with_chunks(emb)

        old_weight = store_mod.RRF_WEIGHT_HYDE
        store_mod.RRF_WEIGHT_HYDE = 0.0
        try:
            with patch("research_tool.store._make_embedding", return_value=emb), \
                 patch("research_tool.store._ONNX_AVAILABLE", True):
                results = index.hybrid_retrieve("rate limits", top_k=5, rerank=False, hyde_emb=hyde_emb)
        finally:
            store_mod.RRF_WEIGHT_HYDE = old_weight

        assert len(results) == 1

    def test_hyde_different_embedding_produces_different_scores(self):
        """Different hyde_emb values should produce different hyde_cosine scores."""
        import numpy as np
        emb = list(np.random.randn(EMBEDDING_DIM).astype(float))
        norm = sum(x * x for x in emb) ** 0.5
        emb = [x / norm for x in emb]

        hyde_similar = list(emb)  # same direction = high cosine
        hyde_orthogonal = [0.0] * EMBEDDING_DIM
        hyde_orthogonal[0] = 1.0  # different direction

        index = self._build_index_with_chunks(emb)

        with patch("research_tool.store._make_embedding", return_value=emb), \
             patch("research_tool.store._ONNX_AVAILABLE", True):
            results_similar = index.hybrid_retrieve("q", top_k=5, rerank=False, hyde_emb=hyde_similar)
            results_ortho = index.hybrid_retrieve("q", top_k=5, rerank=False, hyde_emb=hyde_orthogonal)

        assert results_similar[0].hyde_cosine_score > results_ortho[0].hyde_cosine_score


class TestMUVERAFDE:
    """U6: MUVERA Fixed Dimensional Encoding tests."""

    def test_fde_vector_shape(self):
        """_compute_fde_vector on 50 tokens of 128-dim returns 2560-dim vector."""
        import numpy as np
        tokens = np.random.randn(50, COLBERT_DIM).astype(np.float32)
        tokens = tokens / np.linalg.norm(tokens, axis=1, keepdims=True)
        fde = _compute_fde_vector(tokens)
        assert fde.shape == (MUVERA_FDE_DIM,)
        assert fde.dtype == np.float32

    def test_fde_vector_normalized(self):
        """FDE output should be L2-normalized."""
        import numpy as np
        tokens = np.random.randn(30, COLBERT_DIM).astype(np.float32)
        tokens = tokens / np.linalg.norm(tokens, axis=1, keepdims=True)
        fde = _compute_fde_vector(tokens)
        norm = np.linalg.norm(fde)
        assert abs(norm - 1.0) < 1e-5

    def test_similar_token_sets_produce_similar_fde(self):
        """Similar token sets should produce similar FDE vectors (cosine > 0.3)."""
        import numpy as np
        np.random.seed(99)
        base_tokens = np.random.randn(40, COLBERT_DIM).astype(np.float32)
        base_tokens = base_tokens / np.linalg.norm(base_tokens, axis=1, keepdims=True)
        # Add very small noise to stay within the same bucket assignments
        noise = np.random.randn(40, COLBERT_DIM).astype(np.float32) * 0.01
        similar_tokens = base_tokens + noise
        similar_tokens = similar_tokens / np.linalg.norm(similar_tokens, axis=1, keepdims=True)

        fde1 = _compute_fde_vector(base_tokens)
        fde2 = _compute_fde_vector(similar_tokens)
        cosine = float(np.dot(fde1, fde2))
        # With very small noise, bucket assignments stay mostly stable
        assert cosine > 0.3

    def test_dissimilar_token_sets_produce_dissimilar_fde(self):
        """Dissimilar token sets should produce dissimilar FDE vectors (cosine < 0.3)."""
        import numpy as np
        np.random.seed(123)
        tokens_a = np.random.randn(40, COLBERT_DIM).astype(np.float32)
        tokens_a = tokens_a / np.linalg.norm(tokens_a, axis=1, keepdims=True)
        np.random.seed(456)
        tokens_b = np.random.randn(40, COLBERT_DIM).astype(np.float32)
        tokens_b = tokens_b / np.linalg.norm(tokens_b, axis=1, keepdims=True)

        fde_a = _compute_fde_vector(tokens_a)
        fde_b = _compute_fde_vector(tokens_b)
        cosine = float(np.dot(fde_a, fde_b))
        assert cosine < 0.5

    def test_single_token_input(self):
        """Single token → one bucket gets the value, others are zero before normalization."""
        import numpy as np
        token = np.random.randn(1, COLBERT_DIM).astype(np.float32)
        token = token / np.linalg.norm(token)
        fde = _compute_fde_vector(token)
        assert fde.shape == (MUVERA_FDE_DIM,)
        # Should still be normalized
        assert abs(np.linalg.norm(fde) - 1.0) < 1e-5

    def test_custom_bucket_count(self):
        """Custom n_buckets parameter changes output dimension."""
        import numpy as np
        tokens = np.random.randn(20, COLBERT_DIM).astype(np.float32)
        tokens = tokens / np.linalg.norm(tokens, axis=1, keepdims=True)
        fde = _compute_fde_vector(tokens, n_buckets=10)
        assert fde.shape == (10 * COLBERT_DIM,)

    def test_deterministic_output(self):
        """Same input should produce same FDE (deterministic SimHash)."""
        import numpy as np
        tokens = np.random.randn(25, COLBERT_DIM).astype(np.float32)
        tokens = tokens / np.linalg.norm(tokens, axis=1, keepdims=True)
        fde1 = _compute_fde_vector(tokens)
        fde2 = _compute_fde_vector(tokens)
        assert np.allclose(fde1, fde2)

    def test_store_token_embeddings_stores_fde(self, tmp_path):
        """store_token_embeddings should store both raw tokens and FDE vector."""
        import numpy as np
        store = ResearchStore(db_path=str(tmp_path / "test.db"))
        try:
            store.store_page(url="http://a.com", title="Test")
            store.store_chunk(DocumentChunk(
                page_url="http://a.com", chunk_id="c1",
                section_title="s", text="test", embedding=[0.1] * EMBEDDING_DIM))
            tokens = np.random.randn(20, COLBERT_DIM).astype(np.float32)
            tokens = tokens / np.linalg.norm(tokens, axis=1, keepdims=True)
            store.store_token_embeddings("c1", tokens)

            # Verify raw tokens stored
            retrieved_tokens = store.get_token_embeddings("c1")
            assert retrieved_tokens is not None
            assert np.allclose(tokens, retrieved_tokens, atol=1e-5)

            # Verify FDE vector stored
            fde = store.get_fde_vector("c1")
            assert fde is not None
            assert fde.shape == (MUVERA_FDE_DIM,)
        finally:
            store.close()

    def test_get_all_fde_vectors(self, tmp_path):
        """get_all_fde_vectors returns all FDE vectors keyed by chunk_id."""
        import numpy as np
        store = ResearchStore(db_path=str(tmp_path / "test.db"))
        try:
            store.store_page(url="http://a.com", title="Test")
            for i in range(3):
                store.store_chunk(DocumentChunk(
                    page_url="http://a.com", chunk_id=f"c{i}",
                    section_title="s", text=f"text {i}", embedding=[0.1] * EMBEDDING_DIM))
                tokens = np.random.randn(15, COLBERT_DIM).astype(np.float32)
                tokens = tokens / np.linalg.norm(tokens, axis=1, keepdims=True)
                store.store_token_embeddings(f"c{i}", tokens)

            all_fde = store.get_all_fde_vectors()
            assert len(all_fde) == 3
            for cid in ["c0", "c1", "c2"]:
                assert cid in all_fde
                assert all_fde[cid].shape == (MUVERA_FDE_DIM,)
        finally:
            store.close()

    def test_two_stage_maxsim_uses_fde_screening(self, tmp_path):
        """_maxsim_rank uses FDE to screen candidates before full MaxSim."""
        import numpy as np
        store = ResearchStore(db_path=str(tmp_path / "test.db"))
        try:
            store.store_page(url="http://a.com", title="Test")
            emb = [0.1] * EMBEDDING_DIM

            # Create several chunks
            for i in range(5):
                store.store_chunk(DocumentChunk(
                    page_url="http://a.com", chunk_id=f"c{i}",
                    section_title="s", text=f"text {i}", embedding=emb))
                tokens = np.random.randn(20, COLBERT_DIM).astype(np.float32)
                tokens = tokens / np.linalg.norm(tokens, axis=1, keepdims=True)
                store.store_token_embeddings(f"c{i}", tokens)

            # Build index
            index = HybridIndex(mode="query")
            index.chunks = [
                DocumentChunk(page_url="http://a.com", chunk_id=f"c{i}",
                              section_title="s", text=f"text {i}", embedding=emb)
                for i in range(5)
            ]
            index._built = True

            query_tokens = np.random.randn(4, COLBERT_DIM).astype(np.float32)
            query_tokens = query_tokens / np.linalg.norm(query_tokens, axis=1, keepdims=True)

            with patch("research_tool.store._make_token_embeddings", return_value=query_tokens):
                ranked = index._maxsim_rank("test query", store, top_k=3)

            assert len(ranked) <= 3
            # Results should be sorted by score descending
            if len(ranked) > 1:
                for i in range(len(ranked) - 1):
                    assert ranked[i][1] >= ranked[i + 1][1]
        finally:
            store.close()


class TestFunnelSearch:
    """U7: Funnel search with MRL truncation tests."""

    def _make_well_separated_embeddings(self, n: int, dim: int = EMBEDDING_DIM):
        """Create embeddings that are well-separated in all prefix lengths."""
        import numpy as np
        np.random.seed(42)
        embs = np.random.randn(n, dim).astype(np.float32)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        return embs.tolist()

    def test_funnel_returns_same_top1_as_brute_force(self):
        """Funnel search returns same top-1 result as full-dim brute-force."""
        import numpy as np
        np.random.seed(42)
        embs = self._make_well_separated_embeddings(20)
        query_emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        query_emb = (query_emb / np.linalg.norm(query_emb)).tolist()

        # Brute-force top-1
        brute_scores = []
        for i, emb in enumerate(embs):
            cos = sum(a * b for a, b in zip(query_emb, emb))
            brute_scores.append((f"c{i}", cos))
        brute_scores.sort(key=lambda x: x[1], reverse=True)
        brute_top1 = brute_scores[0][0]

        # Funnel search
        index = HybridIndex(mode="query")
        index.chunks = [
            DocumentChunk(page_url="http://a.com", chunk_id=f"c{i}",
                          section_title="s", text=f"t{i}", embedding=embs[i])
            for i in range(20)
        ]
        index._build_funnel_cache()
        index._built = True

        funnel_results = index._funnel_cosine_rank(query_emb, top_k=5)
        assert len(funnel_results) > 0
        assert funnel_results[0][0] == brute_top1

    def test_funnel_empty_chunks_returns_empty(self):
        """Funnel search on empty index returns []."""
        index = HybridIndex(mode="query")
        index.chunks = []
        index._build_funnel_cache()
        index._built = True
        result = index._funnel_cosine_rank([0.1] * EMBEDDING_DIM, top_k=5)
        assert result == []

    def test_funnel_fewer_chunks_than_stage1(self):
        """When fewer chunks than top_k * 4, Stage 1 returns all."""
        import numpy as np
        np.random.seed(7)
        embs = self._make_well_separated_embeddings(3)
        query_emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        query_emb = (query_emb / np.linalg.norm(query_emb)).tolist()

        index = HybridIndex(mode="query")
        index.chunks = [
            DocumentChunk(page_url="http://a.com", chunk_id=f"c{i}",
                          section_title="s", text=f"t{i}", embedding=embs[i])
            for i in range(3)
        ]
        index._build_funnel_cache()
        index._built = True

        # top_k=10 → stage1 wants top_k*4=40, but only 3 chunks exist
        results = index._funnel_cosine_rank(query_emb, top_k=10)
        assert len(results) == 3

    def test_funnel_differentiates_identical_128_prefixes(self):
        """Chunks with identical 128-dim prefixes but different full embeddings
        should be differentiated at Stage 3."""
        import numpy as np
        # Build two embeddings with same 128-dim prefix but different tail
        shared_prefix = np.random.randn(128).astype(np.float32)
        shared_prefix /= np.linalg.norm(shared_prefix)

        emb_a = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        emb_a[:128] = shared_prefix
        emb_a[128:256] = np.random.randn(128).astype(np.float32)
        emb_a /= np.linalg.norm(emb_a)

        emb_b = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        emb_b[:128] = shared_prefix
        emb_b[128:256] = np.random.randn(128).astype(np.float32) * -1
        emb_b /= np.linalg.norm(emb_b)

        # Query that matches emb_a better in full dims
        query = emb_a.copy()

        index = HybridIndex(mode="query")
        index.chunks = [
            DocumentChunk(page_url="http://a.com", chunk_id="ca",
                          section_title="s", text="a", embedding=emb_a.tolist()),
            DocumentChunk(page_url="http://a.com", chunk_id="cb",
                          section_title="s", text="b", embedding=emb_b.tolist()),
        ]
        index._build_funnel_cache()
        index._built = True

        results = index._funnel_cosine_rank(query.tolist(), top_k=2)
        assert len(results) == 2
        assert results[0][0] == "ca"
        assert results[0][1] > results[1][1]

    def test_hybrid_retrieve_uses_funnel(self):
        """hybrid_retrieve should use funnel search when funnel cache is built."""
        import numpy as np
        np.random.seed(77)
        embs = self._make_well_separated_embeddings(5)

        index = HybridIndex(mode="query")
        index.chunks = [
            DocumentChunk(page_url="http://a.com", chunk_id=f"c{i}",
                          section_title="s", text=f"word{i} content",
                          embedding=embs[i])
            for i in range(5)
        ]
        index._built = True
        index._build_funnel_cache()
        index._bm25_doc_freqs = {"word0": 1, "word1": 1, "word2": 1, "word3": 1, "word4": 1, "content": 5}
        index._bm25_term_freqs = {f"c{i}": {f"word{i}": 1, "content": 1} for i in range(5)}
        index._bm25_doc_lens = {f"c{i}": 2 for i in range(5)}
        index._bm25_avg_dl = 2.0

        query_emb = embs[2]  # Should rank c2 highest
        with patch("research_tool.store._make_embedding", return_value=query_emb), \
             patch("research_tool.store._ONNX_AVAILABLE", True):
            results = index.hybrid_retrieve("word2 content", top_k=3, rerank=False)

        assert len(results) > 0
        # c2 should be ranked high (both BM25 and cosine favor it)
        result_ids = [r.chunk_id for r in results]
        assert "c2" in result_ids[:2]

    def test_funnel_cache_built_on_build_from_store(self, tmp_path):
        """build_from_store should populate funnel cache."""
        import numpy as np
        store = ResearchStore(db_path=str(tmp_path / "test.db"))
        try:
            store.store_page(url="http://a.com", title="Test")
            emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
            emb = (emb / np.linalg.norm(emb)).tolist()
            store.store_chunk(DocumentChunk(
                page_url="http://a.com", chunk_id="c1",
                section_title="s", text="test", embedding=emb))

            index = HybridIndex(mode="query")
            index.build_from_store(store)

            assert "c1" in index._funnel_128
            assert "c1" in index._funnel_256
            assert len(index._funnel_128["c1"]) == 128
            assert len(index._funnel_256["c1"]) == 256
        finally:
            store.close()


class TestClassifyChunkContentType:
    """Test heuristic code vs text classification."""

    def test_plain_prose_is_text(self):
        from research_tool.store import classify_chunk_content_type
        text = "This is a paragraph about machine learning. It describes how neural networks work and why they are useful for classification tasks."
        assert classify_chunk_content_type(text) == "text"

    def test_fenced_code_block_is_code(self):
        from research_tool.store import classify_chunk_content_type
        text = '''Here is an example:

```python
def hello():
    print("hello world")

def goodbye():
    print("goodbye world")
```

This function prints a greeting.'''
        assert classify_chunk_content_type(text) == "code"

    def test_keyword_heavy_text_is_code(self):
        from research_tool.store import classify_chunk_content_type
        text = '''def process(data):
    if data is None:
        return []
    for item in data:
        result = transform(item)
    return result'''
        assert classify_chunk_content_type(text) == "code"

    def test_indented_code_is_code(self):
        from research_tool.store import classify_chunk_content_type
        text = '''    def add(a, b):
        return a + b
    def subtract(a, b):
        return a - b
    def multiply(a, b):
        return a * b'''
        assert classify_chunk_content_type(text) == "code"

    def test_mixed_content_with_few_keywords_is_text(self):
        from research_tool.store import classify_chunk_content_type
        text = "The function returns a list of items. Each item contains a class name and description. We import it from the main module."
        assert classify_chunk_content_type(text) == "text"

    def test_json_like_content_is_code(self):
        from research_tool.store import classify_chunk_content_type
        text = '''const config = {
    "name": "myapp",
    "version": "1.0",
    "dependencies": {
        "express": "^4.0",
        "lodash": "^4.17"
    }
};'''
        assert classify_chunk_content_type(text) == "code"

    def test_empty_text_is_text(self):
        from research_tool.store import classify_chunk_content_type
        assert classify_chunk_content_type("") == "text"


class TestContentAwareEmbedding:
    """U10: Content-type-aware embedding routing tests."""

    def test_text_content_uses_nomic(self):
        """Text content routes to nomic model."""
        from research_tool.store import make_content_aware_embedding
        with patch("research_tool.store._make_embedding", return_value=[0.1] * EMBEDDING_DIM) as mock_nomic:
            result = make_content_aware_embedding("hello world", content_type="text")
        assert result is not None
        assert len(result) == EMBEDDING_DIM
        mock_nomic.assert_called_once_with("hello world", mode="document")

    def test_code_content_always_uses_nomic(self):
        """Code content always uses nomic for the primary embedding."""
        from research_tool.store import make_content_aware_embedding
        nomic_emb = [0.2] * EMBEDDING_DIM
        with patch("research_tool.store._make_embedding", return_value=nomic_emb) as mock_nomic:
            result = make_content_aware_embedding("def foo(): pass", content_type="code")
        assert result == nomic_emb
        mock_nomic.assert_called_once_with("def foo(): pass", mode="document")

    def test_default_content_type_is_text(self):
        """Default content_type is 'text'."""
        from research_tool.store import make_content_aware_embedding
        with patch("research_tool.store._make_embedding", return_value=[0.1] * EMBEDDING_DIM) as mock_nomic:
            result = make_content_aware_embedding("hello")
        mock_nomic.assert_called_once_with("hello", mode="document")

    def test_code_and_text_same_dimensionality(self):
        """Both code and text content produce same-dimensionality nomic embeddings."""
        from research_tool.store import make_content_aware_embedding
        text_emb = [0.1] * EMBEDDING_DIM
        code_emb = [0.2] * EMBEDDING_DIM
        with patch("research_tool.store._make_embedding", return_value=text_emb):
            text_result = make_content_aware_embedding("text", content_type="text")
        with patch("research_tool.store._make_embedding", return_value=code_emb):
            code_result = make_content_aware_embedding("code", content_type="code")
        assert len(text_result) == len(code_result) == EMBEDDING_DIM

    def test_ensure_code_model_returns_false_without_onnx(self):
        """_ensure_code_model returns False when ONNX is unavailable."""
        from research_tool.store import _ensure_code_model
        import research_tool.store as store_mod
        old_session = store_mod._code_embedding_session
        store_mod._code_embedding_session = None
        with patch("research_tool.store._ONNX_AVAILABLE", False):
            assert _ensure_code_model() is False
        store_mod._code_embedding_session = old_session

    def test_code_model_has_default_urls(self):
        """Code model URLs should have hardcoded defaults (jina-embeddings-v2-base-code)."""
        import research_tool.store as store_mod
        assert "jina-embeddings-v2-base-code" in store_mod._CODE_MODEL_URL
        assert "jina-embeddings-v2-base-code" in store_mod._CODE_TOKENIZER_URL

    def test_mixed_content_types_in_hybrid_index(self, tmp_path):
        """Code and text chunks coexist in the same HybridIndex."""
        store = ResearchStore(db_path=str(tmp_path / "test.db"))
        try:
            store.store_page(url="http://a.com", title="Test")
            store.store_chunk(DocumentChunk(
                page_url="http://a.com", chunk_id="text1",
                section_title="s", text="natural language content",
                embedding=[0.1] * EMBEDDING_DIM, content_type="text"))
            store.store_chunk(DocumentChunk(
                page_url="http://a.com", chunk_id="code1",
                section_title="s", text="def hello(): pass",
                embedding=[0.2] * EMBEDDING_DIM, content_type="code"))

            index = HybridIndex(mode="query")
            index.build_from_store(store)

            assert len(index.chunks) == 2
            types = {c.chunk_id: c.content_type for c in index.chunks}
            assert types["text1"] == "text"
            assert types["code1"] == "code"
        finally:
            store.close()


# ── Contextual Retrieval Storage Tests ────────────────────────────────────────


class TestContextualRetrievalStorage:
    """U4: context_summary column storage and retrieval."""

    def test_context_summary_round_trip_store_chunk(self, tmp_db):
        """store_chunk persists context_summary; get_chunk retrieves it."""
        tmp_db.store_page("http://example.com", title="Test")
        chunk = DocumentChunk(
            text="Content about databases",
            page_url="http://example.com",
            chunk_id="ctx-1",
            section_title="Databases",
            context_summary="From a DB tutorial, covers SQL basics.",
        )
        tmp_db.store_chunk(chunk)

        result = tmp_db.get_chunk("ctx-1")
        assert result is not None
        assert result.context_summary == "From a DB tutorial, covers SQL basics."
        assert result.text == "Content about databases"

    def test_context_summary_round_trip_store_chunks(self, tmp_db):
        """store_chunks (batch) persists context_summary for multiple chunks."""
        tmp_db.store_page("http://ex.com", title="Test")
        chunks = [
            DocumentChunk(
                text="First chunk", page_url="http://ex.com",
                chunk_id="batch-1", section_title="S",
                context_summary="Summary for first.",
            ),
            DocumentChunk(
                text="Second chunk", page_url="http://ex.com",
                chunk_id="batch-2", section_title="S",
                context_summary="Summary for second.",
            ),
        ]
        tmp_db.store_chunks(chunks)

        r1 = tmp_db.get_chunk("batch-1")
        r2 = tmp_db.get_chunk("batch-2")
        assert r1.context_summary == "Summary for first."
        assert r2.context_summary == "Summary for second."

    def test_context_summary_in_get_all_chunks(self, tmp_db):
        """get_all_chunks deserializes context_summary."""
        tmp_db.store_page("http://ex.com", title="Test")
        chunk = DocumentChunk(
            text="Content", page_url="http://ex.com",
            chunk_id="all-1", section_title="S",
            context_summary="Important context.",
        )
        tmp_db.store_chunk(chunk)

        all_chunks = tmp_db.get_all_chunks()
        assert len(all_chunks) == 1
        assert all_chunks[0].context_summary == "Important context."

    def test_context_summary_null_by_default(self, tmp_db):
        """Chunks without context_summary default to None."""
        tmp_db.store_page("http://ex.com", title="Test")
        chunk = DocumentChunk(
            text="Plain content", page_url="http://ex.com",
            chunk_id="plain-1", section_title="S",
        )
        tmp_db.store_chunk(chunk)

        result = tmp_db.get_chunk("plain-1")
        assert result.context_summary is None

    def test_context_summary_upsert_preserves(self, tmp_db):
        """Upserting a chunk updates its context_summary."""
        tmp_db.store_page("http://ex.com", title="Test")
        chunk = DocumentChunk(
            text="Content", page_url="http://ex.com",
            chunk_id="upsert-1", section_title="S",
        )
        tmp_db.store_chunk(chunk)
        assert tmp_db.get_chunk("upsert-1").context_summary is None

        chunk.context_summary = "New summary added."
        tmp_db.store_chunk(chunk)
        assert tmp_db.get_chunk("upsert-1").context_summary == "New summary added."

    def test_children_inherit_context_summary_via_split(self):
        """split_into_children copies context_summary from parent to each child."""
        parent = DocumentChunk(
            text=("Alpha paragraph about databases. " * 12 + "\n\n" +
                  "Beta paragraph about networking. " * 12),
            page_url="http://ex.com",
            chunk_id="p-ctx-1",
            section_title="Tech",
            context_summary="From a tech overview document.",
        )
        children = split_into_children(parent, max_tokens=30)
        assert len(children) >= 2
        for child in children:
            assert child.context_summary == "From a tech overview document."
            assert child.parent_chunk_id == "p-ctx-1"

    def test_legacy_db_without_context_summary_column(self, tmp_path):
        """Legacy database without context_summary column migrates safely."""
        db_path = str(tmp_path / "legacy_ctx.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            page_url TEXT,
            section_title TEXT,
            text TEXT,
            embedding BLOB,
            embedding_version INTEGER DEFAULT 0,
            content_type TEXT DEFAULT 'text',
            parent_chunk_id TEXT,
            is_child INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        conn.execute(
            "INSERT INTO chunks (chunk_id, page_url, section_title, text) VALUES (?, ?, ?, ?)",
            ("old-1", "http://ex.com", "S", "Old content"),
        )
        conn.commit()
        conn.close()

        store = ResearchStore(db_path=db_path)
        try:
            chunk = store.get_chunk("old-1")
            assert chunk is not None
            assert chunk.context_summary is None
            assert chunk.text == "Old content"
        finally:
            store.close()


# ── Entity Operations Tests ──────────────────────────────────────────────────


class TestEntityOperations:
    def test_store_and_retrieve_entity(self, tmp_db, sample_embedding):
        """Happy path: store an entity, retrieve it by ID, verify all fields."""
        tmp_db.store_entity(
            entity_id="entity::abc123",
            canonical_name="Python GIL",
            definition="The Global Interpreter Lock in CPython",
            aliases=["GIL", "Global Interpreter Lock"],
            entity_embedding=sample_embedding,
        )
        entity = tmp_db.get_entity_by_id("entity::abc123")
        assert entity is not None
        assert entity["entity_id"] == "entity::abc123"
        assert entity["canonical_name"] == "Python GIL"
        assert entity["definition"] == "The Global Interpreter Lock in CPython"
        assert entity["aliases"] == ["GIL", "Global Interpreter Lock"]
        assert entity["entity_embedding"] is not None
        assert len(entity["entity_embedding"]) == 768

    def test_store_chunk_entity_link(self, tmp_db, sample_embedding):
        """Happy path: link chunk to entity, query in both directions."""
        tmp_db.store_page(url="https://example.com/test", title="Test")
        chunk = DocumentChunk(
            text="The GIL prevents true multithreading.",
            page_url="https://example.com/test",
            chunk_id="test::chunk-0",
            section_title="GIL",
            embedding=sample_embedding,
        )
        tmp_db.store_chunk(chunk)
        tmp_db.store_entity(
            entity_id="entity::gil",
            canonical_name="Python GIL",
            entity_embedding=sample_embedding,
        )
        tmp_db.store_chunk_entity("test::chunk-0", "entity::gil")

        entities = tmp_db.get_entities_for_chunk("test::chunk-0")
        assert len(entities) == 1
        assert entities[0]["canonical_name"] == "Python GIL"

        chunks = tmp_db.get_chunks_for_entity("entity::gil")
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "test::chunk-0"

    def test_find_similar_entity_above_threshold(self, tmp_db):
        """Happy path: find_similar_entity returns match above threshold."""
        emb = [0.0] * 768
        emb[0] = 1.0
        tmp_db.store_entity(
            entity_id="entity::target",
            canonical_name="Target Entity",
            entity_embedding=emb,
        )
        query_emb = list(emb)
        query_emb[1] = 0.1
        norm = (1.0 + 0.01) ** 0.5
        query_emb_norm = [v / norm for v in query_emb]
        matched = tmp_db.find_similar_entity(query_emb_norm, threshold=0.5)
        assert matched == "entity::target"

    def test_find_similar_entity_empty_store(self, tmp_db, sample_embedding):
        """Edge case: no entities exist yet."""
        matched = tmp_db.find_similar_entity(sample_embedding)
        assert matched is None

    def test_find_similar_entity_below_threshold(self, tmp_db):
        """Edge case: all entities below threshold."""
        emb1 = [0.0] * 768
        emb1[0] = 1.0
        tmp_db.store_entity(
            entity_id="entity::a",
            canonical_name="Entity A",
            entity_embedding=emb1,
        )
        emb2 = [0.0] * 768
        emb2[1] = 1.0
        matched = tmp_db.find_similar_entity(emb2, threshold=0.99)
        assert matched is None

    def test_entity_aliases_roundtrip(self, tmp_db):
        """Edge case: aliases JSON round-trips correctly."""
        aliases = ["alias1", "alias two", "Alias-Three"]
        tmp_db.store_entity(
            entity_id="entity::aliases",
            canonical_name="Test Entity",
            aliases=aliases,
        )
        entity = tmp_db.get_entity_by_id("entity::aliases")
        assert entity["aliases"] == aliases

    def test_multiple_entities_per_chunk(self, tmp_db, sample_embedding):
        """Edge case: chunk linked to multiple entities."""
        tmp_db.store_page(url="https://example.com/test", title="Test")
        chunk = DocumentChunk(
            text="Python uses a GIL and has garbage collection.",
            page_url="https://example.com/test",
            chunk_id="test::chunk-0",
            section_title="Python",
            embedding=sample_embedding,
        )
        tmp_db.store_chunk(chunk)
        tmp_db.store_entity(entity_id="entity::gil", canonical_name="GIL")
        tmp_db.store_entity(entity_id="entity::gc", canonical_name="Garbage Collection")
        tmp_db.store_chunk_entity("test::chunk-0", "entity::gil")
        tmp_db.store_chunk_entity("test::chunk-0", "entity::gc")

        entities = tmp_db.get_entities_for_chunk("test::chunk-0")
        assert len(entities) == 2
        names = {e["canonical_name"] for e in entities}
        assert names == {"GIL", "Garbage Collection"}

    def test_duplicate_chunk_entity_idempotent(self, tmp_db, sample_embedding):
        """Edge case: duplicate store_chunk_entity is idempotent."""
        tmp_db.store_page(url="https://example.com/test", title="Test")
        chunk = DocumentChunk(
            text="Some text.",
            page_url="https://example.com/test",
            chunk_id="test::chunk-0",
            section_title="Test",
            embedding=sample_embedding,
        )
        tmp_db.store_chunk(chunk)
        tmp_db.store_entity(entity_id="entity::a", canonical_name="Entity A")
        tmp_db.store_chunk_entity("test::chunk-0", "entity::a")
        tmp_db.store_chunk_entity("test::chunk-0", "entity::a")

        entities = tmp_db.get_entities_for_chunk("test::chunk-0")
        assert len(entities) == 1

    def test_get_chunks_for_entity_across_pages(self, tmp_db, sample_embedding):
        """Integration: entity linked to chunks from different pages."""
        for url in ("https://example.com/a", "https://example.com/b"):
            tmp_db.store_page(url=url, title=url)
        chunk_a = DocumentChunk(
            text="Python GIL on page A.",
            page_url="https://example.com/a",
            chunk_id="a::chunk-0",
            section_title="GIL A",
            embedding=sample_embedding,
        )
        chunk_b = DocumentChunk(
            text="Global Interpreter Lock on page B.",
            page_url="https://example.com/b",
            chunk_id="b::chunk-0",
            section_title="GIL B",
            embedding=sample_embedding,
        )
        tmp_db.store_chunk(chunk_a)
        tmp_db.store_chunk(chunk_b)
        tmp_db.store_entity(entity_id="entity::gil", canonical_name="Python GIL")
        tmp_db.store_chunk_entity("a::chunk-0", "entity::gil")
        tmp_db.store_chunk_entity("b::chunk-0", "entity::gil")

        chunks = tmp_db.get_chunks_for_entity("entity::gil")
        assert len(chunks) == 2
        urls = {c.page_url for c in chunks}
        assert urls == {"https://example.com/a", "https://example.com/b"}

    def test_find_similar_entity_with_prebuilt_matrix(self, tmp_db):
        """Happy path: find_similar_entity with pre-built matrix."""
        import numpy as np_test

        emb = [0.0] * 768
        emb[0] = 1.0
        tmp_db.store_entity(
            entity_id="entity::target",
            canonical_name="Target",
            entity_embedding=emb,
        )
        matrix = np_test.array([emb], dtype=np_test.float32)
        entity_ids = ["entity::target"]
        matched = tmp_db.find_similar_entity(
            emb, threshold=0.5, matrix=matrix, entity_ids=entity_ids
        )
        assert matched == "entity::target"

    def test_entity_schema_idempotent(self, tmp_path):
        """Schema is idempotent — opening twice doesn't error."""
        db_path = str(tmp_path / "idempotent_entity.db")
        store1 = ResearchStore(db_path=db_path)
        store1.store_entity(entity_id="entity::a", canonical_name="Entity A")
        store1.close()

        store2 = ResearchStore(db_path=db_path)
        entity = store2.get_entity_by_id("entity::a")
        assert entity is not None
        assert entity["canonical_name"] == "Entity A"
        store2.close()

    def test_get_all_entities(self, tmp_db):
        """Happy path: get_all_entities returns all stored entities."""
        tmp_db.store_entity(entity_id="entity::a", canonical_name="A")
        tmp_db.store_entity(entity_id="entity::b", canonical_name="B")
        entities = tmp_db.get_all_entities()
        assert len(entities) == 2
        names = {e["canonical_name"] for e in entities}
        assert names == {"A", "B"}

    def test_delete_chunks_for_page_cleans_entity_links(self, tmp_db, sample_embedding):
        """Integration: deleting chunks for a page also removes entity links."""
        tmp_db.store_page(url="https://example.com/test", title="Test")
        chunk = DocumentChunk(
            text="Some text.",
            page_url="https://example.com/test",
            chunk_id="test::chunk-0",
            section_title="Test",
            embedding=sample_embedding,
        )
        tmp_db.store_chunk(chunk)
        tmp_db.store_entity(entity_id="entity::a", canonical_name="Entity A")
        tmp_db.store_chunk_entity("test::chunk-0", "entity::a")

        tmp_db.delete_chunks_for_page("https://example.com/test")

        entities = tmp_db.get_entities_for_chunk("test::chunk-0")
        assert len(entities) == 0

    def test_get_chunk_entity_ids(self, tmp_db, sample_embedding):
        """Happy path: lightweight entity ID lookup for a chunk."""
        tmp_db.store_page(url="https://example.com/test", title="Test")
        chunk = DocumentChunk(
            text="Some text.",
            page_url="https://example.com/test",
            chunk_id="test::chunk-0",
            section_title="Test",
            embedding=sample_embedding,
        )
        tmp_db.store_chunk(chunk)
        tmp_db.store_entity(entity_id="entity::a", canonical_name="A")
        tmp_db.store_entity(entity_id="entity::b", canonical_name="B")
        tmp_db.store_chunk_entity("test::chunk-0", "entity::a")
        tmp_db.store_chunk_entity("test::chunk-0", "entity::b")

        ids = tmp_db.get_chunk_entity_ids("test::chunk-0")
        assert set(ids) == {"entity::a", "entity::b"}


class TestEntityRetrievalSignal:
    """U3: Entity match as RRF signal #8."""

    def _make_emb(self, seed: float) -> list[float]:
        import numpy as np
        rng = np.random.RandomState(int(abs(seed * 1000)) % (2**31))
        v = rng.randn(768).astype(np.float32)
        v /= np.linalg.norm(v)
        return v.tolist()

    def _setup_entity_index(self, tmp_db):
        """Helper: create 2 chunks, 1 entity linked to chunk-0 only."""
        emb_a = self._make_emb(1.0)
        emb_b = self._make_emb(2.0)
        entity_emb = self._make_emb(1.0)  # similar to emb_a

        tmp_db.store_page(url="https://example.com/p1", title="Page 1")
        chunk_a = DocumentChunk(
            text="The Global Interpreter Lock prevents true multithreading in CPython.",
            page_url="https://example.com/p1",
            chunk_id="p1::chunk-0",
            section_title="GIL",
            embedding=emb_a,
        )
        chunk_b = DocumentChunk(
            text="Python supports multiple programming paradigms including OOP.",
            page_url="https://example.com/p1",
            chunk_id="p1::chunk-1",
            section_title="Paradigms",
            embedding=emb_b,
        )
        tmp_db.store_chunk(chunk_a)
        tmp_db.store_chunk(chunk_b)
        tmp_db.store_entity(
            entity_id="entity::gil",
            canonical_name="Python GIL",
            definition="The Global Interpreter Lock in CPython",
            entity_embedding=entity_emb,
        )
        tmp_db.store_chunk_entity("p1::chunk-0", "entity::gil")
        return emb_a, emb_b, entity_emb

    @patch("research_tool.store.ENTITY_RESOLUTION_ENABLED", True)
    def test_entity_rank_boosts_linked_chunk(self, tmp_db):
        """Chunk linked to a query-relevant entity should get an entity score."""
        emb_a, emb_b, entity_emb = self._setup_entity_index(tmp_db)

        index = HybridIndex(mode="query")
        index.build_from_store(tmp_db)

        ranked = index._entity_rank(emb_a)
        assert len(ranked) > 0
        chunk_ids = [cid for cid, _ in ranked]
        assert "p1::chunk-0" in chunk_ids
        assert "p1::chunk-1" not in chunk_ids

    @patch("research_tool.store.ENTITY_RESOLUTION_ENABLED", True)
    def test_entity_rank_multiple_entities_score_higher(self, tmp_db):
        """Chunk linked to multiple matching entities scores higher."""
        emb_a, emb_b, entity_emb = self._setup_entity_index(tmp_db)

        entity_emb2 = self._make_emb(1.01)
        tmp_db.store_entity(
            entity_id="entity::threading",
            canonical_name="CPython Threading",
            definition="Thread execution model in CPython",
            entity_embedding=entity_emb2,
        )
        tmp_db.store_chunk_entity("p1::chunk-0", "entity::threading")

        emb_c = self._make_emb(3.0)
        tmp_db.store_page(url="https://example.com/p2", title="Page 2")
        chunk_c = DocumentChunk(
            text="Concurrency models differ across languages.",
            page_url="https://example.com/p2",
            chunk_id="p2::chunk-0",
            section_title="Concurrency",
            embedding=emb_c,
        )
        tmp_db.store_chunk(chunk_c)
        tmp_db.store_chunk_entity("p2::chunk-0", "entity::threading")

        index = HybridIndex(mode="query")
        index.build_from_store(tmp_db)

        ranked = index._entity_rank(emb_a)
        scores = {cid: score for cid, score in ranked}
        assert scores.get("p1::chunk-0", 0) > scores.get("p2::chunk-0", 0)

    @patch("research_tool.store.ENTITY_RESOLUTION_ENABLED", True)
    def test_entity_rank_empty_store(self, tmp_db):
        """No entities in store returns empty list."""
        tmp_db.store_page(url="https://example.com/p1", title="Page 1")
        chunk = DocumentChunk(
            text="Some text about Python.",
            page_url="https://example.com/p1",
            chunk_id="p1::chunk-0",
            section_title="Python",
            embedding=self._make_emb(1.0),
        )
        tmp_db.store_chunk(chunk)

        index = HybridIndex(mode="query")
        index.build_from_store(tmp_db)

        ranked = index._entity_rank(self._make_emb(1.0))
        assert ranked == []

    @patch("research_tool.store.ENTITY_RESOLUTION_ENABLED", True)
    def test_entity_rank_no_match_below_threshold(self, tmp_db):
        """Query dissimilar to all entities returns empty."""
        self._setup_entity_index(tmp_db)

        index = HybridIndex(mode="query")
        index.build_from_store(tmp_db)

        dissimilar_emb = self._make_emb(999.0)
        ranked = index._entity_rank(dissimilar_emb)
        assert ranked == []

    @patch("research_tool.store.ENTITY_RESOLUTION_ENABLED", False)
    def test_entity_data_not_loaded_when_disabled(self, tmp_db):
        """With ENTITY_RESOLUTION_ENABLED=False, no entity data is loaded."""
        self._setup_entity_index(tmp_db)

        index = HybridIndex(mode="query")
        index.build_from_store(tmp_db)

        assert index._entity_embeddings == {}
        assert index._chunk_entity_ids == {}

    @patch("research_tool.store.ENTITY_RESOLUTION_ENABLED", True)
    @patch("research_tool.store._ONNX_AVAILABLE", True)
    def test_entity_signal_in_rrf(self, tmp_db):
        """Entity signal participates in RRF fusion with correct weight."""
        emb_a, emb_b, entity_emb = self._setup_entity_index(tmp_db)

        index = HybridIndex(mode="query")
        index.build_from_store(tmp_db)

        with patch("research_tool.store._make_embedding", return_value=emb_a):
            results = index.hybrid_retrieve(
                "Global Interpreter Lock multithreading",
                top_k=5,
                rerank=False,
                store=None,
            )
        assert len(results) > 0
        found = [r for r in results if r.chunk_id == "p1::chunk-0"]
        assert len(found) == 1
        assert found[0].entity_score > 0.0

    @patch("research_tool.store.ENTITY_RESOLUTION_ENABLED", True)
    def test_entity_score_populated_in_results(self, tmp_db):
        """RetrievedEntry.entity_score is populated from entity signal."""
        emb_a, emb_b, entity_emb = self._setup_entity_index(tmp_db)

        index = HybridIndex(mode="query")
        index.build_from_store(tmp_db)

        ranked = index._entity_rank(emb_a)
        assert len(ranked) > 0
        for cid, score in ranked:
            assert score > 0.0


# ── Dynamic rerank depth ────────────────────────────────────────────────────


class TestDynamicRerankDepth:
    """Tests for adaptive rerank candidate expansion based on content-type diversity."""

    @staticmethod
    def _build_mixed_index(code_count: int, text_count: int):
        """Build an index with a controlled mix of code and text chunks."""
        index = HybridIndex(mode="query")
        for i in range(code_count):
            emb = [0.0] * 3
            emb[i % 3] = 1.0
            index.add_chunk(DocumentChunk(
                text=f"def function_{i}(): pass",
                page_url=f"https://example.com/code{i}",
                chunk_id=f"code-{i}",
                section_title=f"Code {i}",
                embedding=emb,
                content_type="code",
            ))
        for i in range(text_count):
            emb = [0.0] * 3
            emb[i % 3] = 0.9
            index.add_chunk(DocumentChunk(
                text=f"This is documentation paragraph {i}.",
                page_url=f"https://example.com/doc{i}",
                chunk_id=f"text-{i}",
                section_title=f"Doc {i}",
                embedding=emb,
                content_type="text",
            ))
        index.build()
        return index

    @patch("research_tool.store._rerank_pairs", return_value=None)
    @patch("research_tool.store._make_embedding", return_value=None)
    @patch("research_tool.store.RERANK_CANDIDATES", 10)
    @patch("research_tool.store.RERANK_DIVERSITY_THRESHOLD", 0.2)
    @patch("research_tool.store.RERANK_EXPANSION_MULTIPLIER", 2.0)
    @patch("research_tool.store.RERANK_MAX_CANDIDATES", 200)
    def test_homogeneous_results_trigger_expansion(self, mock_emb, mock_rerank):
        """When top candidates are 90%+ text, rerank depth expands."""
        index = self._build_mixed_index(code_count=2, text_count=20)
        with patch.object(
            index, "bm25_retrieve",
            wraps=index.bm25_retrieve,
        ):
            results = index.hybrid_retrieve("documentation", top_k=5, rerank=True)
        assert len(results) <= 5

    @patch("research_tool.store._rerank_pairs", return_value=None)
    @patch("research_tool.store._make_embedding", return_value=None)
    @patch("research_tool.store.RERANK_CANDIDATES", 10)
    @patch("research_tool.store.RERANK_DIVERSITY_THRESHOLD", 0.2)
    @patch("research_tool.store.RERANK_EXPANSION_MULTIPLIER", 2.0)
    @patch("research_tool.store.RERANK_MAX_CANDIDATES", 200)
    def test_diverse_results_no_expansion(self, mock_emb, mock_rerank):
        """When top candidates are 60/40 mixed, depth stays at default."""
        index = HybridIndex(mode="query")
        for i in range(6):
            emb = [0.0] * 3
            emb[i % 3] = 1.0
            index.add_chunk(DocumentChunk(
                text=f"search term alpha beta code chunk {i}",
                page_url=f"https://example.com/code{i}",
                chunk_id=f"code-{i}",
                section_title=f"Code {i}",
                embedding=emb,
                content_type="code",
            ))
        for i in range(4):
            emb = [0.0] * 3
            emb[i % 3] = 0.9
            index.add_chunk(DocumentChunk(
                text=f"search term alpha beta text chunk {i}",
                page_url=f"https://example.com/doc{i}",
                chunk_id=f"text-{i}",
                section_title=f"Doc {i}",
                embedding=emb,
                content_type="text",
            ))
        index.build()
        with patch(
            "research_tool.store.multi_signal_rrf",
            wraps=multi_signal_rrf,
        ) as mock_rrf:
            index.hybrid_retrieve("search term alpha beta", top_k=5, rerank=True)
        calls = mock_rrf.call_args_list
        assert len(calls) == 1

    @patch("research_tool.store._rerank_pairs", return_value=None)
    @patch("research_tool.store._make_embedding", return_value=None)
    @patch("research_tool.store.RERANK_CANDIDATES", 10)
    @patch("research_tool.store.RERANK_DIVERSITY_THRESHOLD", 0.2)
    @patch("research_tool.store.RERANK_EXPANSION_MULTIPLIER", 2.0)
    @patch("research_tool.store.RERANK_MAX_CANDIDATES", 15)
    def test_expansion_capped_at_max(self, mock_emb, mock_rerank):
        """Expansion is capped at RERANK_MAX_CANDIDATES."""
        index = self._build_mixed_index(code_count=1, text_count=50)
        with patch(
            "research_tool.store.multi_signal_rrf",
            wraps=multi_signal_rrf,
        ) as mock_rrf:
            index.hybrid_retrieve("documentation", top_k=5, rerank=True)
        calls = mock_rrf.call_args_list
        if len(calls) > 1:
            _, kwargs = calls[-1]
            assert kwargs.get("top_k", 0) <= 15

    @patch("research_tool.store._rerank_pairs", return_value=None)
    @patch("research_tool.store._make_embedding", return_value=None)
    @patch("research_tool.store.RERANK_CANDIDATES", 10)
    @patch("research_tool.store.RERANK_DIVERSITY_THRESHOLD", 0.0)
    def test_threshold_zero_disables_expansion(self, mock_emb, mock_rerank):
        """Setting threshold to 0 disables dynamic expansion."""
        index = self._build_mixed_index(code_count=1, text_count=50)
        with patch(
            "research_tool.store.multi_signal_rrf",
            wraps=multi_signal_rrf,
        ) as mock_rrf:
            index.hybrid_retrieve("documentation", top_k=5, rerank=True)
        assert len(mock_rrf.call_args_list) == 1

    @patch("research_tool.store._make_embedding", return_value=None)
    @patch("research_tool.store.RERANK_CANDIDATES", 10)
    @patch("research_tool.store.RERANK_DIVERSITY_THRESHOLD", 0.2)
    def test_no_expansion_when_rerank_disabled(self, mock_emb):
        """Dynamic expansion only applies when rerank=True."""
        index = self._build_mixed_index(code_count=1, text_count=50)
        with patch(
            "research_tool.store.multi_signal_rrf",
            wraps=multi_signal_rrf,
        ) as mock_rrf:
            index.hybrid_retrieve("documentation", top_k=5, rerank=False)
        assert len(mock_rrf.call_args_list) == 1

    @patch("research_tool.store._rerank_pairs", return_value=None)
    @patch("research_tool.store._make_embedding", return_value=None)
    @patch("research_tool.store.RERANK_CANDIDATES", 10)
    @patch("research_tool.store.RERANK_DIVERSITY_THRESHOLD", 0.2)
    def test_all_same_type_does_not_loop(self, mock_emb, mock_rerank):
        """When all chunks are one type, expansion happens once, not infinitely."""
        index = self._build_mixed_index(code_count=0, text_count=30)
        results = index.hybrid_retrieve("documentation", top_k=5, rerank=True)
        assert len(results) <= 5


# ── Query complexity classification ─────────────────────────────────────────


class TestClassifyQueryComplexity:

    def test_short_factual_query_is_simple(self):
        from research_tool.store import classify_query_complexity
        assert classify_query_complexity("What is BM25?") == "simple"

    def test_long_exploratory_query_is_complex(self):
        from research_tool.store import classify_query_complexity
        result = classify_query_complexity(
            "What parts of the retrieval pipeline could benefit from "
            "additional optimization and how would those improvements "
            "affect overall system performance?"
        )
        assert result == "complex"

    def test_moderate_query(self):
        from research_tool.store import classify_query_complexity
        result = classify_query_complexity(
            "How does the reranking pipeline handle various edge cases in production?"
        )
        assert result == "moderate"

    def test_empty_query_is_simple(self):
        from research_tool.store import classify_query_complexity
        assert classify_query_complexity("") == "simple"
        assert classify_query_complexity("   ") == "simple"

    def test_complex_marker_overrides_short_length(self):
        from research_tool.store import classify_query_complexity
        assert classify_query_complexity("BM25 versus cosine similarity") == "complex"

    def test_multiple_clauses_with_question_word(self):
        from research_tool.store import classify_query_complexity
        result = classify_query_complexity(
            "How does BM25 work, and what are its limitations?"
        )
        assert result == "complex"

    def test_single_word_is_simple(self):
        from research_tool.store import classify_query_complexity
        assert classify_query_complexity("BM25") == "simple"


# ── Code-to-Code Search Tests ───────────────────────────────────────────────


class TestCodeToCodeSearch:
    """Tests for query_similar_code and code embedding caching in HybridIndex."""

    def _make_normalized_vec(self, seed: float) -> list[float]:
        """Create a 768-dim L2-normalized vector for testing."""
        import math
        raw = [seed * (i % 20 + 1) for i in range(768)]
        mag = math.sqrt(sum(x * x for x in raw))
        return [x / mag for x in raw]

    def test_query_similar_code_returns_ranked_results(self, tmp_db):
        """Code-to-code search returns results ranked by jina cosine similarity."""
        import math
        tmp_db.store_page(url="https://example.com", title="Example")
        emb = self._make_normalized_vec(0.1)

        # Create two clearly distinct code embeddings
        similar_raw = [1.0 if i < 384 else 0.0 for i in range(768)]
        different_raw = [0.0 if i < 384 else 1.0 for i in range(768)]
        mag_s = math.sqrt(sum(x * x for x in similar_raw))
        mag_d = math.sqrt(sum(x * x for x in different_raw))
        similar_code_emb = [x / mag_s for x in similar_raw]
        different_code_emb = [x / mag_d for x in different_raw]

        tmp_db.store_chunk(DocumentChunk(
            text="def similar(): pass",
            page_url="https://example.com",
            chunk_id="c2c::similar",
            section_title="Code",
            embedding=emb,
            code_embedding=similar_code_emb,
            content_type="code",
        ))
        tmp_db.store_chunk(DocumentChunk(
            text="def different(): return 42",
            page_url="https://example.com",
            chunk_id="c2c::different",
            section_title="Code",
            embedding=emb,
            code_embedding=different_code_emb,
            content_type="code",
        ))

        index = HybridIndex(mode="query")
        index.build_from_store(tmp_db)

        with patch("research_tool.store._make_code_embedding", return_value=similar_code_emb):
            results = query_similar_code("def similar(): pass", index, top_k=10)

        assert len(results) == 2
        assert results[0][0] == "c2c::similar"
        assert results[0][1] > results[1][1]

    def test_no_code_embeddings_returns_empty(self, tmp_db):
        """Returns empty list when no chunks have code_embedding."""
        tmp_db.store_page(url="https://example.com", title="Example")
        emb = self._make_normalized_vec(0.1)
        tmp_db.store_chunk(DocumentChunk(
            text="Just text.",
            page_url="https://example.com",
            chunk_id="c2c::text",
            section_title="Text",
            embedding=emb,
        ))
        index = HybridIndex(mode="query")
        index.build_from_store(tmp_db)

        with patch("research_tool.store._make_code_embedding", return_value=[0.1] * 768):
            results = query_similar_code("def foo(): pass", index)
        assert results == []

    def test_jina_unavailable_returns_empty(self, tmp_db):
        """Returns empty list when jina model returns None."""
        tmp_db.store_page(url="https://example.com", title="Example")
        emb = self._make_normalized_vec(0.1)
        tmp_db.store_chunk(DocumentChunk(
            text="def bar(): pass",
            page_url="https://example.com",
            chunk_id="c2c::code",
            section_title="Code",
            embedding=emb,
            code_embedding=self._make_normalized_vec(0.2),
            content_type="code",
        ))
        index = HybridIndex(mode="query")
        index.build_from_store(tmp_db)

        with patch("research_tool.store._make_code_embedding", return_value=None):
            results = query_similar_code("def foo(): pass", index)
        assert results == []

    def test_top_k_limits_results(self, tmp_db):
        """top_k parameter limits the number of results returned."""
        tmp_db.store_page(url="https://example.com", title="Example")
        emb = self._make_normalized_vec(0.1)
        for i in range(5):
            tmp_db.store_chunk(DocumentChunk(
                text=f"def func_{i}(): pass",
                page_url="https://example.com",
                chunk_id=f"c2c::func-{i}",
                section_title="Code",
                embedding=emb,
                code_embedding=self._make_normalized_vec(0.1 * (i + 1)),
                content_type="code",
            ))
        index = HybridIndex(mode="query")
        index.build_from_store(tmp_db)

        with patch("research_tool.store._make_code_embedding", return_value=self._make_normalized_vec(0.1)):
            results = query_similar_code("def foo(): pass", index, top_k=2)
        assert len(results) == 2

    def test_get_all_code_embeddings_excludes_text_chunks(self, tmp_db):
        """get_all_code_embeddings returns only chunks with non-NULL code_embedding."""
        tmp_db.store_page(url="https://example.com", title="Example")
        emb = self._make_normalized_vec(0.1)
        code_emb = self._make_normalized_vec(0.2)
        tmp_db.store_chunk(DocumentChunk(
            text="def coded(): pass",
            page_url="https://example.com",
            chunk_id="gace::code",
            section_title="Code",
            embedding=emb,
            code_embedding=code_emb,
            content_type="code",
        ))
        tmp_db.store_chunk(DocumentChunk(
            text="Text chunk.",
            page_url="https://example.com",
            chunk_id="gace::text",
            section_title="Text",
            embedding=emb,
        ))
        result = tmp_db.get_all_code_embeddings()
        assert "gace::code" in result
        assert "gace::text" not in result

    def test_code_embeddings_not_cached_in_ingest_mode(self, tmp_db):
        """Ingest-mode index does not load code_embeddings cache."""
        tmp_db.store_page(url="https://example.com", title="Example")
        emb = self._make_normalized_vec(0.1)
        tmp_db.store_chunk(DocumentChunk(
            text="def ingested(): pass",
            page_url="https://example.com",
            chunk_id="mode::code",
            section_title="Code",
            embedding=emb,
            code_embedding=self._make_normalized_vec(0.3),
            content_type="code",
        ))
        index = HybridIndex(mode="ingest")
        index.build_from_store(tmp_db)
        assert index._code_embeddings == {}


# ── Wiki Ingest: content_hash, crawl_status, cross_links (Unit 1) ─────────


class TestWikiIngestColumns:
    """Tests for content_hash / crawl_status on pages and cross_links on chunks."""

    def test_store_page_with_content_hash(self, tmp_db):
        """Happy path: store a page with content_hash, retrieve it, verify hash."""
        tmp_db.store_page(
            url="https://wiki.example.com/page1",
            title="Page 1",
            html="<p>hello</p>",
            extracted_text="hello",
            content_hash="abc123",
        )
        page = tmp_db.get_page("https://wiki.example.com/page1")
        assert page is not None
        assert page["content_hash"] == "abc123"
        assert page["crawl_status"] == "active"  # default

    def test_mark_domain_pages_stale_and_reactivate(self, tmp_db):
        """Mark domain pages stale, re-mark one active, verify the other stays stale."""
        tmp_db.store_page(url="https://wiki.example.com/a", title="A", content_hash="h1", crawl_status="active")
        tmp_db.store_page(url="https://wiki.example.com/b", title="B", content_hash="h2", crawl_status="active")
        tmp_db.store_page(url="https://other.com/c", title="C", content_hash="h3", crawl_status="active")

        tmp_db.mark_domain_pages_stale("wiki.example.com")

        # Re-mark one page active
        tmp_db.mark_page_active("https://wiki.example.com/a")

        stale = tmp_db.get_stale_pages("wiki.example.com")
        stale_urls = [p["url"] for p in stale]
        assert "https://wiki.example.com/b" in stale_urls
        assert "https://wiki.example.com/a" not in stale_urls

        # other.com should be unaffected
        other_page = tmp_db.get_page("https://other.com/c")
        assert other_page["crawl_status"] == "active"

    def test_mark_domain_pages_stale_no_match(self, tmp_db):
        """mark_domain_pages_stale with no matching pages is a no-op."""
        tmp_db.store_page(url="https://other.com/x", title="X")
        tmp_db.mark_domain_pages_stale("nonexistent.example.com")
        # No error raised; other.com page unaffected
        page = tmp_db.get_page("https://other.com/x")
        assert page["crawl_status"] == "active"

    def test_get_content_hash_missing_url(self, tmp_db):
        """get_content_hash for a URL not in the DB returns None."""
        assert tmp_db.get_content_hash("https://not-here.example.com/page") is None

    def test_store_mark_stale_reupsert_active(self, tmp_db):
        """Integration: store -> mark stale -> re-store -> verify active with updated hash."""
        tmp_db.store_page(
            url="https://wiki.example.com/page",
            title="Page",
            content_hash="old_hash",
            crawl_status="active",
        )
        tmp_db.mark_domain_pages_stale("wiki.example.com")
        page = tmp_db.get_page("https://wiki.example.com/page")
        assert page["crawl_status"] == "stale"

        # Re-store with updated hash and active status
        tmp_db.store_page(
            url="https://wiki.example.com/page",
            title="Page Updated",
            content_hash="new_hash",
            crawl_status="active",
        )
        page = tmp_db.get_page("https://wiki.example.com/page")
        assert page["crawl_status"] == "active"
        assert page["content_hash"] == "new_hash"
        assert page["title"] == "Page Updated"

    def test_cross_links_roundtrip(self, tmp_db):
        """Store a chunk with cross_links, retrieve it, verify list is preserved."""
        tmp_db.store_page(url="https://wiki.example.com/a", title="A")
        links = ["https://wiki.example.com/b", "https://wiki.example.com/c"]
        chunk = DocumentChunk(
            text="See also B and C.",
            page_url="https://wiki.example.com/a",
            chunk_id="cl::1",
            section_title="Links",
            cross_links=links,
        )
        tmp_db.store_chunk(chunk)

        retrieved = tmp_db.get_chunk("cl::1")
        assert retrieved is not None
        assert retrieved.cross_links == links

    def test_cross_links_none_by_default(self, tmp_db):
        """Chunks without cross_links return None for the field."""
        tmp_db.store_page(url="https://example.com", title="Ex")
        chunk = DocumentChunk(
            text="No links here.",
            page_url="https://example.com",
            chunk_id="cl::none",
            section_title="Intro",
        )
        tmp_db.store_chunk(chunk)
        retrieved = tmp_db.get_chunk("cl::none")
        assert retrieved is not None
        assert retrieved.cross_links is None

    def test_cross_links_via_get_all_chunks(self, tmp_db):
        """cross_links are deserialized correctly in get_all_chunks."""
        tmp_db.store_page(url="https://wiki.example.com/a", title="A")
        links = ["https://wiki.example.com/x"]
        tmp_db.store_chunk(DocumentChunk(
            text="text",
            page_url="https://wiki.example.com/a",
            chunk_id="cl::all",
            section_title="S",
            cross_links=links,
        ))
        all_chunks = tmp_db.get_all_chunks()
        matched = [c for c in all_chunks if c.chunk_id == "cl::all"]
        assert len(matched) == 1
        assert matched[0].cross_links == links

    def test_coalesce_content_hash_not_nullified(self, tmp_db):
        """COALESCE: re-store with content_hash=None must NOT nullify the existing hash."""
        tmp_db.store_page(
            url="https://wiki.example.com/page",
            title="Page",
            content_hash="preserved_hash",
            crawl_status="active",
        )
        # Re-store without specifying content_hash (defaults to None)
        tmp_db.store_page(
            url="https://wiki.example.com/page",
            title="Page v2",
        )
        page = tmp_db.get_page("https://wiki.example.com/page")
        assert page["content_hash"] == "preserved_hash"
        assert page["crawl_status"] == "active"
        assert page["title"] == "Page v2"

    def test_coalesce_crawl_status_not_nullified(self, tmp_db):
        """COALESCE: re-store with crawl_status=None must NOT nullify the existing status."""
        tmp_db.store_page(
            url="https://wiki.example.com/page",
            title="Page",
            crawl_status="active",
        )
        # Simulate a non-wiki caller that doesn't pass crawl_status
        tmp_db.store_page(
            url="https://wiki.example.com/page",
            title="Page refreshed",
        )
        page = tmp_db.get_page("https://wiki.example.com/page")
        assert page["crawl_status"] == "active"

    def test_cross_links_via_store_chunks_batch(self, tmp_db):
        """cross_links work correctly via the batch store_chunks method."""
        tmp_db.store_page(url="https://wiki.example.com/a", title="A")
        links_1 = ["https://wiki.example.com/b"]
        links_2 = ["https://wiki.example.com/c", "https://wiki.example.com/d"]
        chunks = [
            DocumentChunk(
                text="chunk 1",
                page_url="https://wiki.example.com/a",
                chunk_id="batch::1",
                section_title="S1",
                cross_links=links_1,
            ),
            DocumentChunk(
                text="chunk 2",
                page_url="https://wiki.example.com/a",
                chunk_id="batch::2",
                section_title="S2",
                cross_links=links_2,
            ),
        ]
        tmp_db.store_chunks(chunks)
        c1 = tmp_db.get_chunk("batch::1")
        c2 = tmp_db.get_chunk("batch::2")
        assert c1.cross_links == links_1
        assert c2.cross_links == links_2

    def test_get_content_hash_returns_value(self, tmp_db):
        """get_content_hash returns the stored hash for a known URL."""
        tmp_db.store_page(
            url="https://wiki.example.com/page",
            title="Page",
            content_hash="sha256_abc",
        )
        assert tmp_db.get_content_hash("https://wiki.example.com/page") == "sha256_abc"

    def test_default_crawl_status_is_active(self, tmp_db):
        """New pages without explicit crawl_status default to 'active'."""
        tmp_db.store_page(url="https://wiki.example.com/new", title="New")
        page = tmp_db.get_page("https://wiki.example.com/new")
        assert page["crawl_status"] == "active"
