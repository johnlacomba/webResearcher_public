"""Tests for benchmark configuration matrix and runner."""

from __future__ import annotations

import pytest

from research_tool.benchmarks.configs import (
    BenchmarkConfig,
    generate_matrix,
    get_config_by_name,
)
from research_tool.benchmarks.runner import (
    normalize_text,
    text_overlap,
    text_overlap_score,
    load_corpus,
)
from research_tool.eval import (
    normalize_text as eval_normalize_text,
    text_overlap_score as eval_text_overlap_score,
)


class TestConfigMatrix:
    def test_deterministic_matrix_size(self):
        configs = generate_matrix(include_llm=False)
        assert len(configs) == 96  # 4 chunk × 4 embed × 6 retrieval

    def test_llm_matrix_adds_configs(self):
        det_only = generate_matrix(include_llm=False)
        with_llm = generate_matrix(include_llm=True)
        assert len(with_llm) > len(det_only)
        llm_configs = [c for c in with_llm if c.tier == "llm_required"]
        assert len(llm_configs) == 24  # 4 embed × 6 retrieval

    def test_all_config_names_unique(self):
        configs = generate_matrix(include_llm=True)
        names = [c.name for c in configs]
        assert len(names) == len(set(names))

    def test_config_tiers_correct(self):
        configs = generate_matrix(include_llm=True)
        for c in configs:
            assert c.tier in ("deterministic", "llm_required")

    def test_get_config_by_name(self):
        configs = generate_matrix()
        first = configs[0]
        found = get_config_by_name(first.name)
        assert found is not None
        assert found.name == first.name

    def test_get_config_by_name_missing(self):
        assert get_config_by_name("nonexistent_config") is None

    def test_bm25_only_disables_other_signals(self):
        configs = generate_matrix()
        bm25_configs = [c for c in configs if "bm25-only" in c.name]
        assert len(bm25_configs) > 0
        for c in bm25_configs:
            assert c.module_overrides.get("RRF_WEIGHT_TEXT_COSINE") == 0.0
            assert c.module_overrides.get("RRF_WEIGHT_MAXSIM") == 0.0
            assert c.module_overrides.get("HYDE_ENABLED") is False

    def test_parent_child_config_varies(self):
        configs = generate_matrix()
        pc_enabled = [c for c in configs if c.module_overrides.get("PARENT_CHILD_ENABLED") is True]
        pc_disabled = [c for c in configs if c.module_overrides.get("PARENT_CHILD_ENABLED") is False]
        assert len(pc_enabled) > 0
        assert len(pc_disabled) > 0


class TestTextOverlap:
    def test_exact_match(self):
        assert text_overlap("the quick brown fox jumps", "The Quick Brown Fox Jumps Over The Lazy Dog")

    def test_case_insensitive(self):
        assert text_overlap("HELLO WORLD EXAMPLE TEXT", "hello world example text here")

    def test_whitespace_normalization(self):
        assert text_overlap("hello   world   test  passage", "hello world test passage here")

    def test_html_entity_decoding(self):
        assert text_overlap("less than < greater than >", "less than &lt; greater than &gt;")

    def test_short_passage_rejected(self):
        assert not text_overlap("short", "this is short text", min_length=20)

    def test_no_match(self):
        assert not text_overlap("completely different text here", "nothing related at all found")

    def test_score_all_found(self):
        passages = ["hello world example text", "foo bar baz qux corge"]
        chunks = ["hello world example text is here", "foo bar baz qux corge is here too"]
        assert text_overlap_score(passages, chunks) == 1.0

    def test_score_partial(self):
        passages = ["hello world example text", "not in any chunk at all here"]
        chunks = ["hello world example text is here"]
        assert text_overlap_score(passages, chunks) == 0.5

    def test_score_empty_passages(self):
        assert text_overlap_score([], ["some text"]) == 0.0


class TestEvalTextOverlap:
    """Verify the functions added to eval.py work the same."""

    def test_normalize_text(self):
        assert eval_normalize_text("  Hello   World  ") == "hello world"

    def test_score(self):
        passages = ["hello world example text"]
        chunks = ["Hello   World   Example   Text is here"]
        assert eval_text_overlap_score(passages, chunks) == 1.0


class TestShortPassageMatching:
    """Tests for short-passage matching with word-boundary checks."""

    def test_short_passage_matches_as_whole_word(self):
        """A 7-char passage like 'PENDING' should match when it appears as a word."""
        assert text_overlap("PENDING", "status is PENDING now")

    def test_short_passage_no_match_inside_longer_word(self):
        """Short passage must NOT match as substring of a longer word."""
        assert not text_overlap("or", "move forward quickly")

    def test_passages_under_4_chars_rejected(self):
        """Passages shorter than 4 chars should be rejected."""
        assert not text_overlap("ab", "ab is here in the text")
        assert not text_overlap("abc", "abc is here in the text")

    def test_long_passages_still_work(self):
        """Passages >= 20 chars should work as before (no boundary check needed)."""
        assert text_overlap(
            "the quick brown fox jumps",
            "see the quick brown fox jumps over the lazy dog",
        )

    def test_short_passage_at_start_of_chunk(self):
        """Short passage at the start of a chunk (no preceding char) should match."""
        assert text_overlap("PENDING", "PENDING is the current status")

    def test_short_passage_at_end_of_chunk(self):
        """Short passage at the end of a chunk (no following char) should match."""
        assert text_overlap("PENDING", "the status is PENDING")

    def test_short_passage_with_punctuation_boundary(self):
        """Short passage bounded by punctuation should match."""
        assert text_overlap("PENDING", 'status="PENDING"')
        assert text_overlap("ctx.Err()", "return ctx.Err() // cancelled")

    def test_short_passage_4_chars_matches(self):
        """A 4-char passage should match with boundary check."""
        assert text_overlap("1024", "block size of 1024 bits")

    def test_short_passage_4_chars_no_match_inside_word(self):
        """A 4-char passage should not match inside a longer number/word."""
        assert not text_overlap("1024", "value is 10240 bytes")

    def test_min_length_boundary(self):
        """Exactly 4 chars should be accepted, 3 chars should be rejected."""
        assert text_overlap("abcd", "find abcd here")
        assert not text_overlap("abc", "find abc here")

    def test_short_passage_enum_values(self):
        """Enum-style values like task statuses should match."""
        assert text_overlap("RUNNING", "task is RUNNING now")
        assert text_overlap("FAILED", "the task FAILED yesterday")
        assert text_overlap("TIMED_OUT", "status changed to TIMED_OUT")

    def test_short_passage_code_identifier(self):
        """Code identifiers should match with appropriate boundaries."""
        assert text_overlap("heapq.heappush", "we use heapq.heappush to add items")

    def test_boundary_at_20_chars_no_boundary_check(self):
        """Passages of exactly 20+ chars should NOT need word-boundary checks."""
        # "forwardthinkingideas" is 20 chars and contains "or" - it should match
        # because long passages skip the boundary check
        assert text_overlap(
            "forwardthinkingideas",
            "check forwardthinkingideas in the list",
        )


class TestCorpusLoading:
    def test_load_corpus(self):
        docs, queries = load_corpus()
        assert len(docs) == 7
        assert len(queries) >= 100
        for doc in docs:
            assert doc.text
            assert doc.url
        for q in queries:
            assert q.query
            assert q.expected_passages


class TestChunkModeDifferentiation:
    """Different CHUNK_MODE values must produce measurably different chunk sets."""

    def test_paragraph_vs_wiki_produce_different_chunks(self):
        """paragraph and wiki-aware configs should produce different chunk counts/texts."""
        import tempfile
        from pathlib import Path

        from research_tool.benchmarks.configs import BenchmarkConfig
        from research_tool.benchmarks.runner import (
            CorpusDocument,
            _ingest_document,
            _save_originals,
            _apply_overrides,
            _restore_originals,
        )
        from research_tool.store import ResearchStore

        # HTML with header tags so the wiki chunker detects wiki structure
        # and splits by section, whereas the paragraph chunker treats it as
        # flat text.
        wiki_text = (
            "<h1>Introduction</h1>\n"
            "<p>This is the introduction section with some content. "
            "It has multiple sentences to form a paragraph. "
            "We are providing enough text here so the chunkers behave differently.</p>\n"
            "<h2>Background</h2>\n"
            "<p>Background information about the topic at hand. "
            "This subsection provides important context for the reader.</p>\n"
            "<h1>Methods</h1>\n"
            "<p>Here we describe the methods used in this study. "
            "The methods include several steps that are important to understand.</p>\n"
            "<h2>Data Collection</h2>\n"
            "<p>Data was collected from multiple sources over a period of time. "
            "Each source was validated and cross-referenced for accuracy.</p>\n"
            "<h1>Results</h1>\n"
            "<p>The results show significant improvement across all metrics. "
            "We observed changes across all categories of interest.</p>\n"
            "<h1>Discussion</h1>\n"
            "<p>In this section we discuss the implications of our findings. "
            "Further research is needed in this area to confirm results.</p>\n"
        )

        doc = CorpusDocument(
            path="test_wiki.html",
            url="http://bench.test/wiki_doc",
            doc_type="web",  # doc_type is web — routing must come from CHUNK_MODE
            text=wiki_text,
        )

        originals = _save_originals()

        try:
            # --- paragraph mode ---
            paragraph_cfg = BenchmarkConfig(
                name="test-paragraph",
                tier="deterministic",
                module_overrides={"PARENT_CHILD_ENABLED": False, "CHUNK_MODE": "paragraph"},
            )
            _apply_overrides(paragraph_cfg.module_overrides)
            with tempfile.TemporaryDirectory() as tmpdir:
                store_p = ResearchStore(db_path=str(Path(tmpdir) / "p.db"))
                count_p = _ingest_document(doc, store_p, paragraph_cfg)
                chunks_p = store_p.get_all_chunks()
                texts_p = sorted([c.text for c in chunks_p])
                store_p.close()

            _restore_originals(originals)

            # --- wiki mode ---
            wiki_cfg = BenchmarkConfig(
                name="test-wiki",
                tier="deterministic",
                module_overrides={"PARENT_CHILD_ENABLED": False, "CHUNK_MODE": "wiki"},
            )
            _apply_overrides(wiki_cfg.module_overrides)
            with tempfile.TemporaryDirectory() as tmpdir:
                store_w = ResearchStore(db_path=str(Path(tmpdir) / "w.db"))
                count_w = _ingest_document(doc, store_w, wiki_cfg)
                chunks_w = store_w.get_all_chunks()
                texts_w = sorted([c.text for c in chunks_w])
                store_w.close()
        finally:
            _restore_originals(originals)

        # The two modes must produce different chunk sets
        assert texts_p != texts_w, (
            f"paragraph and wiki modes produced identical chunks "
            f"(count_p={count_p}, count_w={count_w})"
        )

    def test_paragraph_vs_code_produce_different_chunks(self):
        """paragraph and ast-code configs should produce different chunk sets."""
        import tempfile
        from pathlib import Path

        from research_tool.benchmarks.configs import BenchmarkConfig
        from research_tool.benchmarks.runner import (
            CorpusDocument,
            _ingest_document,
            _save_originals,
            _apply_overrides,
            _restore_originals,
        )
        from research_tool.store import ResearchStore

        code_text = (
            "def hello():\n"
            "    print('Hello, world!')\n\n"
            "def goodbye():\n"
            "    print('Goodbye, world!')\n\n"
            "class Greeter:\n"
            "    def greet(self, name):\n"
            "        return f'Hello, {name}!'\n"
        )

        doc = CorpusDocument(
            path="test_code.py",
            url="http://bench.test/code_doc",
            doc_type="web",  # doc_type is web — routing must come from CHUNK_MODE
            language="python",
            text=code_text,
        )

        originals = _save_originals()

        try:
            # --- paragraph mode ---
            paragraph_cfg = BenchmarkConfig(
                name="test-paragraph",
                tier="deterministic",
                module_overrides={"PARENT_CHILD_ENABLED": False, "CHUNK_MODE": "paragraph"},
            )
            _apply_overrides(paragraph_cfg.module_overrides)
            with tempfile.TemporaryDirectory() as tmpdir:
                store_p = ResearchStore(db_path=str(Path(tmpdir) / "p.db"))
                count_p = _ingest_document(doc, store_p, paragraph_cfg)
                chunks_p = store_p.get_all_chunks()
                texts_p = sorted([c.text for c in chunks_p])
                store_p.close()

            _restore_originals(originals)

            # --- code mode ---
            code_cfg = BenchmarkConfig(
                name="test-code",
                tier="deterministic",
                module_overrides={"PARENT_CHILD_ENABLED": False, "CHUNK_MODE": "code"},
            )
            _apply_overrides(code_cfg.module_overrides)
            with tempfile.TemporaryDirectory() as tmpdir:
                store_c = ResearchStore(db_path=str(Path(tmpdir) / "c.db"))
                count_c = _ingest_document(doc, store_c, code_cfg)
                chunks_c = store_c.get_all_chunks()
                texts_c = sorted([c.text for c in chunks_c])
                store_c.close()
        finally:
            _restore_originals(originals)

        # The two modes must produce different chunk sets
        assert texts_p != texts_c, (
            f"paragraph and code modes produced identical chunks "
            f"(count_p={count_p}, count_c={count_c})"
        )

    def test_auto_mode_routes_by_doc_type(self):
        """CHUNK_MODE=auto should preserve current behavior: route by doc_type."""
        import tempfile
        from pathlib import Path

        from research_tool.benchmarks.configs import BenchmarkConfig
        from research_tool.benchmarks.runner import (
            CorpusDocument,
            _ingest_document,
            _save_originals,
            _apply_overrides,
            _restore_originals,
        )
        from research_tool.store import ResearchStore

        # HTML with header structure so the wiki chunker splits by section,
        # while the paragraph chunker treats it as flat text.
        wiki_text = (
            "<h1>Section One</h1>\n"
            "<p>Content for section one with details and enough words to matter. "
            "This paragraph has multiple sentences for chunking.</p>\n"
            "<h1>Section Two</h1>\n"
            "<p>Content for section two with more details and additional text. "
            "This also has multiple sentences to create meaningful chunks.</p>\n"
        )

        doc_wiki = CorpusDocument(
            path="test.html",
            url="http://bench.test/auto_wiki",
            doc_type="wiki",
            text=wiki_text,
        )

        doc_web = CorpusDocument(
            path="test.html",
            url="http://bench.test/auto_web",
            doc_type="web",
            text=wiki_text,
        )

        originals = _save_originals()

        try:
            auto_cfg = BenchmarkConfig(
                name="test-auto",
                tier="deterministic",
                module_overrides={"PARENT_CHILD_ENABLED": False, "CHUNK_MODE": "auto"},
            )
            _apply_overrides(auto_cfg.module_overrides)

            with tempfile.TemporaryDirectory() as tmpdir:
                store_wiki = ResearchStore(db_path=str(Path(tmpdir) / "wiki.db"))
                _ingest_document(doc_wiki, store_wiki, auto_cfg)
                chunks_wiki = store_wiki.get_all_chunks()
                texts_wiki = sorted([c.text for c in chunks_wiki])
                store_wiki.close()

            with tempfile.TemporaryDirectory() as tmpdir:
                store_web = ResearchStore(db_path=str(Path(tmpdir) / "web.db"))
                _ingest_document(doc_web, store_web, auto_cfg)
                chunks_web = store_web.get_all_chunks()
                texts_web = sorted([c.text for c in chunks_web])
                store_web.close()
        finally:
            _restore_originals(originals)

        # auto mode with different doc_types should produce different chunks
        assert texts_wiki != texts_web, (
            "auto mode should route wiki and web doc_types to different chunkers"
        )

    def test_chunking_configs_have_distinct_chunk_modes(self):
        """Each chunking config label should map to a unique CHUNK_MODE value."""
        from research_tool.benchmarks.configs import _chunking_configs

        configs = _chunking_configs()
        modes = [c["overrides"].get("CHUNK_MODE") for c in configs]
        # All configs must specify CHUNK_MODE
        assert all(m is not None for m in modes), (
            f"Some chunking configs lack CHUNK_MODE: {configs}"
        )
        # The four labels should have at least 3 distinct modes
        # (parent-child uses auto, same as default, but differs via PARENT_CHILD_ENABLED)
        labels_by_mode: dict[str, list[str]] = {}
        for c in configs:
            mode = c["overrides"]["CHUNK_MODE"]
            labels_by_mode.setdefault(mode, []).append(c["label"])
        # paragraph, wiki, and code must each be unique
        assert "paragraph" in [c["overrides"]["CHUNK_MODE"] for c in configs]
        assert "wiki" in [c["overrides"]["CHUNK_MODE"] for c in configs]
        assert "code" in [c["overrides"]["CHUNK_MODE"] for c in configs]


class TestEmbeddingConfigDifferentiation:
    """Embedding configs must have distinct overrides and gate embedding calls."""

    def test_embedding_configs_have_distinct_overrides(self):
        """All 4 embedding configs must have different override dictionaries."""
        from research_tool.benchmarks.configs import _embedding_configs

        configs = _embedding_configs()
        assert len(configs) == 4
        override_sets = [frozenset(c["overrides"].items()) for c in configs]
        assert len(set(override_sets)) == 4, (
            f"Embedding configs have duplicate overrides: {configs}"
        )

    def test_embedding_configs_specify_both_flags(self):
        """Each embedding config must specify both JINA and TOKEN flags."""
        from research_tool.benchmarks.configs import _embedding_configs

        for cfg in _embedding_configs():
            assert "JINA_CODE_EMBEDDING_ENABLED" in cfg["overrides"], (
                f"{cfg['label']} missing JINA_CODE_EMBEDDING_ENABLED"
            )
            assert "TOKEN_EMBEDDING_ENABLED" in cfg["overrides"], (
                f"{cfg['label']} missing TOKEN_EMBEDDING_ENABLED"
            )

    def test_nomic_only_disables_jina_and_token(self):
        """nomic-only should disable both code embedding and token embedding."""
        from research_tool.benchmarks.configs import _embedding_configs

        nomic_only = [c for c in _embedding_configs() if c["label"] == "nomic-only"]
        assert len(nomic_only) == 1
        overrides = nomic_only[0]["overrides"]
        assert overrides["JINA_CODE_EMBEDDING_ENABLED"] is False
        assert overrides["TOKEN_EMBEDDING_ENABLED"] is False

    def test_all_embeddings_enables_both(self):
        """all-embeddings should enable both code embedding and token embedding."""
        from research_tool.benchmarks.configs import _embedding_configs

        all_emb = [c for c in _embedding_configs() if c["label"] == "all-embeddings"]
        assert len(all_emb) == 1
        overrides = all_emb[0]["overrides"]
        assert overrides["JINA_CODE_EMBEDDING_ENABLED"] is True
        assert overrides["TOKEN_EMBEDDING_ENABLED"] is True

    def test_nomic_jina_enables_jina_only(self):
        """nomic+jina should enable code embedding but disable token embedding."""
        from research_tool.benchmarks.configs import _embedding_configs

        cfg = [c for c in _embedding_configs() if c["label"] == "nomic+jina"]
        assert len(cfg) == 1
        overrides = cfg[0]["overrides"]
        assert overrides["JINA_CODE_EMBEDDING_ENABLED"] is True
        assert overrides["TOKEN_EMBEDDING_ENABLED"] is False

    def test_nomic_colbert_fde_enables_token_only(self):
        """nomic+colbert+fde should disable code embedding but enable token embedding."""
        from research_tool.benchmarks.configs import _embedding_configs

        cfg = [c for c in _embedding_configs() if c["label"] == "nomic+colbert+fde"]
        assert len(cfg) == 1
        overrides = cfg[0]["overrides"]
        assert overrides["JINA_CODE_EMBEDDING_ENABLED"] is False
        assert overrides["TOKEN_EMBEDDING_ENABLED"] is True

    def test_embedding_flags_in_store_only_attrs(self):
        """Both embedding flags must be in STORE_ONLY_ATTRS."""
        from research_tool.benchmarks.configs import STORE_ONLY_ATTRS

        assert "JINA_CODE_EMBEDDING_ENABLED" in STORE_ONLY_ATTRS
        assert "TOKEN_EMBEDDING_ENABLED" in STORE_ONLY_ATTRS

    def test_store_module_has_embedding_flags(self):
        """store.py must define the two embedding flags as module-level globals."""
        import research_tool.store as store

        assert hasattr(store, "JINA_CODE_EMBEDDING_ENABLED")
        assert hasattr(store, "TOKEN_EMBEDDING_ENABLED")
        # Defaults should be True (production behavior unchanged)
        assert store.JINA_CODE_EMBEDDING_ENABLED is True
        assert store.TOKEN_EMBEDDING_ENABLED is True

    def test_nomic_only_skips_code_and_token_embeddings(self):
        """With nomic-only config, _make_code_embedding and _make_token_embeddings
        should not be called during ingestion."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from research_tool.benchmarks.configs import BenchmarkConfig
        from research_tool.benchmarks.runner import (
            CorpusDocument,
            _ingest_document,
            _save_originals,
            _apply_overrides,
            _restore_originals,
        )
        from research_tool.store import ResearchStore

        doc = CorpusDocument(
            path="test.py",
            url="http://bench.test/emb_nomic",
            doc_type="code",
            language="python",
            text="def hello():\n    return 'world'\n",
        )

        originals = _save_originals()
        try:
            cfg = BenchmarkConfig(
                name="test-nomic-only",
                tier="deterministic",
                module_overrides={
                    "PARENT_CHILD_ENABLED": False,
                    "CHUNK_MODE": "paragraph",
                    "JINA_CODE_EMBEDDING_ENABLED": False,
                    "TOKEN_EMBEDDING_ENABLED": False,
                },
            )
            _apply_overrides(cfg.module_overrides)

            with tempfile.TemporaryDirectory() as tmpdir:
                store_inst = ResearchStore(db_path=str(Path(tmpdir) / "test.db"))
                with patch(
                    "research_tool.store._make_code_embedding"
                ) as mock_code, patch(
                    "research_tool.store._make_token_embeddings"
                ) as mock_token:
                    _ingest_document(doc, store_inst, cfg)
                    mock_code.assert_not_called()
                    mock_token.assert_not_called()
                store_inst.close()
        finally:
            _restore_originals(originals)

    def test_all_embeddings_calls_code_and_token(self):
        """With all-embeddings config, _make_code_embedding and _make_token_embeddings
        should be called during ingestion."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from research_tool.benchmarks.configs import BenchmarkConfig
        from research_tool.benchmarks.runner import (
            CorpusDocument,
            _ingest_document,
            _save_originals,
            _apply_overrides,
            _restore_originals,
        )
        from research_tool.store import ResearchStore

        doc = CorpusDocument(
            path="test.py",
            url="http://bench.test/emb_all",
            doc_type="code",
            language="python",
            text="def hello():\n    return 'world'\n",
        )

        originals = _save_originals()
        try:
            cfg = BenchmarkConfig(
                name="test-all-embeddings",
                tier="deterministic",
                module_overrides={
                    "PARENT_CHILD_ENABLED": False,
                    "CHUNK_MODE": "paragraph",
                    "JINA_CODE_EMBEDDING_ENABLED": True,
                    "TOKEN_EMBEDDING_ENABLED": True,
                },
            )
            _apply_overrides(cfg.module_overrides)

            with tempfile.TemporaryDirectory() as tmpdir:
                store_inst = ResearchStore(db_path=str(Path(tmpdir) / "test.db"))
                with patch(
                    "research_tool.store._make_code_embedding",
                    return_value=None,
                ) as mock_code, patch(
                    "research_tool.store._make_token_embeddings",
                    return_value=None,
                ) as mock_token:
                    _ingest_document(doc, store_inst, cfg)
                    # _make_token_embeddings should definitely be called for all chunks
                    assert mock_token.call_count > 0, (
                        "_make_token_embeddings should be called when TOKEN_EMBEDDING_ENABLED=True"
                    )
                store_inst.close()
        finally:
            _restore_originals(originals)

    def test_matrix_embedding_overrides_propagate(self):
        """Verify that embedding overrides flow into generated BenchmarkConfig objects."""
        configs = generate_matrix(include_llm=False)

        nomic_only_cfgs = [c for c in configs if "nomic-only" in c.name]
        all_emb_cfgs = [c for c in configs if "all-embeddings" in c.name]

        assert len(nomic_only_cfgs) > 0
        assert len(all_emb_cfgs) > 0

        for c in nomic_only_cfgs:
            assert c.module_overrides.get("JINA_CODE_EMBEDDING_ENABLED") is False
            assert c.module_overrides.get("TOKEN_EMBEDDING_ENABLED") is False

        for c in all_emb_cfgs:
            assert c.module_overrides.get("JINA_CODE_EMBEDDING_ENABLED") is True
            assert c.module_overrides.get("TOKEN_EMBEDDING_ENABLED") is True
